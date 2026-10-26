"""Versioned HTTP adapter for the hydrology application services."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import importlib.util
import json
import logging
import os
from pathlib import Path
import secrets
import tempfile
import zipfile

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api_models import (ComparisonRequest, ErrorResponse, ExperimentRequest, JobResponse,
                         ReplayRequest, ScenarioRequest, RunResponse, RunsResponse, PredictionsResponse)
from .application import Application
from .devices import DEVICE_CHOICES
from .jobs import JobStore, QueueConflict
from .paved_scenario import PavedSiteConfig
from .rainfall_replay import forecast_snapshot
from .result_registry import FILTER_FIELDS, available_values, filter_runs
from .serialization import json_safe, public_config
from .server_settings import ServerSettings
from .services import basin_metadata, compare_runs, prepare_replay, replay_episodes, result_summary

log = logging.getLogger(__name__)


def create_app(settings: ServerSettings, *, application=None, store=None):
    app = FastAPI(title="Runoff physics ensemble API", version="1.0.0")
    application = application or Application(settings)
    store = store or JobStore(settings.jobs_path)
    app.state.application, app.state.jobs = application, store
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])
    bearer = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if settings.token and (credentials is None or
                               not secrets.compare_digest(credentials.credentials.encode(), settings.token.encode())):
            raise HTTPException(401, "Invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    router = APIRouter(prefix="/api/v1", dependencies=[Depends(authenticate)],
                       responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
                                  409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})

    def sanitize(message):
        message = str(message)
        roots = [str(settings.storage_dir), str(settings.results_dir)]
        for registration in settings.datasets.values():
            roots.extend(str(v) for v in asdict(registration).values() if v is not None)
        for root in sorted(roots, key=len, reverse=True):
            if root and root != "None":
                message = message.replace(root, "[server data]")
        return message

    def error(code, message, status, details=None, headers=None):
        return JSONResponse(status_code=status, headers=headers,
                            content={"code": code, "message": sanitize(message), "details": details or []})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        details = [{"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
                   for e in exc.errors()]
        return error("validation_error", "Request validation failed", 422, details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return error("http_error", exc.detail, exc.status_code, headers=exc.headers)

    @app.exception_handler(QueueConflict)
    async def conflict_error(request, exc):
        return error("conflict", str(exc), 409)

    @app.exception_handler(KeyError)
    async def missing_error(request, exc):
        return error("not_found", exc.args[0], 404)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return error("invalid_request", str(exc), 422)

    @app.exception_handler(FileNotFoundError)
    async def file_error(request, exc):
        return error("data_unavailable", "Required dataset or result file is unavailable", 422)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        log.error("API request failed", exc_info=(type(exc), exc, exc.__traceback__))
        return error("internal_error", "Unexpected server error; inspect server logs", 500)

    def job_response(job):
        job_id = job["id"]
        links = {"self": f"/api/v1/jobs/{job_id}", "cancel": f"/api/v1/jobs/{job_id}/cancel"}
        if job["result"]:
            links["result"] = f"/api/v1/jobs/{job_id}/result"
            if job["result"].get("run_id"):
                links["run"] = f"/api/v1/runs/{job['result']['run_id']}"
        return {"job_id": job_id, "kind": job["kind"], "status": job["status"],
                "progress": job["progress"], "message": sanitize(job["message"]),
                "cancel_requested": job["cancel_requested"],
                "created_at": datetime.fromtimestamp(job["created_at"], timezone.utc).isoformat(),
                "updated_at": datetime.fromtimestamp(job["updated_at"], timezone.utc).isoformat(),
                "result": json_safe(job["result"]), "error": job["error"], "links": links}

    def submit(kind, payload, key):
        if key is not None and not 1 <= len(key) <= 200:
            raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        return job_response(store.submit(kind, payload, key))

    def filters_from_json(filters):
        parsed = json.loads(filters) if filters else {}
        if not isinstance(parsed, dict) or set(parsed) - {field for field, _ in FILTER_FIELDS}:
            raise ValueError("filters must be a JSON object using supported run-filter fields")
        return parsed

    @router.get("/health")
    def health():
        return {"status": "ok", "api_version": "v1", "worker_ready": store.worker_ready(),
                "registered_datasets": len(settings.datasets)}

    @router.get("/capabilities")
    def capabilities():
        import torch
        from .regional import TRAINING_APPROACHES
        return {"devices": {"cpu": True, "cuda": torch.cuda.is_available(),
                            "mps": torch.backends.mps.is_available()},
                "device_choices": DEVICE_CHOICES, "training_approaches": TRAINING_APPROACHES,
                "complexity_methods": ["morse", "ph", "takens", "lyapunov"],
                "optional_methods": {"kan": importlib.util.find_spec("kan") is not None,
                                     "ph": importlib.util.find_spec("ripser") is not None,
                                     "gif": importlib.util.find_spec("PIL") is not None},
                "run_filters": [field for field, _ in FILTER_FIELDS],
                "checkpoint_inference": False}

    @router.get("/datasets")
    def datasets():
        return {"items": [{"dataset_id": key, "available": registration.data_root.is_dir(),
                            "estreams_configured": registration.estreams_root is not None,
                            "camels_chem_configured": registration.camels_chem_root is not None,
                            "hyd_responses_configured": registration.hyd_responses_root is not None,
                            "bafu_configured": registration.bafu_root is not None,
                            "hanze_configured": registration.hanze_root is not None,
                            "bafu_statistics_configured": registration.bafu_statistics_zip is not None,
                            "bafu_return_levels_configured": registration.bafu_return_levels is not None}
                           for key, registration in settings.datasets.items()]}

    @router.get("/ui/bootstrap")
    def ui_bootstrap():
        """Public configuration defaults for the browser, without server paths."""
        from .regional import (ModelArchitectureConfig, TrainingStrategyConfig,
                               HyperparameterConfig, PhysicsModelConfig,
                               ExtremeEventConfig, ForecastConfig)
        from .benchmarks import BenchmarkConfig
        config = {name: asdict(cls()) for name, cls in (
            ("model", ModelArchitectureConfig), ("strategy", TrainingStrategyConfig),
            ("hyperparameters", HyperparameterConfig), ("physics", PhysicsModelConfig),
            ("extremes", ExtremeEventConfig), ("forecast", ForecastConfig), ("benchmark", BenchmarkConfig))}
        config.update(basins={"target_basins": [], "training_scope": "exclude_targets"},
                      dates={"train": {"start": "2010-01-01", "end": "2016-08-06"},
                             "validation": {"start": "2016-08-07", "end": "2018-10-18"},
                             "test": {"start": "2018-10-19", "end": "2020-12-31"}})
        return {"config": json_safe(config), "leaderboard_enabled": settings.results_dir is not None,
                "schema": ExperimentRequest.model_json_schema()}

    @router.get("/datasets/{dataset_id}/validation")
    def validate_dataset(dataset_id: str):
        from .services import validate_optional_dataset
        registration = application.registration(dataset_id)
        catalog = application.catalog(dataset_id)
        optional = {}
        for kind in ("estreams", "camels_chem", "hyd_responses"):
            path = getattr(registration, kind + "_root")
            if path is not None:
                optional[kind] = {"gauges": validate_optional_dataset(kind, path)}
        if registration.bafu_root is not None:
            from .bafu import observation_root
            observation_root(registration.bafu_root)
            optional["bafu"] = {"configured": True, "date_coverage_checked": False}
        return {"dataset_id": dataset_id, "valid": True,
                "eligible_basins": len(catalog.eligible_ids), "optional": optional}

    @router.get("/datasets/{dataset_id}/basins")
    def basins(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        from .bafu import basin_compatibility
        catalog = application.catalog(dataset_id)
        rows = basin_metadata(catalog)
        compatibility = basin_compatibility(catalog, application.registration(dataset_id).bafu_root)
        for row in rows:
            row['compatibility'] = compatibility[row['basin_id']]
        return {"items": rows[offset:offset + limit], "total": len(rows), "offset": offset, "limit": limit}

    @router.get("/datasets/{dataset_id}/features")
    def features(dataset_id: str):
        from .estreams import ESTREAMS_DYNAMIC_FEATURES
        from .hyd_responses import HYD_RESPONSES_FEATURES
        from .camels_ch_chem import CAMELS_CH_CHEM_DYNAMIC_FEATURES
        registration = application.registration(dataset_id)
        catalog = application.catalog(dataset_id)
        return {"bafu_dynamic": ["previous_discharge", "day_sin", "day_cos"] if registration.bafu_root else [],
                "bafu_targets": ["daily_mean", "daily_max"] if registration.bafu_root else [],
                "dynamic": ["precipitation", "temperature", "pet", "previous_discharge", "day_sin", "day_cos"],
                "static_attributes": sorted({name for row in catalog.records for name in row.static_attributes}),
                "estreams": list(ESTREAMS_DYNAMIC_FEATURES) if registration.estreams_root else [],
                "camels_chem": list(CAMELS_CH_CHEM_DYNAMIC_FEATURES) if registration.camels_chem_root else [],
                "hyd_responses": list(HYD_RESPONSES_FEATURES) if registration.hyd_responses_root else []}

    @router.get("/benchmarks")
    def benchmark_profiles():
        from .benchmarks import profiles
        return {"items": profiles()}

    @router.get("/datasets/{dataset_id}/bafu/statistics")
    def bafu_statistics(dataset_id: str):
        from .bafu import statistics_index, load_return_levels
        registration = application.registration(dataset_id)
        result = statistics_index(registration.bafu_statistics_zip) if registration.bafu_statistics_zip else {
            "records": [], "contains_return_levels": False, "note": "No statistics ZIP registered"}
        result["verified_return_levels"] = list(load_return_levels(registration.bafu_return_levels).values()) if registration.bafu_return_levels else []
        return result

    @router.post("/experiments/preflight")
    def benchmark_preflight(body: ExperimentRequest):
        from .benchmarks import preflight
        config = application.experiment_config(body.model_dump(mode="json"))
        if config.benchmark.mode == "none":
            raise ValueError("Select bafu_only or bafu_combined for benchmark preflight")
        report = preflight(config, application.catalog(body.dataset_id))
        return {"valid": report["cohort_complete"] or not config.benchmark.strict_cohort, "training_started": False, "manifest": json_safe(report)}

    @router.get("/runs/{run_id}/benchmark")
    def run_benchmark(run_id: str):
        return json_safe(application.runs.get(run_id).config.get("benchmark_manifest", {}))

    @router.post("/experiments/validate")
    def validate_experiment(body: ExperimentRequest):
        config = application.experiment_config(body.model_dump(mode="json"))
        if config.benchmark.mode != "none":
            checked_benchmark(body, config)
            return {"valid": True, "config": public_config(config), "dataset_id": body.dataset_id,
                    "data_check": "BAFU station compatibility and selected date windows checked without training"}
        return {"valid": True, "config": public_config(config),
                "dataset_id": body.dataset_id,
                "data_check": "Dataset layout, basin eligibility and configuration checked; date coverage is checked during execution"}

    @router.post("/experiments", status_code=202, response_model=JobResponse)
    def experiments(body: ExperimentRequest, idempotency_key: str | None = Header(None)):
        payload = body.model_dump(mode="json")
        config = application.experiment_config(payload)
        if config.benchmark.mode != "none":
            checked_benchmark(body, config)
        return submit("experiment", payload, idempotency_key)

    def checked_benchmark(body, config):
        from .benchmarks import preflight
        report = preflight(config, application.catalog(body.dataset_id))
        if config.benchmark.strict_cohort and not report['cohort_complete']:
            excluded = set(report['excluded_basins']) | set(report['provenance'].get('missing_stations', []))
            raise ValueError(f"BAFU date coverage is incomplete for {len(excluded)} stations. "
                             "Open BAFU benchmark → Check benchmark coverage, review the exclusions, "
                             "then choose Use stations with complete coverage or adjust the date splits.")
        return report

    @router.get("/jobs/{job_id}", response_model=JobResponse)
    def job(job_id: str):
        return job_response(store.get(job_id))

    @router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel(job_id: str):
        return job_response(store.cancel(job_id))

    @router.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = store.get(job_id)
        if not job["result"]:
            raise HTTPException(409, "This job has no retrievable result yet")
        if job["kind"] == "experiment":
            return result_summary(application.runs.get(job["result"]["run_id"]))
        name = "scenario.json" if job["kind"] == "scenario" else "replay.gif"
        folder = (settings.storage_dir / "job-results" / job["id"]).resolve()
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise KeyError("result artifact not found")
        return FileResponse(path, media_type="application/json" if name.endswith("json") else "image/gif",
                            filename=name)

    @router.get("/runs/facets")
    def run_facets(filters: str | None = None):
        runs, _ = application.runs.list()
        selections = filters_from_json(filters)
        return {field: available_values(runs, selections, field) for field, _ in FILTER_FIELDS}

    @router.get("/ui/runs")
    def ui_runs(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        """Library cards do not need every run's loss traces and event records."""
        all_runs, warnings = application.runs.list()
        items = []
        for run in all_runs[offset:offset + limit]:
            items.append(json_safe({"run_id": run.run_id, "created_at": run.created_at,
                                    "config": public_config(run.config),
                                    "aggregate_metrics": run.aggregate_metrics,
                                    "parameter_count": run.parameter_count,
                                    "runtime_seconds": run.runtime_seconds, "cancelled": run.cancelled}))
        return {"items": items, "total": len(all_runs), "offset": offset, "limit": limit,
                "invalid_entries": len(warnings)}

    @router.get("/runs", response_model=RunsResponse)
    def runs(filters: str | None = None, offset: int = Query(0, ge=0),
             limit: int = Query(100, ge=1, le=1000)):
        all_runs, warnings = application.runs.list()
        selected = filter_runs(all_runs, filters_from_json(filters))
        return {"items": [result_summary(r) for r in selected[offset:offset + limit]],
                "offset": offset, "limit": limit, "total": len(selected), "invalid_entries": len(warnings)}

    @router.get("/runs/{run_id}", response_model=RunResponse)
    def run(run_id: str):
        return result_summary(application.runs.get(run_id))

    @router.post("/runs/{run_id}/save")
    def promote_run(run_id: str):
        run = application.runs.get(run_id)
        if run.cancelled:
            raise ValueError("cancelled partial runs cannot be saved to the leaderboard")
        application.runs.promote(run_id)
        return {"run_id": run_id, "saved_to_leaderboard": True}

    @router.get("/runs/{run_id}/predictions", response_model=PredictionsResponse)
    def predictions(run_id: str, basin_id: str | None = None, seed: int | None = None,
                    start: date | None = None, end: date | None = None,
                    lead_day: int | None = Query(None, ge=1), offset: int = Query(0, ge=0),
                    limit: int = Query(1000, ge=1, le=10000)):
        if start and end and start > end:
            raise ValueError("start must be on or before end")
        items, count = [], 0
        for row in application.runs.predictions(run_id):
            if ((basin_id is not None and row.basin_id != basin_id) or
                (seed is not None and row.seed != seed) or (start and row.target_date < start) or
                (end and row.target_date > end) or (lead_day and row.lead_day != lead_day)):
                continue
            if offset <= count < offset + limit:
                items.append(json_safe(row))
            count += 1
        return {"items": items, "offset": offset, "limit": limit, "total": count}

    @router.get("/runs/{run_id}/hanze")
    def hanze_evaluation(run_id: str):
        return json_safe(application.hanze_for_run(run_id))

    @router.get("/runs/{run_id}/metrics")
    def metrics(run_id: str):
        run = application.runs.get(run_id)
        return json_safe({"aggregate": run.aggregate_metrics, "per_basin": run.per_basin_metrics,
                          "by_lead_day": run.forecast_metrics, "routing": run.routing_diagnostics})

    @router.get("/runs/{run_id}/events")
    def events(run_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        events = application.runs.get(run_id).extreme_events
        return json_safe({"items": events[offset:offset + limit], "total": len(events)})

    @router.get("/runs/{run_id}/losses")
    def losses(run_id: str):
        return json_safe(application.runs.get(run_id).loss_traces)

    @router.get("/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str):
        if name == "hanze":
            return JSONResponse(json_safe(application.hanze_for_run(run_id)),
                                headers={"Content-Disposition": 'attachment; filename="hanze_evaluation.json"'})
        if name == "config":
            return JSONResponse(public_config(application.runs.get(run_id).config),
                                headers={"Content-Disposition": 'attachment; filename="config.json"'})
        if name == "archive":
            # The public bundle contains dataset IDs instead of server filesystem locations.
            handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            handle.close()
            path = Path(handle.name)
            try:
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
                    keys = ["predictions", "metrics"]
                    if application.runs.get(run_id).config.get("extremes", {}).get("mode") == "drought":
                        keys.extend(("drought_metrics", "drought_days", "drought_events"))
                    for key in keys:
                        file = application.runs.artifact(run_id, key)
                        bundle.write(file, file.name)
                    try:
                        hanze_report = application.hanze_for_run(run_id)
                    except (OSError, ValueError):
                        log.exception("Optional HANZE evaluation unavailable for archive")
                        hanze_report = {"status": "error"}
                    if hanze_report.get("status") == "available":
                        bundle.writestr("hanze_evaluation.json", json.dumps(json_safe(hanze_report), allow_nan=False))
                    bundle.writestr("config.json", json.dumps(public_config(application.runs.get(run_id).config), allow_nan=False))
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            return FileResponse(path, media_type="application/zip", filename="regional_experiment.zip",
                                background=BackgroundTask(path.unlink, missing_ok=True))
        if name == "benchmark_manifest":
            return JSONResponse(json_safe(application.runs.get(run_id).config.get("benchmark_manifest", {})))
        return FileResponse(application.runs.artifact(run_id, name), media_type="text/csv",
                            filename=f"{name}.csv")

    @router.post("/comparisons")
    def comparison(body: ComparisonRequest):
        result = compare_runs(application.runs.get(body.run_a), application.runs.get(body.run_b))
        if result.get("benchmark_comparison", {}).get("compatible"):
            from .benchmarks import paired_comparison
            result["paired_comparison"] = json_safe(paired_comparison(
                list(application.runs.predictions(body.run_a)), list(application.runs.predictions(body.run_b))))
        return result

    @router.get("/runs/{run_id}/replay/episodes")
    def episodes(run_id: str, basin_id: str, event_days: int = Query(14, ge=1, le=90),
                 limit: int = Query(8, ge=1, le=100)):
        rows = replay_episodes(application.runs.result(run_id), application.catalog_for_run(run_id),
                               basin_id, event_days=event_days, limit=limit)
        return {"items": [{"label": label, "episode": episode} for label, episode in rows]}

    @router.post("/runs/{run_id}/replay")
    def replay(run_id: str, body: ReplayRequest):
        output = prepare_replay(application.runs.result(run_id), application.catalog_for_run(run_id),
                                 body.basin_id, body.episode, event_days=body.event_days)
        return {"data": json_safe(output),
                "days": [json_safe(forecast_snapshot(output, day)) for day in range(1, body.event_days + 1)],
                "teaching_bucket": "Uncalibrated illustration, separate from trained model storage"}

    @router.post("/runs/{run_id}/replay/gif", status_code=202, response_model=JobResponse)
    def gif(run_id: str, body: ReplayRequest, idempotency_key: str | None = Header(None)):
        application.runs.get(run_id)
        application.dataset_for_run(run_id)
        return submit("gif", {"run_id": run_id, "request": body.model_dump(mode="json")}, idempotency_key)

    @router.post("/runs/{run_id}/scenarios", status_code=202, response_model=JobResponse)
    def scenario(run_id: str, body: ScenarioRequest, idempotency_key: str | None = Header(None)):
        application.runs.get(run_id)
        application.dataset_for_run(run_id)
        PavedSiteConfig(**body.site.model_dump()).validate()
        return submit("scenario", {"run_id": run_id, "request": body.model_dump(mode="json")}, idempotency_key)

    app.include_router(router)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    token = os.environ.get("RUNOFF_API_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("set RUNOFF_API_TOKEN before binding to a non-loopback address")
    import uvicorn
    uvicorn.run(create_app(ServerSettings.load(args.settings, token=token)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
