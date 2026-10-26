"""Application operations shared by Gradio, the CLI, and server workers."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import csv
import re
import shutil
import tempfile

from .regional import (CAMELSCHCatalog, DateRange, ExperimentResult, ExtremeEventConfig,
                       PredictionRecord, run_regional_experiment)
from .result_registry import discover_runs, load_run, save_run
from .rainfall_replay import event_choices, make_replay
from .paved_scenario import run_paved_site_scenario
from .serialization import json_safe, public_config
from .hydrology_drought import drought_summary
from .flood_verification import flood_summary


def execute_experiment(config, progress_callback=None, cancellation_token=None, *, executor=None, **kwargs):
    """Standalone execution remains synchronous; a remote executor is optional."""
    if executor is not None:
        return executor.run(config, progress_callback, cancellation_token)
    return run_regional_experiment(config, progress_callback, cancellation_token, **kwargs)


def validate_optional_dataset(kind, path):
    if kind == "hyd_responses":
        from .hyd_responses import HYDResponses
        return len(HYDResponses(path).gauge_ids)
    if kind == "estreams":
        from .estreams import EStreamsVegetationSnow
        return len(EStreamsVegetationSnow(path).camels_gauge_ids)
    if kind == "camels_chem":
        from .camels_ch_chem import CAMELSCHChemPressures
        return len(CAMELSCHChemPressures(path).gauge_ids)
    raise ValueError("unknown optional dataset kind")


def basin_metadata(catalog):
    records = []
    for record in catalog.records:
        geometry = getattr(record.geometry, "__geo_interface__", None)
        records.append(json_safe({"basin_id": record.basin_id, "name": record.name,
                                  "eligible": record.eligible,
                                  "static_attributes": record.static_attributes,
                                  "geometry": geometry}))
    return records


def _prediction(row):
    def number(key):
        value = row.get(key)
        return float(value) if value not in (None, "") else float("nan")
    return PredictionRecord(
        int(row["seed"]), str(row["basin_id"]), date.fromisoformat(str(row["target_date"])),
        number("observed_mm_day"), number("predicted_mm_day"), number("complex_weight"),
        number("physics_error"), number("ood_score"), number("ood_threshold"),
        date.fromisoformat(str(row["issue_date"])) if row.get("issue_date") else None,
        int(row.get("lead_day") or 1),
    )


class RunRepository:
    def __init__(self, directory, leaderboard=None):
        self.directory = Path(directory)
        self.leaderboard = Path(leaderboard) if leaderboard else None

    def list(self):
        by_id = {}
        warnings = []
        for root in (self.directory, self.leaderboard):
            if root is not None:
                snapshot = discover_runs(root)
                by_id.update({run.run_id: run for run in snapshot.runs})
                warnings.extend(snapshot.warnings)
        return sorted(by_id.values(), key=lambda r: r.created_at, reverse=True), warnings

    def get(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", run_id):
            raise KeyError("run not found")
        for root in (self.directory, self.leaderboard):
            if root is None:
                continue
            folder = (root / run_id).resolve()
            if folder.parent != root.resolve():
                continue
            manifest = folder / "run.json"
            if manifest.is_file() and manifest.resolve().parent == folder:
                return load_run(manifest)
        raise KeyError("run not found")

    def save(self, result, promote=False):
        saved = save_run(result, self.directory)
        if promote and not result.cancelled:
            self.promote(saved.run_id)
        return saved

    def promote(self, run_id):
        if self.leaderboard is None:
            raise ValueError("leaderboard saving is not configured")
        run = self.get(run_id)
        self.leaderboard.mkdir(parents=True, exist_ok=True)
        destination = self.leaderboard / run_id
        if destination.exists():
            return
        staging = Path(tempfile.mkdtemp(prefix=".promote-", dir=self.leaderboard))
        try:
            shutil.copytree(run.path, staging, dirs_exist_ok=True)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def artifact(self, run_id, name):
        names = {"predictions": "predictions.csv", "metrics": "metrics.csv",
                 "archive": "regional_experiment.zip",
                 "drought_metrics": "drought_metrics.csv", "drought_days": "drought_days.csv",
                 "drought_events": "drought_events.csv", "benchmark_manifest": "benchmark_manifest.json", "flood_metrics": "flood_metrics.csv"}
        if name not in names:
            raise KeyError("artifact not found")
        folder = self.get(run_id).path.resolve()
        path = (folder / names[name]).resolve()
        if path.parent != folder or not path.is_file():
            raise KeyError("artifact not found")
        return path

    def predictions(self, run_id):
        with self.artifact(run_id, "predictions").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield _prediction(row)

    def result(self, run_id):
        run = self.get(run_id)
        return ExperimentResult(
            predictions=list(self.predictions(run_id)), aggregate_metrics=run.aggregate_metrics,
            per_basin_metrics=run.per_basin_metrics, routing_diagnostics=run.routing_diagnostics,
            loss_traces=run.loss_traces, extreme_events=run.extreme_events, failures=run.failures,
            resolved_config=run.config, parameter_count=run.parameter_count,
            runtime_seconds=run.runtime_seconds, cancelled=run.cancelled,
            forecast_metrics=run.forecast_metrics,
            drought_details=run.drought_details,
            hanze_details=run.hanze_details,
        )


def result_summary(run):
    config = public_config(run.config)
    requested = config.pop("requested_config", None)
    return json_safe({"run_id": run.run_id, "created_at": run.created_at,
                      "config": config, "requested_config": requested,
                      "aggregate_metrics": run.aggregate_metrics,
                      "per_basin_metrics": run.per_basin_metrics,
                      "forecast_metrics": run.forecast_metrics,
                      "drought_details": run.drought_details,
                      "routing_diagnostics": run.routing_diagnostics, "loss_traces": run.loss_traces,
                      "extreme_events": run.extreme_events, "flood_summary": flood_summary(run.extreme_events), "drought_summary": drought_summary(run.extreme_events),
                      "failures": run.failures,
                      "parameter_count": run.parameter_count, "runtime_seconds": run.runtime_seconds,
                      "cancelled": run.cancelled})


def compare_runs(run_a, run_b):
    if run_a.run_id == run_b.run_id:
        raise ValueError("select two different runs")
    def deltas(a, b):
        return {key: b[key] - a[key] if isinstance(a.get(key), (int, float))
                and isinstance(b.get(key), (int, float)) else None for key in a.keys() | b.keys()}
    first = run_a.config.get("benchmark_manifest", {})
    second = run_b.config.get("benchmark_manifest", {})
    compatible = bool(first.get("evaluation_fingerprint") and first.get("evaluation_fingerprint") == second.get("evaluation_fingerprint"))
    benchmark = {"compatible": compatible,
                 "reason": "Matching evaluation targets, observations, issue policy, thresholds and verifier" if compatible else "Missing or different benchmark evaluation contracts; metric deltas are descriptive only",
                 "same_training_data": bool(first.get("training_fingerprint") and first.get("training_fingerprint") == second.get("training_fingerprint"))}
    return json_safe({"run_a": result_summary(run_a), "run_b": result_summary(run_b),
                      "benchmark_comparison": benchmark,
                      "flood_delta_b_minus_a": deltas(flood_summary(run_a.extreme_events), flood_summary(run_b.extreme_events)),
                      "aggregate_delta_b_minus_a": deltas(run_a.aggregate_metrics, run_b.aggregate_metrics),
                      "drought_delta_b_minus_a": deltas(drought_summary(run_a.extreme_events),
                                                        drought_summary(run_b.extreme_events)),
                      "per_basin_delta_b_minus_a": {
                          basin: deltas(run_a.per_basin_metrics.get(basin, {}),
                                        run_b.per_basin_metrics.get(basin, {}))
                          for basin in run_a.per_basin_metrics.keys() | run_b.per_basin_metrics.keys()}})


def _basin_rows(result, basin):
    if basin not in result.resolved_config["basins"]["target_basins"]:
        raise ValueError("select a prediction target from this run")
    rows = [row for row in result.predictions if row.basin_id == basin]
    if not rows:
        raise ValueError("no predictions for this basin")
    return rows


def replay_episodes(result, catalog, basin, *, event_days=14, limit=8, executor=None):
    if executor is not None:
        return executor.episodes(result, basin, event_days, limit)
    if result.resolved_config.get("benchmark", {}).get("mode", "none") != "none":
        raise ValueError("Rainfall replay and impact scenarios currently support CAMELS-CH target runs only")
    rows = _basin_rows(result, basin)
    context = int(result.resolved_config.get("hyperparameters", {}).get("sequence_length", 30))
    series = catalog.load_series(basin, min(r.target_date for r in rows) - timedelta(days=context + 7),
                                 max(r.target_date for r in rows) + timedelta(days=event_days))
    return event_choices(series, rows, context_days=context, event_days=event_days, limit=limit)


def prepare_replay(result, catalog, basin, episode, *, event_days=14, executor=None):
    if executor is not None:
        return executor.replay(result, basin, episode, event_days)
    if result.resolved_config.get("benchmark", {}).get("mode", "none") != "none":
        raise ValueError("Rainfall replay and impact scenarios currently support CAMELS-CH target runs only")
    rows = _basin_rows(result, basin)
    cfg = result.resolved_config
    context = int(cfg.get("hyperparameters", {}).get("sequence_length", 30))
    first = date.fromisoformat(str(episode))
    series = catalog.load_series(basin, first - timedelta(days=context),
                                 first + timedelta(days=event_days - 1))
    extras = tuple(name for name in cfg.get("feature_names", ()) if name not in {
        "precipitation", "temperature", "pet", "previous_discharge", "day_sin", "day_cos"})
    return make_replay(series, rows, first, context_days=context, event_days=event_days,
                       extra_features=extras, static_feature_count=len(cfg.get("static_feature_names", ())),
                       forecast_mode=cfg.get("forecast", {}).get("mode", "one_day"))


def evaluate_scenario(result, catalog, basin, setup, *, executor=None):
    if executor is not None:
        return executor.scenario(result, basin, setup)
    if result.resolved_config.get("benchmark", {}).get("mode", "none") != "none":
        raise ValueError("Rainfall replay and impact scenarios currently support CAMELS-CH target runs only")
    rows = _basin_rows(result, basin)
    setup.validate()
    train = DateRange(**result.resolved_config["dates"]["train"])
    extremes = ExtremeEventConfig(**result.resolved_config["extremes"])
    first = min(train.resolved().start, setup.start_date) if setup.start_date else train.resolved().start
    series = catalog.load_series(basin, first, max(r.target_date for r in rows))
    return run_paved_site_scenario(rows, series, train, extremes, setup)
