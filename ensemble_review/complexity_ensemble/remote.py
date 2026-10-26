"""Optional HTTP executor used by the existing synchronous CLI and Gradio flows."""
from __future__ import annotations

from datetime import date
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid

from .serialization import public_config


class RemoteExecutor:
    def __init__(self, base_url, dataset_id, token=None):
        if urlsplit(base_url).scheme not in {"http", "https"}:
            raise ValueError("API URL must use http or https")
        if not dataset_id:
            raise ValueError("dataset_id is required for remote execution")
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.dataset_id, self.token = dataset_id, token

    def request(self, method, path, payload=None, *, key=None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if key:
            headers["Idempotency-Key"] = key
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, body, headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                message = json.load(exc).get("message", exc.reason)
            except (ValueError, AttributeError):
                message = exc.reason
            raise RuntimeError(f"API returned {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError("API connection failed") from exc

    def wait(self, job, progress=None, cancellation=None):
        job_id = job["job_id"]
        cancelled = False
        while job["status"] not in {"succeeded", "failed", "cancelled"}:
            if progress:
                progress(job["progress"], job["message"])
            if cancellation and cancellation.is_cancelled() and not cancelled:
                self.request("POST", f"/jobs/{job_id}/cancel")
                cancelled = True
            time.sleep(0.5)
            job = self.request("GET", f"/jobs/{job_id}")
        if job["status"] == "failed":
            raise RuntimeError(job.get("error", {}).get("message", "Remote execution failed"))
        if progress:
            progress(job["progress"], job["message"])
        return job

    def run(self, config, progress=None, cancellation=None):
        from .regional import ExperimentResult
        from .services import _prediction
        request = {"dataset_id": self.dataset_id, "config": public_config(config),
                   "use_estreams": config.estreams_root is not None,
                   "use_camels_chem": config.camels_chem_root is not None,
                   "use_hyd_responses": config.hyd_responses_root is not None}
        job = self.request("POST", "/experiments", request, key=uuid.uuid4().hex)
        if progress:
            progress(0., f"Submitted remote job {job['job_id']}")
        try:
            job = self.wait(job, progress, cancellation)
        except KeyboardInterrupt:
            self.request("POST", f"/jobs/{job['job_id']}/cancel")
            raise
        if not job["result"]:
            resolved = public_config(config) | {
                "data_root": str(config.data_root), "dataset_id": self.dataset_id,
                "effective_training_basins": [], "feature_names": [], "static_feature_names": [],
            }
            return ExperimentResult([], {}, {}, {}, {}, [], [], resolved, cancelled=True)
        run_id = job["result"]["run_id"]
        run = self.request("GET", f"/runs/{run_id}")
        rows, offset = [], 0
        while True:
            page = self.request("GET", f"/runs/{run_id}/predictions?" + urlencode({"offset": offset, "limit": 10000}))
            rows.extend(_prediction(row) for row in page["items"])
            offset += len(page["items"])
            if offset >= page["total"]:
                break
            if not page["items"]:
                raise RuntimeError("Remote predictions pagination made no progress")
        resolved = run["config"] | {"data_root": str(config.data_root),
                                     "api_run_id": run_id, "dataset_id": self.dataset_id,
                                     "requested_config": run.get("requested_config")}
        return ExperimentResult(rows, run["aggregate_metrics"], run["per_basin_metrics"],
                                run["routing_diagnostics"], run["loss_traces"], run["extreme_events"],
                                run["failures"], resolved, run["parameter_count"], run["runtime_seconds"],
                                run["cancelled"], forecast_metrics=run["forecast_metrics"],
                                drought_details=run.get("drought_details", {}))

    def episodes(self, result, basin, event_days, limit):
        run_id = result.resolved_config["api_run_id"]
        response = self.request("GET", f"/runs/{run_id}/replay/episodes?" +
                                urlencode({"basin_id": basin, "event_days": event_days, "limit": limit}))
        return [(row["label"], row["episode"]) for row in response["items"]]

    def replay(self, result, basin, episode, event_days):
        from .rainfall_replay import Replay
        run_id = result.resolved_config["api_run_id"]
        response = self.request("POST", f"/runs/{run_id}/replay", {
            "basin_id": basin, "episode": str(episode), "event_days": event_days})["data"]
        response["dates"] = tuple(date.fromisoformat(day) for day in response["dates"])
        response["issue_dates"] = tuple(date.fromisoformat(day) if day else None for day in response["issue_dates"])
        for key, value in response.items():
            if isinstance(value, list):
                response[key] = tuple(value)
        return Replay(**response)

    def scenario(self, result, basin, setup):
        from .paved_scenario import PavedScenarioResult
        from .serialization import json_safe
        run_id = result.resolved_config["api_run_id"]
        job = self.wait(self.request("POST", f"/runs/{run_id}/scenarios", {
            "basin_id": basin, "site": json_safe(setup)}, key=uuid.uuid4().hex))
        if job["status"] != "succeeded":
            raise RuntimeError("Scenario execution was cancelled")
        response = self.request("GET", f"/jobs/{job['job_id']}/result")
        return PavedScenarioResult(**response)
