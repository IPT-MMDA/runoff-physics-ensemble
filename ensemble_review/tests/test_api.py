"""Synthetic-data HTTP and worker parity tests; CAMELS-CH is not required."""
from dataclasses import asdict, replace
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

try:
    from fastapi.testclient import TestClient
except ImportError:
    raise unittest.SkipTest("Install the api extra to run HTTP integration tests")

import torch

from complexity_ensemble.api import create_app
from complexity_ensemble.api_models import ExperimentRequest
from complexity_ensemble.application import Application
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.jobs import JobStore
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog, run_regional_experiment
from complexity_ensemble.remote import RemoteExecutor
from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings
from complexity_ensemble.services import execute_experiment
from complexity_ensemble.worker import run_job, worker_lock


def series(basin, offset=0.):
    days = tuple(date(2001, 1, 1) + timedelta(days=i) for i in range(110))
    t = torch.arange(110, dtype=torch.float32)
    rain = torch.where(t.remainder(7) < 2, 4., .2)
    forcing = torch.stack((rain, 5 + torch.sin(t / 10), torch.ones(110) * .5), dim=1)
    discharge = 2 + offset + .4 * torch.sin(t / 3) + t * .003
    return HydrologySeries(days, discharge, forcing, ("precipitation", "temperature", "pet"), basin, "CH")


def payload():
    return {"dataset_id": "ch", "config": {
        "basins": {"target_basins": ["target"], "training_scope": "all_basins"},
        "dates": {"train": {"start": "2001-01-08", "end": "2001-01-31"},
                  "validation": {"start": "2001-02-01", "end": "2001-02-15"},
                  "test": {"start": "2001-02-16", "end": "2001-04-10"}},
        "model": {"complex_expert": "mlp", "simple_expert": "fourier",
                  "mlp_widths": [8], "fourier_frequencies": 4},
        "strategy": {"approach": "no_routing", "epochs": 1},
        "hyperparameters": {"sequence_length": 7, "batch_size": 128, "device": "cpu"},
    }}


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = ServerSettings(root / "state", {"ch": DatasetRegistration(root / "data")},
                                       root / "leaderboard", token="test-token")
        self.series = {"a": series("a"), "target": series("target", .2)}
        self.catalog = CAMELSCHCatalog(root / "data", [BasinCatalogRecord(k) for k in self.series])
        self.catalog.load_series = lambda basin, start, end: self.series[basin]
        self.application = Application(self.settings)
        self.application.catalog = lambda dataset_id: self.catalog if dataset_id == "ch" else self.application.registration(dataset_id)
        self.store = JobStore(self.settings.jobs_path)
        self.client = TestClient(create_app(self.settings, application=self.application, store=self.store))
        self.client.headers["Authorization"] = "Bearer test-token"
        self.addCleanup(self.client.close)

    def process_next(self):
        owner = "test-worker"
        self.store.acquire_worker(owner, stale_after=0)
        job = self.store.claim(owner)
        self.assertIsNotNone(job)
        def execute(config, progress_callback=None, cancellation_token=None, **kwargs):
            return run_regional_experiment(config, progress_callback, cancellation_token,
                                           catalog=self.catalog, series_by_basin=self.series)
        with patch("complexity_ensemble.services.run_regional_experiment", execute):
            result = run_job(self.application, self.store, job, owner)
        output = self.store.finish(job["id"], owner, result=result)
        self.store.release_worker(owner)
        return output

    def run_experiment(self, request=None):
        response = self.client.post("/api/v1/experiments", json=request or payload())
        self.assertEqual(response.status_code, 202, response.text)
        job = self.process_next()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertFalse(job["result"]["execution_failed"], job)
        return job["result"]["run_id"]

    def test_authentication_and_openapi(self):
        response = self.client.get("/api/v1/health", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(set(response.json()), {"code", "message", "details"})
        self.assertFalse(self.client.get("/api/v1/health").json()["worker_ready"])
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(self.client.get("/api/v1/datasets/ch/features").status_code, 200)
        self.assertEqual(set(self.client.get("/api/v1/missing").json()), {"code", "message", "details"})
        self.assertIn("/api/v1/experiments", schema["paths"])
        self.assertIn("/api/v1/runs/{run_id}/scenarios", schema["paths"])

    def test_unknown_fields_and_server_paths_are_rejected(self):
        body = payload()
        body["config"]["hyperparameters"]["batch_szie"] = 3
        self.assertEqual(self.client.post("/api/v1/experiments", json=body).status_code, 422)
        body = payload()
        body["config"]["hyperparameters"]["batch_size"] = "128"
        self.assertEqual(self.client.post("/api/v1/experiments", json=body).status_code, 422)
        body = payload()
        body["config"]["data_root"] = "/arbitrary/path"
        self.assertEqual(self.client.post("/api/v1/experiments", json=body).status_code, 422)
        body = payload()
        body["config"]["hyperparameters"]["learning_rate"] = float("nan")
        with self.assertRaises(ValueError):
            ExperimentRequest.model_validate(body)

    def test_domain_validation_and_unknown_dataset(self):
        body = payload()
        body["config"]["hyperparameters"]["batch_size"] = 0
        self.assertEqual(self.client.post("/api/v1/experiments/validate", json=body).status_code, 422)
        body = payload()
        body["dataset_id"] = "missing"
        self.assertEqual(self.client.post("/api/v1/experiments", json=body).status_code, 404)
        body = payload()
        body["config"]["basins"]["target_basins"] = ["unknown"]
        self.assertEqual(self.client.post("/api/v1/experiments", json=body).status_code, 422)

    def test_http_idempotency_and_cancel(self):
        first = self.client.post("/api/v1/experiments", json=payload(), headers={"Idempotency-Key": "same"})
        second = self.client.post("/api/v1/experiments", json=payload(), headers={"Idempotency-Key": "same"})
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        body = payload()
        body["config"]["hyperparameters"]["seeds"] = [1]
        self.assertEqual(self.client.post("/api/v1/experiments", json=body,
                                         headers={"Idempotency-Key": "same"}).status_code, 409)
        job_id = first.json()["job_id"]
        self.assertEqual(self.client.post(f"/api/v1/jobs/{job_id}/cancel").json()["status"], "cancelled")
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/result").status_code, 409)

    def test_worker_results_match_standalone_and_cli_configuration(self):
        run_id = self.run_experiment()
        request = ExperimentRequest.model_validate(payload()).model_dump(mode="json")
        config = self.application.experiment_config(request)
        direct = execute_experiment(config, catalog=self.catalog, series_by_basin=self.series)
        summary = self.client.get(f"/api/v1/runs/{run_id}").json()
        self.assertEqual(summary["aggregate_metrics"], direct.aggregate_metrics)
        self.assertNotIn("data_root", summary["config"])
        self.assertEqual(summary["config"]["dataset_id"], "ch")
        rows = self.client.get(f"/api/v1/runs/{run_id}/predictions?limit=2&offset=1").json()
        self.assertEqual(len(rows["items"]), 2)
        self.assertEqual(rows["total"], len(direct.predictions))
        self.assertIsNone(rows["items"][0]["ood_score"])
        self.assertEqual(rows["items"][0]["target_date"], direct.predictions[1].target_date.isoformat())
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/metrics").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/events").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/losses").status_code, 200)
        from complexity_ensemble.regional_cli import config_from_mapping as cli_parser
        self.assertEqual(cli_parser, config_from_mapping)

    def test_all_training_strategies_and_both_experts_run_through_worker(self):
        for expert in ("mlp", "pinnmamba"):
            for approach in ("soft_routing", "hard_routing", "distillation", "no_routing",
                             "static_50_50", "ood_fallback", "stacking"):
                with self.subTest(expert=expert, approach=approach):
                    request = payload()
                    request["config"]["model"].update({"complex_expert": expert,
                        "mamba_hidden_size": 8, "mamba_layers": 1, "mamba_feed_forward_size": 8})
                    request["config"]["strategy"].update({"approach": approach,
                        "complex_teacher_epochs": 1, "distillation_epochs": 1, "consolidation_epochs": 1})
                    request["config"]["hyperparameters"]["training_noise"] = .1
                    run_id = self.run_experiment(request)
                    summary = self.client.get(f"/api/v1/runs/{run_id}").json()
                    self.assertEqual(summary["failures"], [])
                    self.assertEqual(summary["requested_config"]["strategy"]["approach"], approach)

    def test_result_downloads_and_promotion(self):
        run_id = self.run_experiment()
        self.assertFalse((self.settings.results_dir / run_id).exists())
        response = self.client.get(f"/api/v1/runs/{run_id}/artifacts/archive")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            config = json.loads(bundle.read("config.json"))
            self.assertNotIn("data_root", config)
            self.assertIn("predictions.csv", bundle.namelist())
        self.assertEqual(self.client.post(f"/api/v1/runs/{run_id}/save").status_code, 200)
        self.assertTrue((self.settings.results_dir / run_id / "run.json").is_file())
        self.assertEqual(self.client.get("/api/v1/runs").json()["total"], 1)
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/artifacts/secret").status_code, 404)
        with self.assertRaises(KeyError):
            self.application.runs.get("../escape")

    def test_comparisons_and_filters(self):
        run_a = self.run_experiment()
        request = payload()
        request["config"]["hyperparameters"]["seeds"] = [1]
        run_b = self.run_experiment(request)
        response = self.client.post("/api/v1/comparisons", json={"run_a": run_a, "run_b": run_b})
        self.assertEqual(response.status_code, 200)
        self.assertIn("nse", response.json()["aggregate_delta_b_minus_a"])
        self.assertEqual(self.client.get("/api/v1/runs", params={"filters": json.dumps({"strategy.approach": "missing"})}).json()["total"], 0)
        self.assertEqual(self.client.get("/api/v1/runs/facets").status_code, 200)

    def test_replay_and_scenario_use_saved_run_data(self):
        run_id = self.run_experiment()
        response = self.client.get(f"/api/v1/runs/{run_id}/replay/episodes?basin_id=target")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["items"])
        episode = response.json()["items"][0]["episode"]
        replay = self.client.post(f"/api/v1/runs/{run_id}/replay", json={"basin_id": "target", "episode": episode})
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(len(replay.json()["days"]), 14)
        scenario = self.client.post(f"/api/v1/runs/{run_id}/scenarios", json={"basin_id": "target", "site": {}})
        self.assertEqual(scenario.status_code, 202)
        job = self.process_next()
        response = self.client.get(f"/api/v1/jobs/{job['id']}/result")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["rows"])
        self.assertEqual(response.json()["summary"]["basin_id"], "target")

    def test_remote_adapter_hydrates_domain_results(self):
        executor = RemoteExecutor("http://testserver", "ch", token="test-token")
        def request(method, path, payload=None, key=None):
            headers = {"Idempotency-Key": key} if key else {}
            response = self.client.request(method, "/api/v1" + path, json=payload, headers=headers)
            self.assertLess(response.status_code, 400, response.text)
            if method == "POST" and path == "/experiments":
                self.process_next()
            return response.json()
        executor.request = request
        request_body = ExperimentRequest.model_validate(payload()).model_dump(mode="json")
        config = self.application.experiment_config(request_body)
        with patch("complexity_ensemble.remote.time.sleep"):
            result = execute_experiment(config, executor=executor)
        self.assertTrue(result.predictions)
        self.assertIn("api_run_id", result.resolved_config)
        self.assertIsInstance(result.predictions[0].target_date, date)
        self.assertEqual(result.resolved_config["hyperparameters"]["device"], "cpu")

    def test_remote_queued_cancellation_returns_a_renderable_empty_result(self):
        executor = RemoteExecutor("http://testserver", "ch")
        executor.request = lambda *a, **k: {"job_id": "cancelled", "status": "cancelled", "result": None}
        config = self.application.experiment_config(ExperimentRequest.model_validate(payload()).model_dump(mode="json"))
        result = executor.run(config)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.resolved_config["effective_training_basins"], [])

    def test_running_cancellation_keeps_inspectable_partial_run(self):
        response = self.client.post("/api/v1/experiments", json=payload())
        self.store.acquire_worker("cancel-worker")
        job = self.store.claim("cancel-worker")
        def cancel_then_execute(config, progress_callback=None, cancellation_token=None, **kwargs):
            self.store.cancel(job["id"])
            return run_regional_experiment(config, progress_callback, cancellation_token,
                                           catalog=self.catalog, series_by_basin=self.series)
        with patch("complexity_ensemble.services.run_regional_experiment", cancel_then_execute):
            result = run_job(self.application, self.store, job, "cancel-worker")
        output = self.store.finish(job["id"], "cancel-worker", result=result)
        self.assertEqual(output["status"], "cancelled")
        self.assertTrue(output["result"]["partial"])
        run_id = output["result"]["run_id"]
        self.assertTrue(self.client.get(f"/api/v1/runs/{run_id}").json()["cancelled"])
        self.assertEqual(self.client.post(f"/api/v1/runs/{run_id}/save").status_code, 422)
        self.store.release_worker("cancel-worker")

    def test_health_without_datasets_and_worker_restart(self):
        from complexity_ensemble.worker import serve
        settings = ServerSettings(self.settings.storage_dir / "empty")
        with TestClient(create_app(settings)) as client:
            self.assertEqual(client.get("/api/v1/health").status_code, 200)
            self.assertEqual(client.get("/api/v1/datasets").json()["items"], [])
        serve(settings, once=True)
        self.assertFalse(JobStore(settings.jobs_path).worker_ready())

    def test_gradio_build_preserves_interface_with_remote_executor(self):
        import importlib.util
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Gradio is not installed")
        from complexity_ensemble.app import build_app
        import os
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            with patch("complexity_ensemble.app.CAMELSCHCatalog", return_value=self.catalog):
                ui = build_app(self.catalog.root, executor=RemoteExecutor("http://testserver", "ch"))
        self.assertTrue(ui.config["components"])
        self.assertTrue(ui.config["dependencies"])
        ui.close()

    def test_gif_export_runs_as_a_job(self):
        import importlib.util
        if importlib.util.find_spec("PIL") is None or importlib.util.find_spec("matplotlib") is None:
            self.skipTest("GIF dependencies are not installed")
        run_id = self.run_experiment()
        episodes = self.client.get(f"/api/v1/runs/{run_id}/replay/episodes?basin_id=target").json()["items"]
        submitted = self.client.post(f"/api/v1/runs/{run_id}/replay/gif", json={
            "basin_id": "target", "episode": episodes[0]["episode"], "event_days": 2})
        self.assertEqual(submitted.status_code, 202)
        job = self.process_next()
        response = self.client.get(f"/api/v1/jobs/{job['id']}/result")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"GIF"))

    def test_worker_os_lock_is_exclusive(self):
        with worker_lock(self.settings.storage_dir / "worker.lock"):
            with self.assertRaises(RuntimeError):
                with worker_lock(self.settings.storage_dir / "worker.lock"):
                    pass


if __name__ == "__main__":
    unittest.main()
