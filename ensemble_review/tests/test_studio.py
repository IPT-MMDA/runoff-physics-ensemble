"""Custom workspace hosting, authentication and HTTP job integration."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import gradio
    from fastapi.testclient import TestClient
except ImportError:
    raise unittest.SkipTest("Install the api extra and Gradio to test the workspace")

from complexity_ensemble.api_models import ExperimentRequest
from complexity_ensemble.application import Application
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog
from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings
from complexity_ensemble.studio_server import create_studio


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = ServerSettings(root / "state", {"ch": DatasetRegistration(root / "data")},
                                       token="studio-secret")
        application = Application(self.settings)
        self.application = application
        catalog = CAMELSCHCatalog(root / "data", [BasinCatalogRecord("4009")])
        application.catalog = lambda dataset_id: catalog
        app = create_studio(self.settings, application=application)
        self.client = TestClient(app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_public_shell_does_not_expose_authenticated_data(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("/studio-assets/app.js", page.text)
        self.assertNotIn("studio-secret", page.text)
        self.assertEqual(self.client.get("/api/v1/ui/bootstrap").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/datasets").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/ui/runs").status_code, 401)
        self.assertEqual(self.client.get("/studio-assets/core.js").status_code, 200)
        script = self.client.get("/studio-assets/app.js?v=fixed-sky-blue-1")
        self.assertEqual(script.status_code, 200)
        self.assertEqual(script.headers["Cache-Control"], "no-cache")
        self.assertNotIn('id="appearance-toggle"', page.text)
        self.assertNotIn('<dialog id="appearance"', page.text)
        self.assertNotIn("COMPLEXITY → UNDERSTANDING", page.text)
        self.assertNotIn('class="orbit-art"', page.text)
        self.assertNotIn("studio-secret", self.client.get("/gradio/").text)

    def test_defaults_match_contract_and_job_runs_outside_ui(self):
        self.client.headers["Authorization"] = "Bearer studio-secret"
        bootstrap = self.client.get("/api/v1/ui/bootstrap").json()
        self.assertFalse(bootstrap["leaderboard_enabled"])
        self.assertNotIn(str(self.settings.storage_dir), str(bootstrap))
        body = {"dataset_id": "ch", "config": bootstrap["config"]}
        body["config"]["basins"]["target_basins"] = ["4009"]
        ExperimentRequest.model_validate(body)
        submitted = self.client.post("/api/v1/experiments", json=body)
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job = submitted.json()
        self.assertEqual(job["status"], "queued")
        self.assertFalse(self.client.get("/api/v1/health").json()["worker_ready"])
        self.assertEqual(self.client.get(job["links"]["self"]).json()["job_id"], job["job_id"])
        self.assertEqual(self.client.post(job["links"]["cancel"]).json()["status"], "cancelled")
        self.assertEqual(self.client.get("/api/v1/ui/runs").json()["items"], [])

    def test_library_cards_exclude_heavy_result_arrays_and_server_paths(self):
        self.client.headers["Authorization"] = "Bearer studio-secret"
        run = SimpleNamespace(run_id="run-one", created_at="2026-10-17T12:00:00Z",
                              config={"data_root": str(self.settings.datasets["ch"].data_root),
                                      "dataset_id": "ch", "strategy": {"approach": "soft_routing"}},
                              aggregate_metrics={"rmse": 1.2}, parameter_count=123,
                              runtime_seconds=4.2, cancelled=False)
        with patch.object(self.application.runs, "list", return_value=([run], [])):
            page = self.client.get("/api/v1/ui/runs").json()
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["aggregate_metrics"], {"rmse": 1.2})
        self.assertNotIn("loss_traces", page["items"][0])
        self.assertNotIn("extreme_events", page["items"][0])
        self.assertNotIn(str(self.settings.storage_dir.parent), str(page))
