"""Hydropower works through the API without native Gradio or a worker."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
from fastapi.testclient import TestClient
from shapely.geometry import Point, box

from complexity_ensemble.api import create_app
from complexity_ensemble.application import Application
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog
from complexity_ensemble.server_settings import DatasetRegistration, HydropowerSettings, ServerSettings


class HydropowerAPITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.settings = ServerSettings(
            self.root / "state", {"ch": DatasetRegistration(self.root / "data")},
            token="test-token", hydropower=HydropowerSettings(geopackage=self.root / "missing.gpkg"))
        self.application = Application(self.settings)
        self.application.hydropower_sites = gpd.GeoDataFrame(
            {"_site_id": ["inside", "outside"], "_site_name": ["Matched plant", "Outside plant"]},
            geometry=[Point(1, 1), Point(10, 10)], crs="EPSG:4326")
        self.catalog = CAMELSCHCatalog(self.root / "data", [BasinCatalogRecord(
            "basin", "Test basin", {"area": 86.4}, eligible=True, geometry=box(0, 0, 2, 2))])
        self.application.catalog = lambda dataset_id: self.catalog
        self.predictions = self.root / "predictions.csv"
        self.predictions.write_text(
            "seed,basin_id,target_date,predicted_mm_day\n"
            "1,basin,2024-01-01,4\n2,basin,2024-01-01,4\n"
            "1,basin,2024-01-02,7\n2,basin,2024-01-02,9\n")
        self.runs = {
            "one": SimpleNamespace(run_id="one", created_at="2024-01-03", config={"dataset_id": "ch"}),
            "legacy": SimpleNamespace(run_id="legacy", created_at="2024-01-02",
                                      config={"data_root": str(self.root / "data")}),
            "unregistered": SimpleNamespace(run_id="unregistered", created_at="2024-01-01",
                                            config={"dataset_id": "missing"}),
        }
        def get(run_id):
            if run_id not in self.runs:
                raise KeyError("run not found")
            return self.runs[run_id]
        for name, replacement in (
            ("get", get), ("list", lambda: (list(self.runs.values()), [])),
            ("artifact", lambda run_id, kind: self.predictions),
        ):
            patcher = patch.object(self.application.runs, name, replacement)
            patcher.start(); self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(self.settings, application=self.application))
        self.client.headers["Authorization"] = "Bearer test-token"
        self.addCleanup(self.client.close)

    def test_all_hydropower_endpoints_require_authentication(self):
        self.client.headers.clear()
        for path in ("/hydropower/sites", "/hydropower/runs", "/runs/one/hydropower"):
            self.assertEqual(self.client.get("/api/v1" + path).status_code, 401)

    def test_inventory_needs_no_model_or_worker(self):
        response = self.client.get("/api/v1/hydropower/sites")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["features"]), 2)
        self.assertIsNone(response.json()["target_date"])
        self.assertIsNone(response.json()["features"][0]["properties"]["change_pct"])

    def test_run_picker_resolves_legacy_datasets_and_skips_unregistered_sources(self):
        response = self.client.get("/api/v1/hydropower/runs")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([r["run_id"] for r in response.json()["items"]], ["one", "legacy"])
        self.assertEqual(response.json()["items"][1]["dataset_id"], "ch")
        self.assertEqual(response.json()["skipped"], 1)
        self.assertNotIn(str(self.root), response.text)

    def test_latest_forecast_averages_seeds_and_matches_registered_catchments(self):
        response = self.client.get("/api/v1/runs/one/hydropower")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["dataset_id"], "ch")
        self.assertEqual(payload["available_dates"], ["2024-01-01", "2024-01-02"])
        self.assertEqual(payload["map"]["target_date"], "2024-01-02")
        self.assertEqual((payload["matched"], payload["with_change"], payload["total"]), (1, 1, 2))
        plant = payload["map"]["features"][0]["properties"]
        self.assertEqual(plant["predicted_m3_s"], 8)
        self.assertEqual(plant["change_pct"], 100)
        self.assertIsNone(payload["map"]["features"][1]["properties"]["change_pct"])
        self.assertNotIn(str(self.root), response.text)

    def test_explicit_date_and_missing_previous_day(self):
        response = self.client.get("/api/v1/runs/legacy/hydropower?target_date=2024-01-01")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["matched"], 1)
        self.assertEqual(response.json()["with_change"], 0)

    def test_invalid_dates_unknown_runs_and_unregistered_sources(self):
        for run_id, query, status in (
            ("one", "?target_date=bad", 422), ("one", "?target_date=2025-01-01", 422),
            ("missing", "", 404), ("unregistered", "", 404),
        ):
            response = self.client.get(f"/api/v1/runs/{run_id}/hydropower{query}")
            self.assertEqual(response.status_code, status, response.text)

    def test_disabled_feature_rejects_inventory_and_forecasts(self):
        settings = replace(self.settings, hydropower=HydropowerSettings(enabled=False))
        with TestClient(create_app(settings, application=self.application)) as client:
            for path in ("/hydropower/sites", "/hydropower/runs", "/runs/one/hydropower"):
                response = client.get("/api/v1" + path, headers={"Authorization": "Bearer test-token"})
                self.assertEqual(response.status_code, 409)
                self.assertIn("disabled", response.json()["message"])

    def test_missing_inventory_is_reported_without_exposing_server_paths(self):
        del self.application.hydropower_sites
        response = self.client.get("/api/v1/hydropower/sites")
        self.assertEqual(response.status_code, 422, response.text)
        self.assertNotIn(str(self.root), response.text)
