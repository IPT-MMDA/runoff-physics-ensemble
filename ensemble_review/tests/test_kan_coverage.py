"""Training-cutoff cohort counts, sparsity, weather gaps and API access."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from complexity_ensemble.kan_coverage import coverage_inventory, planned_usage
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog
from test_kan_case_study import small_config, synthetic_series


class CoverageTests(unittest.TestCase):
    def test_complex_model_requires_budgeted_target_labels_and_source_validation(self):
        row = {"training_observations": 0, "validation_observations": 0, "test_observations": 15}
        target = planned_usage(row, "target", [30], 3, ["kan_complex"])
        self.assertIn("30", str(target["train_used"]))
        self.assertNotEqual(target["readiness"], "Ready")
        validation = planned_usage(row, "validation", [30], 3, ["kan_complex"])
        self.assertNotEqual(validation["readiness"], "Not used by selected models")
        self.assertNotEqual(validation["readiness"], "Ready")

    def catalog(self, root="."):
        return CAMELSCHCatalog(
            root, [BasinCatalogRecord(b) for b in ("source", "validation", "target")]
        )

    def test_counts_gaps_and_months_inform_manual_roles(self):
        series = synthetic_series()
        q = series["target"].discharge.clone()
        q[8:30] = float("nan")
        q[[8, 10, 12]] = 1.0
        series["target"] = replace(series["target"], discharge=q)
        report = coverage_inventory(small_config(), self.catalog(), series_by_basin=series)
        rows = {r["basin_id"]: r for r in report["items"]}
        self.assertEqual(report["as_of"], "2001-01-30")
        self.assertEqual(rows["source"]["training_observations"], 22)
        self.assertEqual(rows["source"]["validation_observations"], 11)
        self.assertEqual(rows["source"]["test_observations"], 15)
        self.assertEqual(report["dates"]["test"], {"start": "2001-02-11", "end": "2001-02-25"})
        self.assertTrue(rows["source"]["source_ready"])
        target = rows["target"]
        self.assertEqual(target["training_observations"], 3)
        self.assertNotIn("data_regime", target)
        self.assertEqual(target["longest_missing_spell_days"], 17)
        self.assertEqual(target["longest_observed_spell_days"], 1)
        self.assertEqual(target["calendar_months_represented"], 1)
        self.assertEqual(target["observations_by_month"]["01"], 3)
        self.assertFalse(target["target_budget_ready"])

    def test_future_discharge_cannot_change_ranking_and_weather_gaps_are_explicit(self):
        series = synthetic_series()
        config = replace(small_config(), train_end="2001-01-14")
        before = coverage_inventory(config, self.catalog(), series_by_basin=series)
        for basin, item in series.items():
            q = item.discharge.clone()
            q[14:] = float("nan")
            series[basin] = replace(item, discharge=q)
        after = coverage_inventory(config, self.catalog(), series_by_basin=series)
        self.assertEqual(
            [(r["basin_id"], r["training_observations"]) for r in before["items"]],
            [(r["basin_id"], r["training_observations"]) for r in after["items"]],
        )
        self.assertTrue(
            all(
                r["validation_observations"] == 0 and r["test_observations"] == 0
                for r in after["items"]
            )
        )
        series = synthetic_series()
        weather = series["source"].forcings.clone()
        weather[0, 0] = float("nan")
        series["source"] = replace(series["source"], forcings=weather)
        row = next(
            r
            for r in coverage_inventory(small_config(), self.catalog(), series_by_basin=series)[
                "items"
            ]
            if r["basin_id"] == "source"
        )
        self.assertEqual(row["training_observations"], 22)
        self.assertEqual(row["weather_missing_days"], 1)
        self.assertFalse(row["source_ready"])

    def test_file_inventory_refreshes_when_labels_are_added(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observed = root / "timeseries" / "observation_based"
            simulated = root / "timeseries" / "simulation_based"
            observed.mkdir(parents=True)
            simulated.mkdir(parents=True)
            path = observed / "CAMELS_CH_obs_based_target.csv"
            path.write_text("date,discharge_spec(mm/d)\n2001-01-01,1\n2001-01-02,1\n2001-01-03,\n")
            (simulated / "CAMELS_CH_sim_based_target.csv").write_text(
                "date,precipitation_sim(mm/d),temperature_sim(degC),pet_sim(mm/d)\n"
                + "2000-12-31,1,5,0\n"
                + "".join(f"2001-01-{i:02d},1,5,0\n" for i in range(1, 6))
            )
            config = replace(
                small_config(),
                source_basins=(),
                validation_basins=(),
                target_basins=(),
                train_start="2001-01-01",
                train_end="2001-01-03",
                validation_start="2001-01-04",
                validation_end="2001-01-04",
                test_start="2001-01-05",
                test_end="2001-01-05",
                warmup_days=1,
            )
            catalog = CAMELSCHCatalog(root, [BasinCatalogRecord("target")])
            first = coverage_inventory(config, catalog)["items"][0]
            self.assertEqual(first["training_observations"], 2)
            self.assertEqual(first["weather_missing_days"], 0)
            path.write_text(path.read_text().replace("2001-01-03,\n", "2001-01-03,1\n"))
            second = coverage_inventory(config, catalog)["items"][0]
            self.assertEqual(second["training_observations"], 3)

    def test_authenticated_coverage_endpoint_does_not_require_selected_cohorts(self):
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.application import Application
        from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = ServerSettings(
                root / "state", {"ch": DatasetRegistration(root / "data")}, token="secret"
            )
            app = Application(settings)
            app.catalog = lambda dataset: self.catalog(root / "data")
            with TestClient(create_app(settings, application=app)) as client:
                payload = {
                    "dataset_id": "ch",
                    "config": asdict(
                        replace(
                            small_config(), source_basins=(), validation_basins=(), target_basins=()
                        )
                    ),
                }
                path = "/api/v1/case-studies/kan/coverage"
                self.assertEqual(client.post(path, json=payload).status_code, 401)
                client.headers["Authorization"] = "Bearer secret"
                response = client.post(path, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()["items"]), 3)
                self.assertTrue(
                    all(r["training_observations"] == 0 for r in response.json()["items"])
                )
                self.assertNotIn("data_regime", response.text)
                self.assertTrue(
                    all(
                        "test_observations" in r and "validation_observations" in r
                        for r in response.json()["items"]
                    )
                )
                self.assertNotIn(str(root), response.text)

    def test_native_scan_accepts_empty_cohorts_and_preserves_search_filter(self):
        import gradio as gr
        from complexity_ensemble.kan_case_study import CaseStudyConfig
        from complexity_ensemble.kan_case_study_ui import build_case_study_tab

        catalog = self.catalog()
        report = coverage_inventory(small_config(), catalog, series_by_basin=synthetic_series())
        defaults = asdict(CaseStudyConfig())
        fields = [
            "source_basins",
            "validation_basins",
            "target_basins",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
            "observation_budgets",
            "seeds",
            "device",
            "selected_models",
        ]
        fields.extend(k for k in defaults if k not in fields and k != "target_observation_budgets")
        values = [
            (
                ",".join(map(str, defaults[k]))
                if k in {"seeds", "observation_budgets", "mlp_widths"}
                else defaults[k]
            )
            for k in fields
        ]
        with tempfile.TemporaryDirectory() as folder:
            with gr.Blocks() as ui:
                build_case_study_tab(
                    gr, catalog, [(r.basin_id, r.basin_id) for r in catalog.records], Path(folder)
                )
            try:
                callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "scan_records")
                with patch(
                    "complexity_ensemble.kan_case_study_ui.coverage_inventory", return_value=report
                ) as scan:
                    output = callback(*values, [], "target", "all")
                self.assertEqual(scan.call_args.args[0].source_basins, ())
                self.assertEqual(len(output), 6)
                self.assertEqual(output[2]["basin_id"].tolist(), ["target"])
                self.assertEqual(len(output[3]["choices"]), 3)
                self.assertNotIn("data_regime", output[2].columns)
                self.assertIn("train_available", output[2].columns)
                self.assertIn("validation_available", output[2].columns)
                self.assertIn("test_available", output[2].columns)
                coverage_callback = next(
                    fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "coverage_rows"
                )
                target_row = coverage_callback(report, "target", "target", [], [], ["target"]).iloc[
                    0
                ]
                self.assertEqual(target_row["train_used"], "3, 6")
                self.assertEqual(target_row["validation_used"], 0)
                self.assertEqual(target_row["test_used"], 15)
                self.assertTrue(
                    all(
                        "high_data" not in text and "low_data" not in text
                        for text, _ in output[3]["choices"]
                    )
                )
                controls = [
                    component.label
                    for component in ui.blocks.values()
                    if hasattr(component, "label")
                ]
                self.assertIn("Target catchments · your selection", controls)
                self.assertIn("Your target calibration budgets", controls)
                self.assertIn("Compute device", controls)
                cohort_groups = [
                    component
                    for component in ui.blocks.values()
                    if getattr(component, "label", "")
                    in {
                        "Source catchments · your selection",
                        "Source-validation catchments · your selection",
                        "Target catchments · your selection",
                    }
                ]
                self.assertEqual(len(cohort_groups), 3)
                self.assertTrue(
                    all(isinstance(component, gr.CheckboxGroup) for component in cohort_groups)
                )
            finally:
                ui.close()

    def test_per_target_budgets_control_readiness_without_assigning_roles(self):
        config = replace(
            small_config(), observation_budgets=(40,), target_observation_budgets={"target": (3, 6)}
        )
        report = coverage_inventory(config, self.catalog(), series_by_basin=synthetic_series())
        target = next(row for row in report["items"] if row["basin_id"] == "target")
        self.assertEqual(target["requested_target_budget"], 6)
        self.assertTrue(target["target_budget_ready"])

    def test_planned_usage_flags_empty_test_split_despite_large_training_count(self):
        row = {
            "training_observations": 1096,
            "validation_observations": 0,
            "test_observations": 0,
            "weather_missing_days": 0,
        }
        target = planned_usage(row, "target", [30, 90, 180], 3)
        self.assertEqual(target["train_used"], "30, 90, 180")
        self.assertEqual(target["test_used"], 0)
        self.assertIn("Needs 3 test observations; has 0", target["readiness"])
        source = planned_usage(row, "source", [30], 3)
        self.assertEqual(source["train_used"], 1096)
        self.assertEqual(source["validation_used"], 0)
        self.assertEqual(source["test_used"], 0)
        self.assertEqual(source["readiness"], "Ready")
        validation = planned_usage(row, "validation", [30], 3)
        self.assertIn("Needs 3 validation observations", validation["readiness"])

    def test_unavailable_device_is_rejected_before_case_study_is_queued(self):
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.application import Application
        from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = ServerSettings(root / "state", {"ch": DatasetRegistration(root / "data")})
            app = Application(settings)
            with TestClient(create_app(settings, application=app)) as client:
                payload = {
                    "dataset_id": "ch",
                    "config": asdict(replace(small_config(), device="cuda")),
                }
                with patch("torch.cuda.is_available", return_value=False):
                    response = client.post("/api/v1/case-studies/kan", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn("CUDA was requested but is not available", response.json()["message"])
                self.assertEqual(client.get("/api/v1/case-studies/kan").json()["items"], [])


if __name__ == "__main__":
    unittest.main()
