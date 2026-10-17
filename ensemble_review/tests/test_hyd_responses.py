import csv
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from complexity_ensemble.hyd_responses import HYDResponses, HYD_RESPONSES_FEATURES, HYD_RESPONSES_SOURCES
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, DateRange, DateSplitConfig,
    HyperparameterConfig, ModelArchitectureConfig, RegionalExperimentConfig,
    TrainingStrategyConfig, run_regional_experiment,
    ForecastConfig, make_regional_hydrology_data, _rolling_test_predictions,
)
from complexity_ensemble.serialization import json_safe, public_config


def write_hyd(root, days, basins=("2011",), overlap=7.56):
    base = root / "data/01_timeseries/01_base_variables"
    units = root / "data/01_timeseries/00_meta_information/units"
    descriptive = root / "data/03_descriptive"
    for p in (base, units, descriptive):
        p.mkdir(parents=True, exist_ok=True)
    with (units / "01_base_variables.csv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(("variable", "unit"))
        writer.writerows((column, unit) for column, unit, _ in HYD_RESPONSES_SOURCES.values())
    with (descriptive / "HYDRESPONSES_descriptive_general_station_information.csv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(("catchmentnr", "percentage_overlap_CH"))
        writer.writerows((basin, overlap) for basin in basins)
    with (descriptive / "HYDRESPONSES_descriptive_hydro_geological_terrestrial_info.csv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(("catchmentnr", "drainage_density", "SOILDEPTH_deep"))
        writer.writerows((basin, 2, 30) for basin in basins)
    for basin in basins:
        with (base / f"HYDRESPONSES_basevars_catchment_{basin}.csv").open("w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(("catchmentnr", "date", *(v[0] for v in HYD_RESPONSES_SOURCES.values())))
            for i, day in enumerate(days):
                writer.writerow((basin, day, *("NA" if i == 1 else i / 100 for _ in HYD_RESPONSES_FEATURES)))


class HYDResponsesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        days = tuple(date(2001, 1, 1) + timedelta(days=i) for i in range(65))
        self.series = HydrologySeries(days, torch.linspace(1, 2, 65), torch.ones(65, 3),
                                      ("precipitation", "temperature", "pet"), "2011", "CH")
        write_hyd(self.root, days)
        self.config = RegionalExperimentConfig(
            self.root, BasinScopeConfig(("2011",), "targets_only"),
            DateSplitConfig(DateRange("2001-01-08", "2001-02-02"),
                            DateRange("2001-02-03", "2001-02-12"),
                            DateRange("2001-02-13", "2001-03-01")),
            model=ModelArchitectureConfig(complex_expert="mlp", simple_expert="fourier",
                                          mlp_widths=(8,), fourier_frequencies=4),
            strategy=TrainingStrategyConfig("no_routing", 1, 1, 1, 1),
            hyperparameters=HyperparameterConfig(sequence_length=7, batch_size=128),
            hyd_responses_root=self.root,
        )

    def test_exact_dates_units_and_partial_coverage(self):
        hyd = HYDResponses(self.root)
        augmented = hyd.augment({"2011": self.series}, ("hyd_soil_moisture_layer_1", "hyd_swe_oshd"))["2011"]
        self.assertEqual(augmented.forcings[0, 3:7].tolist(), [0, 1, 0, 1])
        self.assertEqual(augmented.forcings[1, 3:7].tolist(), [0, 0, 0, 0])
        self.assertAlmostEqual(float(augmented.forcings[2, 3]), .02)
        self.assertAlmostEqual(float(augmented.forcings[2, -2]), .0756)
        self.assertEqual(self.series.forcings.shape[-1], 3)
        missing = replace(self.series, basin_id="4008")
        result = hyd.augment({"4008": missing}, HYD_RESPONSES_FEATURES)["4008"]
        self.assertEqual(float(result.forcings[:, 3:].sum()), 0)
        hyd.units["swvl1_mean_mean"] = "mm"
        with self.assertRaisesRegex(ValueError, "units"):
            hyd.augment({"2011": self.series}, ("hyd_soil_moisture_layer_1",))
        with self.assertRaisesRegex(ValueError, "unknown HYD-RESPONSES"):
            hyd.augment({}, ("Q",))

    def test_training_static_selection_and_saved_config(self):
        catalog = CAMELSCHCatalog(self.root, [BasinCatalogRecord("2011")])
        for selected, static in ((("hyd_soil_moisture_layer_2",), False),
                                 (HYD_RESPONSES_FEATURES, True), ((), True)):
            config = replace(self.config, hyd_responses_features=selected, hyd_responses_static=static)
            if selected == HYD_RESPONSES_FEATURES:
                config = replace(config,
                    model=ModelArchitectureConfig(complex_expert="pinnmamba", simple_expert="fourier",
                                                  mamba_hidden_size=8, mamba_layers=1,
                                                  mamba_feed_forward_size=16, fourier_frequencies=4),
                    strategy=TrainingStrategyConfig("soft_routing", 1, 1, 1, 1))
            result = run_regional_experiment(config, catalog=catalog, series_by_basin={"2011": self.series})
            self.assertTrue(result.predictions)
            self.assertFalse(result.failures)
            for name in HYD_RESPONSES_FEATURES:
                self.assertEqual(name in result.resolved_config["feature_names"], name in selected)
            self.assertEqual("hyd_drainage_density" in result.resolved_config["static_feature_names"], static)
            restored = config_from_mapping(json_safe(result.resolved_config))
            self.assertEqual(restored.hyd_responses_features, selected)
            self.assertEqual(restored.hyd_responses_static, static)
        with self.assertRaisesRegex(ValueError, "at least one"):
            replace(self.config, hyd_responses_features=()).validate()
        self.assertNotIn("hyd_responses_root", public_config(self.config))

    def test_rolling_forecasts_do_not_read_future_water_states(self):
        series = HYDResponses(self.root).augment({"2011": self.series}, ("hyd_soil_moisture_layer_1",))
        data = make_regional_hydrology_data(series, {}, self.config.basins,
                                           self.config.dates, sequence_length=7)
        config = replace(self.config, forecast=ForecastConfig("rolling", 3))
        index = data.feature_names.index("hyd_soil_moisture_layer_1")

        class Model:
            def complex_expert(self, inputs, static):
                return inputs[:, -1, index:index + 1] * data.feature_scale[index] + data.feature_mean[index]

            def _positive_discharge(self, raw):
                return raw / data.discharge_scale

            def physics_residual(self, predicted, physical):
                return torch.zeros_like(predicted)

        def predict(data):
            return _rolling_test_predictions(Model(), data, config, 0, torch.device("cpu"),
                                              stacking_weight=None, ood_detector=None)

        original = predict(data)
        inputs = data.test.inputs.clone()
        physical = data.test.physical_inputs.clone()
        inputs[1:3, -1, index] = 1000
        physical[1:3, -1, index] = 1000
        changed = predict(replace(data, test=replace(data.test, inputs=inputs, physical_inputs=physical)))
        self.assertEqual([r.predicted_mm_day for r in original[:3]], [r.predicted_mm_day for r in changed[:3]])
        self.assertEqual(len({r.predicted_mm_day for r in original[:3]}), 1)

    def test_gradio_selection_and_registered_api_worker(self):
        from complexity_ensemble.app import build_app, _config_from_ui_values
        from complexity_ensemble.api_models import ExperimentRequest
        from complexity_ensemble.application import Application
        from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings
        import os
        catalog = CAMELSCHCatalog(self.root, [BasinCatalogRecord("2011")])
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            with patch("complexity_ensemble.app.CAMELSCHCatalog", return_value=catalog):
                ui = build_app(self.root, hyd_responses_root=self.root)
        self.addCleanup(ui.close)
        components = {c._id: c for c in ui.blocks.values() if hasattr(c, "value")}
        feature = next(c for c in components.values() if getattr(c, "label", None) == "HYD-RESPONSES water-state features")
        static = next(c for c in components.values() if getattr(c, "label", None) == "Include HYD-RESPONSES physical catchment descriptors")
        enabled = next(c for c in components.values() if getattr(c, "label", None) == "Include HYD-RESPONSES water states and descriptors")
        dependency = next(d for d in ui.config["dependencies"] if feature._id in d["inputs"] and len(d["inputs"]) > 70)
        values = [components[i].value for i in dependency["inputs"]]
        values[dependency["inputs"].index(feature._id)] = ["hyd_soil_moisture_layer_3"]
        values[dependency["inputs"].index(static._id)] = True
        configured = _config_from_ui_values(self.root, ["2011"], "targets_only", values[3:-10])
        self.assertEqual(configured.hyd_responses_features, ("hyd_soil_moisture_layer_3",))
        self.assertTrue(configured.hyd_responses_static)
        values[dependency["inputs"].index(enabled._id)] = False
        self.assertIsNone(_config_from_ui_values(self.root, ["2011"], "targets_only", values[3:-10]).hyd_responses_root)
        request = ExperimentRequest.model_validate({"dataset_id": "ch", "use_hyd_responses": True,
                                                    "config": public_config(self.config)})
        app = Application(ServerSettings(self.root / "server", {"ch": DatasetRegistration(self.root, hyd_responses_root=self.root)}))
        app.catalog = lambda _: catalog
        restored = app.experiment_config(request.model_dump(mode="json"))
        self.assertEqual(restored.hyd_responses_root, self.root)
        result = run_regional_experiment(restored, catalog=catalog, series_by_basin={"2011": self.series})
        self.assertTrue(result.predictions)
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.jobs import JobStore
        from complexity_ensemble.worker import run_job
        store = JobStore(app.settings.jobs_path)
        with TestClient(create_app(app.settings, application=app, store=store)) as client:
            listed = client.get("/api/v1/datasets/ch/features").json()
            self.assertEqual(listed["hyd_responses"], list(HYD_RESPONSES_FEATURES))
            submitted = client.post("/api/v1/experiments", json=request.model_dump(mode="json"))
            self.assertEqual(submitted.status_code, 202, submitted.text)
            store.acquire_worker("test-worker", stale_after=0)
            job = store.claim("test-worker")
            def execute(config, progress_callback=None, cancellation_token=None, **kwargs):
                return run_regional_experiment(config, progress_callback, cancellation_token,
                                               catalog=catalog, series_by_basin={"2011": self.series})
            with patch("complexity_ensemble.services.run_regional_experiment", execute):
                output = run_job(app, store, job, "test-worker")
            store.finish(job["id"], "test-worker", result=output)
            store.release_worker("test-worker")
            saved = app.runs.get(output["run_id"])
            self.assertIn("hyd_soil_moisture_layer_1", saved.config["feature_names"])
            self.assertNotIn("hyd_responses_root", saved.config["requested_config"])
        with self.assertRaisesRegex(ValueError, "not configured"):
            Application(ServerSettings(self.root / "other", {"ch": DatasetRegistration(self.root)})).experiment_config(request.model_dump(mode="json"))


if __name__ == "__main__":
    unittest.main()
