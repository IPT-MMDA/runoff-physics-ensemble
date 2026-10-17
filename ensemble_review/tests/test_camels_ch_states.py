from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch

import torch

from complexity_ensemble.camels_ch_states import augment_camels_ch_states
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, DateRange, DateSplitConfig,
    HyperparameterConfig, ModelArchitectureConfig, RegionalExperimentConfig,
    TrainingStrategyConfig, make_regional_hydrology_data, run_regional_experiment,
)
from complexity_ensemble.serialization import json_safe, public_config


class CAMELSCHStatesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        days = tuple(date(2001, 1, 1) + timedelta(days=i) for i in range(65))
        self.series = HydrologySeries(
            days, torch.linspace(1, 2, 65), torch.ones(65, 3),
            ("precipitation", "temperature", "pet"), "2011", "Switzerland",
        )
        self.config = RegionalExperimentConfig(
            self.root, BasinScopeConfig(("2011",), "targets_only"),
            DateSplitConfig(DateRange("2001-01-08", "2001-02-02"),
                            DateRange("2001-02-03", "2001-02-12"),
                            DateRange("2001-02-13", "2001-03-01")),
            model=ModelArchitectureConfig(complex_expert="mlp", simple_expert="fourier",
                                          mlp_widths=(8,), fourier_frequencies=4),
            strategy=TrainingStrategyConfig("no_routing", 1, 1, 1, 1),
            hyperparameters=HyperparameterConfig(sequence_length=7, batch_size=128),
        )
        for group, prefix, column in (
            ("observation_based", "obs", "swe(mm)"),
            ("simulation_based", "sim", "intercept_storage_sim(mm)"),
        ):
            directory = self.root / "timeseries" / group
            directory.mkdir(parents=True)
            path = directory / f"CAMELS_CH_{prefix}_based_2011.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(("date", column))
                # Reverse order exercises date joining rather than row position.
                for i in reversed(range(65)):
                    value = {1: "NaN", 2: -999, 3: "", 4: "inf"}.get(i, i)
                    writer.writerow((days[i], value))

    def test_selection_exact_dates_and_missing_masks(self):
        augmented = augment_camels_ch_states(
            self.root, {"2011": self.series}, ("snow_water_equivalent",),
        )["2011"]
        self.assertEqual(augmented.forcing_names[-2:],
                         ("snow_water_equivalent", "snow_water_equivalent_available"))
        self.assertEqual(augmented.forcings[:6, -2:].tolist(),
                         [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [5, 1]])
        self.assertEqual(self.series.forcings.shape[-1], 3)
        missing = replace(self.series, basin_id="missing")
        result = augment_camels_ch_states(self.root, {"missing": missing},
                                         ("canopy_interception_storage",))["missing"]
        self.assertEqual(float(result.forcings[:, -2:].sum()), 0)
        self.assertEqual(result.dates, missing.dates)

    def test_states_are_past_only_in_training_windows(self):
        series = augment_camels_ch_states(self.root, {"2011": self.series},
                                          ("snow_water_equivalent",))
        data = make_regional_hydrology_data(series, {}, self.config.basins,
                                           self.config.dates, sequence_length=7)
        index = data.feature_names.index("snow_water_equivalent")
        for split in (data.train, data.validation, data.test):
            for window, target in zip(split.physical_inputs, split.target_dates):
                expected = (target - self.series.dates[0]).days - 1
                self.assertEqual(float(window[-1, index]), expected)

    def test_training_and_configuration_roundtrip(self):
        catalog = CAMELSCHCatalog(self.root, [BasinCatalogRecord("2011")])
        for selected in ((), ("snow_water_equivalent",),
                         ("canopy_interception_storage",),
                         ("snow_water_equivalent", "canopy_interception_storage")):
            with self.subTest(selected=selected):
                config = replace(self.config, camels_state_features=selected)
                result = run_regional_experiment(
                    config, catalog=catalog, series_by_basin={"2011": self.series},
                )
                self.assertTrue(result.predictions)
                self.assertFalse(result.failures)
                names = result.resolved_config["feature_names"]
                for name in ("snow_water_equivalent", "canopy_interception_storage"):
                    self.assertEqual(name in names, name in selected)
                restored = config_from_mapping(json_safe(result.resolved_config))
                self.assertEqual(restored.camels_state_features, selected)
        legacy = json_safe(self.config)
        del legacy["camels_state_features"]
        self.assertEqual(config_from_mapping(legacy).camels_state_features, ())
        with self.assertRaisesRegex(ValueError, "unknown CAMELS-CH water-state"):
            replace(self.config, camels_state_features=("invalid",)).validate()

    def test_api_and_gradio_selection(self):
        from complexity_ensemble.api_models import ExperimentRequest
        from complexity_ensemble.app import _config_from_ui_values, build_app
        selected = ("canopy_interception_storage",)
        request = ExperimentRequest.model_validate({
            "dataset_id": "ch", "config": public_config(
                replace(self.config, camels_state_features=selected)),
        })
        self.assertEqual(request.config.camels_state_features, selected)
        import os
        catalog = CAMELSCHCatalog(self.root, [BasinCatalogRecord("2011")])
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            with patch("complexity_ensemble.app.CAMELSCHCatalog", return_value=catalog):
                ui = build_app(self.root)
        self.addCleanup(ui.close)
        components = {component._id: component for component in ui.blocks.values()
                      if hasattr(component, "value")}
        water = next(c for c in components.values()
                     if getattr(c, "label", None) == "Optional CAMELS-CH water states")
        self.assertEqual(water.value, [])
        # Exercise the actual Gradio callback order, including the new final control.
        dependencies = ui.config["dependencies"]
        dependency = next(d for d in dependencies
                          if water._id in d["inputs"] and len(d["inputs"]) > 70)
        ordered = [components[i].value for i in dependency["inputs"]]
        ordered[0] = ["2011"]
        ordered[2] = None  # cancellation token state
        ordered[dependency["inputs"].index(water._id)] = list(selected)
        config = _config_from_ui_values(self.root, ordered[0], ordered[1], ordered[3:-10])
        self.assertEqual(config.camels_state_features, selected)


if __name__ == "__main__":
    unittest.main()
