"""Routing integration tests using synthetic daily discharge, without CAMELS."""
from dataclasses import asdict, replace
from datetime import date, timedelta
import importlib.util
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError

from complexity_ensemble.api_models import Hyperparameters
from complexity_ensemble.app import _config_from_ui_values, build_app
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.hydrology_complexity import (
    TakensDimensionEstimator, TakensPersistenceEstimator,
    build_hydrology_complexity_estimator,
)
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, CancellationToken, DateRange, DateSplitConfig,
    ForecastConfig, HyperparameterConfig, ModelArchitectureConfig,
    RegionalExperimentConfig, TrainingStrategyConfig,
    make_regional_hydrology_data, run_regional_experiment,
)


def configuration(method="takens", approach="soft_routing", **hyperparameters):
    return RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "targets_only"),
        DateSplitConfig(DateRange("2001-03-12", "2001-04-10"),
                        DateRange("2001-04-11", "2001-04-20"),
                        DateRange("2001-04-21", "2001-04-27")),
        ModelArchitectureConfig(complex_expert="mlp", simple_expert="fourier",
                                mlp_widths=(8,), fourier_frequencies=4),
        TrainingStrategyConfig(approach, 1, 1, 1, 1),
        HyperparameterConfig(sequence_length=7, batch_size=128,
                             complexity_method=method, routing_context_length=70,
                             embedding_dim=3, embedding_delay=2,
                             lyapunov_theiler_window=10, lyapunov_fit_horizon=8,
                             **hyperparameters),
    )


def daily_series():
    t = torch.arange(130, dtype=torch.float32)
    days = tuple(date(2001, 1, 1) + timedelta(days=i) for i in range(len(t)))
    discharge = 2 + 0.4 * torch.sin(t / 3) + 0.2 * torch.sin(t / 7) + t * 0.003
    forcings = torch.stack((2 + torch.sin(t / 5), 5 + torch.cos(t / 10),
                            torch.full_like(t, 0.5)), dim=1)
    return {"target": HydrologySeries(days, discharge, forcings,
                                     ("precipitation", "temperature", "pet"),
                                     "target", "CH")}


def execute(config, series=None):
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord("target")])
    return run_regional_experiment(config, catalog=catalog,
                                   series_by_basin=series or daily_series())


class TakensScoreTests(unittest.TestCase):
    def test_effective_dimension_separates_line_from_full_dimensional_cloud(self):
        generator = torch.Generator().manual_seed(3)
        windows = torch.stack((torch.arange(512.0),
                               torch.randn(512, generator=generator),
                               torch.ones(512)))[:, :, None]
        estimates = TakensDimensionEstimator(0, delay=1).estimate(windows)
        self.assertAlmostEqual(float(estimates[0, 0]), 1.0, places=5)
        self.assertGreater(float(estimates[1, 0]), 2.8)
        self.assertEqual(estimates[:, 1].tolist(), [1.0, 1.0, 0.0])
        rescaled = TakensDimensionEstimator(0, delay=1).estimate(10 * windows + 100)
        torch.testing.assert_close(estimates, rescaled)

    def test_ph_alias_retains_existing_estimator(self):
        for method in ("ph", "tda"):
            self.assertIsInstance(build_hydrology_complexity_estimator(
                method, ("previous_discharge",)), TakensPersistenceEstimator)


class ConfigurationTests(unittest.TestCase):
    def test_api_schema_and_saved_configuration_round_trip(self):
        schema = Hyperparameters.model_json_schema()["properties"]
        self.assertEqual(schema["complexity_method"]["default"], "morse")
        self.assertEqual(set(schema["complexity_method"]["enum"]),
                         {"morse", "ph", "takens", "lyapunov", "tda"})
        for method in ("morse", "ph", "takens", "lyapunov", "tda"):
            config = configuration(method)
            parsed = config_from_mapping(asdict(config))
            parsed.validate()
            self.assertEqual(parsed.hyperparameters, config.hyperparameters)
        with self.assertRaises(ValidationError):
            Hyperparameters(complexity_method="unknown")

    def test_rejects_insufficient_history_and_invalid_embedding(self):
        config = configuration()
        for updates in ({"routing_context_length": 5}, {"embedding_dim": 1},
                        {"embedding_delay": 0}, {"ph_score": "invalid"},
                        {"complexity_method": "lyapunov", "lyapunov_theiler_window": 100}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(config, hyperparameters=replace(config.hyperparameters, **updates)).validate()

    def test_gradio_values_include_routing_controls_without_shifting_old_fields(self):
        values = [
            "2001-03-12", "2001-04-10", "2001-04-11", "2001-04-20",
            "2001-04-21", "2001-04-27", "hard_routing", "mlp", "fourier",
            1, 7, "0", 128, 0.001, 0, 80, 0.15, 0.05, 0.05,
            1, 1, 1, "8", 16, 1, 64, 32, 4, "none", 0.05, 0, "cpu",
            "automatic_q95", None, 0.99, 0.1, "none", "hard_routing",
            3, 50, 3, 3, 2, 1, 0.001, "0.05,0.1", 0.05, False,
            False, "", [], False, "", [], "one_day", 7, [],
            0.8, 0.2, 0.95, 3, 7, 2, 0.95,
            "lyapunov", 70, 3, 2, 0.2, "persistent_holes", 10, 8, 0.7,
        ]
        config = _config_from_ui_values(".", ["target"], "targets_only", values)
        config.validate()
        self.assertEqual(config.hyperparameters.complexity_method, "lyapunov")
        self.assertEqual(config.hyperparameters.ph_score, "persistent_holes")
        self.assertEqual(config.hyperparameters.lyapunov_min_r2, 0.7)
        self.assertEqual(config.model.fourier_frequencies, 4)
        self.assertEqual(config.extremes.fall_quantile, 0.95)
        legacy = _config_from_ui_values(".", ["target"], "targets_only", values[:64])
        self.assertEqual(legacy.hyperparameters.complexity_method, "morse")

    def test_gradio_app_exposes_all_choices(self):
        catalog = CAMELSCHCatalog(".", [BasinCatalogRecord("target")])
        with tempfile.TemporaryDirectory() as results, patch(
            "complexity_ensemble.app.CAMELSCHCatalog", return_value=catalog
        ):
            app = build_app(".", results_dir=results)
        components = app.get_config_file()["components"]
        selector = next(item for item in components
                        if item["props"].get("label") == "Complexity routing method")
        self.assertEqual([choice[1] for choice in selector["props"]["choices"]],
                         ["morse", "ph", "takens", "lyapunov"])


class RegionalRoutingTests(unittest.TestCase):
    def test_long_routing_history_is_aligned_and_contains_only_past_values(self):
        config = configuration()
        series = daily_series()
        data = make_regional_hydrology_data(series, {}, config.basins, config.dates,
                                            sequence_length=7, routing_context_length=70)
        flow_index = data.feature_names.index("previous_discharge")
        source = series["target"]
        for split in (data.train, data.validation, data.test):
            self.assertEqual(split.routing_inputs.shape[1], 70)
            torch.testing.assert_close(split.inputs, split.routing_inputs[:, -7:])
            for index, day in enumerate(split.target_dates):
                target_index = source.dates.index(day)
                recovered = (split.routing_inputs[index, :, flow_index]
                             * data.feature_scale[flow_index] + data.feature_mean[flow_index])
                torch.testing.assert_close(recovered, source.discharge[target_index - 70:target_index])

    def test_history_based_routes_train_across_routed_approaches(self):
        for method in ("takens", "lyapunov"):
            for approach in ("soft_routing", "hard_routing", "distillation"):
                with self.subTest(method=method, approach=approach):
                    result = execute(configuration(method, approach))
                    self.assertFalse(result.failures, result.failures)
                    self.assertEqual(len(result.predictions), 7)
                    self.assertEqual(result.routing_diagnostics["complexity_method"], method)
                    self.assertTrue(result.routing_diagnostics["complexity_routing_active"])
                    self.assertEqual(result.resolved_config["hyperparameters"]["complexity_method"], method)

    def test_noise_recomputes_scores_with_the_long_history(self):
        result = execute(configuration(training_noise=0.1))
        self.assertFalse(result.failures, result.failures)
        self.assertEqual(len(result.predictions), 7)

    def test_cancellation_during_score_preparation_returns_an_inspectable_result(self):
        token = CancellationToken()

        def progress(fraction, message):
            if message.startswith("Computing"):
                token.cancel()

        result = run_regional_experiment(
            configuration(), progress, token,
            catalog=CAMELSCHCatalog(".", [BasinCatalogRecord("target")]),
            series_by_basin=daily_series(),
        )
        self.assertTrue(result.cancelled)
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.resolved_config["hyperparameters"]["complexity_method"], "takens")

    def test_no_routing_ignores_ph_and_does_not_require_ripser(self):
        with patch("complexity_ensemble.regional.build_hydrology_complexity_estimator",
                   side_effect=AssertionError("unused estimator was constructed")):
            result = execute(configuration("ph", "no_routing"))
        self.assertFalse(result.failures, result.failures)
        self.assertFalse(result.routing_diagnostics["complexity_routing_active"])

    def test_ph_diagrams_feed_both_routing_modes_and_distillation(self):
        # Isolate pipeline wiring from the optional PH engine. Numerical PH
        # behavior is covered by the real-Ripser tests when that extra is installed.
        for approach in ("soft_routing", "hard_routing", "distillation"):
            calls = []

            def persistence(points, maxdim):
                calls.append((len(points), maxdim))
                lifetime = 0.2 + (len(calls) % 5) * 0.2
                return {"dgms": [torch.empty(0, 2).numpy(),
                                 torch.tensor([[0.0, lifetime]]).numpy()]}

            with self.subTest(approach=approach), patch.dict(
                "sys.modules", {"ripser": types.SimpleNamespace(ripser=persistence)}
            ):
                result = execute(configuration("ph", approach))
                self.assertFalse(result.failures, result.failures)
                self.assertEqual(len(result.predictions), 7)
                self.assertEqual(result.routing_diagnostics["valid_complexity_fraction"], 1.0)
                self.assertTrue(all(count == 48 and dimension == 1 for count, dimension in calls))
                weights = [item.complex_weight for item in result.predictions]
                self.assertGreater(max(weights), min(weights))

    @unittest.skipUnless(importlib.util.find_spec("ripser"), "ripser is optional")
    def test_ph_routes_train_and_predict(self):
        for method in ("ph", "tda"):
            for approach in ("soft_routing", "hard_routing", "distillation"):
                with self.subTest(method=method, approach=approach):
                    result = execute(configuration(method, approach))
                    self.assertFalse(result.failures, result.failures)
                    self.assertEqual(len(result.predictions), 7)

    def test_rolling_history_uses_predictions_after_issue_date(self):
        config = replace(configuration(), forecast=ForecastConfig("rolling", 7))
        captured = []
        original = TakensDimensionEstimator.estimate

        def capture(estimator, windows):
            if len(windows) == 1:
                captured.append(windows.detach().clone())
            return original(estimator, windows)

        with patch.object(TakensDimensionEstimator, "estimate", capture):
            result = execute(config)
        self.assertFalse(result.failures, result.failures)
        self.assertEqual(len(captured), 7)
        data = make_regional_hydrology_data(daily_series(), {}, config.basins, config.dates,
                                            sequence_length=7, routing_context_length=70)
        flow_index = data.feature_names.index("previous_discharge")
        used_scores = torch.cat([original(TakensDimensionEstimator(flow_index, embedding_dim=3,
                                                                  delay=2, context_length=70),
                                          history) for history in captured])
        self.assertAlmostEqual(result.routing_diagnostics["mean_complexity_score"],
                               float(used_scores[:, 0].mean()), places=5)
        for lead, history in enumerate(captured):
            # Every flow after issue date must equal a previously generated prediction.
            for earlier in range(lead):
                position = 70 - lead + earlier
                recovered = (history[0, position, flow_index] * data.feature_scale[flow_index]
                             + data.feature_mean[flow_index])
                self.assertAlmostEqual(float(recovered), result.predictions[earlier].predicted_mm_day,
                                       places=5)
            self.assertEqual(result.predictions[lead].lead_day, lead + 1)


if __name__ == "__main__":
    unittest.main()
