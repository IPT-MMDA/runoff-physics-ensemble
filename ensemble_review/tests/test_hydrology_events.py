from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import torch

from complexity_ensemble.hydrology_events import basin_event_metrics, regional_event_metrics
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, DateRange, DateSplitConfig,
    ExtremeEventConfig, ForecastConfig, HyperparameterConfig, ModelArchitectureConfig,
    RegionalExperimentConfig, TrainingStrategyConfig, _rolling_test_predictions,
    make_regional_hydrology_data, run_regional_experiment,
)
from complexity_ensemble.hydrology_data import HydrologySeries


def test_spell_duration_deficit_and_missing_day_break():
    days = [date(2020, 1, 1) + timedelta(days=index) for index in (0, 1, 2, 4, 5)]
    flow = [4.0, 1.0, 1.0, 1.0, 1.0]
    score = basin_event_metrics(
        days, flow, flow, event_type="low_flow_spell", high_threshold=8.0,
        low_threshold=2.0, rise_threshold=3.0, fall_threshold=3.0,
        minimum_days=2, pulse_gap_days=1,
    )
    assert score["observed_events"] == 2
    assert score["observed_event_days"] == 4
    assert score["observed_severity"] == 4.0
    assert score["event_recall"] == 1.0


def test_repeated_pulses_and_regional_concurrence():
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(7)]
    flow = [0.0, 9.0, 9.0, 0.0, 9.0, 9.0, 0.0]
    pulse = basin_event_metrics(
        days, flow, flow, event_type="repeated_high_flow", high_threshold=8.0,
        low_threshold=2.0, rise_threshold=3.0, fall_threshold=3.0,
        minimum_days=3, pulse_gap_days=1,
    )
    assert pulse["observed_events"] == 1
    assert pulse["observed_event_days"] == 4
    rows = {basin: list(zip(days, flow, flow)) for basin in ("a", "b")}
    regional = regional_event_metrics(rows, {"a": 8.0, "b": 8.0}, 2, 2)
    assert regional["observed_events"] == 2
    low_regional = regional_event_metrics(rows, {"a": 2.0, "b": 2.0}, 2, 1,
                                          direction="low")
    assert low_regional["observed_events"] == 3
    fall = basin_event_metrics(
        days, flow, flow, event_type="rapid_fall", high_threshold=8.0,
        low_threshold=2.0, rise_threshold=3.0, fall_threshold=3.0,
        minimum_days=2, pulse_gap_days=1,
    )
    assert fall["observed_events"] == 2


def test_rolling_rollout_ignores_observed_discharge_inside_forecast_block():
    dates = tuple(date(2001, 1, 1) + timedelta(days=index) for index in range(27))
    discharge = torch.arange(27, dtype=torch.float32) + 1
    forcings = torch.ones(27, 3)
    series = HydrologySeries(dates, discharge, forcings,
                             ("precipitation", "temperature", "pet"), "a", "Switzerland")
    data = make_regional_hydrology_data(
        {"a": series}, {}, BasinScopeConfig(("a",), "targets_only"),
        DateSplitConfig(DateRange("2001-01-05", "2001-01-14"),
                        DateRange("2001-01-15", "2001-01-18"),
                        DateRange("2001-01-19", "2001-01-27")),
        sequence_length=3,
    )
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("a",), "targets_only"),
        DateSplitConfig(DateRange("2001-01-05", "2001-01-14"),
                        DateRange("2001-01-15", "2001-01-18"),
                        DateRange("2001-01-19", "2001-01-27")),
        strategy=TrainingStrategyConfig(approach="no_routing"),
        forecast=ForecastConfig("rolling", 3),
    )

    class Model:
        def complex_expert(self, inputs, static):
            return inputs[:, -1, q_index:q_index + 1] * data.feature_scale[q_index] + data.feature_mean[q_index]

        def _positive_discharge(self, raw):
            return raw / data.discharge_scale

        def physics_residual(self, predicted, physical):
            return torch.zeros_like(predicted)

    q_index = data.feature_names.index("previous_discharge")
    original = _rolling_test_predictions(Model(), data, config, 0, torch.device("cpu"),
                                         stacking_weight=None, ood_detector=None)
    changed_inputs = data.test.inputs.clone()
    changed_physical = data.test.physical_inputs.clone()
    changed_inputs[1:3, :, q_index] = 1000
    changed_physical[1:3, :, q_index] = 1000
    changed = replace(data, test=replace(data.test, inputs=changed_inputs,
                                          physical_inputs=changed_physical))
    rerun = _rolling_test_predictions(Model(), changed, config, 0, torch.device("cpu"),
                                      stacking_weight=None, ood_detector=None)
    assert [row.predicted_mm_day for row in original[:3]] == pytest.approx(
        [row.predicted_mm_day for row in rerun[:3]])
    assert [row.lead_day for row in original] == [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert original[0].issue_date == original[0].target_date - timedelta(days=1)


@pytest.mark.parametrize("approach", ["no_routing", "soft_routing", "ood_fallback", "stacking"])
def test_event_based_rolling_run_produces_forecasts_and_event_scores(approach):
    days = tuple(date(2001, 1, 1) + timedelta(days=index) for index in range(45))
    forcings = torch.stack((torch.arange(45).float() % 6, torch.ones(45),
                            torch.full((45,), 0.5)), dim=1)
    series = {
        basin: HydrologySeries(
            days, torch.tensor([1.0 + (index % 7) * 0.5 + offset
                                for index in range(45)]), forcings,
            ("precipitation", "temperature", "pet"), basin, "Switzerland",
        )
        for basin, offset in (("a", 0.0), ("b", 0.2))
    }
    dates = DateSplitConfig(DateRange("2001-01-05", "2001-01-24"),
                            DateRange("2001-01-25", "2001-01-31"),
                            DateRange("2001-02-01", "2001-02-14"))
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("a", "b"), "all_basins"), dates,
        model=ModelArchitectureConfig(complex_expert="mlp", mlp_widths=(8,)),
        strategy=TrainingStrategyConfig(approach, 1),
        hyperparameters=HyperparameterConfig(sequence_length=3, batch_size=128),
        extremes=ExtremeEventConfig(mode="event_based", minimum_days=2),
        forecast=ForecastConfig("rolling", 4),
    )
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(basin) for basin in series])
    result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)
    assert result.predictions and not result.failures
    assert len(result.predictions) == 28
    assert {row["event_type"] for row in result.extreme_events} == set(config.extremes.event_types)
    assert result.resolved_config["forecast"] == {"mode": "rolling", "horizon_days": 4}
    assert {row.lead_day for row in result.predictions} == {1, 2, 3, 4}
    assert set(result.forecast_metrics) == {"1", "2", "3", "4"}
