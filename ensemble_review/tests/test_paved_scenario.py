from datetime import date, timedelta

import pytest
import torch

from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.paved_scenario import PavedSiteConfig, run_paved_site_scenario
from complexity_ensemble.regional import (
    DateRange, ExperimentResult, ExtremeEventConfig, PredictionRecord,
)


def _inputs(rain=(10.0, 10.0, 0.0)):
    first = date(2020, 1, 1)
    days = tuple(first + timedelta(days=index) for index in range(15))
    rainfall = [0.0] * 10 + list(rain) + [0.0] * 2
    forcings = torch.tensor([[amount, 5.0, 1.0] for amount in rainfall])
    flow = torch.tensor([1.0 + index * 0.1 for index in range(15)])
    series = HydrologySeries(days, flow, forcings,
                             ("precipitation", "temperature", "pet"), "A", "CH")
    forecasts = [PredictionRecord(0, "A", days[index], 2.0, 2.0, 0.5)
                 for index in range(10, 13)]
    return series, forecasts, DateRange(days[0], days[9])


def test_paving_adds_only_change_from_original_site_runoff():
    series, forecasts, train = _inputs()
    setup = PavedSiteConfig(
        site_fraction=0.1, original_paved_fraction=0,
        proposed_paved_fraction=1, pervious_runoff_coefficient=0.2,
        paved_runoff_coefficient=1,
    )
    result = run_paved_site_scenario(
        forecasts, series, train, ExtremeEventConfig(), setup,
    )
    assert [row["delta_mm_day"] for row in result.rows] == pytest.approx([0.8, 0.8, 0])
    assert [row["scenario_mm_day"] for row in result.rows] == pytest.approx([2.8, 2.8, 2])
    assert result.summary["clipped_negative_days"] == 0
    assert any(row["event_type"] == "daily_high_flow" for row in result.events)


def test_retention_delays_runoff_and_preserves_storage_balance():
    series, forecasts, train = _inputs()
    setup = PavedSiteConfig(
        site_fraction=0.1, original_paved_fraction=0,
        proposed_paved_fraction=1, pervious_runoff_coefficient=0.2,
        paved_runoff_coefficient=1, storage_capacity_mm=10, release_mm_day=2,
    )
    result = run_paved_site_scenario(
        forecasts, series, train, ExtremeEventConfig(), setup,
    )
    assert [row["delta_mm_day"] for row in result.rows] == pytest.approx([0, 0.8, 0.2])
    assert result.summary["end_storage_mm_over_site"] == pytest.approx(6)


def test_start_date_keeps_earlier_predictions_at_baseline():
    series, forecasts, train = _inputs()
    setup = PavedSiteConfig(start_date=series.dates[11])
    result = run_paved_site_scenario(
        forecasts, series, train, ExtremeEventConfig(), setup,
    )
    assert result.rows[0]["delta_mm_day"] == 0
    assert result.rows[1]["delta_mm_day"] > 0


def test_scenario_parameters_are_bounded():
    with pytest.raises(ValueError, match="site_fraction"):
        PavedSiteConfig(site_fraction=1.1).validate()
    with pytest.raises(ValueError, match="non-negative"):
        PavedSiteConfig(storage_capacity_mm=-1).validate()


@pytest.mark.parametrize("start_text", ["", "2020-01-11"])
def test_gradio_scenario_callback_uses_configured_start_date(tmp_path, start_text):
    pytest.importorskip("gradio")
    from complexity_ensemble.app import build_app

    series, forecasts, _ = _inputs()
    observed = tmp_path / "timeseries" / "observation_based"
    simulated = tmp_path / "timeseries" / "simulation_based"
    observed.mkdir(parents=True)
    simulated.mkdir(parents=True)
    (observed / "CAMELS_CH_obs_based_A.csv").write_text(
        "date,discharge_spec(mm/d)\n" + "".join(
            f"{day.isoformat()},{float(value)}\n"
            for day, value in zip(series.dates, series.discharge)
        ), encoding="utf-8",
    )
    (simulated / "CAMELS_CH_sim_based_A.csv").write_text(
        "date,precipitation_sim(mm/d),temperature_sim(degC),pet_sim(mm/d)\n"
        + "".join(
            f"{day.isoformat()},{float(forcing[0])},5,1\n"
            for day, forcing in zip(series.dates, series.forcings)
        ), encoding="utf-8",
    )
    app = build_app(tmp_path)
    compare = next(function.fn for function in app.fns.values()
                   if function.fn.__name__ == "compare_paved_site")
    result = ExperimentResult(
        predictions=forecasts, aggregate_metrics={}, per_basin_metrics={},
        routing_diagnostics={}, loss_traces={}, extreme_events=[], failures=[],
        resolved_config={
            "basins": {"target_basins": ["A"]},
            "dates": {"train": {"start": "2020-01-01", "end": "2020-01-10"}},
            "extremes": {},
        },
    )
    summary, figure, events = compare(
        result, "A", 0.1, 10, 80, 100, 0.25, 0.9, 20, 2, start_text,
    )
    assert summary["start_date"] == "2020-01-11"
    assert len(figure.data) == 2
    assert not events.empty
