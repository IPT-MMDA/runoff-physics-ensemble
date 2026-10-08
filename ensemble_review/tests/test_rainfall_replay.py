from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import torch

from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.rainfall_replay import (
    event_choices, forecast_snapshot, illustrative_bucket, make_replay, replay_figures,
)


def test_replay_uses_prior_context_and_test_day_predictions():
    start = date(2020, 1, 1)
    days = tuple(start + timedelta(days=i) for i in range(70))
    forcings = torch.tensor([
        [20.0 if 39 <= i <= 41 else 1.0, 5.0, 2.0]
        for i in range(70)
    ])
    series = HydrologySeries(
        days, torch.arange(70, dtype=torch.float32), forcings,
        ("precipitation", "temperature", "pet"), "A", "Switzerland",
    )
    predictions = [SimpleNamespace(basin_id="A", target_date=day,
                                   predicted_mm_day=float(i + 1))
                   for i, day in enumerate(days[30:])]
    choices = event_choices(series, predictions)
    assert choices
    replay = make_replay(series, predictions, choices[0][1])
    assert replay.dates[30] == date.fromisoformat(choices[0][1])
    assert all(value != value for value in replay.predicted[:30])  # no predicted context
    assert replay.predicted[30] == predictions[days[30:].index(replay.dates[30])].predicted_mm_day
    first = replay_figures(replay, 1)
    last = replay_figures(replay, 14)
    assert len(first) == len(last) == 4
    assert all(figure.layout.showlegend is None or figure.layout.showlegend
               for figure in first[:3])
    assert [[trace.name for trace in figure.data if trace.showlegend is not False]
            for figure in first[:3]] == [
        ["Rainfall", "PET"],
        ["Observed", "Predicted (seed mean)"],
        ["Storage", "Evaporation", "Stream release"],
    ]
    assert all(not figure.frames for figure in first[:3])
    assert replay.dates[29].isoformat() in first[3]
    assert replay.dates[30].isoformat() in first[3]
    assert "Target-day weather is not used" in first[3]
    assert first[3] != last[3]
    assert first[1].data[2].x[0] != last[1].data[2].x[0]
    assert forecast_snapshot(replay, 1)["end_date"] == replay.dates[29]


def test_teaching_bucket_conserves_water_and_caps_evaporation():
    rain = (10.0, 0.0, 0.0)
    pet = (3.0, 20.0, 1.0)
    storage, evaporation, runoff = illustrative_bucket(rain, pet)
    previous = 0.0
    for i in range(len(rain)):
        assert storage[i] + evaporation[i] + runoff[i] == pytest.approx(previous + rain[i])
        assert evaporation[i] <= pet[i]
        previous = storage[i]


def test_rolling_handoff_uses_previous_prediction_for_feedback():
    days = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(50))
    series = HydrologySeries(
        days, torch.ones(50), torch.tensor([[2.0, 5.0, 1.0]] * 50),
        ("precipitation", "temperature", "pet"), "A", "Switzerland",
    )
    rows = [SimpleNamespace(
        basin_id="A", target_date=day, predicted_mm_day=float(i + 2),
        seed=0, issue_date=days[29], lead_day=i + 1,
    ) for i, day in enumerate(days[30:44])]
    replay = make_replay(series, rows, days[30], forecast_mode="rolling")
    assert forecast_snapshot(replay, 1)["previous_flow"] == pytest.approx(1.0)
    assert forecast_snapshot(replay, 2)["previous_flow"] == pytest.approx(2.0)
    assert "Predicted feedback" in replay_figures(replay, 2)[3]
