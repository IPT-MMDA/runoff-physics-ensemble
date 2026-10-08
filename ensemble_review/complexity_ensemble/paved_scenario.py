"""Daily, user-defined paved-site runoff and retention sensitivity scenario.

The historical forecast is the baseline. Only the difference between the
original and proposed site's connected runoff is applied to that forecast.
This is a daily bucket calculation, not a calibrated SWMM simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import torch

from .hydrology_data import HydrologySeries
from .hydrology_events import basin_event_metrics
from .regional import (
    DateRange, ExtremeEventConfig, PredictionRecord, _event_thresholds, _thresholds,
)


@dataclass(frozen=True)
class PavedSiteConfig:
    site_fraction: float = 0.001  # fraction of upstream basin area
    original_paved_fraction: float = 0.1
    proposed_paved_fraction: float = 0.8
    pervious_runoff_coefficient: float = 0.25
    paved_runoff_coefficient: float = 0.9
    connected_fraction: float = 1.0
    storage_capacity_mm: float = 0.0  # depth over the entire site
    release_mm_day: float = 0.0  # depth over the entire site
    start_date: date | None = None

    def validate(self) -> None:
        for name in (
            "site_fraction", "original_paved_fraction", "proposed_paved_fraction",
            "pervious_runoff_coefficient", "paved_runoff_coefficient", "connected_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("storage_capacity_mm", "release_mm_day"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.paved_runoff_coefficient < self.pervious_runoff_coefficient:
            raise ValueError("paved runoff coefficient must be at least the pervious coefficient")


@dataclass(frozen=True)
class PavedScenarioResult:
    # One row per seed and forecast date; baseline and scenario share the model run.
    rows: list[dict[str, object]]
    events: list[dict[str, object]]
    summary: dict[str, object]


def run_paved_site_scenario(
    predictions: Sequence[PredictionRecord], series: HydrologySeries,
    train_range: DateRange, extremes: ExtremeEventConfig, config: PavedSiteConfig,
) -> PavedScenarioResult:
    """Apply one upstream site's daily runoff difference to test predictions."""
    config.validate()
    if "precipitation" not in series.forcing_names:
        raise ValueError("daily precipitation is required for the paved-site scenario")
    selected = sorted(
        (row for row in predictions if row.basin_id == series.basin_id),
        key=lambda row: (row.seed, row.target_date),
    )
    if not selected:
        raise ValueError(f"no predictions are available for basin {series.basin_id}")
    start = config.start_date or min(row.target_date for row in selected)
    last = max(row.target_date for row in selected)
    if start > last:
        raise ValueError("scenario start date must be on or before the last predicted day")
    precipitation_index = series.forcing_names.index("precipitation")
    old_coefficient = (
        config.original_paved_fraction * config.paved_runoff_coefficient
        + (1 - config.original_paved_fraction) * config.pervious_runoff_coefficient
    )
    new_coefficient = (
        config.proposed_paved_fraction * config.paved_runoff_coefficient
        + (1 - config.proposed_paved_fraction) * config.pervious_runoff_coefficient
    )
    daily_delta: dict[date, float] = {}
    storage = 0.0
    previous: date | None = None
    gaps = 0
    for index, day in enumerate(series.dates):
        if day < start or day > last:
            continue
        if previous is not None and day != previous + timedelta(days=1):
            storage = 0.0
            gaps += 1
        previous = day
        rain = max(0.0, float(series.forcings[index, precipitation_index]))
        original = rain * old_coefficient
        proposed = rain * new_coefficient
        available = storage + proposed
        overflow = max(0.0, available - config.storage_capacity_mm)
        storage = min(available, config.storage_capacity_mm)
        release = min(storage, config.release_mm_day)
        storage -= release
        daily_delta[day] = (
            config.site_fraction * config.connected_fraction
            * (overflow + release - original)
        )
    missing = sorted({row.target_date for row in selected if row.target_date >= start}
                     - daily_delta.keys())
    if missing:
        raise ValueError(f"precipitation is unavailable for predicted day {missing[0]}")
    rows: list[dict[str, object]] = []
    clipped = 0
    for row in selected:
        delta = daily_delta.get(row.target_date, 0.0)
        scenario = row.predicted_mm_day + delta
        if scenario < 0:
            scenario = 0.0
            clipped += 1
        rows.append({
            "seed": row.seed, "date": row.target_date,
            "baseline_mm_day": row.predicted_mm_day,
            "scenario_mm_day": scenario,
            "delta_mm_day": scenario - row.predicted_mm_day,
        })

    # Fixed training-period thresholds make before/after event counts comparable.
    high_q95 = _thresholds(
        ExtremeEventConfig(mode="statistical", definition="automatic_q95"),
        (series.basin_id,), {series.basin_id: series}, train_range,
    )[series.basin_id]
    if extremes.mode == "statistical":
        high_threshold = _thresholds(
            extremes, (series.basin_id,), {series.basin_id: series}, train_range,
        )[series.basin_id]
    else:
        high_threshold = high_q95
    high, low, rise, fall = _event_thresholds(
        extremes, (series.basin_id,), {series.basin_id: series}, train_range,
    )[series.basin_id]
    event_types = (
        tuple(kind for kind in extremes.event_types if not kind.startswith("regional_"))
        if extremes.mode == "event_based" else
        ("high_flow_spell", "low_flow_spell", "rapid_rise", "rapid_fall", "repeated_high_flow")
    )
    events: list[dict[str, object]] = []
    for seed in sorted({row.seed for row in selected}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        days = [row["date"] for row in seed_rows]
        baseline = [float(row["baseline_mm_day"]) for row in seed_rows]
        scenario = [float(row["scenario_mm_day"]) for row in seed_rows]
        events.append({
            "seed": seed, "event_type": "daily_high_flow",
            "threshold_mm_day": high_threshold,
            "baseline_events": sum(value > high_threshold for value in baseline),
            "scenario_events": sum(value > high_threshold for value in scenario),
            "baseline_event_days": sum(value > high_threshold for value in baseline),
            "scenario_event_days": sum(value > high_threshold for value in scenario),
        })
        for kind in event_types:
            metrics = basin_event_metrics(
                days, baseline, scenario, event_type=kind,
                high_threshold=high, low_threshold=low,
                rise_threshold=rise, fall_threshold=fall,
                minimum_days=extremes.minimum_days,
                pulse_gap_days=extremes.pulse_gap_days,
            )
            events.append({
                "seed": seed, "event_type": kind,
                "threshold_mm_day": high if kind in {"high_flow_spell", "repeated_high_flow"}
                else low if kind == "low_flow_spell" else rise if kind == "rapid_rise" else fall,
                "baseline_events": metrics["observed_events"],
                "scenario_events": metrics["predicted_events"],
                "baseline_event_days": metrics["observed_event_days"],
                "scenario_event_days": metrics["predicted_event_days"],
            })
    summary = {
        "basin_id": series.basin_id, "start_date": start.isoformat(),
        "site_fraction_of_basin": config.site_fraction,
        "original_runoff_coefficient": old_coefficient,
        "proposed_runoff_coefficient": new_coefficient,
        "end_storage_mm_over_site": storage,
        "mean_delta_mm_day": sum(float(row["delta_mm_day"]) for row in rows) / len(rows),
        "max_delta_mm_day": max(float(row["delta_mm_day"]) for row in rows),
        "clipped_negative_days": clipped,
        "rainfall_gap_resets": gaps,
        "q95_training_mm_day": high_q95,
    }
    return PavedScenarioResult(rows, events, summary)
