"""Daily, basin-aware hydrological event definitions and scores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping, Sequence


EVENT_TYPES = (
    "high_flow_spell", "low_flow_spell", "rapid_rise", "rapid_fall",
    "repeated_high_flow", "regional_concurrence", "regional_low_flow",
)


@dataclass(frozen=True)
class Event:
    start: date
    end: date
    duration_days: int
    severity: float


def _runs(dates: Sequence[date], active: Sequence[bool], severity: Sequence[float],
          minimum_days: int = 1) -> list[Event]:
    events: list[Event] = []
    start = None
    total = 0.0
    previous = None
    for day, flag, amount in zip(dates, active, severity):
        if start is not None and (not flag or day != previous + timedelta(days=1)):
            duration = (previous - start).days + 1
            if duration >= minimum_days:
                events.append(Event(start, previous, duration, total))
            start, total = None, 0.0
        if flag:
            if start is None:
                start = day
            total += amount
            previous = day
    if start is not None:
        duration = (previous - start).days + 1
        if duration >= minimum_days:
            events.append(Event(start, previous, duration, total))
    return events


def _pulses(spells: Sequence[Event], max_gap_days: int,
            available_dates: set[date]) -> list[Event]:
    clusters: list[list[Event]] = []
    for spell in spells:
        gap_days = (spell.start - clusters[-1][-1].end).days - 1 if clusters else 0
        gap_is_observed = bool(clusters) and all(
            clusters[-1][-1].end + timedelta(days=step) in available_dates
            for step in range(1, gap_days + 1)
        )
        if clusters and gap_days <= max_gap_days and gap_is_observed:
            clusters[-1].append(spell)
        else:
            clusters.append([spell])
    return [Event(group[0].start, group[-1].end,
                  sum(event.duration_days for event in group),
                  sum(event.severity for event in group))
            for group in clusters if len(group) >= 2]


def _event_scores(observed: Sequence[Event], predicted: Sequence[Event]) -> dict[str, float | int]:
    def overlaps(left: Event, right: Event) -> bool:
        return left.start <= right.end and right.start <= left.end

    matched_observed = [event for event in observed if any(overlaps(event, other) for other in predicted)]
    matched_predicted = [event for event in predicted if any(overlaps(event, other) for other in observed)]
    return {
        "observed_events": len(observed),
        "predicted_events": len(predicted),
        "event_recall": len(matched_observed) / len(observed) if observed else float("nan"),
        "event_precision": len(matched_predicted) / len(predicted) if predicted else float("nan"),
        "observed_event_days": sum(event.duration_days for event in observed),
        "predicted_event_days": sum(event.duration_days for event in predicted),
        "observed_severity": sum(event.severity for event in observed),
        "predicted_severity": sum(event.severity for event in predicted),
    }


def basin_event_metrics(
    dates: Sequence[date], observed: Sequence[float], predicted: Sequence[float],
    *, event_type: str, high_threshold: float, low_threshold: float,
    rise_threshold: float, fall_threshold: float,
    minimum_days: int, pulse_gap_days: int,
) -> dict[str, float | int]:
    if len(dates) != len(observed) or len(dates) != len(predicted):
        raise ValueError("event dates and values must have equal lengths")
    if tuple(dates) != tuple(sorted(dates)) or len(set(dates)) != len(dates):
        raise ValueError("event dates must be unique and sorted")

    def events(values: Sequence[float]) -> list[Event]:
        if event_type in {"high_flow_spell", "repeated_high_flow"}:
            amounts = [max(value - high_threshold, 0.0) for value in values]
            spells = _runs(dates, [value > high_threshold for value in values],
                          amounts, 1 if event_type == "repeated_high_flow" else minimum_days)
            return _pulses(spells, pulse_gap_days, set(dates)) if event_type == "repeated_high_flow" else spells
        if event_type == "low_flow_spell":
            return _runs(dates, [value < low_threshold for value in values],
                         [max(low_threshold - value, 0.0) for value in values], minimum_days)
        if event_type in {"rapid_rise", "rapid_fall"}:
            changes = [0.0] + [
                (values[index] - values[index - 1]) * (1 if event_type == "rapid_rise" else -1)
                if dates[index] == dates[index - 1] + timedelta(days=1) else 0.0
                for index in range(1, len(values))
            ]
            threshold = rise_threshold if event_type == "rapid_rise" else fall_threshold
            return _runs(dates, [change > threshold for change in changes],
                         [max(change - threshold, 0.0) for change in changes])
        raise ValueError(f"unknown basin event type: {event_type}")

    return _event_scores(events(observed), events(predicted))


def regional_event_metrics(
    rows_by_basin: Mapping[str, Sequence[tuple[date, float, float]]],
    thresholds: Mapping[str, float], minimum_basins: int,
    minimum_days: int, *, direction: str = "high",
) -> dict[str, float | int]:
    if direction not in {"high", "low"}:
        raise ValueError("regional event direction must be high or low")
    observed_counts: dict[date, int] = {}
    predicted_counts: dict[date, int] = {}
    coverage: dict[date, int] = {}
    for basin, rows in rows_by_basin.items():
        threshold = thresholds[basin]
        for day, observed, predicted in rows:
            coverage[day] = coverage.get(day, 0) + 1
            observed_counts[day] = observed_counts.get(day, 0) + int(
                observed > threshold if direction == "high" else observed < threshold)
            predicted_counts[day] = predicted_counts.get(day, 0) + int(
                predicted > threshold if direction == "high" else predicted < threshold)
    dates = sorted(day for day in coverage if coverage[day] >= minimum_basins)
    observed = _runs(dates, [observed_counts.get(day, 0) >= minimum_basins for day in dates],
                     [float(observed_counts.get(day, 0)) for day in dates], minimum_days)
    predicted = _runs(dates, [predicted_counts.get(day, 0) >= minimum_basins for day in dates],
                      [float(predicted_counts.get(day, 0)) for day in dates], minimum_days)
    return _event_scores(observed, predicted)
