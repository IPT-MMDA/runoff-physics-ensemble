"""HYD-RESPONSES-style drought verification, independent of models and UI.

Thresholds use training observations only. Each forecast issue is evaluated
separately; post-issue observations are never used in its predicted smoothing.
Existing hydrological event scoring is deliberately left untouched.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import mean
from typing import Mapping

DAY = timedelta(days=1)
VERIFICATION = "drought_one_to_one_overlap_v1"


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def smooth(flow: Mapping[date, float], day: date, smoothing: str):
    if smoothing not in {"trailing", "centered"}:
        raise ValueError("Drought smoothing must be trailing or centered")
    offsets = range(-6, 1) if smoothing == "trailing" else range(-3, 4)
    values = [flow.get(day + offset * DAY, float("nan")) for offset in offsets]
    return mean(values) if all(math.isfinite(value) for value in values) else None


def calendar_day(day):
    # A common leap-year calendar keeps March dates aligned across all years.
    return date(2000, day.month, day.day).timetuple().tm_yday


@dataclass(frozen=True)
class DroughtThresholds:
    definition: str
    fixed: float | None
    seasonal: Mapping[int, float]

    def at(self, day):
        return self.fixed if self.fixed is not None else self.seasonal.get(calendar_day(day))


def fit_thresholds(flow, start, end, *, definition="seasonal_quantile",
                   smoothing="trailing", probability=.15, calendar_window=31):
    if definition not in {"seasonal_quantile", "q347"}:
        raise ValueError("Unknown drought threshold definition")
    training = {day: value for day, value in flow.items()
                if start <= day <= end and math.isfinite(value) and value >= 0}
    if not training:
        raise ValueError("Drought thresholds require finite training-period discharge observations")
    if definition == "q347":
        return DroughtThresholds(definition, quantile(list(training.values()), 18 / 365), {})
    smoothed = [(calendar_day(day), value) for day in sorted(training)
                if (value := smooth(training, day, smoothing)) is not None]
    if not smoothed:
        raise ValueError("Drought thresholds need at least seven consecutive training days")
    thresholds = {}
    radius = calendar_window // 2
    samples = defaultdict(list)
    for doy, value in smoothed:
        samples[doy].append(value)
    for doy in range(1, 367):
        values = [value for offset in range(-radius, radius + 1)
                  for value in samples[(doy - 1 + offset) % 366 + 1]]
        if values:
            thresholds[doy] = quantile(values, probability)
    return DroughtThresholds(definition, None, thresholds)


@dataclass(frozen=True)
class DroughtEvent:
    start: date
    end: date
    duration: int
    deficit: float
    left_censored: bool
    right_censored: bool


def extract_events(rows, field, minimum_days, left_context=None):
    """Invalid days split segments; boundary events have unknown onset/recovery."""
    events = []
    for index, row in enumerate(rows):
        if row[field] >= row["threshold_mm_day"]:
            continue
        if (index and rows[index - 1]["date"] == row["date"] - DAY
                and rows[index - 1][field] < rows[index - 1]["threshold_mm_day"]):
            continue
        end = index
        while (end + 1 < len(rows) and rows[end + 1]["date"] == rows[end]["date"] + DAY
               and rows[end + 1][field] < rows[end + 1]["threshold_mm_day"]):
            end += 1
        if end - index + 1 < minimum_days:
            continue
        left = index == 0 or rows[index - 1]["date"] != row["date"] - DAY
        if (index == 0 and left_context is not None
                and left_context["date"] == row["date"] - DAY
                and left_context[field] >= left_context["threshold_mm_day"]):
            left = False
        right = end == len(rows) - 1 or rows[end + 1]["date"] != rows[end]["date"] + DAY
        events.append(DroughtEvent(row["date"], rows[end]["date"], end - index + 1,
                      sum(item["threshold_mm_day"] - item[field] for item in rows[index:end + 1]), left, right))
    return events


def match_events(observed, predicted):
    """Maximum-total-overlap one-to-one matching, preserving temporal order.

    Disjoint ordered intervals cannot have crossing positive-overlap matches.
    Dynamic programming avoids a greedy choice hiding a better global match.
    """
    n, m = len(observed), len(predicted)
    scores = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    actions = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            overlap = (min(observed[i - 1].end, predicted[j - 1].end)
                       - max(observed[i - 1].start, predicted[j - 1].start)).days + 1
            choices = [(scores[i - 1][j], "observed"), (scores[i][j - 1], "predicted")]
            if overlap > 0:
                previous = scores[i - 1][j - 1]
                choices.append(((previous[0] + overlap, previous[1] + 1), "match"))
            scores[i][j], actions[i, j] = max(choices, key=lambda item: item[0])
    pairs = []
    i, j = n, m
    while i and j:
        action = actions[i, j]
        if action == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif action == "observed":
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def _day_metrics(rows):
    observed = [row["observed_m7_mm_day"] < row["threshold_mm_day"] for row in rows]
    predicted = [row["predicted_m7_mm_day"] < row["threshold_mm_day"] for row in rows]
    hits = sum(a and b for a, b in zip(observed, predicted))
    misses = sum(a and not b for a, b in zip(observed, predicted))
    alarms = sum(b and not a for a, b in zip(observed, predicted))
    return {"evaluated_days": len(rows), "observed_drought_days": sum(observed),
            "predicted_drought_days": sum(predicted), "hits": hits, "misses": misses,
            "false_alarms": alarms, "precision": _ratio(hits, hits + alarms),
            "recall": _ratio(hits, hits + misses), "csi": _ratio(hits, hits + misses + alarms),
            "false_alarm_ratio": _ratio(alarms, hits + alarms),
            "low_flow_m7_bias_mm_day": _ratio(sum(row["predicted_m7_mm_day"] - row["observed_m7_mm_day"]
                for row, active in zip(rows, observed) if active), sum(observed)),
            "low_flow_m7_mae_mm_day": _ratio(sum(abs(row["predicted_m7_mm_day"] - row["observed_m7_mm_day"])
                for row, active in zip(rows, observed) if active), sum(observed)),
            "m7_mae_mm_day": _ratio(sum(abs(row["predicted_m7_mm_day"] - row["observed_m7_mm_day"])
                                       for row in rows), len(rows))}


def verify_drought(predictions, series_by_basin, train_start, train_end, *,
                   definition="seasonal_quantile", smoothing="trailing", probability=.15,
                   calendar_window=31, minimum_days=1):
    """Return metric rows and auditable day/event tables, without any training."""
    groups = defaultdict(lambda: defaultdict(list))
    for prediction in predictions:
        issue = prediction.issue_date or prediction.target_date - DAY
        if prediction.target_date <= issue:
            raise ValueError("Drought forecast targets must be after their issue date")
        groups[prediction.basin_id][prediction.seed, issue].append(prediction)
    scores, day_rows, event_rows = [], [], []
    thresholds_by_basin = {}
    for basin in sorted(groups):
        series = series_by_basin[basin]
        flow = {day: float(value) for day, value in zip(series.dates, series.discharge)}
        thresholds_by_basin[basin] = fit_thresholds(
            flow, train_start, train_end, definition=definition, smoothing=smoothing,
            probability=probability, calendar_window=calendar_window)
        for (seed, issue), records in sorted(groups[basin].items()):
            records.sort(key=lambda item: item.target_date)
            future = {item.target_date: item.predicted_mm_day for item in records}
            if len(future) != len(records):
                raise ValueError("Duplicate drought prediction dates within a forecast issue")
            # Seven historical days also allow checking whether onset precedes the window.
            history = {issue - offset * DAY: flow.get(issue - offset * DAY, float("nan"))
                       for offset in range(7)}
            path = {**history, **future}
            common = {"basin_id": basin, "seed": seed, "issue_date": issue.isoformat(),
                      "event_type": "hydrological_drought", "smoothing": smoothing,
                      "drought_definition": definition, "verification": VERIFICATION,
                      "minimum_days": minimum_days, "seasonal_quantile": probability,
                      "calendar_window_days": calendar_window}
            valid = []
            for record in records:
                day = record.target_date
                observed = smooth(flow, day, smoothing)
                predicted = smooth(path, day, smoothing)
                threshold = thresholds_by_basin[basin].at(day)
                reasons = [reason for value, reason in (
                    (observed, "missing_observed_window"), (predicted, "missing_same_issue_prediction_window"),
                    (threshold, "missing_training_calendar_reference")) if value is None]
                row = {**common, "date": day, "lead_day": record.lead_day,
                       "threshold_mm_day": threshold, "observed_m7_mm_day": observed,
                       "predicted_m7_mm_day": predicted, "eligible": not reasons,
                       "exclusion_reason": ";".join(reasons)}
                day_rows.append({**row, "date": day.isoformat()})
                if not reasons:
                    valid.append(row)
            metrics = {**common, "evaluation_scope": "issue", "lead_day": "all",
                       "requested_days": len(records), "excluded_days": len(records) - len(valid),
                       **_day_metrics(valid)}
            context_day = valid[0]["date"] - DAY if valid else issue
            context_threshold = thresholds_by_basin[basin].at(context_day)
            context_observed = smooth(flow, context_day, smoothing)
            context_predicted = smooth(path, context_day, smoothing)
            def context(field, value):
                return {"date": context_day, "threshold_mm_day": context_threshold, field: value} \
                    if value is not None and context_threshold is not None else None
            observed_events = extract_events(valid, "observed_m7_mm_day", minimum_days,
                context("observed_m7_mm_day", context_observed))
            predicted_events = extract_events(valid, "predicted_m7_mm_day", minimum_days,
                context("predicted_m7_mm_day", context_predicted))
            pairs = match_events(observed_events, predicted_events)
            metrics.update({"observed_events": len(observed_events), "predicted_events": len(predicted_events),
                            "matched_events": len(pairs),
                            "event_precision": _ratio(len(pairs), len(predicted_events)),
                            "event_recall": _ratio(len(pairs), len(observed_events)),
                            "overlap_event_precision": _ratio(sum(any(a.start <= b.end and b.start <= a.end
                                for a in observed_events) for b in predicted_events), len(predicted_events)),
                            "overlap_event_recall": _ratio(sum(any(a.start <= b.end and b.start <= a.end
                                for b in predicted_events) for a in observed_events), len(observed_events)),
                            "observed_window_deficit_mm": sum(event.deficit for event in observed_events),
                            "predicted_window_deficit_mm": sum(event.deficit for event in predicted_events)})
            onset_errors, recovery_errors, duration_errors, deficit_errors = [], [], [], []
            for oi, pi in pairs:
                a, b = observed_events[oi], predicted_events[pi]
                if not a.left_censored and not b.left_censored:
                    onset_errors.append((b.start - a.start).days)
                if not a.right_censored and not b.right_censored:
                    recovery_errors.append((b.end - a.end).days)
                if not any((a.left_censored, a.right_censored, b.left_censored, b.right_censored)):
                    duration_errors.append(b.duration - a.duration)
                    deficit_errors.append(b.deficit - a.deficit)
            for name, values in (("onset", onset_errors), ("recovery", recovery_errors),
                                 ("duration", duration_errors), ("deficit", deficit_errors)):
                units = "mm" if name == "deficit" else "days"
                metrics[f"{name}_pairs"] = len(values)
                metrics[f"{name}_bias_{units}"] = _ratio(sum(values), len(values))
                metrics[f"{name}_mae_{units}"] = _ratio(sum(map(abs, values)), len(values))
            scores.append(metrics)
            for kind, events in (("observed", observed_events), ("predicted", predicted_events)):
                matched = {a: b for a, b in pairs} if kind == "observed" else {b: a for a, b in pairs}
                for index, event in enumerate(events):
                    event_rows.append({**common, "kind": kind, "event_index": index,
                        "start": event.start.isoformat(), "end": event.end.isoformat(),
                        "duration_days": event.duration, "window_deficit_mm": event.deficit,
                        "left_censored": event.left_censored, "right_censored": event.right_censored,
                        "matched_event_index": matched.get(index, "")})
    # Day scores by lead are pooled only within a basin and seed, never across issues
    # for event extraction. Each saved issue remains independently auditable.
    by_lead = defaultdict(list)
    for row in day_rows:
        by_lead[row["basin_id"], row["seed"], row["lead_day"]].append(row)
    for (basin, seed, lead), rows in sorted(by_lead.items()):
        valid = [row for row in rows if row["eligible"]]
        scores.append({"basin_id": basin, "seed": seed, "event_type": "hydrological_drought",
                       "evaluation_scope": "lead", "lead_day": lead, "smoothing": smoothing,
                       "drought_definition": definition, "verification": VERIFICATION,
                       "minimum_days": minimum_days, "seasonal_quantile": probability,
                       "calendar_window_days": calendar_window, "requested_days": len(rows),
                       "excluded_days": len(rows) - len(valid), **_day_metrics(valid)})
    return scores, {"days": day_rows, "events": event_rows}


def drought_summary(metrics):
    """Mean finite lead scores across basin/seed rows, with contributing counts."""
    grouped = defaultdict(list)
    for row in metrics:
        if row.get("evaluation_scope") != "lead":
            continue
        for metric in ("precision", "recall", "csi", "false_alarm_ratio", "m7_mae_mm_day",
                       "low_flow_m7_bias_mm_day", "low_flow_m7_mae_mm_day"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                grouped[f"lead_{row['lead_day']}.{metric}"].append(value)
    result = {}
    for key, values in grouped.items():
        result[key] = mean(values)
        result[f"{key}.basin_seed_count"] = len(values)
    return result
