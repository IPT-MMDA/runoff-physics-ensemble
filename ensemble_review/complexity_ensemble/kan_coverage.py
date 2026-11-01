"""Date-aware split availability and planned usage for catchment roles."""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from .hydrology_data import _parse_date


@lru_cache(maxsize=2048)
def _file_dates(path, mtime_ns, size, kind):
    """Cache only dates of valid records, invalidating on file changes."""
    columns = (
        ("discharge_spec(mm/d)",)
        if kind == "discharge"
        else ("precipitation_sim(mm/d)", "temperature_sim(degC)", "pet_sim(mm/d)")
    )
    valid, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        if not set(("date", *columns)).issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing {kind} columns")
        for row in reader:
            if not row.get("date"):
                raise ValueError(f"Missing {kind} date")
            day = _parse_date(row["date"])
            if day in seen:
                raise ValueError(f"Duplicate {kind} dates")
            seen.add(day)
            try:
                numbers = [float(row[column]) for column in columns]
            except (ValueError, TypeError):
                continue
            if not all(math.isfinite(value) for value in numbers):
                continue
            if numbers[0] < 0 or (kind == "forcing" and numbers[2] < 0):
                continue
            valid.append(day)
    # Cache contiguous ordinal intervals rather than thousands of date objects
    # for each file in a large catalogue.
    intervals: list[tuple[int, int]] = []
    for day in sorted(valid):
        ordinal = day.toordinal()
        if intervals and ordinal == intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], ordinal)
        else:
            intervals.append((ordinal, ordinal))
    return tuple(intervals)


def _load_dates(root, basin, kind, start, end):
    folder, prefix = (
        ("observation_based", "obs_based")
        if kind == "discharge"
        else ("simulation_based", "sim_based")
    )
    path = Path(root) / "timeseries" / folder / f"CAMELS_CH_{prefix}_{basin}.csv"
    stat = path.stat()
    intervals = _file_dates(str(path.resolve()), stat.st_mtime_ns, stat.st_size, kind)
    a, b = start.toordinal(), end.toordinal()
    return {
        date.fromordinal(i)
        for first, last in intervals
        for i in range(max(first, a), min(last, b) + 1)
    }


def _longest_run(flags, value):
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag == value else 0
        longest = max(longest, current)
    return longest


def planned_usage(row, role, budgets, minimum_source_days, selected_models=None):
    """Counts are per run; target budgets are alternatives, never summed."""
    train, validation, test = (
        row.get(k, 0)
        for k in ("training_observations", "validation_observations", "test_observations")
    )
    train_used: int | str = 0
    validation_used = test_used = 0
    issues = []
    needs_sources = selected_models is None or any(
        m.startswith("kan_") or "_source_" in m or "_mixed_" in m for m in selected_models
    )
    needs_validation = selected_models is None or any(m.startswith("kan_") for m in selected_models)
    needs_targets = selected_models is None or any(
        m in {"kan_tuned", "kan_complex"} or "_target_target" in m or "_mixed_" in m
        for m in selected_models
    )
    if (role == "source" and not needs_sources) or (role == "validation" and not needs_validation):
        return {
            "train_used": 0,
            "validation_used": 0,
            "test_used": 0,
            "readiness": "Not used by selected models",
        }
    if role == "source":
        train_used = train
        if train < minimum_source_days:
            issues.append(f"Needs {minimum_source_days} training observations; has {train}")
    elif role == "validation":
        validation_used = validation
        if validation < 3:
            issues.append(f"Needs 3 validation observations; has {validation}")
    elif role == "target":
        train_used = (
            ", ".join(
                str(n) if n <= train else f"{n} requested (only {train} available)" for n in budgets
            )
            if needs_targets
            else 0
        )
        test_used = test
        if needs_targets and train < max(budgets):
            issues.append(f"Largest training budget is {max(budgets)}; has {train}")
        if test < 3:
            issues.append(f"Needs 3 test observations; has {test}")
    else:
        return {
            "train_used": "—",
            "validation_used": "—",
            "test_used": "—",
            "readiness": "Choose one role" if role == "conflict" else "Assign a role",
        }
    if row.get("weather_missing_days", 0):
        issues.append(f"Missing {row['weather_missing_days']} weather days")
    if row.get("error"):
        issues.append(row["error"])
    return {
        "train_used": train_used,
        "validation_used": validation_used,
        "test_used": test_used,
        "readiness": "; ".join(issues) if issues else "Ready",
    }


def coverage_inventory(config, catalog, *, series_by_basin=None):
    """Inspect all eligible basins before cohorts are chosen; never fit a model.

    Roles are assigned by the user. This reports counts and fitting readiness.
    Future discharge availability and discharge magnitudes do not affect ranking.
    """
    config.validate(require_cohorts=False)
    start, cutoff = date.fromisoformat(config.train_start), date.fromisoformat(config.train_end)
    split_dates = {}
    split_ranges = {}
    for split in ("train", "validation", "test"):
        a, b = (date.fromisoformat(getattr(config, f"{split}_{key}")) for key in ("start", "end"))
        split_dates[split] = tuple(a + timedelta(days=i) for i in range((b - a).days + 1))
        split_ranges[split] = {"start": a.isoformat(), "end": b.isoformat()}
    train_dates = split_dates["train"]
    warm_start = start - timedelta(days=config.warmup_days)
    forcing_end = date.fromisoformat(config.test_end)
    forcing_dates = {
        warm_start + timedelta(days=i) for i in range((forcing_end - warm_start).days + 1)
    }
    budget = max(config.observation_budgets)
    rows = []
    for record in catalog.records:
        if not record.eligible:
            continue
        basin = record.basin_id
        error = ""
        try:
            if series_by_basin is None:
                discharge = _load_dates(catalog.root, basin, "discharge", start, forcing_end)
                weather = _load_dates(catalog.root, basin, "forcing", warm_start, forcing_end)
            else:
                series = series_by_basin[basin]
                discharge = {
                    d
                    for d, q in zip(series.dates, series.discharge)
                    if math.isfinite(float(q)) and float(q) >= 0
                }
                columns = [
                    series.forcing_names.index(n) for n in ("precipitation", "temperature", "pet")
                ]
                weather = {
                    d
                    for d, values in zip(series.dates, series.forcings[:, columns])
                    if all(math.isfinite(float(v)) for v in values)
                    and values[0] >= 0
                    and values[2] >= 0
                }
            # Count labels usable by this pipeline, with valid weather on that day.
            available = [d in discharge and d in weather for d in train_dates]
            observed_dates = [d for d, valid in zip(train_dates, available) if valid]
            count = len(observed_dates)
            split_counts = {
                split: sum(d in discharge and d in weather for d in days)
                for split, days in split_dates.items()
            }
            missing_weather = len(forcing_dates - weather)
        except (OSError, ValueError, KeyError, csv.Error):
            available = [False] * len(train_dates)
            observed_dates, count, missing_weather = [], 0, len(forcing_dates)
            split_counts = {split: 0 for split in split_dates}
            error = "Record files unavailable, malformed, or missing required columns/dates"
        months = {
            f"{month:02d}": sum(d.month == month for d in observed_dates) for month in range(1, 13)
        }
        rows.append(
            {
                "basin_id": basin,
                "name": record.name,
                "training_observations": count,
                "training_calendar_days": len(train_dates),
                "validation_observations": split_counts["validation"],
                "test_observations": split_counts["test"],
                "validation_calendar_days": len(split_dates["validation"]),
                "test_calendar_days": len(split_dates["test"]),
                "coverage_percent": round(100 * count / len(train_dates), 1),
                "first_observation": observed_dates[0].isoformat() if count else None,
                "last_observation": observed_dates[-1].isoformat() if count else None,
                "longest_missing_spell_days": _longest_run(available, False),
                "longest_observed_spell_days": _longest_run(available, True),
                "calendar_months_represented": sum(v > 0 for v in months.values()),
                "observations_by_month": months,
                "weather_missing_days": missing_weather,
                "source_ready": count >= config.minimum_source_days and missing_weather == 0,
                "validation_ready": split_counts["validation"] >= 3 and missing_weather == 0,
                "target_test_ready": split_counts["test"] >= 3 and missing_weather == 0,
                "requested_target_budget": max(config.budgets_for(basin)),
                "target_budget_ready": count >= max(config.budgets_for(basin))
                and missing_weather == 0,
                "error": error,
            }
        )
    rows.sort(key=lambda r: (-r["training_observations"], r["basin_id"]))
    return {
        "training_start": start.isoformat(),
        "as_of": cutoff.isoformat(),
        "dates": split_ranges,
        "default_target_budgets": list(config.observation_budgets),
        "target_observation_budgets": {
            b: list(v) for b, v in config.target_observation_budgets.items()
        },
        "selected_models": list(config.selected_models),
        "minimum_source_days": config.minimum_source_days,
        "maximum_target_budget": budget,
        "weather_start": warm_start.isoformat(),
        "weather_end": forcing_end.isoformat(),
        "role_policy": "Cohort roles are assigned by the user; no high/low-data categories or automatic assignments are applied.",
        "interpretation": "Counts and seasonal coverage inform your manual roles. Per-target budgets restrict the local data used. Full preflight checks source-validation and target-test labels.",
        "items": rows,
    }
