"""Paired discharge verification and an explicit daily energy-cost scenario."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from .hydropower import _basin_area_km2, prepare_hydropower_forecast


@dataclass(frozen=True)
class EnergyScenario:
    head_m: float = 100.0
    efficiency: float = 0.9
    environmental_flow_m3_s: float = 0.0
    max_turbine_flow_m3_s: float | None = None
    shortfall_eur_mwh: float = 50.0
    surplus_eur_mwh: float = 20.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.head_m <= 0 or not 0 < self.efficiency <= 1:
            raise ValueError("Head must be positive and efficiency must be in (0, 1]")
        if self.max_turbine_flow_m3_s is not None and self.max_turbine_flow_m3_s <= 0:
            raise ValueError("Maximum turbine flow must be positive or omitted")

    def energy(self, flow):
        turbine = max(0.0, flow - self.environmental_flow_m3_s)
        if self.max_turbine_flow_m3_s is not None:
            turbine = min(turbine, self.max_turbine_flow_m3_s)
        return 0.23544 * self.efficiency * self.head_m * turbine

    def cost(self, predicted_energy, observed_energy):
        error = predicted_energy - observed_energy
        return self.shortfall_eur_mwh * max(error, 0) + self.surplus_eur_mwh * max(-error, 0)


def read_comparison_predictions(path, *, lead_day=1, rolling=False):
    """Average seeds without mixing forecast issues or lead times."""
    groups, seen = defaultdict(list), set()
    audit = {"invalid_rows": 0, "other_lead_rows": 0, "inferred_issue_rows": 0}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"seed", "basin_id", "target_date", "observed_mm_day", "predicted_mm_day"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Comparison requires predictions with observed_mm_day and predicted_mm_day"
            )
        for row in reader:
            try:
                target = date.fromisoformat(row["target_date"])
                raw_lead, raw_issue = row.get("lead_day"), row.get("issue_date")
                if not raw_lead or not raw_issue:
                    if rolling or raw_lead or raw_issue:
                        raise ValueError("missing rolling issue metadata")
                    lead, issue = 1, target - timedelta(days=1)
                    audit["inferred_issue_rows"] += 1
                else:
                    lead, issue = int(raw_lead), date.fromisoformat(raw_issue)
                if lead < 1 or target - issue != timedelta(days=lead):
                    raise ValueError("inconsistent issue and lead")
                if lead != lead_day:
                    audit["other_lead_rows"] += 1
                    continue
                observed, predicted = float(row["observed_mm_day"]), float(row["predicted_mm_day"])
                if not math.isfinite(observed) or observed < 0 or not math.isfinite(predicted):
                    raise ValueError("invalid flow")
                basin = row["basin_id"].strip()
                if not basin:
                    raise ValueError("missing basin")
                key = (basin, target, issue, lead)
                seed_key = (*key, int(row["seed"]))
            except (ValueError, TypeError):
                audit["invalid_rows"] += 1
                continue
            if seed_key in seen:
                raise ValueError(
                    "Duplicate prediction for the same seed, basin, target, issue and lead"
                )
            seen.add(seed_key)
            groups[key].append((observed, predicted))
    result = {}
    for key, values in groups.items():
        observed = values[0][0]
        if any(not math.isclose(observed, obs, rel_tol=1e-6, abs_tol=1e-6) for obs, _ in values):
            raise ValueError("Observations differ across seeds within a run")
        result[key] = (observed, sum(p for _, p in values) / len(values))
    return result, audit


def _metrics(rows, name):
    errors = [r[name] - r["observed"] for r in rows]
    groups = defaultdict(list)
    for row in rows:
        groups[row["basin_id"]].append(row)
    nses = []
    for items in groups.values():
        mean = sum(r["observed"] for r in items) / len(items)
        denominator = sum((r["observed"] - mean) ** 2 for r in items)
        if denominator > 1e-12:
            nses.append(1 - sum((r[name] - r["observed"]) ** 2 for r in items) / denominator)
    return {
        "mae_m3_s": sum(abs(e) for e in errors) / len(errors),
        "rmse_m3_s": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "bias_m3_s": sum(errors) / len(errors),
        "mean_basin_nse": sum(nses) / len(nses) if nses else None,
        "nse_basins": len(nses),
    }


def compare_hydropower(
    path_a,
    path_b,
    catalog,
    sites,
    *,
    config_a=None,
    config_b=None,
    start=None,
    end=None,
    target_date=None,
    lead_day=1,
    scenario=None,
):
    scenario = scenario or EnergyScenario()
    if lead_day < 1:
        raise ValueError("Lead day must be positive")
    if start and end and start > end:
        raise ValueError("Start must be on or before end")
    configs = [config_a or {}, config_b or {}]
    if any(
        (config.get("benchmark") or {}).get("target", "daily_mean") != "daily_mean"
        for config in configs
    ):
        raise ValueError(
            "The daily energy scenario requires daily_mean discharge, not daily maxima"
        )

    # Observed values alone cannot reveal a different target definition.
    def definition(config):
        benchmark = config.get("benchmark") or {}
        mode = benchmark.get("mode", "none")
        return (
            mode,
            benchmark.get("target", "daily_mean"),
            benchmark.get("day_timezone", "Europe/Zurich"),
            tuple(sorted(benchmark.get("release_states", [2, 3]))),
        )

    if definition(configs[0]) != definition(configs[1]):
        raise ValueError(
            "Runs must use the same observation source, target, timezone and release states"
        )
    a, audit_a = read_comparison_predictions(
        path_a,
        lead_day=lead_day,
        rolling=(configs[0].get("forecast") or {}).get("mode") == "rolling",
    )
    b, audit_b = read_comparison_predictions(
        path_b,
        lead_day=lead_day,
        rolling=(configs[1].get("forecast") or {}).get("mode") == "rolling",
    )
    common = set(a) & set(b)
    audit = {
        "run_a": audit_a,
        "run_b": audit_b,
        "only_a": len(set(a) - set(b)),
        "only_b": len(set(b) - set(a)),
        "observation_conflicts": 0,
        "missing_area_or_geometry": 0,
    }
    records = {str(r.basin_id): r for r in catalog.records if r.geometry is not None}
    rows = []
    for key in sorted(common):
        basin, day, issue, lead = key
        if (start and day < start) or (end and day > end):
            continue
        if not math.isclose(a[key][0], b[key][0], rel_tol=1e-6, abs_tol=1e-6):
            audit["observation_conflicts"] += 1
            continue
        area = _basin_area_km2(records[basin]) if basin in records else None
        if area is None:
            audit["missing_area_or_geometry"] += 1
            continue
        rows.append(
            {
                "basin_id": basin,
                "date": day.isoformat(),
                "issue_date": issue.isoformat(),
                "lead_day": lead,
                "observed": a[key][0] * area / 86.4,
                "a": a[key][1] * area / 86.4,
                "b": b[key][1] * area / 86.4,
            }
        )
    if not rows:
        raise ValueError(
            "No paired predictions with shared observations, issue/lead, area and geometry in this period"
        )
    # One stable assignment across both runs and the entire evaluation period.
    basins = {r["basin_id"] for r in rows}
    dummy_date = date(2000, 1, 1)
    dummy = SimpleNamespace(
        predictions=[
            SimpleNamespace(basin_id=basin, target_date=dummy_date, predicted_mm_day=0)
            for basin in basins
        ]
    )
    mapped, _ = prepare_hydropower_forecast(dummy, catalog, sites, dummy_date)
    mapped_basins = set(mapped.basin_id.dropna())
    audit["unmapped_basin_samples"] = sum(r["basin_id"] not in mapped_basins for r in rows)
    rows = [r for r in rows if r["basin_id"] in mapped_basins]
    if not rows:
        raise ValueError("No hydropower plants match the paired catchments")
    dates = sorted({r["date"] for r in rows})
    selected = target_date.isoformat() if target_date else dates[-1]
    if selected not in dates:
        raise ValueError("Map date has no paired predictions in the selected period")
    for row in rows:
        row["energy_observed_mwh"] = scenario.energy(row["observed"])
        for name in ("a", "b"):
            row[f"energy_{name}_mwh"] = scenario.energy(row[name])
            row[f"cost_{name}_eur"] = scenario.cost(
                row[f"energy_{name}_mwh"], row["energy_observed_mwh"]
            )
    by_basin = defaultdict(list)
    for row in rows:
        by_basin[row["basin_id"]].append(row)
    basin_rows = []
    for basin, items in sorted(by_basin.items()):
        cost_a = sum(r["cost_a_eur"] for r in items)
        cost_b = sum(r["cost_b_eur"] for r in items)
        basin_rows.append(
            {
                "basin_id": basin,
                "name": records[basin].name,
                "days": len(items),
                "metrics_a": _metrics(items, "a"),
                "metrics_b": _metrics(items, "b"),
                "cost_a_eur": cost_a,
                "cost_b_eur": cost_b,
                "saving_eur": cost_b - cost_a,
            }
        )
    day_rows = {r["basin_id"]: r for r in rows if r["date"] == selected}
    features = []
    for _, site in mapped.iterrows():
        basin = site.get("basin_id")
        if basin not in mapped_basins:
            basin = None
        row = day_rows.get(basin)
        props = {
            "name": site["_site_name"],
            "site_id": site["_site_id"],
            "basin_id": basin,
            "status": "Paired forecast" if row else "No paired forecast on map date",
        }
        for key in (
            "observed",
            "a",
            "b",
            "energy_observed_mwh",
            "energy_a_mwh",
            "energy_b_mwh",
            "cost_a_eur",
            "cost_b_eur",
        ):
            props[key] = row[key] if row else None
        props["error_a_m3_s"] = row["a"] - row["observed"] if row else None
        props["error_b_m3_s"] = row["b"] - row["observed"] if row else None
        features.append(
            {"type": "Feature", "geometry": site.geometry.__geo_interface__, "properties": props}
        )
    cost_a = sum(r["cost_a_eur"] for r in rows)
    cost_b = sum(r["cost_b_eur"] for r in rows)
    scale = (
        max(
            (
                abs(f["properties"][f"error_{name}_m3_s"])
                for f in features
                for name in ("a", "b")
                if f["properties"][f"error_{name}_m3_s"] is not None
            ),
            default=0,
        )
        or 1
    )
    metrics_a, metrics_b = _metrics(rows, "a"), _metrics(rows, "b")
    changes = {}
    for metric in ("mae", "rmse"):
        baseline = metrics_b[f"{metric}_m3_s"]
        changes[f"{metric}_reduction_pct"] = (
            100 * (baseline - metrics_a[f"{metric}_m3_s"]) / baseline if baseline > 0 else None
        )
    changes["mean_basin_nse_gain"] = (
        metrics_a["mean_basin_nse"] - metrics_b["mean_basin_nse"]
        if metrics_a["mean_basin_nse"] is not None and metrics_b["mean_basin_nse"] is not None
        else None
    )
    return {
        "available_dates": dates,
        "target_date": selected,
        "lead_day": lead_day,
        "period": {"start": dates[0], "end": dates[-1]},
        "scenario": asdict(scenario),
        "samples": len(rows),
        "basins": len(by_basin),
        "audit": audit,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "accuracy_change": changes,
        "economics": {
            "cost_a_eur": cost_a,
            "cost_b_eur": cost_b,
            "saving_eur": cost_b - cost_a,
            "saving_pct": 100 * (cost_b - cost_a) / cost_b if cost_b > 0 else None,
            "cost_index_a": 100 * (cost_a / cost_b) if cost_b > 0 else None,
            "cost_index_b": 100 if cost_b > 0 else None,
            "unit": "One abstract plant per uniquely mapped basin; period totals, not annualized",
        },
        "map": {"type": "FeatureCollection", "features": features, "error_scale_m3_s": scale},
        "basin_results": basin_rows,
    }
