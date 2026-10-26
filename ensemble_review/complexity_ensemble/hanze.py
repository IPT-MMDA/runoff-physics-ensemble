"""Independent regional-catalogue agreement, separate from gauge flood verification.

HANZE reports impacted administrative regions, not gauge-level flood labels. An
unlinked prediction is not a confirmed false alarm. Only Swiss River records are
used; flash/pluvial events require a different observation and matching model.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path

from .flood_verification import _events

VERSION = "hanze_swiss_river_gauge_region_v2"
EVENT_FILE = "HANZE_events_v3_0_1b.csv"
REGION_FILE = "Regions_v2024_simplified/NUTS3_regions_v2024_simplified.shp"
DAY = timedelta(days=1)


@dataclass(frozen=True)
class CatalogueEvent:
    event_id: str
    start: date
    end: date
    regions: tuple[str, ...]
    sources: str = ""


def load_events(root):
    path = Path(root) / EVENT_FILE
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"ID", "Country code", "Type", "Start date", "End date", "Regions affected (NUTS 3)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("HANZE event catalogue has an unsupported schema")
        selected = []
        seen = set()
        for row in reader:
            if row["Country code"] != "CH" or row["Type"] != "River":
                continue
            start, end = date.fromisoformat(row["Start date"]), date.fromisoformat(row["End date"])
            regions = tuple(sorted(set(r.strip() for r in row["Regions affected (NUTS 3)"].split(";") if r.strip())))
            if start > end or not regions or row["ID"] in seen or any(not r.startswith("CH") for r in regions):
                raise ValueError("Invalid or duplicated Swiss HANZE River event")
            seen.add(row["ID"])
            selected.append(CatalogueEvent(row["ID"], start, end, regions, row.get("Sources", "")))
    return sorted(selected, key=lambda e: (e.start, e.end, e.event_id))


def source_provenance(root):
    root = Path(root)
    shape = root / REGION_FILE
    paths = [root / EVENT_FILE, *(shape.with_suffix(ext) for ext in (".shp", ".shx", ".dbf", ".prj"))]
    return {"source": "https://zenodo.org/records/20478847", "version": "v3.0.1 beta",
            "files": [{"file": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in paths]}


def gauge_regions(root, records):
    """Use gauge points, not any upstream catchment overlap with a canton.

    The simplified boundaries and rounded gauge coordinates introduce spatial
    uncertainty. Missing, outside, or ambiguous points are excluded, never assigned
    to the nearest canton or to a centroid of the upstream catchment.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    frame = gpd.read_file(Path(root) / REGION_FILE)
    if frame.crs is None or "Code" not in frame:
        raise ValueError("HANZE region boundaries require a CRS and Code column")
    frame = frame[frame["Code"].str.startswith("CH", na=False)].copy()
    # The shipped simplified Swiss boundaries include self-intersecting rings.
    # Repair topology before reprojection; preserve the downloaded source files.
    frame.geometry = frame.geometry.make_valid()
    frame = frame.to_crs("EPSG:4326")
    if frame.empty or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError("HANZE Swiss region boundaries are missing or invalid")
    mapping = {}
    for record in records:
        lat, lon = (record.static_attributes.get(k) for k in ("gauge_lat", "gauge_lon"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lat, lon)):
            mapping[record.basin_id] = {"region": None, "reason": "missing_gauge_coordinates"}
            continue
        point = Point(lon, lat)
        hits = frame[frame.geometry.covers(point)]
        if len(hits) != 1:
            mapping[record.basin_id] = {"region": None, "reason": "ambiguous_or_outside_region"}
        else:
            row = hits.iloc[0]
            mapping[record.basin_id] = {"region": str(row["Code"]), "region_name": str(row.get("Name", "")),
                                        "reason": "", "gauge_lat": lat, "gauge_lon": lon}
    return mapping


def match_catalogue_events(observed, predicted):
    """Maximum-cardinality one-to-one overlap, including overlapping catalogue windows."""
    edges = []
    for event in observed:
        candidates = []
        for j, forecast in enumerate(predicted):
            overlap = (min(event.end, forecast.end) - max(event.start, forecast.start)).days + 1
            if overlap > 0:
                candidates.append((overlap, j))
        edges.append([j for _, j in sorted(candidates, reverse=True)])
    assigned = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(observed)):
        visit(i, set())
    return sorted((i, j) for j, i in assigned.items())


def evaluate(predictions, config, events, regions, provenance=None):
    """Evaluate fixed leads and the scheduled rolling series without pooling seeds."""
    thresholds = config.get("benchmark_manifest", {}).get("thresholds_mm_day", {})
    period = config.get("dates", {}).get("test", {})
    if not period.get("start") or not period.get("end"):
        raise ValueError("HANZE evaluation requires the saved test date range")
    start, end = date.fromisoformat(str(period["start"])), date.fromisoformat(str(period["end"]))
    targets = list(config.get("basins", {}).get("target_basins", []))
    groups = defaultdict(list)
    for row in predictions:
        if start <= row.target_date <= end and row.basin_id in targets:
            groups[row.basin_id, row.seed, row.lead_day].append(row)
    seeds = config.get("hyperparameters", {}).get("seeds", sorted({key[1] for key in groups}))
    horizon = config.get("forecast", {}).get("horizon_days", 1) if config.get("forecast", {}).get("mode") == "rolling" else 1
    rolling = config.get("forecast", {}).get("mode") == "rolling"
    if rolling:
        # Non-overlapping horizon-sized blocks have daily coverage collectively,
        # while each fixed lead is sampled only once per block. Build one scheduled
        # forecast series per seed. If blocks overlap, use the shortest available
        # lead on each target date; never average forecasts or mix training seeds.
        scheduled = defaultdict(dict)
        for (basin, seed, _), rows in list(groups.items()):
            for row in rows:
                previous = scheduled[basin, seed].get(row.target_date)
                if previous is None or row.lead_day < previous.lead_day:
                    scheduled[basin, seed][row.target_date] = row
        for (basin, seed), rows in scheduled.items():
            groups[basin, seed, "all"] = list(rows.values())
    scopes = ["all", *range(1, horizon + 1)] if rolling else [1]
    metrics, details, forecasts = [], [], []
    for basin in targets:
        region = regions.get(basin, {"region": None, "reason": "missing_gauge_metadata"})
        threshold = thresholds.get(basin)
        mapped = bool(region.get("region"))
        usable = mapped and isinstance(threshold, (int, float)) and math.isfinite(threshold)
        known = [e for e in events if region.get("region") in e.regions and e.start <= end and e.end >= start] if mapped else []
        for seed in seeds:
            for lead in scopes:
                common = {"basin_id": basin, "seed": seed, "lead_day": lead, "region": region.get("region"),
                          "region_name": region.get("region_name"), "threshold_mm_day": threshold,
                          "evaluation_scope": "rolling_blocks" if lead == "all" else "fixed_lead"}
                rows = sorted(groups[basin, seed, lead], key=lambda r: r.target_date)
                if len({r.target_date for r in rows}) != len(rows):
                    raise ValueError("Duplicate target dates within a HANZE basin/seed/lead series")
                valid = [r for r in rows if math.isfinite(r.predicted_mm_day) and math.isfinite(r.observed_mm_day)]
                covered = {r.target_date for r in valid}
                eligible = []
                excluded = []
                for event in known:
                    total = (event.end - event.start).days + 1
                    covered_event_days = sum(event.start + DAY * offset in covered for offset in range(total))
                    complete = usable and covered_event_days == total
                    reason = "" if complete else region.get("reason") or ("missing_threshold" if not usable else "incomplete_prediction_coverage")
                    entry = {**common, "hanze_id": event.event_id, "start": event.start.isoformat(), "end": event.end.isoformat(),
                             "sources": event.sources, "covered_event_days": covered_event_days, "required_event_days": total, "eligible": complete, "exclusion_reason": reason, "matched": None}
                    details.append(entry)
                    if complete:
                        eligible.append((event, entry))
                    else:
                        excluded.append(event)
                spells = _events(valid, "predicted_mm_day", threshold) if usable else []
                # Boundary/gap-touching forecasts and windows overlapping unscorable
                # catalogue events do not enter the precision denominator.
                scored = [p for p in spells if not p.censored and not any(p.start <= e.end and p.end >= e.start for e in excluded)]
                pairs = match_catalogue_events([e for e, _ in eligible], scored)
                matched_forecasts = {j: eligible[i][0].event_id for i, j in pairs}
                matched_catalogue = {i: j for i, j in pairs}
                for i, (_, entry) in enumerate(eligible):
                    entry["matched"] = i in matched_catalogue
                    if entry["matched"]:
                        forecast = scored[matched_catalogue[i]]
                        entry.update(predicted_start=forecast.start.isoformat(), predicted_end=forecast.end.isoformat(),
                                     predicted_peak_day=forecast.peak_day.isoformat())
                for j, spell in enumerate(scored):
                    forecasts.append({**common, "start": spell.start.isoformat(), "end": spell.end.isoformat(),
                                      "peak_day": spell.peak_day.isoformat(), "predicted_peak_mm_day": spell.peak,
                                      "hanze_id": matched_forecasts.get(j), "catalogue_linked": j in matched_forecasts})
                metrics.append({**common, "status": "evaluated" if usable and valid else "unavailable",
                                "reason": region.get("reason") or ("missing_threshold" if not usable else "no_predictions" if not valid else ""),
                                "catalogue_events": len(known), "eligible_events": len(eligible), "excluded_events": len(excluded),
                                "predicted_events": len(scored), "excluded_predicted_events": len(spells) - len(scored),
                                "matched_events": len(pairs), "unlinked_predicted_events": len(scored) - len(pairs),
                                "catalogue_precision": len(pairs) / len(scored) if scored else None,
                                "catalogue_recall": len(pairs) / len(eligible) if eligible else None,
                                "covered_days": len(covered)})
    return {"status": "available", "verification": VERSION, "provenance": provenance or {},
            "evaluation_period": {"start": start.isoformat(), "end": end.isoformat()},
            "default_scope": "all" if rolling else "1",
            "evaluation_scopes": [str(scope) for scope in scopes],
            "matching": {"country": "CH", "event_types": ["River"], "spatial": "gauge_point_in_reported_canton",
                         "temporal": "one_to_one_date_overlap", "date_tolerance_days": 0,
                         "geometry_processing": "make_valid_before_reprojection_to_EPSG4326",
                         "series": "separate_seed_fixed_leads_and_scheduled_rolling_blocks",
                         "rolling_overlap_policy": "shortest_available_lead_per_target_date", "coverage": "entire_catalogue_window_required",
                         "precision_denominator": "complete_predicted_events_excluding_unscorable_catalogue_windows"},
            "limitations": ["Regional catalogue agreement is not confirmed gauge-level flood classification.",
                            "An unlinked prediction is not a confirmed false alarm; HANZE is an impact catalogue with incomplete reporting.",
                            "Swiss River events only; Flash events are excluded. Rounded gauge coordinates and simplified canton boundaries limit spatial precision.",
                            "Rolling-block scores mix forecast leads along the saved forecast schedule; they are not fixed-lead skill scores.",
                            "No catalogue-linked peak error is computed: HANZE event dates are not gauge peak observations."],
            "basin_regions": regions, "metrics": metrics, "events": details, "predicted_events": forecasts}


def evaluate_run(root, predictions, config, catalog):
    if config.get("benchmark", {}).get("mode", "none") == "none" or config.get("extremes", {}).get("mode") != "statistical":
        return {"status": "not_applicable", "message": "HANZE evaluation applies to BAFU high-flow runs."}
    events = load_events(root)
    targets = set(config.get("basins", {}).get("target_basins", []))
    regions = gauge_regions(root, [r for r in catalog.records if r.basin_id in targets])
    return evaluate(predictions, config, events, regions, source_provenance(root))
