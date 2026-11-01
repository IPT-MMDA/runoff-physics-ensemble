"""Stable catchment preset for iterating on the Hydropower case study."""

from datetime import date
from types import SimpleNamespace

from .hydropower import _basin_area_km2, prepare_hydropower_forecast


# Top 50 by assigned plant count (ties by basin ID), using full-resolution
# catchments and the 183-basin cohort of run 20261021T124615Z-0bae752b7d-dce609.
# The reference assignment covers 705 plants; these 50 account for 544 of them.
# Freeze IDs so changing dates, architecture or source does not reshuffle trials.
HYDROPOWER_50_BASINS = (
    "2009",
    "2011",
    "2016",
    "2019",
    "2024",
    "2029",
    "2030",
    "2033",
    "2044",
    "2053",
    "2056",
    "2063",
    "2078",
    "2084",
    "2085",
    "2091",
    "2099",
    "2102",
    "2104",
    "2106",
    "2117",
    "2119",
    "2141",
    "2143",
    "2150",
    "2152",
    "2155",
    "2160",
    "2185",
    "2187",
    "2203",
    "2205",
    "2243",
    "2265",
    "2303",
    "2346",
    "2351",
    "2352",
    "2372",
    "2387",
    "2418",
    "2420",
    "2469",
    "2473",
    "2475",
    "2481",
    "2494",
    "2602",
    "2606",
    "2620",
)


def hydropower_training_cohort(catalog, sites, *, mode="none", compatibility=None):
    """Validate the frozen preset against this dataset, source and plant inventory.

    Metadata/spatial checks only. Run's automatic audit checks the chosen dates.
    Never replace unavailable IDs with other catchments silently.
    """
    if mode not in {"none", "bafu_only", "bafu_combined"}:
        raise ValueError("Unknown data source for the hydropower preset")
    records = {str(r.basin_id): r for r in catalog.records}
    reasons = {}
    for basin in HYDROPOWER_50_BASINS:
        record = records.get(basin)
        if record is None:
            reasons[basin] = "Not present in this catchment dataset"
        elif record.geometry is None or _basin_area_km2(record) is None:
            reasons[basin] = "Catchment geometry or positive area is unavailable"
        elif mode == "none" and not record.eligible:
            reasons[basin] = "CAMELS observations or predictors unavailable"
        elif mode != "none":
            available = (compatibility or {}).get(basin, {}).get(mode, {})
            if not available.get("eligible"):
                reasons[basin] = (
                    "; ".join(available.get("reasons", [])) or "BAFU source unavailable"
                )
    day = date(2000, 1, 1)
    dummy = SimpleNamespace(
        predictions=[
            SimpleNamespace(basin_id=b, target_date=day, predicted_mm_day=0)
            for b in HYDROPOWER_50_BASINS
            if b not in reasons
        ]
    )
    counts = {}
    if dummy.predictions:
        mapped, _ = prepare_hydropower_forecast(dummy, catalog, sites, day)
        counts = {str(b): int(n) for b, n in mapped.basin_id.value_counts().items()}
        for basin in HYDROPOWER_50_BASINS:
            if basin not in reasons and not counts.get(basin):
                reasons[basin] = "No assigned hydropower plant in the configured inventory"
    return {
        "id": "hydropower-50-v1",
        "name": "Hydropower · 50 catchments",
        "basin_ids": list(HYDROPOWER_50_BASINS),
        "size": 50,
        "available": not reasons,
        "unavailable_basins": reasons,
        "matched_plants": sum(counts.values()),
        "training_scope": "targets_only",
        "date_coverage_checked": False,
        "catchments": [
            {
                "basin_id": basin,
                "name": records[basin].name if basin in records else basin,
                "matched_plants": counts.get(basin, 0),
            }
            for basin in HYDROPOWER_50_BASINS
        ],
    }
