"""Optional daily water states from the existing CAMELS-CH files."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .hydrology_data import HydrologySeries, _parse_date, _read_rows


CAMELS_CH_STATE_SOURCES = {
    "snow_water_equivalent": ("observation_based", "obs", "swe(mm)"),
    "canopy_interception_storage": (
        "simulation_based", "sim", "intercept_storage_sim(mm)",
    ),
}
CAMELS_CH_STATE_FEATURES = tuple(CAMELS_CH_STATE_SOURCES)


def augment_camels_ch_states(
    root: str | Path,
    series_by_basin: Mapping[str, HydrologySeries],
    features: Sequence[str],
) -> dict[str, HydrologySeries]:
    """Join exact-date states, using zero plus a mask for missing observations.

    States are never carried forward or interpolated. The regional window
    builder uses only dates preceding the prediction target.
    """
    selected = tuple(dict.fromkeys(features))
    unknown = set(selected) - set(CAMELS_CH_STATE_FEATURES)
    if unknown:
        raise ValueError(f"unknown CAMELS-CH water-state features: {sorted(unknown)}")
    if not selected:
        return dict(series_by_basin)
    names = tuple(item for name in selected for item in (name, f"{name}_available"))
    result = {}
    for basin, series in series_by_basin.items():
        observations = {}
        for name in selected:
            group, prefix, column = CAMELS_CH_STATE_SOURCES[name]
            path = Path(root) / "timeseries" / group / f"CAMELS_CH_{prefix}_based_{basin}.csv"
            values = {}
            if path.is_file():
                for row in _read_rows(path):
                    raw = row.get(column, "")
                    try:
                        value = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(value) and value >= 0:
                        day = _parse_date(row["date"])
                        if day in values:
                            raise ValueError(f"duplicate water-state date in {path}: {day}")
                        values[day] = value
            observations[name] = values
        external = torch.tensor([
            [item for name in selected for item in (
                observations[name].get(day, 0.0), float(day in observations[name]),
            )] for day in series.dates
        ], dtype=series.forcings.dtype, device=series.forcings.device)
        result[basin] = replace(
            series, forcings=torch.cat((series.forcings, external), dim=1),
            forcing_names=(*series.forcing_names, *names),
        )
    return result
