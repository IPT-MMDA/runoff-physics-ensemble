"""Read-only integration of HYD-RESPONSES base water states and metadata."""
from __future__ import annotations

import csv
import math
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .hydrology_data import HydrologySeries


# Public name -> source column, expected units, Switzerland-only spatial support.
HYD_RESPONSES_SOURCES = {
    **{f"hyd_soil_moisture_layer_{i}": (f"swvl{i}_mean_mean", "m3 m-3", False)
       for i in range(1, 5)},
    "hyd_snowmelt_oshd": ("romc_mean", "mm", True),
    "hyd_swe_oshd": ("swee_mean", "mm", True),
    "hyd_swe_spass": ("SWECLQMD_mean", "mm", True),
}
HYD_RESPONSES_FEATURES = tuple(HYD_RESPONSES_SOURCES)
DEFAULT_HYD_RESPONSES_FEATURES = HYD_RESPONSES_FEATURES[:5]


def _identifier(value: str) -> str:
    # Preserve historical IDs (e.g. 0488), rather than guessing a modern gauge.
    return value.strip()


def _number(value: str | None) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


class HYDResponses:
    def __init__(self, root: str | Path):
        self.root = self.validate_layout(root)
        self.base = self.root / "data/01_timeseries/01_base_variables"
        with (self.root / "data/03_descriptive/HYDRESPONSES_descriptive_general_station_information.csv").open(
            encoding="utf-8-sig", newline="",
        ) as source:
            self.stations = {_identifier(row["catchmentnr"]): row
                             for row in csv.DictReader(source, delimiter=";")}
        with (self.root / "data/01_timeseries/00_meta_information/units/01_base_variables.csv").open(
            encoding="utf-8-sig", newline="",
        ) as source:
            self.units = {row[0].strip(): row[1].strip()
                          for row in csv.reader(source, delimiter=";") if len(row) >= 2}

    @staticmethod
    def validate_layout(root: str | Path) -> Path:
        root = Path(root)
        required = (
            "data/01_timeseries/01_base_variables",
            "data/01_timeseries/00_meta_information/units/01_base_variables.csv",
            "data/03_descriptive/HYDRESPONSES_descriptive_general_station_information.csv",
        )
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"invalid HYD-RESPONSES layout under {root}; missing: {', '.join(missing)}")
        return root

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return tuple(basin for basin in self.stations
                     if (self.base / f"HYDRESPONSES_basevars_catchment_{basin}.csv").is_file())

    def augment(self, series_by_basin: Mapping[str, HydrologySeries], features: Sequence[str]):
        selected = tuple(dict.fromkeys(features))
        unknown = set(selected) - set(HYD_RESPONSES_FEATURES)
        if unknown:
            raise ValueError(f"unknown HYD-RESPONSES features: {sorted(unknown)}")
        if not selected:
            return dict(series_by_basin)
        for name in selected:
            column, unit, _ = HYD_RESPONSES_SOURCES[name]
            if self.units.get(column) != unit:
                raise ValueError(f"HYD-RESPONSES units for {column} must be {unit!r}")
        snow = any(HYD_RESPONSES_SOURCES[name][2] for name in selected)
        names = tuple(item for name in selected for item in (name, f"{name}_available"))
        if snow:
            names += ("hyd_swiss_area_fraction", "hyd_swiss_area_fraction_available")
        result = {}
        for basin, series in series_by_basin.items():
            observations = {}
            path = self.base / f"HYDRESPONSES_basevars_catchment_{basin}.csv"
            if basin in self.stations and path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as source:
                    reader = csv.DictReader(source, delimiter=";")
                    required = {"date", "catchmentnr", *(HYD_RESPONSES_SOURCES[n][0] for n in selected)}
                    missing = required - set(reader.fieldnames or ())
                    if missing:
                        raise ValueError(f"HYD-RESPONSES {path.name} lacks columns: {sorted(missing)}")
                    for row in reader:
                        day = date.fromisoformat(row["date"])
                        if _identifier(row["catchmentnr"]) != basin:
                            raise ValueError(f"HYD-RESPONSES catchment ID mismatch in {path.name}")
                        if day in observations:
                            raise ValueError(f"duplicate HYD-RESPONSES date in {path.name}: {day}")
                        observations[day] = tuple(_number(row[HYD_RESPONSES_SOURCES[n][0]]) for n in selected)
            overlap = _number(self.stations.get(basin, {}).get("percentage_overlap_CH"))
            if overlap is not None and overlap > 100:
                raise ValueError(f"invalid Swiss area percentage for HYD-RESPONSES basin {basin}")
            rows = []
            for day in series.dates:
                values = observations.get(day, (None,) * len(selected))
                row = [item for value in values for item in
                       (value if value is not None else 0.0, float(value is not None))]
                if snow:
                    row.extend((overlap / 100 if overlap is not None else 0.0,
                                float(overlap is not None)))
                rows.append(row)
            external = torch.tensor(rows, dtype=series.forcings.dtype, device=series.forcings.device)
            result[basin] = replace(series, forcings=torch.cat((series.forcings, external), dim=1),
                                    forcing_names=(*series.forcing_names, *names))
        return result

    def physical_attributes(self) -> dict[str, dict[str, float]]:
        """Only physical descriptors; discharge signatures and climatology are excluded."""
        path = self.root / "data/03_descriptive/HYDRESPONSES_descriptive_hydro_geological_terrestrial_info.csv"
        if not path.is_file():
            raise FileNotFoundError(f"HYD-RESPONSES physical descriptors missing: {path}")
        result = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source, delimiter=";"):
                basin = _identifier(row["catchmentnr"])
                result[basin] = {f"hyd_{name}": value for name, raw in row.items()
                                 if name != "catchmentnr" and (value := _number(raw)) is not None}
        return result
