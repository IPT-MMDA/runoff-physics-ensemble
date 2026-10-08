"""Read-only CAMELS-CH-Chem catchment-pressure integration."""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .hydrology_data import HydrologySeries


# Public feature name -> (source group, source column). Land cover is omitted
# because it duplicates CAMELS-CH's CORINE-derived product. Stream chemistry is
# kept out of predictors because it is intended to become a target in a
# dedicated water-quality task, not a leakage-prone discharge covariate.
_FEATURE_SOURCES = {
    "chem_agriculture_cereal": ("agriculture", "cereal"),
    "chem_agriculture_maize": ("agriculture", "maize"),
    "chem_agriculture_sugarbeet": ("agriculture", "sugarbeet"),
    "chem_agriculture_potato": ("agriculture", "potato"),
    "chem_agriculture_rapeseed": ("agriculture", "rapeseed"),
    "chem_agriculture_pulse": ("agriculture", "pulse"),
    "chem_agriculture_vegetable": ("agriculture", "vegetable"),
    "chem_agriculture_total_arable": ("agriculture", "total_arable"),
    "chem_agriculture_grapevine": ("agriculture", "grapevine"),
    "chem_agriculture_orchard": ("agriculture", "orchard"),
    "chem_livestock_gve": ("livestock", "gve_sum"),
    "chem_livestock_gve_per_ha": ("livestock", "gve_ha"),
    "chem_deposition_hno3_gas": ("deposition", "dhno3gas"),
    "chem_deposition_nh3_gas": ("deposition", "dnh3gas"),
    "chem_deposition_nh4_total": ("deposition", "dnh4total"),
    "chem_deposition_no2_gas": ("deposition", "dno2gas"),
    "chem_deposition_no3_total": ("deposition", "dno3total"),
    "chem_deposition_n_total": ("deposition", "dntotal"),
    "chem_rain_delta_2h": ("rain_isotopes", "delta_2h"),
    "chem_rain_delta_18o": ("rain_isotopes", "delta_18o"),
}
CAMELS_CH_CHEM_DYNAMIC_FEATURES = tuple(_FEATURE_SOURCES)
DEFAULT_CAMELS_CH_CHEM_FEATURES = (
    "chem_agriculture_total_arable",
    "chem_livestock_gve_per_ha",
    "chem_deposition_n_total",
    "chem_rain_delta_18o",
)

_GROUPS = {
    "agriculture": (
        "catchment_aggregated_data/agricultural_data",
        "camels_ch_chem_swisscrops_{basin}.csv",
        "annual",
    ),
    "livestock": (
        "catchment_aggregated_data/livestock_data",
        "camels_ch_chem_livestock_{basin}.csv",
        "annual",
    ),
    "deposition": (
        "catchment_aggregated_data/atmospheric_deposition",
        "camels_ch_chem_atmdepo_{basin}.csv",
        "annual",
    ),
    "rain_isotopes": (
        "catchment_aggregated_data/rain_water_isotopes",
        "camels_ch_chem_rainisotopes_{basin}.csv",
        "dated",
    ),
}


class CAMELSCHChemPressures:
    """Add selected CAMELS-CH-Chem pressures to CAMELS-CH daily series.

    Annual values become available on 31 December of their source year and are
    carried forward. This conservative alignment prevents an annual aggregate
    from leaking into forecasts made before that year has finished. Dated rain
    isotope values become available on their recorded date.
    """

    def __init__(self, root: str | Path):
        self.root = self.validate_layout(root)
        self._gauge_ids = self._read_gauge_ids(
            self.root / "gauges_metadata" / "camels_ch_chem_gauges_metadata.csv"
        )

    @staticmethod
    def validate_layout(root: str | Path) -> Path:
        path = Path(root)
        required = [
            path / "gauges_metadata" / "camels_ch_chem_gauges_metadata.csv",
            *(path / group[0] for group in _GROUPS.values()),
        ]
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise FileNotFoundError(
                "invalid CAMELS-CH-Chem layout; missing: " + ", ".join(missing)
            )
        return path

    @staticmethod
    def _read_gauge_ids(path: Path) -> tuple[str, ...]:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return tuple(
                row["gauge_id"].strip()
                for row in csv.DictReader(source)
                if row.get("gauge_id", "").strip()
            )

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return self._gauge_ids

    def _observations(
        self, basin_id: str, selected: Sequence[str]
    ) -> dict[str, tuple[tuple[date, float], ...]]:
        result: dict[str, list[tuple[date, float]]] = {
            feature: [] for feature in selected
        }
        by_group: dict[str, list[tuple[str, str]]] = {}
        for feature in selected:
            group, column = _FEATURE_SOURCES[feature]
            by_group.setdefault(group, []).append((feature, column))
        for group, features in by_group.items():
            directory, template, cadence = _GROUPS[group]
            path = self.root / directory / template.format(basin=basin_id)
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    raw_date = row.get("date", "").strip()
                    if not raw_date:
                        continue
                    timestamp = (
                        date(int(raw_date), 12, 31)
                        if cadence == "annual"
                        else date.fromisoformat(raw_date[:10])
                    )
                    for feature, column in features:
                        raw = row.get(column, "").strip()
                        if not raw or raw.lower() in {"na", "nan"}:
                            continue
                        value = float(raw)
                        if math.isfinite(value):
                            result[feature].append((timestamp, value))
        return {feature: tuple(values) for feature, values in result.items()}

    @staticmethod
    def _causal_daily_values(
        dates: tuple[date, ...], observations: tuple[tuple[date, float], ...]
    ) -> tuple[list[float], list[float]]:
        observation_dates = [item[0] for item in observations]
        values: list[float] = []
        available: list[float] = []
        for day in dates:
            index = bisect_right(observation_dates, day) - 1
            if index < 0:
                values.append(0.0)
                available.append(0.0)
            else:
                values.append(observations[index][1])
                available.append(1.0)
        return values, available

    def augment(
        self,
        series_by_basin: Mapping[str, HydrologySeries],
        features: Sequence[str] | None = None,
    ) -> dict[str, HydrologySeries]:
        selected = tuple(dict.fromkeys(
            DEFAULT_CAMELS_CH_CHEM_FEATURES if features is None else features
        ))
        unknown = set(selected) - set(CAMELS_CH_CHEM_DYNAMIC_FEATURES)
        if unknown:
            raise ValueError(f"unknown CAMELS-CH-Chem features: {sorted(unknown)}")
        if not selected:
            raise ValueError("at least one CAMELS-CH-Chem feature must be selected")
        supported = set(self._gauge_ids)
        result: dict[str, HydrologySeries] = {}
        forcing_names = tuple(
            item
            for feature in selected
            for item in (feature, f"{feature}_available")
        )
        for basin_id, series in series_by_basin.items():
            observations = (
                self._observations(basin_id, selected)
                if basin_id in supported
                else {feature: () for feature in selected}
            )
            columns: list[list[float]] = []
            for feature in selected:
                values, available = self._causal_daily_values(
                    series.dates, observations[feature]
                )
                columns.extend((values, available))
            external = torch.tensor(columns, dtype=series.forcings.dtype).T
            result[basin_id] = HydrologySeries(
                dates=series.dates,
                discharge=series.discharge,
                forcings=torch.cat((series.forcings, external), dim=1),
                forcing_names=(*series.forcing_names, *forcing_names),
                basin_id=series.basin_id,
                country=series.country,
            )
        return result


__all__ = [
    "CAMELS_CH_CHEM_DYNAMIC_FEATURES",
    "DEFAULT_CAMELS_CH_CHEM_FEATURES",
    "CAMELSCHChemPressures",
]
