"""Read-only integration of EStreams vegetation and snow with CAMELS-CH."""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .hydrology_data import HydrologySeries


_MONTHLY_FILES = {
    "estreams_lai": ("estreams_LAI_monhtly.csv", 1.0),
    "estreams_ndvi": ("estreams_NDVI_monhtly.csv", 1.0),
    "estreams_snow_cover_fraction": ("estreams_snowcover_monhtly.csv", 0.01),
}
ESTREAMS_DYNAMIC_FEATURES = tuple(_MONTHLY_FILES)


class EStreamsVegetationSnow:
    """Join EStreams monthly basin values to CAMELS-CH daily series.

    EStreams timestamps monthly composites at month end. Values are exposed on
    that date and carried forward until the next composite, so a forecast never
    receives information from a monthly composite before its source timestamp.
    Availability columns distinguish a real zero from a missing crosswalk or a
    date before the first EStreams observation.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._paths = self.validate_layout(self.root)
        self._camels_to_estreams = self._read_camels_crosswalk(
            self._paths["gauges"]
        )

    @staticmethod
    def _find_file(root: Path, filename: str) -> Path:
        matches = [
            path
            for path in root.rglob(filename)
            if "__MACOSX" not in path.parts and not path.name.startswith("._")
        ]
        if not matches:
            raise FileNotFoundError(
                f"invalid EStreams layout under {root}; missing {filename}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous EStreams layout under {root}; found multiple {filename} files"
            )
        return matches[0]

    @classmethod
    def validate_layout(cls, root: str | Path) -> dict[str, Path]:
        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(f"EStreams root is not a directory: {path}")
        result = {
            name: cls._find_file(path, filename)
            for name, (filename, _) in _MONTHLY_FILES.items()
        }
        result["gauges"] = cls._find_file(path, "estreams_gauging_stations.csv")
        return result

    @staticmethod
    def _read_camels_crosswalk(path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("gauge_provider", "").strip() != "CH_CAMELS":
                    continue
                gauge_id = row.get("gauge_id", "").strip()
                basin_id = row.get("basin_id", "").strip()
                if not gauge_id or not basin_id:
                    continue
                previous = result.setdefault(gauge_id, basin_id)
                if previous != basin_id:
                    raise ValueError(
                        f"CAMELS-CH gauge {gauge_id} maps to multiple EStreams basins"
                    )
        return result

    @property
    def camels_gauge_ids(self) -> tuple[str, ...]:
        return tuple(self._camels_to_estreams)

    @staticmethod
    def _read_monthly_columns(
        path: Path, basin_ids: set[str], scale: float
    ) -> dict[str, tuple[tuple[date, float], ...]]:
        values: dict[str, list[tuple[date, float]]] = {
            basin_id: [] for basin_id in basin_ids
        }
        if not basin_ids:
            return {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(f"empty EStreams file: {path}") from error
            columns = {
                index: name for index, name in enumerate(header) if name in basin_ids
            }
            missing = basin_ids - set(columns.values())
            if missing:
                raise ValueError(
                    f"EStreams file {path.name} lacks basin columns: {sorted(missing)}"
                )
            for row in reader:
                if not row:
                    continue
                timestamp = date.fromisoformat(row[0].strip())
                for index, basin_id in columns.items():
                    raw = row[index].strip()
                    if not raw or raw.lower() in {"na", "nan"}:
                        continue
                    value = float(raw) * scale
                    if math.isfinite(value):
                        values[basin_id].append((timestamp, value))
        return {basin: tuple(items) for basin, items in values.items()}

    @staticmethod
    def _causal_daily_values(
        dates: tuple[date, ...], observations: tuple[tuple[date, float], ...]
    ) -> tuple[list[float], list[float]]:
        observation_dates = [item[0] for item in observations]
        result: list[float] = []
        available: list[float] = []
        for day in dates:
            index = bisect_right(observation_dates, day) - 1
            if index < 0:
                result.append(0.0)
                available.append(0.0)
            else:
                result.append(observations[index][1])
                available.append(1.0)
        return result, available

    def augment(
        self, series_by_basin: Mapping[str, HydrologySeries],
        features: Sequence[str] | None = None,
    ) -> dict[str, HydrologySeries]:
        """Return new daily series with LAI, NDVI, snow, and availability masks."""
        selected = tuple(dict.fromkeys(
            ESTREAMS_DYNAMIC_FEATURES if features is None else features
        ))
        unknown = set(selected) - set(ESTREAMS_DYNAMIC_FEATURES)
        if unknown:
            raise ValueError(f"unknown EStreams dynamic features: {sorted(unknown)}")
        if not selected:
            raise ValueError("at least one EStreams dynamic feature must be selected")
        mapped = {
            gauge_id: self._camels_to_estreams[gauge_id]
            for gauge_id in series_by_basin
            if gauge_id in self._camels_to_estreams
        }
        requested = set(mapped.values())
        monthly = {
            name: self._read_monthly_columns(
                self._paths[name], requested, _MONTHLY_FILES[name][1]
            )
            for name in selected
        }
        forcing_names = tuple(
            item
            for name in selected
            for item in (name, f"{name}_available")
        )
        result: dict[str, HydrologySeries] = {}
        for gauge_id, series in series_by_basin.items():
            estreams_id = mapped.get(gauge_id)
            columns: list[list[float]] = []
            for name in selected:
                observations = monthly[name].get(estreams_id, ()) if estreams_id else ()
                values, available = self._causal_daily_values(series.dates, observations)
                columns.extend((values, available))
            external = torch.tensor(columns, dtype=series.forcings.dtype).T
            result[gauge_id] = HydrologySeries(
                dates=series.dates,
                discharge=series.discharge,
                forcings=torch.cat((series.forcings, external), dim=1),
                forcing_names=(*series.forcing_names, *forcing_names),
                basin_id=series.basin_id,
                country=series.country,
            )
        return result


__all__ = ["ESTREAMS_DYNAMIC_FEATURES", "EStreamsVegetationSnow"]
