from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import torch


CANONICAL_COLUMNS = {
    "date": "date",
    "discharge": "discharge_mm_day",
    "precipitation": "precipitation_mm_day",
    "temperature": "temperature_c",
    "pet": "pet_mm_day",
}
LANDCOVER_COLUMNS = ("crop", "grass", "forest", "urban", "ice")


@dataclass(frozen=True)
class HydrologySeries:
    dates: tuple[date, ...]
    discharge: torch.Tensor
    forcings: torch.Tensor
    forcing_names: tuple[str, ...]
    basin_id: str
    country: str


@dataclass(frozen=True)
class HydrologySplit:
    inputs: torch.Tensor
    physical_inputs: torch.Tensor
    targets: torch.Tensor
    target_dates: tuple[date, ...]
    routing_inputs: torch.Tensor | None = None


@dataclass(frozen=True)
class HydrologyData:
    train: HydrologySplit
    validation: HydrologySplit
    test: HydrologySplit
    feature_names: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    discharge_scale: torch.Tensor
    basin_id: str
    country: str


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return datetime.strptime(value.strip(), "%Y/%m/%d").date()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(line for line in source if not line.startswith("#")))


def _finite_float(row: Mapping[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    if not value or value.lower() == "nan":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _series_from_records(
    records: Sequence[tuple[date, float, Sequence[float]]],
    forcing_names: Sequence[str],
    *,
    basin_id: str,
    country: str,
) -> HydrologySeries:
    if len(records) < 3:
        raise ValueError("at least three complete daily records are required")
    ordered = sorted(records, key=lambda item: item[0])
    dates = tuple(item[0] for item in ordered)
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate dates are not allowed")
    return HydrologySeries(
        dates=dates,
        discharge=torch.tensor([item[1] for item in ordered], dtype=torch.float32),
        forcings=torch.tensor([item[2] for item in ordered], dtype=torch.float32),
        forcing_names=tuple(forcing_names),
        basin_id=str(basin_id),
        country=country,
    )


def _load_landcover(root: Path, basin_id: str) -> dict[int, tuple[float, ...]]:
    path = root / "annual_timeseries" / f"CAMELS_CH_annual_data_{basin_id}.csv"
    if not path.exists():
        return {}
    result: dict[int, tuple[float, ...]] = {}
    for row in _read_rows(path):
        values = (
            _finite_float(row, "crop_perc"),
            _finite_float(row, "grass_perc"),
            sum(
                value or 0.0
                for value in (
                    _finite_float(row, "dwood_perc"),
                    _finite_float(row, "mix_wood_perc"),
                    _finite_float(row, "ewood_perc"),
                )
            ),
            _finite_float(row, "urban_perc"),
            _finite_float(row, "ice_perc"),
        )
        if all(value is not None for value in values):
            result[int(row["year"])] = tuple(float(value) / 100.0 for value in values)
    return result


def load_camels_ch(
    root: str | Path,
    basin_id: str = "2011",
    *,
    start: str | None = "2000-01-01",
    end: str | None = "2020-12-31",
    include_landcover: bool = True,
) -> HydrologySeries:
    """Load observed specific discharge and simulation-based meteorological forcing."""
    root = Path(root)
    observed_path = root / "timeseries" / "observation_based" / f"CAMELS_CH_obs_based_{basin_id}.csv"
    simulated_path = root / "timeseries" / "simulation_based" / f"CAMELS_CH_sim_based_{basin_id}.csv"
    if not observed_path.exists() or not simulated_path.exists():
        raise FileNotFoundError(f"CAMELS-CH basin {basin_id} was not found under {root}")

    # Key by parsed date rather than raw text.  Some CAMELS-CH releases use
    # different but equivalent date formatting in the observation and
    # simulation exports.
    observed = {
        _parse_date(row["date"]): _finite_float(row, "discharge_spec(mm/d)")
        for row in _read_rows(observed_path)
    }
    landcover = _load_landcover(root, basin_id) if include_landcover else {}
    forcing_names = ["precipitation", "temperature", "pet"]
    if include_landcover:
        forcing_names.extend(f"landcover_{name}" for name in LANDCOVER_COLUMNS)
    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    records = []
    for row in _read_rows(simulated_path):
        day = _parse_date(row["date"])
        if (start_date and day < start_date) or (end_date and day > end_date):
            continue
        discharge = observed.get(day)
        values = [
            _finite_float(row, "precipitation_sim(mm/d)"),
            _finite_float(row, "temperature_sim(degC)"),
            _finite_float(row, "pet_sim(mm/d)"),
        ]
        if include_landcover:
            cover = landcover.get(day.year)
            if cover is None:
                continue
            values.extend(cover)
        if discharge is None or discharge < 0.0 or any(value is None for value in values):
            continue
        records.append((day, discharge, [float(value) for value in values]))
    return _series_from_records(records, forcing_names, basin_id=basin_id, country="Switzerland")


def load_ukraine_csv(
    path: str | Path,
    *,
    basin_id: str,
    columns: Mapping[str, str] | None = None,
) -> HydrologySeries:
    """Load a pre-aligned observed Ukrainian daily series.

    The default schema uses the canonical names in ``CANONICAL_COLUMNS``.
    ``columns`` maps canonical roles (date, discharge, precipitation,
    temperature, pet) to source column names, which supports GRDC/EStreams
    exports after their licensing-compliant local preprocessing step.
    """
    names = dict(CANONICAL_COLUMNS)
    names.update(columns or {})
    missing_roles = set(CANONICAL_COLUMNS) - set(names)
    if missing_roles:
        raise ValueError(f"missing column roles: {sorted(missing_roles)}")
    records = []
    for row in _read_rows(Path(path)):
        values = [
            _finite_float(row, names["precipitation"]),
            _finite_float(row, names["temperature"]),
            _finite_float(row, names["pet"]),
        ]
        discharge = _finite_float(row, names["discharge"])
        if discharge is None or discharge < 0.0 or any(value is None for value in values):
            continue
        records.append(
            (_parse_date(row[names["date"]]), discharge, [float(value) for value in values])
        )
    return _series_from_records(
        records, ("precipitation", "temperature", "pet"),
        basin_id=basin_id, country="Ukraine",
    )


def make_hydrology_data(
    series: HydrologySeries,
    *,
    sequence_length: int = 30,
    routing_context_length: int | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> HydrologyData:
    """Build chronological, leakage-free one-day-ahead windows."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    routing_context_length = routing_context_length or sequence_length
    if routing_context_length < sequence_length:
        raise ValueError("routing_context_length must be at least sequence_length")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and leave a non-empty test fraction")

    day_of_year = torch.tensor([day.timetuple().tm_yday for day in series.dates], dtype=torch.float32)
    angle = 2.0 * torch.pi * day_of_year / 365.25
    features = torch.cat(
        (
            series.forcings,
            series.discharge[:, None],
            angle.sin()[:, None],
            angle.cos()[:, None],
        ),
        dim=1,
    )
    feature_names = (*series.forcing_names, "previous_discharge", "day_sin", "day_cos")
    physical_windows, routing_windows, targets, target_dates = [], [], [], []
    for target_index in range(routing_context_length, len(series.dates)):
        first = target_index - routing_context_length
        expected = series.dates[first] + timedelta(days=routing_context_length)
        if series.dates[target_index] != expected:
            continue
        if any(
            series.dates[index + 1] - series.dates[index] != timedelta(days=1)
            for index in range(first, target_index)
        ):
            continue
        physical_windows.append(features[target_index - sequence_length:target_index])
        routing_windows.append(features[first:target_index])
        targets.append(series.discharge[target_index])
        target_dates.append(series.dates[target_index])
    if len(targets) < 15:
        raise ValueError("not enough contiguous records to create train/validation/test windows")

    physical = torch.stack(physical_windows)
    routing = torch.stack(routing_windows)
    target = torch.stack(targets)
    train_end = int(len(target) * train_fraction)
    validation_end = int(len(target) * (train_fraction + validation_fraction))
    if min(train_end, validation_end - train_end, len(target) - validation_end) == 0:
        raise ValueError("each chronological split must contain at least one window")
    feature_mean = physical[:train_end].reshape(-1, physical.shape[-1]).mean(dim=0)
    feature_scale = physical[:train_end].reshape(-1, physical.shape[-1]).std(dim=0, unbiased=False)
    feature_scale = feature_scale.clamp_min(1e-6)
    normalized = (physical - feature_mean) / feature_scale
    normalized_routing = (routing - feature_mean) / feature_scale
    discharge_scale = target[:train_end].std(unbiased=False).clamp_min(1e-6)

    def split(start: int, stop: int) -> HydrologySplit:
        return HydrologySplit(
            normalized[start:stop], physical[start:stop], target[start:stop] / discharge_scale,
            tuple(target_dates[start:stop]), normalized_routing[start:stop],
        )

    return HydrologyData(
        split(0, train_end), split(train_end, validation_end), split(validation_end, len(target)),
        tuple(feature_names), feature_mean, feature_scale, discharge_scale,
        series.basin_id, series.country,
    )
