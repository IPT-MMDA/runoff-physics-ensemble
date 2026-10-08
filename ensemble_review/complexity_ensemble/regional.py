"""UI-independent regional hydrology experiments.

The module deliberately has no Gradio dependency.  Both command-line and web
front ends can construct :class:`RegionalExperimentConfig` and call
``run_regional_experiment``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Mapping, Sequence

import torch

from .camels_ch_chem import (
    CAMELS_CH_CHEM_DYNAMIC_FEATURES, DEFAULT_CAMELS_CH_CHEM_FEATURES,
    CAMELSCHChemPressures,
)
from .hydrology import HydrologyRoutedOutput, RoutedHydrologyModel
from .hydrology_comparison import _add_observation_noise, _metrics
from .hydrology_data import HydrologySeries, load_camels_ch
from .estreams import ESTREAMS_DYNAMIC_FEATURES, EStreamsVegetationSnow
from .hydrology_extreme_comparison import extreme_metrics
from .hydrology_events import EVENT_TYPES, basin_event_metrics, regional_event_metrics
from .hydrology_hard_routing_consolidation_comparison import hard_routed_losses
from .ood import MahalanobisOODDetector
from .stacking import fit_convex_stacking_weight
from .kan_physics import (
    KANPhysicsSettings, PHYSICS_OPTIMIZATION_MODES, discover_kan_physics_correction,
    standalone_kan_prediction,
)


TRAINING_SCOPES = ("exclude_targets", "all_basins", "targets_only")
_SCOPE_ALIASES = {
    "exclude_targets": "exclude_targets", "all_except_targets": "exclude_targets",
    "all_basins": "all_basins", "all": "all_basins",
    "targets_only": "targets_only", "selected_only": "targets_only",
}
TRAINING_APPROACHES = (
    "soft_routing", "hard_routing", "distillation", "no_routing",
    "static_50_50", "ood_fallback", "stacking",
)
PHYSICS_MODELS = ("linear_reservoir",)
PHYSICS_MODEL_REGISTRY = {"linear_reservoir": "linear reservoir water balance"}


def _day(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass(frozen=True)
class DateRange:
    start: date | str
    end: date | str

    def resolved(self) -> "DateRange":
        return DateRange(_day(self.start), _day(self.end))

    def contains(self, value: date) -> bool:
        item = self.resolved()
        return item.start <= value <= item.end  # type: ignore[operator]


@dataclass(frozen=True)
class BasinScopeConfig:
    target_basins: tuple[str, ...]
    training_scope: str = "exclude_targets"
    eligible_basins: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DateSplitConfig:
    train: DateRange
    validation: DateRange
    test: DateRange


@dataclass(frozen=True)
class ModelArchitectureConfig:
    complex_expert: str = "pinnmamba"
    simple_expert: str = "rbf"
    mlp_widths: tuple[int, ...] = (64, 64)
    mamba_hidden_size: int = 16
    mamba_layers: int = 1
    mamba_feed_forward_size: int = 64
    rbf_centers: int = 32
    rbf_width: float | None = None
    fourier_frequencies: int = 32
    fourier_scale: float | None = None
    static_embedding_size: int = 8


@dataclass(frozen=True)
class TrainingStrategyConfig:
    approach: str = "soft_routing"
    epochs: int = 20
    complex_teacher_epochs: int = 20
    distillation_epochs: int = 20
    consolidation_epochs: int = 20


@dataclass(frozen=True)
class HyperparameterConfig:
    sequence_length: int = 30
    seeds: tuple[int, ...] = (0,)
    batch_size: int = 64
    learning_rate: float = 1e-3
    training_noise: float = 0.0
    complexity_percentile: float = 80.0
    gate_temperature: float = 0.15
    physics_weight: float = 0.05
    interface_weight: float = 0.05
    routing_weight: float = 0.05
    compute_weight: float = 0.0
    device: str = "cpu"
    ood_quantile: float = 0.99
    ood_shrinkage: float = 0.1


@dataclass(frozen=True)
class PhysicsModelConfig:
    name: str = "linear_reservoir"
    optimization_mode: str = "none"
    distillation_teacher: str = "hard_routing"
    kan_candidates: int = 3
    kan_steps: int = 50
    kan_grid: int = 3
    kan_spline_order: int = 3
    kan_additive_nodes: int = 2
    kan_multiplicative_nodes: int = 1
    kan_sparsity_weight: float = 1e-3
    robustness_noise_levels: tuple[float, ...] = (0.05, 0.1, 0.2)
    robustness_accuracy_tolerance: float = 0.05
    symbolic: bool = True


@dataclass(frozen=True)
class ExtremeEventConfig:
    mode: str = "none"
    definition: str = "automatic_q95"
    value: float | None = None
    per_basin_values: Mapping[str, float] = field(default_factory=dict)
    event_types: tuple[str, ...] = EVENT_TYPES
    high_quantile: float = 0.8
    low_quantile: float = 0.2
    rise_quantile: float = 0.95
    fall_quantile: float = 0.95
    minimum_days: int = 3
    pulse_gap_days: int = 7
    minimum_basins: int = 2


@dataclass(frozen=True)
class ForecastConfig:
    mode: str = "one_day"
    horizon_days: int = 7


@dataclass(frozen=True)
class RegionalExperimentConfig:
    data_root: str | Path
    basins: BasinScopeConfig
    dates: DateSplitConfig
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    strategy: TrainingStrategyConfig = field(default_factory=TrainingStrategyConfig)
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    physics: PhysicsModelConfig = field(default_factory=PhysicsModelConfig)
    extremes: ExtremeEventConfig = field(default_factory=ExtremeEventConfig)
    estreams_root: str | Path | None = None
    estreams_features: tuple[str, ...] = ESTREAMS_DYNAMIC_FEATURES
    camels_chem_root: str | Path | None = None
    camels_chem_features: tuple[str, ...] = DEFAULT_CAMELS_CH_CHEM_FEATURES
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    def validate(self) -> None:
        targets = tuple(dict.fromkeys(map(str, self.basins.target_basins)))
        if not targets:
            raise ValueError("at least one prediction target basin is required")
        if self.basins.training_scope not in _SCOPE_ALIASES:
            raise ValueError(f"training_scope must be one of {tuple(_SCOPE_ALIASES)}")
        ranges = [item.resolved() for item in (self.dates.train, self.dates.validation, self.dates.test)]
        if any(item.start > item.end for item in ranges):  # type: ignore[operator]
            raise ValueError("each date range must start on or before it ends")
        if not (ranges[0].end < ranges[1].start and ranges[1].end < ranges[2].start):  # type: ignore[operator]
            raise ValueError("train, validation, and test ranges must be ordered and non-overlapping")
        if self.model.complex_expert not in {"mlp", "pinnmamba"}:
            raise ValueError("complex_expert must be 'mlp' or 'pinnmamba'")
        if self.model.simple_expert not in {"rbf", "fourier"}:
            raise ValueError("simple_expert must be 'rbf' or 'fourier'")
        if self.strategy.approach not in TRAINING_APPROACHES:
            raise ValueError(f"approach must be one of {TRAINING_APPROACHES}")
        if (
            self.strategy.approach == "stacking"
            and _SCOPE_ALIASES[self.basins.training_scope] == "exclude_targets"
        ):
            raise ValueError(
                "stacking requires target basins in training; choose all_basins or targets_only"
            )
        epochs = (self.strategy.epochs, self.strategy.complex_teacher_epochs,
                  self.strategy.distillation_epochs, self.strategy.consolidation_epochs)
        hp = self.hyperparameters
        if min(epochs) < 1 or hp.sequence_length < 2 or hp.batch_size < 1 or hp.learning_rate <= 0:
            raise ValueError("epoch counts, batch size, sequence length, and learning rate must be positive")
        if not hp.seeds:
            raise ValueError("at least one seed is required")
        if hp.training_noise < 0 or any(value < 0 for value in (
            hp.physics_weight, hp.interface_weight, hp.routing_weight, hp.compute_weight
        )):
            raise ValueError("noise and loss weights must be non-negative")
        if not 0 < hp.complexity_percentile < 100 or hp.gate_temperature <= 0:
            raise ValueError("complexity percentile and gate temperature are invalid")
        if not 0 < hp.ood_quantile < 1 or not 0 <= hp.ood_shrinkage <= 1:
            raise ValueError("OOD quantile or covariance shrinkage is invalid")
        if self.physics.name not in PHYSICS_MODELS:
            raise ValueError(f"physics model must be one of {PHYSICS_MODELS}")
        if self.physics.optimization_mode not in ("none", *PHYSICS_OPTIMIZATION_MODES):
            raise ValueError(
                "physics optimization mode must be none, accuracy, robustness, balanced, "
                "or distillation"
            )
        if (
            self.physics.optimization_mode != "none"
            and _SCOPE_ALIASES[self.basins.training_scope] == "exclude_targets"
        ):
            raise ValueError(
                "KAN physics optimization requires target basins in training; choose "
                "all_basins or targets_only"
            )
        if self.physics.distillation_teacher not in TRAINING_APPROACHES:
            raise ValueError(f"distillation teacher must be one of {TRAINING_APPROACHES}")
        if min(
            self.physics.kan_candidates, self.physics.kan_steps, self.physics.kan_grid,
            self.physics.kan_spline_order, self.physics.kan_additive_nodes,
        ) < 1 or self.physics.kan_multiplicative_nodes < 0:
            raise ValueError("KAN sizes, candidate count, and step count are invalid")
        if self.physics.kan_sparsity_weight < 0 or not 0 <= self.physics.robustness_accuracy_tolerance <= 1:
            raise ValueError("KAN sparsity and robustness tolerance are invalid")
        if any(level <= 0 for level in self.physics.robustness_noise_levels):
            raise ValueError("KAN robustness noise levels must be positive")
        if (
            self.physics.optimization_mode in {"robustness", "balanced"}
            and not self.physics.robustness_noise_levels
        ):
            raise ValueError("robustness and balanced KAN modes require noise levels")
        if self.extremes.mode not in {"none", "statistical", "event_based"}:
            raise ValueError("extreme mode must be 'none', 'statistical', or 'event_based'")
        if self.forecast.mode not in {"one_day", "rolling"} or self.forecast.horizon_days < 2:
            raise ValueError("forecast mode or horizon days are invalid")
        if (self.extremes.mode == "event_based" and
                (not self.extremes.event_types or set(self.extremes.event_types) - set(EVENT_TYPES)
                or not 0 < self.extremes.low_quantile < self.extremes.high_quantile < 1
                or not 0 < self.extremes.rise_quantile < 1
                or not 0 < self.extremes.fall_quantile < 1
                or self.extremes.minimum_days < 1 or self.extremes.pulse_gap_days < 0
                or self.extremes.minimum_basins < 2)):
            raise ValueError("event definitions or thresholds are invalid")
        if self.extremes.definition not in {"automatic_q95", "absolute", "quantile"}:
            raise ValueError("unknown extreme-event definition")
        values = list(self.extremes.per_basin_values.values())
        if self.extremes.value is not None:
            values.append(self.extremes.value)
        if self.extremes.definition == "absolute" and any(value < 0 for value in values):
            raise ValueError("absolute discharge thresholds must be non-negative")
        if self.extremes.definition in {"quantile", "automatic_q95"} and any(
            not 0 < value < 1 for value in values
        ):
            raise ValueError("quantiles must be strictly between zero and one")
        unknown_estreams = set(self.estreams_features) - set(ESTREAMS_DYNAMIC_FEATURES)
        if unknown_estreams:
            raise ValueError(f"unknown EStreams features: {sorted(unknown_estreams)}")
        if self.estreams_root is not None and not self.estreams_features:
            raise ValueError("enabled EStreams integration requires at least one feature")
        unknown_chem = set(self.camels_chem_features) - set(
            CAMELS_CH_CHEM_DYNAMIC_FEATURES
        )
        if unknown_chem:
            raise ValueError(f"unknown CAMELS-CH-Chem features: {sorted(unknown_chem)}")
        if self.camels_chem_root is not None and not self.camels_chem_features:
            raise ValueError(
                "enabled CAMELS-CH-Chem integration requires at least one feature"
            )


@dataclass(frozen=True)
class BasinCatalogRecord:
    basin_id: str
    name: str = ""
    static_attributes: Mapping[str, float | None] = field(default_factory=dict)
    eligible: bool = True
    geometry: object | None = None


_LEAKAGE_WORDS = (
    "discharge", "runoff", "streamflow", "precip", "temperature", "climate",
    "aridity", "pet_mean", "snow_fraction", "high_flow", "low_flow", "q_mean",
)


def _normalize_basin_id(value: object) -> str:
    """Normalize identifiers read as either text or spreadsheet-style numbers."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


class CAMELSCHCatalog:
    """Catalog joining local CAMELS-CH series, metadata, attributes, and polygons."""

    def __init__(self, root: str | Path, records: Sequence[BasinCatalogRecord] | None = None):
        self.root = Path(root)
        self.records = tuple(records) if records is not None else self._discover()
        self._by_id = {record.basin_id: record for record in self.records}

    @staticmethod
    def validate_layout(root: str | Path) -> Path:
        path = Path(root)
        required = path / "timeseries" / "observation_based", path / "timeseries" / "simulation_based"
        missing = [str(item) for item in required if not item.is_dir()]
        if missing:
            raise FileNotFoundError("invalid CAMELS-CH layout; missing: " + ", ".join(missing))
        return path

    def _discover(self) -> tuple[BasinCatalogRecord, ...]:
        self.validate_layout(self.root)
        observed = self.root / "timeseries" / "observation_based"
        basin_ids = {
            path.stem.removeprefix("CAMELS_CH_obs_based_") for path in observed.glob("CAMELS_CH_obs_based_*.csv")
        }
        attributes: dict[str, dict[str, float | None]] = {item: {} for item in basin_ids}
        names: dict[str, str] = {}
        # Attribute releases differ slightly in directory naming.  Joining all
        # tabular metadata by a recognizable gauge id keeps the loader robust.
        for path in self.root.rglob("*.csv"):
            if "timeseries" in path.parts or "annual_timeseries" in path.parts:
                continue
            try:
                with path.open(encoding="utf-8-sig", newline="") as source:
                    rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
            except (OSError, UnicodeError, csv.Error):
                continue
            for row in rows:
                identifier = next((_normalize_basin_id(row[key]) for key in row if key.lower() in {
                    "gauge_id", "station_id", "basin_id", "id", "gauge_code"
                } and _normalize_basin_id(row[key]) in basin_ids), None)
                if identifier is None:
                    continue
                for key, raw in row.items():
                    lower = key.lower()
                    if any(word in lower for word in _LEAKAGE_WORDS):
                        continue
                    if lower in {"name", "station_name", "gauge_name"} and raw:
                        names[identifier] = raw.strip()
                    try:
                        value = float(raw) if raw not in (None, "", "NA", "NaN", "nan") else None
                    except ValueError:
                        continue
                    if value is None or math.isfinite(value):
                        attributes[identifier][key] = value
        geometry: dict[str, object] = {}
        try:
            import geopandas as gpd
            preferred = self.root / "catchment_delineations" / "CAMELS_CH_catchments.shp"
            shape = preferred if preferred.is_file() else next(
                (path for path in self.root.rglob("*.shp")
                 if "catchment" in path.stem.lower()
                 and "sub_catchment" not in path.stem.lower()
                 and "station" not in path.stem.lower()),
                None,
            )
            if shape is not None:
                frame = gpd.read_file(shape)
                # Browser mapping libraries consume longitude/latitude GeoJSON.
                # CAMELS-CH releases commonly store catchments in a projected
                # Swiss CRS, so normalize geometries while the source CRS is
                # still available on the GeoDataFrame.
                if frame.crs is not None:
                    frame = frame.to_crs(epsg=4326)
                # Catchment boundaries are much more detailed than a browser
                # map at national scale can display. Keeping the source
                # vertices made each Gradio map payload well over 100 MB.
                # This only affects catalog display geometry; model inputs and
                # all hydrological calculations remain unchanged.
                frame.geometry = frame.geometry.simplify(
                    tolerance=0.002, preserve_topology=True
                )
                id_column = next((column for column in frame.columns if column.lower() in {
                    "gauge_id", "station_id", "basin_id", "id", "gauge_code"
                }), None)
                if id_column:
                    name_column = next((column for column in frame.columns if column.lower() in {
                        "name", "station_name", "gauge_name"
                    }), None)
                    for _, row in frame.iterrows():
                        identifier = _normalize_basin_id(row[id_column])
                        geometry[identifier] = row.geometry
                        if name_column and str(row[name_column]).strip():
                            names[identifier] = str(row[name_column]).strip().replace("_", " ")
        except (ImportError, OSError, ValueError):
            pass
        simulated = self.root / "timeseries" / "simulation_based"
        return tuple(BasinCatalogRecord(
            basin_id=item, name=names.get(item, item), static_attributes=attributes[item],
            eligible=(simulated / f"CAMELS_CH_sim_based_{item}.csv").exists(),
            geometry=geometry.get(item),
        ) for item in sorted(basin_ids))

    def get(self, basin_id: str) -> BasinCatalogRecord:
        try:
            return self._by_id[str(basin_id)]
        except KeyError as error:
            raise KeyError(f"unknown CAMELS-CH basin {basin_id}") from error

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return tuple(record.basin_id for record in self.records if record.eligible)

    def load_series(self, basin_id: str, start: date, end: date) -> HydrologySeries:
        # Regional models receive land-cover and other catchment descriptors
        # through the static-attribute path. Requiring annual land-cover files
        # here would otherwise discard every daily record for releases that do
        # not ship that optional product.
        return load_camels_ch(
            self.root, basin_id, start=start.isoformat(), end=end.isoformat(),
            include_landcover=False,
        )


@dataclass(frozen=True)
class RegionalSplit:
    inputs: torch.Tensor
    physical_inputs: torch.Tensor
    static_inputs: torch.Tensor
    targets: torch.Tensor
    target_dates: tuple[date, ...]
    basin_ids: tuple[str, ...]
    routing_inputs: torch.Tensor | None = None


@dataclass(frozen=True)
class RegionalHydrologyData:
    train: RegionalSplit
    validation: RegionalSplit
    test: RegionalSplit
    feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    static_mean: torch.Tensor
    static_scale: torch.Tensor
    discharge_scale: torch.Tensor
    training_basins: tuple[str, ...]
    target_basins: tuple[str, ...]


def effective_training_basins(scope: BasinScopeConfig, eligible: Iterable[str]) -> tuple[str, ...]:
    allowed = tuple(dict.fromkeys(map(str, scope.eligible_basins or tuple(eligible))))
    targets = set(map(str, scope.target_basins))
    if not targets.issubset(set(allowed)):
        raise ValueError("all target basins must be eligible")
    mode = _SCOPE_ALIASES.get(scope.training_scope)
    if mode == "exclude_targets":
        result = tuple(item for item in allowed if item not in targets)
    elif mode == "all_basins":
        result = allowed
    elif mode == "targets_only":
        result = tuple(item for item in allowed if item in targets)
    else:
        raise ValueError(f"unknown training scope {scope.training_scope!r}")
    if not result:
        raise ValueError("the selected training scope produces an empty training-basin set")
    return result


def _empty_split(sequence_length: int, feature_count: int, static_count: int) -> RegionalSplit:
    return RegionalSplit(torch.empty(0, sequence_length, feature_count),
                         torch.empty(0, sequence_length, feature_count),
                         torch.empty(0, static_count), torch.empty(0), (), (), None)


def make_regional_hydrology_data(
    series_by_basin: Mapping[str, HydrologySeries],
    static_attributes: Mapping[str, Mapping[str, float | None]],
    scope: BasinScopeConfig,
    dates: DateSplitConfig,
    *,
    sequence_length: int = 30,
) -> RegionalHydrologyData:
    """Create windows first, then assign them by target date without leakage."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    targets = tuple(dict.fromkeys(map(str, scope.target_basins)))
    training_basins = effective_training_basins(scope, series_by_basin)
    relevant = tuple(dict.fromkeys((*training_basins, *targets)))
    missing = set(relevant) - set(series_by_basin)
    if missing:
        raise ValueError(f"series missing for basins: {sorted(missing)}")
    forcing_names = series_by_basin[relevant[0]].forcing_names
    if any(series_by_basin[item].forcing_names != forcing_names for item in relevant):
        raise ValueError("all regional series must have identical forcing columns")
    feature_names = (*forcing_names, "previous_discharge", "day_sin", "day_cos")
    static_names = tuple(sorted({key for basin in training_basins
                                 for key in static_attributes.get(basin, {})}))
    raw_static = torch.tensor([
        [float(static_attributes.get(basin, {}).get(name, float("nan")) or 0.0)
         if static_attributes.get(basin, {}).get(name) is not None else float("nan")
         for name in static_names] for basin in training_basins
    ], dtype=torch.float32)
    if static_names:
        static_mean = torch.nanmean(raw_static, dim=0)
        static_mean = torch.nan_to_num(static_mean)
        filled = torch.where(torch.isnan(raw_static), static_mean, raw_static)
        static_scale = filled.std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        static_mean, static_scale = torch.empty(0), torch.empty(0)

    ranges = {"train": dates.train.resolved(), "validation": dates.validation.resolved(),
              "test": dates.test.resolved()}
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor, float, date, str, torch.Tensor]]] = {
        "train": [], "validation": [], "test": []
    }
    all_train_physical: list[torch.Tensor] = []
    all_train_targets: list[float] = []
    for basin in relevant:
        series = series_by_basin[basin]
        day_of_year = torch.tensor([item.timetuple().tm_yday for item in series.dates], dtype=torch.float32)
        angle = 2.0 * torch.pi * day_of_year / 365.25
        features = torch.cat((series.forcings, series.discharge[:, None],
                              angle.sin()[:, None], angle.cos()[:, None]), dim=1)
        raw = [static_attributes.get(basin, {}).get(name) for name in static_names]
        static = torch.tensor([static_mean[index] if value is None or not math.isfinite(float(value))
                               else float(value) for index, value in enumerate(raw)])
        for target_index in range(sequence_length, len(series.dates)):
            first = target_index - sequence_length
            if series.dates[target_index] - series.dates[first] != timedelta(days=sequence_length):
                continue
            target_date = series.dates[target_index]
            split_name = next((name for name, item in ranges.items() if item.contains(target_date)), None)
            if split_name is None:
                continue
            # Only effective model-training basins contribute training samples;
            # validation and test are prediction-target views.
            if split_name == "train" and basin not in training_basins:
                continue
            if split_name != "train" and basin not in targets:
                continue
            window = features[first:target_index]
            buckets[split_name].append((window, window, float(series.discharge[target_index]),
                                        target_date, basin, static))
            if split_name == "train":
                all_train_physical.append(window)
                all_train_targets.append(float(series.discharge[target_index]))
    if not all_train_physical:
        raise ValueError("no training windows fall in the requested training date range")
    if not buckets["validation"] or not buckets["test"]:
        raise ValueError("validation and test ranges must each contain target-basin windows")
    train_tensor = torch.stack(all_train_physical)
    feature_mean = train_tensor.reshape(-1, train_tensor.shape[-1]).mean(0)
    feature_scale = train_tensor.reshape(-1, train_tensor.shape[-1]).std(0, unbiased=False).clamp_min(1e-6)
    discharge_scale = torch.tensor(all_train_targets).std(unbiased=False).clamp_min(1e-6)

    def build(name: str) -> RegionalSplit:
        rows = buckets[name]
        if not rows:
            return _empty_split(sequence_length, len(feature_names), len(static_names))
        physical = torch.stack([row[0] for row in rows])
        normalized = (physical - feature_mean) / feature_scale
        statics = torch.stack([row[5] for row in rows]) if static_names else torch.empty(len(rows), 0)
        statics = (statics - static_mean) / static_scale if static_names else statics
        return RegionalSplit(normalized, physical, statics,
                             torch.tensor([row[2] for row in rows]) / discharge_scale,
                             tuple(row[3] for row in rows), tuple(row[4] for row in rows), normalized)
    return RegionalHydrologyData(build("train"), build("validation"), build("test"),
                                 tuple(feature_names), static_names, feature_mean, feature_scale,
                                 static_mean, static_scale, discharge_scale,
                                 training_basins, targets)


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self.cancelled


@dataclass(frozen=True)
class PredictionRecord:
    seed: int
    basin_id: str
    target_date: date
    observed_mm_day: float
    predicted_mm_day: float
    complex_weight: float
    physics_error: float = float("nan")
    ood_score: float = float("nan")
    ood_threshold: float = float("nan")
    issue_date: date | None = None
    lead_day: int = 1


@dataclass
class ExperimentResult:
    predictions: list[PredictionRecord]
    aggregate_metrics: dict[str, float]
    per_basin_metrics: dict[str, dict[str, float]]
    routing_diagnostics: dict[str, object]
    loss_traces: dict[str, list[float]]
    extreme_events: list[dict[str, object]]
    failures: list[str]
    resolved_config: dict[str, object]
    parameter_count: int = 0
    runtime_seconds: float = 0.0
    cancelled: bool = False
    artifact_paths: dict[str, str] = field(default_factory=dict)
    forecast_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)


ProgressCallback = Callable[[float, str], None]


def _cancelled(token: object | None) -> bool:
    if token is None:
        return False
    value = getattr(token, "cancelled", False)
    return bool(value() if callable(value) else value) or bool(
        getattr(token, "is_cancelled", lambda: False)()
    )


def _architecture_kwargs(config: ModelArchitectureConfig, input_dim: int) -> tuple[dict, dict]:
    complex_kwargs = ({"hidden_dims": config.mlp_widths} if config.complex_expert == "mlp" else {
        "hidden_dim": config.mamba_hidden_size, "num_layers": config.mamba_layers,
        "hidden_d_ff": config.mamba_feed_forward_size,
        "static_embedding_dim": config.static_embedding_size,
    })
    if config.simple_expert == "rbf":
        simple_kwargs = {"n_centers": config.rbf_centers}
        if config.rbf_width is not None:
            simple_kwargs["initial_width"] = config.rbf_width
    else:
        simple_kwargs = {"n_frequencies": config.fourier_frequencies}
        if config.fourier_scale is not None:
            simple_kwargs["frequency_scale"] = config.fourier_scale
    return simple_kwargs, complex_kwargs


def _train(
    model: RoutedHydrologyModel, split: RegionalSplit, config: RegionalExperimentConfig,
    seed: int, device: torch.device, progress: ProgressCallback | None,
    token: object | None, traces: dict[str, list[float]], seed_index: int, seed_count: int,
    feature_names: tuple[str, ...],
) -> None:
    hp, strategy = config.hyperparameters, config.strategy
    inputs, physical = split.inputs.to(device), split.physical_inputs.to(device)
    static, targets = split.static_inputs.to(device), split.targets.to(device)
    if strategy.approach in {"soft_routing", "hard_routing", "distillation"}:
        model.fit_router(inputs)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)

    def batches(epoch: int):
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), hp.batch_size):
            yield permutation[start:start + hp.batch_size]

    def run_joint(epochs: int, phase: str, hard: bool = False) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
        for epoch in range(epochs):
            if _cancelled(token):
                return
            total = 0.0
            for indices in batches(epoch):
                if _cancelled(token):
                    return
                noisy = _add_observation_noise(inputs[indices], config_data_names, hp.training_noise, generator)
                optimizer.zero_grad()
                if hard:
                    # Inline the static-aware hard loss.
                    soft = model(noisy, return_details=True, static_inputs=static[indices])
                    assert isinstance(soft, HydrologyRoutedOutput)
                    route = model.router.complex_weight(model._router_features(noisy, None), hard=True)
                    route = route + soft.complex_weight - soft.complex_weight.detach()
                    discharge = model._positive_discharge(torch.lerp(
                        soft.simple_raw, soft.complex_raw, route[:, None]))
                    data_loss = (discharge.squeeze(-1) - targets[indices]).square().mean()
                    physics = model.physics_residual(discharge, physical[indices]).square().mean()
                    transition = 4 * soft.complex_weight * (1 - soft.complex_weight)
                    interface = (transition * (soft.simple_raw-soft.complex_raw).square().squeeze(-1)).mean()
                    routing = model.router.regularization(model._router_features(noisy, None))
                    loss = data_loss + hp.physics_weight*physics + hp.interface_weight*interface + hp.routing_weight*routing + hp.compute_weight*soft.complex_weight.mean()
                else:
                    loss = model.losses(noisy, physical[indices], targets[indices],
                                        static_inputs=static[indices], physics_weight=hp.physics_weight,
                                        interface_weight=hp.interface_weight,
                                        routing_weight=hp.routing_weight,
                                        compute_weight=hp.compute_weight).total
                loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(indices)
            traces.setdefault(phase, []).append(total / len(inputs))
            if progress:
                progress((seed_index + (epoch + 1) / epochs) / seed_count,
                         f"seed {seed}: {phase} epoch {epoch + 1}/{epochs}")

    config_data_names = feature_names
    if strategy.approach in {"no_routing", "static_50_50", "ood_fallback", "stacking"}:
        parameters = [*model.complex_expert.parameters(), model.raw_response, model.raw_recession]
        if strategy.approach != "no_routing":
            parameters.extend(model.simple_expert.parameters())
        optimizer = torch.optim.Adam(parameters, lr=hp.learning_rate)
        for epoch in range(strategy.epochs):
            if _cancelled(token):
                return
            total = 0.0
            for indices in batches(epoch):
                if _cancelled(token):
                    return
                noisy = _add_observation_noise(
                    inputs[indices], config_data_names, hp.training_noise, generator
                )
                optimizer.zero_grad()
                complex_prediction = model._positive_discharge(
                    model.complex_expert(noisy, static[indices])
                )
                complex_loss = (
                    (complex_prediction.squeeze(-1) - targets[indices]).square().mean()
                    + hp.physics_weight
                    * model.physics_residual(complex_prediction, physical[indices]).square().mean()
                )
                if strategy.approach == "no_routing":
                    loss = complex_loss
                else:
                    simple_prediction = model._positive_discharge(
                        model.simple_expert(noisy, static[indices])
                    )
                    simple_loss = (
                        (simple_prediction.squeeze(-1) - targets[indices]).square().mean()
                        + hp.physics_weight
                        * model.physics_residual(simple_prediction, physical[indices]).square().mean()
                    )
                    loss = 0.5 * (simple_loss + complex_loss)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            traces.setdefault(strategy.approach, []).append(total / len(inputs))
            if progress:
                progress(
                    (seed_index + (epoch + 1) / strategy.epochs) / seed_count,
                    f"seed {seed}: {strategy.approach} epoch {epoch + 1}/{strategy.epochs}",
                )
        return
    if strategy.approach in {"soft_routing", "hard_routing"}:
        run_joint(strategy.epochs, strategy.approach, strategy.approach == "hard_routing")
        return

    # Complex teacher -> simple student on hard simple-space -> consolidation.
    complex_optimizer = torch.optim.Adam([
        *model.complex_expert.parameters(), model.raw_response, model.raw_recession
    ], lr=hp.learning_rate)
    for epoch in range(strategy.complex_teacher_epochs):
        if _cancelled(token): return
        total = 0.0
        for indices in batches(epoch):
            if _cancelled(token): return
            noisy = _add_observation_noise(inputs[indices], config_data_names, hp.training_noise, generator)
            complex_optimizer.zero_grad()
            prediction = model._positive_discharge(model.complex_expert(noisy, static[indices]))
            loss = (prediction.squeeze(-1)-targets[indices]).square().mean() + hp.physics_weight * model.physics_residual(prediction, physical[indices]).square().mean()
            loss.backward(); complex_optimizer.step(); total += float(loss.detach())*len(indices)
        traces.setdefault("complex_teacher", []).append(total/len(inputs))
        if progress:
            progress((seed_index + (epoch + 1) / strategy.complex_teacher_epochs / 3) / seed_count,
                     f"seed {seed}: complex teacher epoch {epoch + 1}/{strategy.complex_teacher_epochs}")
    with torch.no_grad():
        simple_indices = torch.nonzero(model.router.complex_weight(model._router_features(inputs, None), hard=True) < .5).squeeze(-1)
    if not len(simple_indices):
        raise ValueError("router assigned no samples to the simple expert")
    simple_optimizer = torch.optim.Adam(model.simple_expert.parameters(), lr=hp.learning_rate)
    for epoch in range(strategy.distillation_epochs):
        if _cancelled(token): return
        total = 0.0
        perm = simple_indices[torch.randperm(len(simple_indices), generator=generator, device=device)]
        for start in range(0, len(perm), hp.batch_size):
            if _cancelled(token): return
            indices = perm[start:start+hp.batch_size]
            noisy = _add_observation_noise(inputs[indices], config_data_names, hp.training_noise, generator)
            with torch.no_grad(): teacher = model._positive_discharge(model.complex_expert(noisy, static[indices]))
            simple_optimizer.zero_grad(); student = model._positive_discharge(model.simple_expert(noisy, static[indices]))
            loss = (student-teacher).square().mean(); loss.backward(); simple_optimizer.step()
            total += float(loss.detach())*len(indices)
        traces.setdefault("distillation", []).append(total/len(simple_indices))
        if progress:
            progress((seed_index + (1 + (epoch + 1) / strategy.distillation_epochs) / 3) / seed_count,
                     f"seed {seed}: distillation epoch {epoch + 1}/{strategy.distillation_epochs}")
    run_joint(strategy.consolidation_epochs, "consolidation")


def _thresholds(config: ExtremeEventConfig, targets: tuple[str, ...],
                series: Mapping[str, HydrologySeries], train_range: DateRange) -> dict[str, float]:
    if config.mode == "none": return {}
    result = {}
    period = train_range.resolved()
    for basin in targets:
        values = torch.tensor([float(value) for day, value in zip(series[basin].dates, series[basin].discharge)
                               if period.contains(day)])
        if not len(values): raise ValueError(f"no historical target observations for {basin}")
        setting = config.per_basin_values.get(basin, config.value)
        if config.definition == "absolute":
            if setting is None: raise ValueError("an absolute threshold value is required")
            result[basin] = float(setting)
        else:
            quantile = .95 if config.definition == "automatic_q95" else setting
            if quantile is None or not 0 < quantile < 1: raise ValueError("a valid quantile is required")
            result[basin] = float(torch.quantile(values, float(quantile)))
    return result


def _event_thresholds(config: ExtremeEventConfig, targets: tuple[str, ...],
                      series: Mapping[str, HydrologySeries], train_range: DateRange
                      ) -> dict[str, tuple[float, float, float, float]]:
    period = train_range.resolved()
    thresholds = {}
    for basin in targets:
        days = series[basin].dates
        flow = series[basin].discharge
        indices = [index for index, day in enumerate(days) if period.contains(day)]
        if not indices:
            raise ValueError(f"no training-period observations for basin {basin}")
        values = flow[indices]
        changes = torch.tensor([
            float(flow[index] - flow[index - 1])
            for index in indices if index > 0 and period.contains(days[index - 1])
            and days[index] == days[index - 1] + timedelta(days=1)
        ])
        if not len(changes):
            raise ValueError(f"no consecutive training-period observations for basin {basin}")
        thresholds[basin] = (
            float(torch.quantile(values, config.high_quantile)),
            float(torch.quantile(values, config.low_quantile)),
            float(torch.quantile(changes.clamp_min(0), config.rise_quantile)),
            float(torch.quantile((-changes).clamp_min(0), config.fall_quantile)),
        )
    return thresholds


def _teacher_prediction(
    model: RoutedHydrologyModel,
    split: RegionalSplit,
    approach: str,
    device: torch.device,
    *,
    train_split: RegionalSplit,
    validation_split: RegionalSplit,
    ood_quantile: float,
    ood_shrinkage: float,
) -> torch.Tensor:
    """Return a frozen approach's final normalized discharge prediction."""
    inputs = split.inputs.to(device)
    static = split.static_inputs.to(device)
    with torch.no_grad():
        if approach not in {"no_routing", "static_50_50", "ood_fallback", "stacking"}:
            output = model(
                inputs, static_inputs=static, return_details=True,
                hard=approach == "hard_routing",
            )
            assert isinstance(output, HydrologyRoutedOutput)
            return output.discharge.squeeze(-1)
        complex_prediction = model._positive_discharge(model.complex_expert(inputs, static))
        if approach == "no_routing":
            return complex_prediction.squeeze(-1)
        simple_prediction = model._positive_discharge(model.simple_expert(inputs, static))
        if approach == "static_50_50":
            weight = torch.full((len(inputs),), 0.5, device=device)
        elif approach == "stacking":
            validation_inputs = validation_split.inputs.to(device)
            validation_static = validation_split.static_inputs.to(device)
            validation_simple = model._positive_discharge(
                model.simple_expert(validation_inputs, validation_static)
            )
            validation_complex = model._positive_discharge(
                model.complex_expert(validation_inputs, validation_static)
            )
            fitted_weight, _ = fit_convex_stacking_weight(
                validation_simple, validation_complex,
                validation_split.targets.to(device),
            )
            weight = torch.full((len(inputs),), fitted_weight, device=device)
        else:
            detector = MahalanobisOODDetector(ood_quantile, ood_shrinkage).fit(
                train_split.inputs.to(device), train_split.static_inputs.to(device)
            )
            weight = (~detector.is_ood(inputs, static)).to(inputs.dtype)
        return torch.lerp(simple_prediction, complex_prediction, weight[:, None]).squeeze(-1)


def _predict_with_approach(
    model: RoutedHydrologyModel, inputs: torch.Tensor, static: torch.Tensor,
    approach: str, *, stacking_weight: float | None = None,
    ood_detector: MahalanobisOODDetector | None = None,
) -> tuple[HydrologyRoutedOutput, torch.Tensor]:
    """Apply one trained approach to a batch of normalized input windows."""
    ood_score = torch.full((len(inputs),), float("nan"), device=inputs.device)
    if approach not in {"no_routing", "static_50_50", "ood_fallback", "stacking"}:
        output = model(inputs, return_details=True, static_inputs=static,
                       hard=approach == "hard_routing")
        assert isinstance(output, HydrologyRoutedOutput)
        return output, ood_score
    complex_raw = model.complex_expert(inputs, static)
    complex_prediction = model._positive_discharge(complex_raw)
    if approach == "no_routing":
        simple_raw = torch.zeros_like(complex_raw)
        simple_prediction = torch.zeros_like(complex_prediction)
        weight = torch.ones(len(inputs), device=inputs.device)
    else:
        simple_raw = model.simple_expert(inputs, static)
        simple_prediction = model._positive_discharge(simple_raw)
        if approach == "static_50_50":
            weight = torch.full((len(inputs),), 0.5, device=inputs.device)
        elif approach == "stacking":
            assert stacking_weight is not None
            weight = torch.full((len(inputs),), stacking_weight, device=inputs.device)
        else:
            assert ood_detector is not None and ood_detector.threshold is not None
            ood_score = ood_detector.score(inputs, static)
            weight = (~ood_detector.is_ood(inputs, static)).to(inputs.dtype)
    output = HydrologyRoutedOutput(
        torch.lerp(simple_prediction, complex_prediction, weight[:, None]),
        weight, simple_raw, complex_raw,
    )
    return output, ood_score


def _rolling_test_predictions(
    model: RoutedHydrologyModel, data: RegionalHydrologyData,
    config: RegionalExperimentConfig, seed: int, device: torch.device,
    *, stacking_weight: float | None, ood_detector: MahalanobisOODDetector | None,
) -> list[PredictionRecord]:
    """Forecast non-overlapping blocks, replacing future observed flow with predictions.

    Future meteorological covariates are historical observations, so this is a
    perfect-weather hindcast rather than an operational weather forecast.
    """
    result: list[PredictionRecord] = []
    q_index = data.feature_names.index("previous_discharge")
    scale = float(data.discharge_scale)
    mean = float(data.feature_mean[q_index])
    std = float(data.feature_scale[q_index])
    horizon = config.forecast.horizon_days
    for basin in data.target_basins:
        indices = [index for index, item in enumerate(data.test.basin_ids) if item == basin]
        indices.sort(key=lambda index: data.test.target_dates[index])
        continuous: list[list[int]] = []
        for index in indices:
            if (not continuous or data.test.target_dates[index]
                    != data.test.target_dates[continuous[-1][-1]] + timedelta(days=1)):
                continuous.append([])
            continuous[-1].append(index)
        for segment in continuous:
            for offset in range(0, len(segment), horizon):
                block = segment[offset:offset + horizon]
                issue_date = data.test.target_dates[block[0]] - timedelta(days=1)
                predicted_flow: dict[date, float] = {}
                for lead, index in enumerate(block, 1):
                    day = data.test.target_dates[index]
                    normalized = data.test.inputs[index:index + 1].to(device).clone()
                    physical = data.test.physical_inputs[index:index + 1].to(device).clone()
                    for sequence_index in range(normalized.shape[1]):
                        input_date = day - timedelta(days=normalized.shape[1] - sequence_index)
                        if input_date in predicted_flow:
                            value = predicted_flow[input_date]
                            physical[0, sequence_index, q_index] = value
                            normalized[0, sequence_index, q_index] = (value - mean) / std
                    static = data.test.static_inputs[index:index + 1].to(device)
                    output, ood_score = _predict_with_approach(
                        model, normalized, static, config.strategy.approach,
                        stacking_weight=stacking_weight, ood_detector=ood_detector,
                    )
                    value = float(output.discharge[0, 0].cpu()) * scale
                    predicted_flow[day] = value
                    residual = float(model.physics_residual(output.discharge, physical).square()[0].cpu())
                    result.append(PredictionRecord(
                        seed, basin, day, float(data.test.targets[index]) * scale, value,
                        float(output.complex_weight[0].cpu()), residual,
                        float(ood_score[0].cpu()),
                        float(ood_detector.threshold.cpu()) if ood_detector is not None else float("nan"),
                        issue_date, lead,
                    ))
    return result


def _resolved(config: RegionalExperimentConfig, data: RegionalHydrologyData) -> dict[str, object]:
    value = asdict(config)
    value["data_root"] = str(config.data_root)
    value["analysis_task"] = (
        "daily_discharge_with_extremes" if config.extremes.mode == "statistical"
        else "daily_discharge_with_hydrological_events" if config.extremes.mode == "event_based"
        else "daily_discharge_forecasting"
    )
    if config.estreams_root is not None:
        value["estreams_root"] = str(config.estreams_root)
    if config.camels_chem_root is not None:
        value["camels_chem_root"] = str(config.camels_chem_root)
    for split in ("train", "validation", "test"):
        item = value["dates"][split]  # type: ignore[index]
        item["start"], item["end"] = str(item["start"]), str(item["end"])
    value["effective_training_basins"] = list(data.training_basins)
    value["feature_names"] = list(data.feature_names)
    value["static_feature_names"] = list(data.static_feature_names)
    return value


def run_regional_experiment(
    config: RegionalExperimentConfig,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: object | None = None,
    *,
    catalog: CAMELSCHCatalog | None = None,
    series_by_basin: Mapping[str, HydrologySeries] | None = None,
) -> ExperimentResult:
    """Train one shared regional model per seed and predict selected targets."""
    started = time.perf_counter(); config.validate()
    catalog = catalog or CAMELSCHCatalog(config.data_root)
    eligible = tuple(config.basins.eligible_basins or catalog.eligible_ids)
    training = effective_training_basins(config.basins, eligible)
    relevant = tuple(dict.fromkeys((*training, *config.basins.target_basins)))
    full_start = config.dates.train.resolved().start - timedelta(days=config.hyperparameters.sequence_length)  # type: ignore[operator]
    full_end = config.dates.test.resolved().end
    series = dict(series_by_basin or {})
    load_failures: list[str] = []
    target_set = set(config.basins.target_basins)
    for basin in relevant:
        if basin not in series:
            try:
                series[basin] = catalog.load_series(basin, full_start, full_end)  # type: ignore[arg-type]
            except (ValueError, FileNotFoundError) as error:
                message = f"basin {basin} could not be loaded: {error}"
                if basin in target_set:
                    raise ValueError(message) from error
                load_failures.append(message)
    if config.estreams_root is not None:
        series = EStreamsVegetationSnow(config.estreams_root).augment(
            series, config.estreams_features
        )
    if config.camels_chem_root is not None:
        series = CAMELSCHChemPressures(config.camels_chem_root).augment(
            series, config.camels_chem_features
        )
    loaded = tuple(basin for basin in relevant if basin in series)
    attrs = {basin: catalog.get(basin).static_attributes for basin in loaded}
    data = make_regional_hydrology_data(series, attrs, config.basins, config.dates,
                                        sequence_length=config.hyperparameters.sequence_length)
    resolved = _resolved(config, data)
    traces: dict[str, list[float]] = {}; predictions: list[PredictionRecord] = []
    ood_thresholds: list[float] = []
    stacking_weights: list[float] = []
    stacking_validation_mses: list[float] = []
    physics_formulas: list[str] = []
    physics_candidate_metrics: list[list[dict[str, float]]] = []
    physics_candidate_indices: list[int] = []
    failures = load_failures
    device = torch.device(config.hyperparameters.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    parameter_count = 0
    for seed_index, seed in enumerate(config.hyperparameters.seeds):
        if _cancelled(cancellation_token): break
        try:
            torch.manual_seed(seed)
            simple_kwargs, complex_kwargs = _architecture_kwargs(config.model,
                data.train.inputs.shape[1] * data.train.inputs.shape[2] + data.train.static_inputs.shape[1])

            def build_model() -> RoutedHydrologyModel:
                return RoutedHydrologyModel(
                    data.feature_names, data.train.inputs.shape[1], data.discharge_scale,
                    simple_kind=config.model.simple_expert,
                    complex_kind=config.model.complex_expert, routing="morse",
                    complexity_percentile=config.hyperparameters.complexity_percentile,
                    gate_temperature=config.hyperparameters.gate_temperature,
                    simple_kwargs=simple_kwargs, complex_kwargs=complex_kwargs,
                    static_dim=data.train.static_inputs.shape[1],
                ).to(device)

            model = build_model()
            if config.physics.optimization_mode != "none":
                teacher_approach = (
                    config.physics.distillation_teacher
                    if config.physics.optimization_mode == "distillation"
                    else config.strategy.approach
                )
                teacher_config = replace(
                    config,
                    strategy=replace(config.strategy, approach=teacher_approach),
                    physics=replace(config.physics, optimization_mode="none"),
                )
                teacher_traces: dict[str, list[float]] = {}
                _train(
                    model, data.train, teacher_config, seed, device, progress_callback,
                    cancellation_token, teacher_traces, seed_index,
                    len(config.hyperparameters.seeds), data.feature_names,
                )
                if _cancelled(cancellation_token):
                    break
                model.eval()
                response_tensor, recession_tensor = model.reservoir_parameters()
                response = float(response_tensor.detach().cpu())
                recession = float(recession_tensor.detach().cpu())
                train_teacher = validation_teacher = None
                if config.physics.optimization_mode == "distillation":
                    train_teacher = _teacher_prediction(
                        model, data.train, teacher_approach, device,
                        train_split=data.train, validation_split=data.validation,
                        ood_quantile=config.hyperparameters.ood_quantile,
                        ood_shrinkage=config.hyperparameters.ood_shrinkage,
                    ) * data.discharge_scale.to(device)
                    validation_teacher = _teacher_prediction(
                        model, data.validation, teacher_approach, device,
                        train_split=data.train, validation_split=data.validation,
                        ood_quantile=config.hyperparameters.ood_quantile,
                        ood_shrinkage=config.hyperparameters.ood_shrinkage,
                    ) * data.discharge_scale.to(device)
                settings = KANPhysicsSettings(
                    mode=config.physics.optimization_mode,
                    candidates=config.physics.kan_candidates,
                    steps=config.physics.kan_steps,
                    grid=config.physics.kan_grid,
                    spline_order=config.physics.kan_spline_order,
                    additive_nodes=config.physics.kan_additive_nodes,
                    multiplicative_nodes=config.physics.kan_multiplicative_nodes,
                    sparsity_weight=config.physics.kan_sparsity_weight,
                    noise_levels=config.physics.robustness_noise_levels,
                    robustness_accuracy_tolerance=(
                        config.physics.robustness_accuracy_tolerance
                    ),
                    symbolic=config.physics.symbolic,
                )
                discovery = discover_kan_physics_correction(
                    train_physical=data.train.physical_inputs,
                    validation_physical=data.validation.physical_inputs,
                    train_observed_flow=(
                        data.train.targets * data.discharge_scale
                    ),
                    validation_observed_flow=(
                        data.validation.targets * data.discharge_scale
                    ),
                    feature_names=data.feature_names,
                    response=response,
                    recession=recession,
                    settings=settings,
                    seed=seed,
                    device=device,
                    train_teacher_flow=train_teacher,
                    validation_teacher_flow=validation_teacher,
                    train_basin_ids=data.train.basin_ids,
                    validation_basin_ids=data.validation.basin_ids,
                )
                physics_formulas.append(
                    f"delta_q = {response:.8g}*max(precipitation-pet, 0) "
                    f"- {recession:.8g}*previous_discharge + ({discovery.formula})"
                )
                physics_candidate_metrics.append(discovery.candidate_metrics)
                physics_candidate_indices.append(discovery.candidate_index)
                for phase, values in teacher_traces.items():
                    traces.setdefault(f"physics_teacher_{phase}", []).extend(values)
                # Report the frozen equation's direct observation error before a
                # fresh predictive model is trained with it. These are one-step
                # predictions because each split supplies observed previous flow.
                with torch.no_grad():
                    for split_name, split in (
                        ("train", data.train),
                        ("validation", data.validation),
                        ("test", data.test),
                    ):
                        standalone = standalone_kan_prediction(
                            split.physical_inputs.to(device), data.feature_names,
                            response, recession, discovery.correction,
                        )
                        observed = split.targets.to(device) * data.discharge_scale.to(device)
                        normalized_mse = (
                            (standalone - observed) / data.discharge_scale.to(device)
                        ).square().mean()
                        traces.setdefault(
                            f"kan_standalone_{split_name}_observation_mse", []
                        ).append(float(normalized_mse.cpu()))
                teacher_raw_response = model.raw_response.detach().clone()
                teacher_raw_recession = model.raw_recession.detach().clone()
                model = build_model()
                model.raw_response.data.copy_(teacher_raw_response)
                model.raw_recession.data.copy_(teacher_raw_recession)
                model.raw_response.requires_grad_(False)
                model.raw_recession.requires_grad_(False)
                model.set_physics_correction(discovery.correction)
            if config.strategy.approach == "no_routing":
                parameter_count = sum(
                    item.numel() for item in (
                        *model.complex_expert.parameters(), model.raw_response, model.raw_recession
                    )
                )
            else:
                parameter_count = sum(item.numel() for item in model.parameters() if item.requires_grad)
            _train(model, data.train, config, seed, device, progress_callback,
                   cancellation_token, traces, seed_index, len(config.hyperparameters.seeds),
                   data.feature_names)
            if _cancelled(cancellation_token): break
            model.eval()
            with torch.no_grad():
                stacking_weight = None
                if config.strategy.approach == "stacking":
                    validation_inputs = data.validation.inputs.to(device)
                    validation_static = data.validation.static_inputs.to(device)
                    validation_simple = model._positive_discharge(
                        model.simple_expert(validation_inputs, validation_static)
                    )
                    validation_complex = model._positive_discharge(
                        model.complex_expert(validation_inputs, validation_static)
                    )
                    stacking_weight, validation_mse = fit_convex_stacking_weight(
                        validation_simple, validation_complex,
                        data.validation.targets.to(device),
                    )
                    stacking_weights.append(stacking_weight)
                    stacking_validation_mses.append(
                        validation_mse * float(data.discharge_scale.square())
                    )
                detector = None
                if config.strategy.approach == "ood_fallback":
                    detector = MahalanobisOODDetector(
                        config.hyperparameters.ood_quantile,
                        config.hyperparameters.ood_shrinkage,
                    ).fit(data.train.inputs.to(device), data.train.static_inputs.to(device))
                    assert detector.threshold is not None
                    ood_thresholds.append(float(detector.threshold.cpu()))
                if config.forecast.mode == "rolling":
                    predictions.extend(_rolling_test_predictions(
                        model, data, config, seed, device,
                        stacking_weight=stacking_weight, ood_detector=detector,
                    ))
                else:
                    output, ood_score = _predict_with_approach(
                        model, data.test.inputs.to(device), data.test.static_inputs.to(device),
                        config.strategy.approach, stacking_weight=stacking_weight,
                        ood_detector=detector,
                    )
                    predicted = output.discharge.squeeze(-1).cpu() * data.discharge_scale
                    observed = data.test.targets * data.discharge_scale
                    residual = model.physics_residual(
                        output.discharge, data.test.physical_inputs.to(device)
                    ).square().cpu()
                    for index in range(len(predicted)):
                        predictions.append(PredictionRecord(
                            seed, data.test.basin_ids[index], data.test.target_dates[index],
                            float(observed[index]), float(predicted[index]),
                            float(output.complex_weight[index].cpu()), float(residual[index]),
                            float(ood_score[index].cpu()),
                            float(detector.threshold.cpu()) if detector is not None else float("nan"),
                            data.test.target_dates[index] - timedelta(days=1), 1,
                        ))
        except Exception as error:
            failures.append(f"seed {seed}: {error}")
    per_basin: dict[str, dict[str, float]] = {}
    for basin in data.target_basins:
        rows = [item for item in predictions if item.basin_id == basin]
        if rows:
            pred = torch.tensor([item.predicted_mm_day for item in rows]); obs = torch.tensor([item.observed_mm_day for item in rows])
            nse, kge, rmse = _metrics(pred, obs)
            per_basin[basin] = {"nse": nse, "kge": kge, "rmse_mm_day": rmse,
                                "physics_error": sum(item.physics_error for item in rows)/len(rows)}
    if predictions:
        pred = torch.tensor([item.predicted_mm_day for item in predictions]); obs = torch.tensor([item.observed_mm_day for item in predictions])
        nse, kge, rmse = _metrics(pred, obs); aggregate = {
            "nse": nse, "kge": kge, "rmse_mm_day": rmse,
            "physics_error": sum(item.physics_error for item in predictions)/len(predictions),
        }
        routing = {"mean_complex_weight": sum(item.complex_weight for item in predictions)/len(predictions)}
        finite_ood = [item.ood_score for item in predictions if math.isfinite(item.ood_score)]
        if finite_ood:
            routing.update({
                "mean_ood_score": sum(finite_ood) / len(finite_ood),
                "ood_fraction": sum(item.complex_weight < 0.5 for item in predictions) / len(predictions),
                "mean_ood_threshold": sum(ood_thresholds) / len(ood_thresholds),
            })
        if stacking_weights:
            routing.update({
                "fitted_complex_weight": sum(stacking_weights) / len(stacking_weights),
                "stacking_validation_mse_mm2_day2": (
                    sum(stacking_validation_mses) / len(stacking_validation_mses)
                ),
                "stacking_calibration_scope": "pooled_target_validation",
            })
        if physics_formulas:
            routing.update({
                "physics_optimization_mode": config.physics.optimization_mode,
                "physics_distillation_teacher": (
                    config.physics.distillation_teacher
                    if config.physics.optimization_mode == "distillation" else "none"
                ),
                "kan_selected_formulas": physics_formulas,
                "kan_selected_candidate_indices": physics_candidate_indices,
                "kan_candidate_metrics": physics_candidate_metrics,
                "kan_selection_scope": "target_validation",
            })
    else: aggregate, routing = {}, {}
    resolved["fitted_diagnostics"] = dict(routing)
    extremes = []
    if config.extremes.mode == "statistical":
        for basin, threshold in _thresholds(config.extremes, data.target_basins, series, config.dates.train).items():
            rows = [item for item in predictions if item.basin_id == basin]
            # Metrics are evaluated per seed because repeated dates are not one event sequence.
            for seed in config.hyperparameters.seeds:
                selected = sorted((item for item in rows if item.seed == seed),
                                  key=lambda item: item.target_date)
                if selected:
                    metrics = extreme_metrics(torch.tensor([x.predicted_mm_day for x in selected]),
                        torch.tensor([x.observed_mm_day for x in selected]),
                        tuple(x.target_date for x in selected), threshold)
                    extremes.append({"basin_id": basin, "seed": seed, "event_type": "statistical_high_flow",
                                     "threshold_mm_day": threshold, **metrics})
    elif config.extremes.mode == "event_based":
        thresholds = _event_thresholds(config.extremes, data.target_basins,
                                       series, config.dates.train)
        for seed in config.hyperparameters.seeds:
            rows_by_basin = {}
            for basin in data.target_basins:
                selected = sorted((item for item in predictions
                                   if item.seed == seed and item.basin_id == basin),
                                  key=lambda item: item.target_date)
                if not selected:
                    continue
                rows_by_basin[basin] = [(item.target_date, item.observed_mm_day,
                                        item.predicted_mm_day) for item in selected]
                high, low, rise, fall = thresholds[basin]
                for event_type in config.extremes.event_types:
                    if event_type in {"regional_concurrence", "regional_low_flow"}:
                        continue
                    metrics = basin_event_metrics(
                        [item.target_date for item in selected],
                        [item.observed_mm_day for item in selected],
                        [item.predicted_mm_day for item in selected],
                        event_type=event_type, high_threshold=high,
                        low_threshold=low, rise_threshold=rise, fall_threshold=fall,
                        minimum_days=config.extremes.minimum_days,
                        pulse_gap_days=config.extremes.pulse_gap_days,
                    )
                    extremes.append({"basin_id": basin, "seed": seed,
                                     "event_type": event_type,
                                     "high_threshold_mm_day": high,
                                     "low_threshold_mm_day": low,
                                     "rise_threshold_mm_day": rise,
                                     "fall_threshold_mm_day": fall, **metrics})
            if len(rows_by_basin) >= config.extremes.minimum_basins:
                for event_type, threshold_index, direction in (
                    ("regional_concurrence", 0, "high"),
                    ("regional_low_flow", 1, "low"),
                ):
                    if event_type in config.extremes.event_types:
                        metrics = regional_event_metrics(
                            rows_by_basin,
                            {basin: thresholds[basin][threshold_index] for basin in rows_by_basin},
                            config.extremes.minimum_basins, config.extremes.minimum_days,
                            direction=direction,
                        )
                        extremes.append({"basin_id": "all", "seed": seed,
                                         "event_type": event_type, **metrics})
    result = ExperimentResult(predictions, aggregate, per_basin, routing, traces, extremes,
                              failures, resolved, parameter_count, time.perf_counter()-started,
                              _cancelled(cancellation_token))
    if config.forecast.mode == "rolling":
        for lead in sorted({item.lead_day for item in predictions}):
            rows = [item for item in predictions if item.lead_day == lead]
            prediction = torch.tensor([item.predicted_mm_day for item in rows])
            observation = torch.tensor([item.observed_mm_day for item in rows])
            nse, kge, rmse = _metrics(prediction, observation)
            result.forecast_metrics[str(lead)] = {
                "samples": len(rows), "nse": nse, "kge": kge, "rmse_mm_day": rmse,
            }
    return result


def write_result_artifacts(result: ExperimentResult, directory: str | Path | None = None) -> dict[str, str]:
    """Write stable CSV/JSON schemas and a combined ZIP archive."""
    root = Path(directory or tempfile.mkdtemp(prefix="regional-hydrology-")); root.mkdir(parents=True, exist_ok=True)
    predictions_path, metrics_path, config_path = root/"predictions.csv", root/"metrics.csv", root/"config.json"
    with predictions_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("seed","basin_id","target_date","observed_mm_day","predicted_mm_day","complex_weight","physics_error","ood_score","ood_threshold","issue_date","lead_day")); writer.writeheader()
        for item in result.predictions:
            row = asdict(item); row["target_date"] = item.target_date.isoformat()
            row["issue_date"] = item.issue_date.isoformat() if item.issue_date else ""
            writer.writerow(row)
    with metrics_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("scope","basin_id","nse","kge","rmse_mm_day","physics_error")); writer.writeheader()
        writer.writerow({"scope":"aggregate","basin_id":"all",**result.aggregate_metrics})
        for basin, values in result.per_basin_metrics.items(): writer.writerow({"scope":"basin","basin_id":basin,**values})
        for lead, values in result.forecast_metrics.items():
            writer.writerow({"scope":f"lead_{lead}", "basin_id":"all",
                             "nse":values["nse"], "kge":values["kge"],
                             "rmse_mm_day":values["rmse_mm_day"]})
    config_path.write_text(json.dumps(result.resolved_config, indent=2, allow_nan=False), encoding="utf-8")
    archive = root/"regional_experiment.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (predictions_path, metrics_path, config_path): bundle.write(path, path.name)
    result.artifact_paths = {"predictions":str(predictions_path), "metrics":str(metrics_path),
                             "config":str(config_path), "archive":str(archive)}
    return result.artifact_paths


# Concise aliases useful to callers that prefer domain terms without the suffix.
ExperimentConfig = RegionalExperimentConfig
ModelConfig = ModelArchitectureConfig
TrainingConfig = TrainingStrategyConfig
ExtremeConfig = ExtremeEventConfig
DateRangesConfig = DateSplitConfig
PooledHydrologyData = RegionalHydrologyData
PooledHydrologySplit = RegionalSplit


__all__ = ["BasinCatalogRecord", "BasinScopeConfig", "CAMELSCHCatalog", "CancellationToken",
           "DateRange", "DateSplitConfig", "ExperimentConfig", "ExperimentResult",
           "ExtremeEventConfig", "ForecastConfig", "HyperparameterConfig", "ModelArchitectureConfig",
           "PhysicsModelConfig", "PredictionRecord", "RegionalExperimentConfig",
           "RegionalHydrologyData", "RegionalSplit", "TrainingStrategyConfig",
           "DateRangesConfig", "PooledHydrologyData", "PooledHydrologySplit",
           "PHYSICS_MODEL_REGISTRY",
           "effective_training_basins", "make_regional_hydrology_data",
           "run_regional_experiment", "write_result_artifacts"]
