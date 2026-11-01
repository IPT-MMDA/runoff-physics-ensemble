"""Configuration mapping shared by CLI, application services, and HTTP adapters."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .benchmarks import BenchmarkConfig
from .camels_ch_chem import DEFAULT_CAMELS_CH_CHEM_FEATURES
from .estreams import ESTREAMS_DYNAMIC_FEATURES
from .hyd_responses import DEFAULT_HYD_RESPONSES_FEATURES
from .regional import (
    BasinScopeConfig,
    DateRange,
    DateSplitConfig,
    ExtremeEventConfig,
    ForecastConfig,
    HyperparameterConfig,
    ModelArchitectureConfig,
    PhysicsModelConfig,
    RegionalExperimentConfig,
    TrainingStrategyConfig,
)


def _dataclass_values(cls: type[object], value: Mapping[str, object]) -> dict[str, Any]:
    # JSON supplies heterogeneous field values; the concrete dataclass constructor
    # and its validation own their types after this shared conversion boundary.
    if not is_dataclass(cls):
        raise TypeError("configuration sections must be dataclasses")
    allowed = {item.name for item in fields(cls)}
    result = {key: item for key, item in value.items() if key in allowed}
    for key in (
        "mlp_widths",
        "seeds",
        "robustness_noise_levels",
        "event_types",
        "release_states",
    ):
        item = result.get(key)
        if isinstance(item, list):
            result[key] = tuple(item)
    return result


def config_from_mapping(
    value: Mapping[str, object],
    *,
    data_root: str | Path | None = None,
    estreams_root: str | Path | None = None,
    camels_chem_root: str | Path | None = None,
    hyd_responses_root: str | Path | None = None,
    bafu_root: str | Path | None = None,
    bafu_statistics_zip: str | Path | None = None,
    bafu_return_levels: str | Path | None = None,
) -> RegionalExperimentConfig:
    """Build the typed service configuration from saved/resolved JSON."""
    try:
        basin_value = value["basins"]
        date_value = value["dates"]
    except KeyError as error:
        raise ValueError(f"configuration is missing {error.args[0]!r}") from error
    if not isinstance(basin_value, Mapping) or not isinstance(date_value, Mapping):
        raise ValueError("basins and dates must be JSON objects")

    def section(name: str) -> Mapping[str, object]:
        item = value.get(name, {})
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        return item

    def date_range(name: str) -> DateRange:
        item = date_value.get(name)
        if not isinstance(item, Mapping) or "start" not in item or "end" not in item:
            raise ValueError(f"dates.{name} must contain start and end")
        return DateRange(str(item["start"]), str(item["end"]))

    resolved_root = data_root if data_root is not None else value.get("data_root")
    if not isinstance(resolved_root, (str, Path)) or resolved_root == "":
        raise ValueError("data_root is required in the config or as --data-root")
    targets = basin_value.get("target_basins", ())
    eligible = basin_value.get("eligible_basins")
    estreams_features = value.get("estreams_features", ESTREAMS_DYNAMIC_FEATURES)
    chem_features = value.get("camels_chem_features", DEFAULT_CAMELS_CH_CHEM_FEATURES)
    state_features = value.get("camels_state_features", ())
    hyd_features = value.get("hyd_responses_features", DEFAULT_HYD_RESPONSES_FEATURES)
    if not isinstance(hyd_features, (list, tuple)):
        raise ValueError("hyd_responses_features must be a list of feature names")
    hyd_static = value.get("hyd_responses_static", False)
    if not isinstance(hyd_static, bool):
        raise ValueError("hyd_responses_static must be a boolean")
    if not isinstance(state_features, (list, tuple)):
        raise ValueError("camels_state_features must be a list of feature names")

    def optional_path(name: str, override: str | Path | None) -> str | Path | None:
        item = override if override is not None else value.get(name)
        if item in (None, ""):
            return None
        if not isinstance(item, (str, Path)):
            raise ValueError(f"{name} must be a path string")
        return item

    return RegionalExperimentConfig(
        resolved_root,
        BasinScopeConfig(
            tuple(map(str, targets)) if isinstance(targets, (list, tuple)) else (),
            str(basin_value.get("training_scope", "exclude_targets")),
            tuple(map(str, eligible)) if isinstance(eligible, (list, tuple)) else None,
        ),
        DateSplitConfig(date_range("train"), date_range("validation"), date_range("test")),
        ModelArchitectureConfig(**_dataclass_values(ModelArchitectureConfig, section("model"))),
        TrainingStrategyConfig(**_dataclass_values(TrainingStrategyConfig, section("strategy"))),
        HyperparameterConfig(**_dataclass_values(HyperparameterConfig, section("hyperparameters"))),
        PhysicsModelConfig(**_dataclass_values(PhysicsModelConfig, section("physics"))),
        ExtremeEventConfig(**_dataclass_values(ExtremeEventConfig, section("extremes"))),
        estreams_root=optional_path("estreams_root", estreams_root),
        estreams_features=(
            tuple(map(str, estreams_features))
            if isinstance(estreams_features, (list, tuple))
            else ESTREAMS_DYNAMIC_FEATURES
        ),
        camels_chem_root=optional_path("camels_chem_root", camels_chem_root),
        camels_chem_features=(
            tuple(map(str, chem_features))
            if isinstance(chem_features, (list, tuple))
            else DEFAULT_CAMELS_CH_CHEM_FEATURES
        ),
        forecast=ForecastConfig(**_dataclass_values(ForecastConfig, section("forecast"))),
        camels_state_features=tuple(map(str, state_features)),
        hyd_responses_root=optional_path("hyd_responses_root", hyd_responses_root),
        hyd_responses_features=tuple(map(str, hyd_features)),
        hyd_responses_static=hyd_static,
        benchmark=BenchmarkConfig(**_dataclass_values(BenchmarkConfig, section("benchmark"))),
        bafu_root=optional_path("bafu_root", bafu_root),
        bafu_statistics_zip=optional_path("bafu_statistics_zip", bafu_statistics_zip),
        bafu_return_levels=optional_path("bafu_return_levels", bafu_return_levels),
    )
