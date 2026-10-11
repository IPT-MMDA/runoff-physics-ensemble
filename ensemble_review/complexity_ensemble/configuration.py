"""Configuration mapping shared by CLI, application services, and HTTP adapters."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Mapping, TypeVar

from .camels_ch_chem import DEFAULT_CAMELS_CH_CHEM_FEATURES
from .estreams import ESTREAMS_DYNAMIC_FEATURES
from .regional import (
    BasinScopeConfig, DateRange, DateSplitConfig, ExtremeEventConfig, ForecastConfig,
    HyperparameterConfig, ModelArchitectureConfig, PhysicsModelConfig,
    RegionalExperimentConfig, TrainingStrategyConfig,
)

T = TypeVar("T")


def _dataclass_values(cls: type[T], value: Mapping[str, object]) -> dict[str, object]:
    allowed = {item.name for item in fields(cls)}
    result = {key: item for key, item in value.items() if key in allowed}
    for key in (
        "mlp_widths", "seeds", "robustness_noise_levels", "event_types",
    ):
        if key in result and isinstance(result[key], list):
            result[key] = tuple(result[key])  # type: ignore[index]
    return result


def config_from_mapping(
    value: Mapping[str, object], *, data_root: str | Path | None = None,
    estreams_root: str | Path | None = None,
    camels_chem_root: str | Path | None = None,
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
    if resolved_root in (None, ""):
        raise ValueError("data_root is required in the config or as --data-root")
    targets = basin_value.get("target_basins", ())
    eligible = basin_value.get("eligible_basins")
    resolved_estreams = (
        estreams_root if estreams_root is not None else value.get("estreams_root")
    )
    estreams_features = value.get("estreams_features", ESTREAMS_DYNAMIC_FEATURES)
    resolved_chem = (
        camels_chem_root if camels_chem_root is not None else value.get("camels_chem_root")
    )
    chem_features = value.get("camels_chem_features", DEFAULT_CAMELS_CH_CHEM_FEATURES)
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
        estreams_root=resolved_estreams if resolved_estreams not in (None, "") else None,
        estreams_features=(
            tuple(map(str, estreams_features))
            if isinstance(estreams_features, (list, tuple))
            else ESTREAMS_DYNAMIC_FEATURES
        ),
        camels_chem_root=resolved_chem if resolved_chem not in (None, "") else None,
        camels_chem_features=(
            tuple(map(str, chem_features))
            if isinstance(chem_features, (list, tuple))
            else DEFAULT_CAMELS_CH_CHEM_FEATURES
        ),
        forecast=ForecastConfig(**_dataclass_values(ForecastConfig, section("forecast"))),
    )


