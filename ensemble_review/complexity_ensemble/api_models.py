"""Version-one HTTP contracts, derived from the shared configuration defaults."""
from __future__ import annotations

from dataclasses import MISSING, fields
from datetime import date
from typing import Literal, Union, get_args, get_origin, get_type_hints
from collections.abc import Mapping
import types

from pydantic import (BaseModel, ConfigDict, Field, StrictBool, StrictFloat,
                      StrictInt, StrictStr, create_model)

from .benchmarks import BenchmarkConfig
from .regional import (BasinScopeConfig, HyperparameterConfig, ModelArchitectureConfig,
                       TrainingStrategyConfig, PhysicsModelConfig, ExtremeEventConfig, ForecastConfig)
from .estreams import ESTREAMS_DYNAMIC_FEATURES
from .camels_ch_chem import DEFAULT_CAMELS_CH_CHEM_FEATURES
from .hyd_responses import DEFAULT_HYD_RESPONSES_FEATURES


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def configuration_model(cls):
    hints = get_type_hints(cls)
    definitions = {}
    for f in fields(cls):
        default = ...
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:
            default = Field(default_factory=f.default_factory)
        definitions[f.name] = (strict_type(hints[f.name]), default)
    return create_model(cls.__name__.removesuffix("Config") + "Request", __base__=Contract, **definitions)


def strict_type(annotation):
    primitives = {int: StrictInt, float: StrictFloat, str: StrictStr, bool: StrictBool}
    if annotation in primitives:
        return primitives[annotation]
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return Union[tuple(strict_type(arg) for arg in args)]
    if origin is tuple:
        return tuple[strict_type(args[0]), ...]
    if origin in (dict, Mapping):
        return dict[strict_type(args[0]), strict_type(args[1])]
    return annotation


Basins = configuration_model(BasinScopeConfig)
Model = configuration_model(ModelArchitectureConfig)
Strategy = configuration_model(TrainingStrategyConfig)
Hyperparameters = configuration_model(HyperparameterConfig)
Physics = configuration_model(PhysicsModelConfig)
Extremes = configuration_model(ExtremeEventConfig)
Forecast = configuration_model(ForecastConfig)
Benchmark = configuration_model(BenchmarkConfig)


class DateRangeRequest(Contract):
    start: date
    end: date


class DateSplitsRequest(Contract):
    train: DateRangeRequest
    validation: DateRangeRequest
    test: DateRangeRequest


class ExperimentConfiguration(Contract):
    basins: Basins
    dates: DateSplitsRequest
    model: Model = Field(default_factory=Model)
    strategy: Strategy = Field(default_factory=Strategy)
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    physics: Physics = Field(default_factory=Physics)
    extremes: Extremes = Field(default_factory=Extremes)
    forecast: Forecast = Field(default_factory=Forecast)
    benchmark: Benchmark = Field(default_factory=Benchmark)
    estreams_features: tuple[str, ...] = ESTREAMS_DYNAMIC_FEATURES
    camels_chem_features: tuple[str, ...] = DEFAULT_CAMELS_CH_CHEM_FEATURES
    camels_state_features: tuple[str, ...] = ()
    hyd_responses_features: tuple[str, ...] = DEFAULT_HYD_RESPONSES_FEATURES
    hyd_responses_static: StrictBool = False


class ExperimentRequest(Contract):
    dataset_id: str = Field(min_length=1, max_length=80)
    config: ExperimentConfiguration
    use_estreams: StrictBool = False
    use_camels_chem: StrictBool = False
    use_hyd_responses: StrictBool = False
    save_to_leaderboard: StrictBool = False


class ComparisonRequest(Contract):
    run_a: str
    run_b: str


class ReplayRequest(Contract):
    basin_id: str
    episode: date
    event_days: int = Field(default=14, ge=1, le=90)


class PavedSiteRequest(Contract):
    site_fraction: float = Field(default=0.001, ge=0, le=1)
    original_paved_fraction: float = Field(default=0.1, ge=0, le=1)
    proposed_paved_fraction: float = Field(default=0.8, ge=0, le=1)
    pervious_runoff_coefficient: float = Field(default=0.25, ge=0, le=1)
    paved_runoff_coefficient: float = Field(default=0.9, ge=0, le=1)
    connected_fraction: float = Field(default=1., ge=0, le=1)
    storage_capacity_mm: float = Field(default=0., ge=0)
    release_mm_day: float = Field(default=0., ge=0)
    start_date: date | None = None


class ScenarioRequest(Contract):
    basin_id: str
    site: PavedSiteRequest = Field(default_factory=PavedSiteRequest)


class ErrorResponse(Contract):
    code: str
    message: str
    details: list[dict] = Field(default_factory=list)


class JobResponse(Contract):
    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    progress: float
    message: str
    cancel_requested: bool
    created_at: str
    updated_at: str
    result: dict | None = None
    error: dict | None = None
    links: dict[str, str]


class RunResponse(Contract):
    run_id: str
    created_at: str
    config: dict
    requested_config: dict | None = None
    aggregate_metrics: dict[str, float | None]
    per_basin_metrics: dict[str, dict[str, float | None]]
    forecast_metrics: dict[str, dict[str, float | int | None]]
    routing_diagnostics: dict
    loss_traces: dict[str, list[float | None]]
    extreme_events: list[dict]
    drought_details: dict = Field(default_factory=dict)
    drought_summary: dict = Field(default_factory=dict)
    flood_summary: dict = Field(default_factory=dict)
    failures: list[str]
    parameter_count: int
    runtime_seconds: float
    cancelled: bool


class PredictionResponse(Contract):
    seed: int
    basin_id: str
    target_date: date
    observed_mm_day: float | None
    predicted_mm_day: float | None
    complex_weight: float | None
    physics_error: float | None
    ood_score: float | None
    ood_threshold: float | None
    issue_date: date | None
    lead_day: int


class PredictionsResponse(Contract):
    items: list[PredictionResponse]
    offset: int
    limit: int
    total: int


class RunsResponse(Contract):
    items: list[RunResponse]
    offset: int
    limit: int
    total: int
    invalid_entries: int
