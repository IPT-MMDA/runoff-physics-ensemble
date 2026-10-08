"""Complexity-routed hydrology ensemble building blocks."""

from .experts import FourierExpert, MLPExpert, RBFExpert, build_expert
from .camels_ch_chem import (
    CAMELS_CH_CHEM_DYNAMIC_FEATURES, DEFAULT_CAMELS_CH_CHEM_FEATURES,
    CAMELSCHChemPressures,
)
from .hydrology import (
    HydrologyPINNMambaExpert,
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
)
from .hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from .estreams import ESTREAMS_DYNAMIC_FEATURES, EStreamsVegetationSnow
from .hydrology_extreme_comparison import (
    ENSEMBLE_APPROACHES,
    ExtremeComparisonConfig,
    ExtremeEnsembleRecord,
    compare_extreme_event_ensembles,
)
from .hydrology_complexity import (
    HydrologyComplexityEstimator,
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
    build_hydrology_complexity_estimator,
    takens_delay_embedding,
)
from .routing import LearnedRouter, MorseRouter, ScoreRouter
from .pinnmamba import PINNMamba, PINNMambaExpert, evaluate_reaction_pinnmamba
from .regional import (
    BasinScopeConfig,
    CAMELSCHCatalog,
    CancellationToken,
    DateRange,
    DateSplitConfig,
    ExperimentResult,
    ExtremeEventConfig,
    ForecastConfig,
    HyperparameterConfig,
    ModelArchitectureConfig,
    PhysicsModelConfig,
    RegionalExperimentConfig,
    TrainingStrategyConfig,
    make_regional_hydrology_data,
    run_regional_experiment,
    write_result_artifacts,
)
from .result_registry import (
    DEFAULT_RESULTS_DIRECTORY,
    SavedRun,
    discover_runs,
    filter_runs,
    save_run,
)

__all__ = [
    "FourierExpert",
    "CAMELS_CH_CHEM_DYNAMIC_FEATURES",
    "DEFAULT_CAMELS_CH_CHEM_FEATURES",
    "CAMELSCHChemPressures",
    "ENSEMBLE_APPROACHES",
    "ExtremeComparisonConfig",
    "ExtremeEnsembleRecord",
    "ESTREAMS_DYNAMIC_FEATURES",
    "EStreamsVegetationSnow",
    "HydrologyPINNMambaExpert",
    "HydrologyComplexityEstimator",
    "LearnedHydrologyMorsePotential",
    "MLPExpert",
    "LearnedRouter",
    "MorseRouter",
    "LyapunovComplexityEstimator",
    "PINNMamba",
    "PINNMambaExpert",
    "RBFExpert",
    "RoutedHydrologyModel",
    "SingleComplexHydrologyModel",
    "ScoreRouter",
    "TakensPersistenceEstimator",
    "build_expert",
    "build_hydrology_complexity_estimator",
    "compare_extreme_event_ensembles",
    "evaluate_reaction_pinnmamba",
    "load_camels_ch",
    "load_ukraine_csv",
    "make_hydrology_data",
    "make_regional_hydrology_data",
    "BasinScopeConfig",
    "CAMELSCHCatalog",
    "CancellationToken",
    "DateRange",
    "DateSplitConfig",
    "ExperimentResult",
    "ExtremeEventConfig",
    "ForecastConfig",
    "HyperparameterConfig",
    "ModelArchitectureConfig",
    "PhysicsModelConfig",
    "RegionalExperimentConfig",
    "TrainingStrategyConfig",
    "run_regional_experiment",
    "write_result_artifacts",
    "DEFAULT_RESULTS_DIRECTORY",
    "SavedRun",
    "discover_runs",
    "filter_runs",
    "save_run",
    "takens_delay_embedding",
]
