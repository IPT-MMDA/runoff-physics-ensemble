import argparse
import json
from pathlib import Path

import torch

from complexity_ensemble.hydrology_consolidation_comparison import (
    compare_hydrology_models_with_consolidation,
    hydrology_comparison_summary,
    plot_hydrology_comparison,
    save_hydrology_comparison,
)
from complexity_ensemble.hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from complexity_ensemble.hydrology_complexity import (
    MODEL_NAMES,
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
)
from complexity_ensemble.hydrology_morse_learning import (
    assess_learned_morse,
    calibrate_learned_morse,
    save_calibration_report,
    save_potential_assessment,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")
MODEL_CHOICES = MODEL_NAMES
DEFAULT_MODELS = ("morse", "learned_morse", "learned", "single_complex")


def parse_models(value: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """Parse the CLI model selection while preserving the requested order."""
    models = tuple(dict.fromkeys(model.strip() for model in value.split(",") if model.strip()))
    if not models:
        parser.error("--models must contain at least one model")
    invalid_models = set(models) - set(MODEL_CHOICES)
    if invalid_models:
        parser.error(
            "unknown --models value(s): "
            f"{', '.join(sorted(invalid_models))}; choose from {', '.join(MODEL_CHOICES)}"
        )
    return models


def main(
    comparison_function=compare_hydrology_models_with_consolidation,
    *,
    description: str = "Compare selectable hydrology routing models with isolated boundary consolidation",
    default_output_dir: Path = Path("artifacts/hydrology/learned_morse_consolidation"),
    stem_suffix: str = "",
    result_label: str = "isolated consolidation",
) -> None:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--tda-context-length", type=int, default=128,
        help="Past observations used by TDA routing (default: 128)",
    )
    parser.add_argument(
        "--tda-embedding-dim", type=int, default=3,
        help="Takens embedding dimension for TDA routing (default: 3)",
    )
    parser.add_argument("--tda-delay", type=int, default=7)
    parser.add_argument("--tda-min-persistence", type=float, default=0.1)
    parser.add_argument(
        "--tda-score",
        choices=("total_persistence", "persistent_holes", "persistence_entropy"),
        default="total_persistence",
    )
    parser.add_argument(
        "--lyapunov-context-length", type=int, default=365,
        help="Past observations used by Lyapunov routing (default: 365)",
    )
    parser.add_argument("--lyapunov-embedding-dim", type=int, default=3)
    parser.add_argument("--lyapunov-delay", type=int, default=7)
    parser.add_argument("--lyapunov-theiler-window", type=int, default=30)
    parser.add_argument("--lyapunov-fit-horizon", type=int, default=20)
    parser.add_argument("--lyapunov-min-r2", type=float, default=0.8)
    parser.add_argument(
        "--complexity-percentile", type=float, default=80.0,
        help="Training-score percentile used as the complex-route threshold (default: 80)",
    )
    parser.add_argument(
        "--gate-temperature", type=float, default=0.15,
        help="Soft routing transition width relative to score IQR (default: 0.15)",
    )
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            f"Comma-separated models to train: {', '.join(MODEL_CHOICES)}. "
            "Learned-Morse calibration runs only when learned_morse is selected."
        ),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pilot-epochs", type=int, default=5)
    parser.add_argument("--potential-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20, help="Final ensemble epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--training-noise", type=float, default=0.1,
        help="Noise standard deviation for dynamic training inputs (default: 0.1)",
    )
    parser.add_argument(
        "--inference-noise", type=float, default=0.1,
        help="Noise standard deviation for dynamic inference inputs (default: 0.1)",
    )
    parser.add_argument(
        "--consolidation-weight",
        type=float,
        default=0.05,
        help="Boundary agreement weight; set to 0 to disable (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    args = parser.parse_args()
    models = parse_models(args.models, parser)
    if args.consolidation_weight < 0.0:
        parser.error("--consolidation-weight must be non-negative")
    if args.training_noise < 0.0 or args.inference_noise < 0.0:
        parser.error("noise levels must be non-negative")
    if not 0.0 < args.complexity_percentile < 100.0:
        parser.error("--complexity-percentile must be strictly between 0 and 100")
    if args.gate_temperature <= 0.0:
        parser.error("--gate-temperature must be positive")
    if "tda" in models and args.tda_context_length < args.sequence_length:
        parser.error("--tda-context-length must be at least --sequence-length")
    if "lyapunov" in models and args.lyapunov_context_length < args.sequence_length:
        parser.error("--lyapunov-context-length must be at least --sequence-length")

    if args.source == "camels_ch":
        series = load_camels_ch(
            args.data_root,
            args.basin,
            start=args.start,
            end=args.end,
            include_landcover=not args.without_landcover,
        )
    else:
        if args.csv is None:
            parser.error("--csv is required with --source ukraine_csv")
        series = load_ukraine_csv(args.csv, basin_id=args.basin)
    routing_context_lengths = [args.sequence_length]
    if "tda" in models:
        routing_context_lengths.append(args.tda_context_length)
    if "lyapunov" in models:
        routing_context_lengths.append(args.lyapunov_context_length)
    data = make_hydrology_data(
        series,
        sequence_length=args.sequence_length,
        routing_context_length=max(routing_context_lengths),
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    potential = None
    calibration = None
    assessment = None
    if "learned_morse" in models:
        potential, calibration = calibrate_learned_morse(
            data,
            simple_kind=args.simple,
            complex_kind=args.complex,
            folds=args.folds,
            pilot_epochs=args.pilot_epochs,
            potential_epochs=args.potential_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        assessment = assess_learned_morse(
            data,
            potential,
            simple_kind=args.simple,
            complex_kind=args.complex,
            pilot_epochs=args.pilot_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
    flow_index = data.feature_names.index("previous_discharge")
    complexity_estimators = {}
    if "tda" in models:
        complexity_estimators["tda"] = TakensPersistenceEstimator(
            flow_index,
            embedding_dim=args.tda_embedding_dim,
            delay=args.tda_delay,
            min_persistence=args.tda_min_persistence,
            score=args.tda_score,
            context_length=args.tda_context_length,
        )
    if "lyapunov" in models:
        complexity_estimators["lyapunov"] = LyapunovComplexityEstimator(
            flow_index,
            embedding_dim=args.lyapunov_embedding_dim,
            delay=args.lyapunov_delay,
            theiler_window=args.lyapunov_theiler_window,
            fit_horizon=args.lyapunov_fit_horizon,
            min_r2=args.lyapunov_min_r2,
            context_length=args.lyapunov_context_length,
        )
    records = comparison_function(
        data,
        models=models,
        simple_kind=args.simple,
        complex_kind=args.complex,
        seeds=(args.seed,),
        training_noise_levels=(0.0, args.training_noise),
        inference_noise_levels=(0.0, args.inference_noise),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        learned_morse_potential=potential,
        complexity_estimators=complexity_estimators,
        complexity_percentile=args.complexity_percentile,
        gate_temperature=args.gate_temperature,
        consolidation_weight=args.consolidation_weight,
    )
    stem = (
        f"{args.source}_{args.basin}_{args.simple}_{args.complex}_{'-'.join(models)}"
        f"_consolidation_{args.consolidation_weight:g}{stem_suffix}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["models"] = list(models)
    with (args.output_dir / f"{stem}_config.json").open("w", encoding="utf-8") as output:
        json.dump(configuration, output, indent=2, sort_keys=True)
        output.write("\n")
    if calibration is not None and assessment is not None and potential is not None:
        save_calibration_report(calibration, args.output_dir / f"{stem}_calibration.csv")
        save_potential_assessment(assessment, args.output_dir / f"{stem}_transfer.csv")
        torch.save(potential.state_dict(), args.output_dir / f"{stem}_potential.pt")
    save_hydrology_comparison(records, args.output_dir / f"{stem}_comparison.csv")
    plot_hydrology_comparison(records, args.output_dir / f"{stem}.png")
    if calibration is not None and assessment is not None:
        print(
            f"calibration samples={calibration.samples}, "
            f"positive utility={calibration.positive_utility_fraction:.3f}, "
            f"score/utility Spearman={calibration.score_utility_spearman:+.3f}"
        )
        for item in assessment:
            print(
                f"{item.split} transfer: score/utility Spearman="
                f"{item.score_utility_spearman:+.3f}, top-20% precision="
                f"{item.top_utility_precision:.3f}, complex-useful high/low="
                f"{item.positive_utility_high_score:.3f}/{item.positive_utility_low_score:.3f}"
            )
    print(hydrology_comparison_summary(records))
    print(f"consolidation weight={args.consolidation_weight:g}")
    print(f"saved {result_label} results under {args.output_dir}")


if __name__ == "__main__":
    main()
