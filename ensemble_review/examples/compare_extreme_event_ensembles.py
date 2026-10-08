"""Compare selectable ensemble approaches on one-day high-flow extremes.

This file is only the command-line adapter.  The reusable comparison API lives
in ``complexity_ensemble.hydrology_extreme_comparison`` so a future app can
construct a configuration and call it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from complexity_ensemble.hydrology_data import (  # noqa: E402
    load_camels_ch,
    load_ukraine_csv,
    make_hydrology_data,
)
from complexity_ensemble.hydrology_extreme_comparison import (  # noqa: E402
    ENSEMBLE_APPROACHES,
    ExtremeComparisonConfig,
    compare_extreme_event_ensembles,
    format_extreme_ensemble_summary,
    save_extreme_ensemble_records,
)


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not values:
        raise argparse.ArgumentTypeError("comma-separated selection must not be empty")
    return values


def _csv_integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_CAMELS_CH_ROOT))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--approaches",
        type=_csv_strings,
        default=ENSEMBLE_APPROACHES,
        help=(
            "Comma-separated: soft_routing,hard_routing,distillation,no_routing,"
            "static_50_50,ood_fallback,stacking"
        ),
    )
    parser.add_argument("--routing", choices=("morse", "learned"), default="morse")
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="pinnmamba")
    parser.add_argument("--seeds", type=_csv_integers, default=(0, 1, 2))
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Epochs for routing and independently trained ensemble approaches",
    )
    parser.add_argument("--complex-epochs", type=int, default=150)
    parser.add_argument("--distillation-epochs", type=int, default=150)
    parser.add_argument("--consolidation-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-noise", type=float, default=0.0)
    parser.add_argument("--inference-noise", type=float, default=0.0)
    parser.add_argument("--consolidation-weight", type=float, default=0.0)
    parser.add_argument("--complexity-percentile", type=float, default=80.0)
    parser.add_argument("--gate-temperature", type=float, default=0.15)
    parser.add_argument("--ood-quantile", type=float, default=0.99)
    parser.add_argument("--ood-shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--hard-inference",
        action="store_true",
        help="Use binary routing at inference for the hard-routing approach",
    )
    parser.add_argument("--without-baselines", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hydrology/extreme_ensemble_comparison/results.csv"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ExtremeComparisonConfig(
        approaches=args.approaches,
        routing=args.routing,
        simple_kind=args.simple,
        complex_kind=args.complex,
        seeds=args.seeds,
        threshold_quantile=args.quantile,
        epochs=args.epochs,
        complex_epochs=args.complex_epochs,
        distillation_epochs=args.distillation_epochs,
        consolidation_epochs=args.consolidation_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_noise=args.training_noise,
        inference_noise=args.inference_noise,
        consolidation_weight=args.consolidation_weight,
        complexity_percentile=args.complexity_percentile,
        gate_temperature=args.gate_temperature,
        hard_inference=args.hard_inference,
        include_baselines=not args.without_baselines,
        ood_quantile=args.ood_quantile,
        ood_shrinkage=args.ood_shrinkage,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))

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
    data = make_hydrology_data(series, sequence_length=args.sequence_length)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    records = compare_extreme_event_ensembles(
        data,
        config,
        device=device,
        verbose=args.verbose,
    )
    save_extreme_ensemble_records(records, args.output)
    run_config = {
        "data": {
            "source": args.source,
            "data_root": str(args.data_root),
            "csv": str(args.csv) if args.csv else None,
            "basin": args.basin,
            "start": args.start,
            "end": args.end,
            "include_landcover": not args.without_landcover,
            "sequence_length": args.sequence_length,
        },
        "comparison": asdict(config),
        "device": str(device),
    }
    config_path = args.output.with_suffix(".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as output:
        json.dump(run_config, output, indent=2, sort_keys=True)
        output.write("\n")

    print(
        f"Training-only Q{100 * config.threshold_quantile:g} threshold: "
        f"{records[0].threshold_mm_day:.3f} mm/day"
    )
    print(format_extreme_ensemble_summary(records))
    print(f"saved {args.output} and {config_path}")


if __name__ == "__main__":
    main()
