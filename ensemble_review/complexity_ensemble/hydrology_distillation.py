from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import torch

from .hydrology import HydrologyRoutedOutput, RoutedHydrologyModel
from .hydrology_comparison import (
    _add_observation_noise,
    _evaluate,
    _make_model,
    _parameter_count,
    plot_hydrology_comparison,
)
from .hydrology_consolidation_comparison import train_hydrology_model_with_consolidation
from .hydrology_data import (
    HydrologySeries,
    HydrologySplit,
    load_camels_ch,
    load_ukraine_csv,
    make_hydrology_data,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")
VERBOSE = False

@dataclass(frozen=True)
class EvaluationPeriod:
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class HydrologyDistillationRecord:
    country: str
    basin: str
    period: str
    period_start: str
    period_end: str
    model: str
    training_method: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise: float
    inference_noise: float
    inference_noise_seed: int
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    physics_error: float
    parameters: int
    mean_complex_weight: float
    mean_complexity_score: float
    valid_complexity_fraction: float
    training_seconds: float
    complex_epochs: int
    distillation_epochs: int
    consolidation_epochs: int


@dataclass(frozen=True)
class HydrologyDistillationReport:
    """Loss traces and routing coverage for staged expert training."""

    complex_losses: tuple[float, ...]
    distillation_losses: tuple[float, ...]
    consolidation_losses: tuple[float, ...]
    simple_sample_count: int
    total_sample_count: int
    complex_seconds: float
    distillation_seconds: float
    consolidation_seconds: float

    @property
    def simple_fraction(self) -> float:
        return self.simple_sample_count / self.total_sample_count


def _epoch_average(total: float, samples: int) -> float:
    if samples == 0:
        raise RuntimeError("the training phase did not receive any samples")
    return total / samples


def train_hydrology_distill_to_simple(
    model: RoutedHydrologyModel,
    split: HydrologySplit,
    *,
    complex_epochs: int,
    distillation_epochs: int,
    consolidation_epochs: int,
    batch_size: int,
    learning_rate: float,
    training_noise: float,
    seed: int,
    device: torch.device | str,
    feature_names: tuple[str, ...],
    routing_features: torch.Tensor | None = None,
    physics_weight: float = 0.05,
    consolidation_weight: float = 0.05,
    verbose: bool = False,
) -> HydrologyDistillationReport:
    """Train a routed hydrology ensemble using a complex teacher.

    The router is fitted first and its hard assignments define a fixed simple
    subset. The complex expert learns from all observed targets, the simple
    expert learns the complex expert's predictions only on that subset, and a
    short joint phase tunes the final routed ensemble and its interface.
    """
    if min(complex_epochs, distillation_epochs, consolidation_epochs) < 1:
        raise ValueError("all phase epoch counts must be positive")
    if batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("batch_size and learning_rate must be positive")
    if training_noise < 0.0:
        raise ValueError("training_noise must be non-negative")
    if physics_weight < 0.0 or consolidation_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if not len(split.inputs):
        raise ValueError("the training split must not be empty")
    if model.routing_kind == "learned":
        raise ValueError(
            "distill-to-simple requires a fixed complexity-space router; "
            "the jointly learned gate has no meaningful pre-training partition"
        )

    device = torch.device(device)
    model.to(device)
    model.train()
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    targets = split.targets.to(device)
    routing_features = None if routing_features is None else routing_features.to(device)

    # Complexity is established before either expert is trained. Keeping these
    # labels fixed prevents teacher errors or later co-adjustment from moving the
    # definition of the simple space during distillation.
    model.fit_router(inputs, routing_features)
    with torch.no_grad():
        router_inputs = model._router_features(inputs, routing_features)
        simple_mask = model.router.complex_weight(router_inputs, hard=True) < 0.5
    simple_sample_count = int(simple_mask.sum())
    if simple_sample_count == 0:
        raise ValueError("the fitted router assigned no training samples to simple space")

    complex_parameters = [
        *model.complex_expert.parameters(),
        model.raw_response,
        model.raw_recession,
    ]
    complex_optimizer = torch.optim.Adam(complex_parameters, lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)
    complex_losses = []
    complex_started = time.perf_counter()
    if verbose:
        print(f"[1/3] training complex teacher on {len(inputs)} samples")
    for epoch in range(complex_epochs):
        epoch_total = 0.0
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names, training_noise, generator
            )
            complex_optimizer.zero_grad()
            prediction = model._positive_discharge(model.complex_expert(batch_inputs))
            data_loss = (prediction.squeeze(-1) - targets[indices]).square().mean()
            physics_loss = model.physics_residual(
                prediction, physical[indices]
            ).square().mean()
            loss = data_loss + physics_weight * physics_loss
            loss.backward()
            complex_optimizer.step()
            epoch_total += float(loss.detach()) * len(indices)
        complex_losses.append(_epoch_average(epoch_total, len(inputs)))
        if verbose:
            print(
                f"  complex epoch {epoch + 1}/{complex_epochs}: "
                f"loss={complex_losses[-1]:.6f}"
            )
    complex_seconds = time.perf_counter() - complex_started

    model.complex_expert.eval()
    model.simple_expert.train()
    simple_optimizer = torch.optim.Adam(model.simple_expert.parameters(), lr=learning_rate)
    distillation_generator = torch.Generator(device=device).manual_seed(seed + 130363)
    simple_indices = torch.nonzero(simple_mask, as_tuple=False).squeeze(-1)
    distillation_losses = []
    distillation_started = time.perf_counter()
    if verbose:
        print(
            f"[2/3] distilling into simple expert on {len(simple_indices)}/{len(inputs)} "
            "simple-space samples"
        )
    for epoch in range(distillation_epochs):
        epoch_total = 0.0
        permutation = simple_indices[
            torch.randperm(len(simple_indices), generator=distillation_generator, device=device)
        ]
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names, training_noise, distillation_generator
            )
            with torch.no_grad():
                teacher = model._positive_discharge(model.complex_expert(batch_inputs))
            simple_optimizer.zero_grad()
            student = model._positive_discharge(model.simple_expert(batch_inputs))
            loss = (student - teacher).square().mean()
            loss.backward()
            simple_optimizer.step()
            epoch_total += float(loss.detach()) * len(indices)
        distillation_losses.append(_epoch_average(epoch_total, len(simple_indices)))
        if verbose:
            print(
                f"  distillation epoch {epoch + 1}/{distillation_epochs}: "
                f"loss={distillation_losses[-1]:.6f}"
            )
    distillation_seconds = time.perf_counter() - distillation_started

    model.train()
    joint_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    consolidation_generator = torch.Generator(device=device).manual_seed(seed + 155921)
    consolidation_losses = []
    consolidation_started = time.perf_counter()
    if verbose:
        print("[3/3] jointly consolidating routed experts")
    for epoch in range(consolidation_epochs):
        epoch_total = 0.0
        permutation = torch.randperm(
            len(inputs), generator=consolidation_generator, device=device
        )
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names, training_noise, consolidation_generator
            )
            batch_routing = None if routing_features is None else routing_features[indices]
            joint_optimizer.zero_grad()
            losses = model.losses(
                batch_inputs,
                physical[indices],
                targets[indices],
                routing_features=batch_routing,
                physics_weight=physics_weight,
                interface_weight=consolidation_weight,
            )
            losses.total.backward()
            joint_optimizer.step()
            epoch_total += float(losses.total.detach()) * len(indices)
        consolidation_losses.append(_epoch_average(epoch_total, len(inputs)))
        if verbose:
            print(
                f"  consolidation epoch {epoch + 1}/{consolidation_epochs}: "
                f"loss={consolidation_losses[-1]:.6f}"
            )
    consolidation_seconds = time.perf_counter() - consolidation_started

    return HydrologyDistillationReport(
        tuple(complex_losses),
        tuple(distillation_losses),
        tuple(consolidation_losses),
        simple_sample_count,
        len(inputs),
        complex_seconds,
        distillation_seconds,
        consolidation_seconds,
    )


@torch.no_grad()
def simple_space_distillation_error(
    model: RoutedHydrologyModel,
    inputs: torch.Tensor,
    *,
    routing_features: torch.Tensor | None = None,
) -> float:
    """Return student/teacher MSE on the router's hard simple subset."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    routing_features = None if routing_features is None else routing_features.to(device)
    details = model(inputs, routing_features=routing_features, return_details=True)
    assert isinstance(details, HydrologyRoutedOutput)
    simple = details.complex_weight < 0.5
    if not bool(simple.any()):
        return float("nan")
    student = model._positive_discharge(details.simple_raw[simple])
    teacher = model._positive_discharge(details.complex_raw[simple])
    return float((student - teacher).square().mean())


__all__ = [
    "HydrologyDistillationReport",
    "HydrologyDistillationRecord",
    "format_basin_results_table",
    "simple_space_distillation_error",
    "summarize_basin_results",
    "train_hydrology_distill_to_simple",
]



def _csv_values(value: str, cast: type) -> tuple:
    values = tuple(
        dict.fromkeys(cast(item.strip()) for item in value.split(",") if item.strip())
    )
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value must not be empty")
    return values


def _parse_period(value: str) -> EvaluationPeriod:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("period must be LABEL:START:END")
    label, start, end = parts
    try:
        if date.fromisoformat(end) <= date.fromisoformat(start):
            raise ValueError("end must be after start")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid period: {error}") from error
    return EvaluationPeriod(label, start, end)


def _parse_boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in ("true", "false"):
        return normalized == "true"
    raise argparse.ArgumentTypeError("expected true or false")


def _load_period(
    args: argparse.Namespace, basin: str, period: EvaluationPeriod
) -> HydrologySeries:
    if args.source == "camels_ch":
        return load_camels_ch(
            args.data_root,
            basin,
            start=period.start,
            end=period.end,
            include_landcover=not args.without_landcover,
        )
    if args.csv_template is not None:
        path = Path(str(args.csv_template).format(basin=basin))
    elif args.csv is not None and len(args.basins) == 1:
        path = args.csv
    else:
        raise ValueError("Ukraine multi-basin runs require --csv-template containing {basin}")
    series = load_ukraine_csv(path, basin_id=basin)
    start, end = date.fromisoformat(period.start), date.fromisoformat(period.end)
    keep = [index for index, day in enumerate(series.dates) if start <= day <= end]
    if not keep:
        raise ValueError(f"basin {basin} has no observations in period {period.label}")
    indices = torch.tensor(keep)
    return replace(
        series,
        dates=tuple(series.dates[index] for index in keep),
        discharge=series.discharge[indices],
        forcings=series.forcings[indices],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-basin distill-to-simple hydrology robustness comparison"
    )
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--csv-template", type=Path)
    parser.add_argument("--basin")
    parser.add_argument("--basins")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--period", action="append", type=_parse_period)
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument("--models", default="morse,learned,single_complex")
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--complex-epochs", "--epochs", dest="complex_epochs", type=int, default=20,
        help="Complex-teacher epochs; --epochs is an alias",
    )
    parser.add_argument("--distillation-epochs", type=int, default=150)
    parser.add_argument("--consolidation-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-noise", type=float)
    parser.add_argument("--training-noise-levels", default="0")
    parser.add_argument("--inference-noise-levels", default="0")
    parser.add_argument("--inference-noise-draws", type=int, default=1)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--consolidation-weight", type=float, default=0.2)
    parser.add_argument("--complexity-percentile", type=float, default=80.0)
    parser.add_argument("--gate-temperature", type=float, default=0.15)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--viz", type=_parse_boolean, default=True, metavar="true|false")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/hydrology/distill_to_simple")
    )
    return parser


def _configuration(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = [asdict(item) if isinstance(item, EvaluationPeriod) else item for item in value]
        else:
            result[key] = value
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return mean(finite), stdev(finite) if len(finite) > 1 else 0.0


def summarize_basin_results(
    records: list[HydrologyDistillationRecord],
) -> list[dict[str, object]]:
    """Aggregate final metrics across seeds and inference-noise draws per basin."""
    keys = sorted({
        (
            record.country,
            record.basin,
            record.period,
            record.period_start,
            record.period_end,
            record.model,
            record.training_method,
            record.training_noise,
            record.inference_noise,
        )
        for record in records
    })
    rows = []
    metric_fields = (
        "validation_nse",
        "test_nse",
        "test_kge",
        "test_rmse_mm_day",
        "physics_error",
    )
    for key in keys:
        country, basin, period, start, end, model, strategy, train_noise, infer_noise = key
        group = [
            record
            for record in records
            if (
                record.country,
                record.basin,
                record.period,
                record.period_start,
                record.period_end,
                record.model,
                record.training_method,
                record.training_noise,
                record.inference_noise,
            ) == key
        ]
        row: dict[str, object] = {
            "country": country,
            "basin": basin,
            "period": period,
            "period_start": start,
            "period_end": end,
            "model": model,
            "training_method": strategy,
            "training_noise": train_noise,
            "inference_noise": infer_noise,
            "evaluation_count": len(group),
            "seed_count": len({record.seed for record in group}),
        }
        for field in metric_fields:
            average, spread = _mean_and_std([getattr(record, field) for record in group])
            row[f"{field}_mean"] = average
            row[f"{field}_std"] = spread
        # Evaluation draws repeat the same fitted model, so timing statistics
        # include each seed's training run once rather than once per draw.
        timings_by_seed = {record.seed: record.training_seconds for record in group}
        timing_mean, timing_std = _mean_and_std(list(timings_by_seed.values()))
        row["training_seconds_mean"] = timing_mean
        row["training_seconds_std"] = timing_std
        rows.append(row)
    return rows


def format_basin_results_table(rows: list[dict[str, object]]) -> str:
    """Format per-basin aggregates as a compact terminal table."""
    if not rows:
        return "No completed basin results."

    def metric(row: dict[str, object], field: str) -> str:
        average = float(row[f"{field}_mean"])
        spread = float(row[f"{field}_std"])
        return f"{average:.4f}±{spread:.4f}"

    header = (
        f"{'basin':<10} {'period':<10} {'model':<15} {'train σ':>7} {'infer σ':>7} "
        f"{'n':>4} {'test NSE':>17} {'test KGE':>17} {'RMSE mm/d':>17} "
        f"{'physics':>17} {'train s':>12}"
    )
    lines = ["Final per-basin results (mean ± sample SD across seeds/draws)", header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row['basin']):<10} {str(row['period']):<10} "
            f"{str(row['model']):<15} {float(row['training_noise']):7.3g} "
            f"{float(row['inference_noise']):7.3g} {int(row['evaluation_count']):4d} "
            f"{metric(row, 'test_nse'):>17} {metric(row, 'test_kge'):>17} "
            f"{metric(row, 'test_rmse_mm_day'):>17} "
            f"{metric(row, 'physics_error'):>17} "
            f"{float(row['training_seconds_mean']):12.2f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def _routing_diagnostics(
    model: torch.nn.Module, inputs: torch.Tensor, device: torch.device
) -> tuple[float, float]:
    if not isinstance(model, RoutedHydrologyModel):
        return float("nan"), float("nan")
    router_inputs = model._router_features(inputs.to(device), None)
    scores = model.router.complexity(router_inputs)
    finite = torch.isfinite(scores)
    if not bool(finite.any()):
        return float("nan"), 0.0
    return float(scores[finite].mean()), float(finite.float().mean())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.basins = _csv_values(args.basins or args.basin or "2011", str)
    args.models = _csv_values(args.models, str)
    args.seeds = _csv_values(args.seeds, int)
    training_noise_value = (
        str(args.training_noise)
        if args.training_noise is not None
        else args.training_noise_levels
    )
    args.training_noise_levels = _csv_values(training_noise_value, float)
    args.inference_noise_levels = _csv_values(args.inference_noise_levels, float)
    args.period = tuple(args.period or [EvaluationPeriod("full", args.start, args.end)])
    supported_models = {"morse", "learned", "single_complex"}
    invalid_models = set(args.models) - supported_models
    if invalid_models:
        parser.error(f"unsupported --models values: {', '.join(sorted(invalid_models))}")
    if min(args.complex_epochs, args.distillation_epochs, args.consolidation_epochs) < 1:
        parser.error("all epoch counts must be positive")
    if args.inference_noise_draws < 1:
        parser.error("--inference-noise-draws must be positive")
    if any(value < 0.0 for value in (*args.training_noise_levels, *args.inference_noise_levels)):
        parser.error("noise levels must be non-negative")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[HydrologyDistillationRecord] = []
    failures = []
    for basin in args.basins:
        for period in args.period:
            run_name = f"{args.source}_{basin}_{period.label}"
            run_records = []
            try:
                print(f"loading {run_name}", flush=True)
                data = make_hydrology_data(
                    _load_period(args, basin, period), sequence_length=args.sequence_length
                )
                print(
                    f"prepared {len(data.train.inputs)}/{len(data.validation.inputs)}/"
                    f"{len(data.test.inputs)} train/validation/test windows on {device}",
                    flush=True,
                )
                for seed in args.seeds:
                    for training_noise in args.training_noise_levels:
                        for model_name in args.models:
                            print(
                                f"run model={model_name} seed={seed} "
                                f"training_noise={training_noise:g}", flush=True,
                            )
                            torch.manual_seed(seed)
                            model = _make_model(
                                model_name, data, args.simple, args.complex, None,
                                args.complexity_percentile, args.gate_temperature,
                            ).to(device)
                            started = time.perf_counter()
                            if model_name == "morse":
                                report = train_hydrology_distill_to_simple(
                                    model, data.train,
                                    complex_epochs=args.complex_epochs,
                                    distillation_epochs=args.distillation_epochs,
                                    consolidation_epochs=args.consolidation_epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    training_noise=training_noise, seed=seed,
                                    device=device, feature_names=data.feature_names,
                                    physics_weight=args.physics_weight,
                                    consolidation_weight=args.consolidation_weight,
                                    verbose=VERBOSE,
                                )
                                elapsed = sum((
                                    report.complex_seconds,
                                    report.distillation_seconds,
                                    report.consolidation_seconds,
                                ))
                                phase_epochs = (
                                    args.complex_epochs,
                                    args.distillation_epochs,
                                    args.consolidation_epochs,
                                )
                                strategy = "distill_to_simple"
                            else:
                                train_hydrology_model_with_consolidation(
                                    model, data.train, epochs=args.complex_epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    training_noise=training_noise, seed=seed,
                                    device=device,
                                    consolidation_weight=args.consolidation_weight,
                                    feature_names=data.feature_names,
                                    verbose=VERBOSE,
                                )
                                elapsed = time.perf_counter() - started
                                phase_epochs = (args.complex_epochs, 0, 0)
                                strategy = (
                                    "joint_baseline" if model_name == "learned"
                                    else "single_baseline"
                                )
                                print(f"  baseline training finished in {elapsed:.2f}s", flush=True)
                            for draw in range(args.inference_noise_draws):
                                noise_seed = seed + 300007 + 1000003 * draw
                                for inference_noise in args.inference_noise_levels:
                                    validation = _evaluate(
                                        model, data.validation, data,
                                        inference_noise=inference_noise,
                                        seed=noise_seed + 17, device=device,
                                    )
                                    test = _evaluate(
                                        model, data.test, data,
                                        inference_noise=inference_noise,
                                        seed=noise_seed, device=device,
                                    )
                                    mean_score, valid_fraction = _routing_diagnostics(
                                        model, data.test.inputs, device
                                    )
                                    record = HydrologyDistillationRecord(
                                        data.country, basin, period.label,
                                        period.start, period.end, model_name, strategy,
                                        args.simple if model_name != "single_complex" else "none",
                                        args.complex, seed, training_noise,
                                        inference_noise, noise_seed, validation[0],
                                        test[0], test[1], test[2], test[3],
                                        _parameter_count(model), test[4],
                                        mean_score, valid_fraction, elapsed,
                                        *phase_epochs,
                                    )
                                    records.append(record)
                                    run_records.append(record)
                if run_records and args.viz:
                    plot_hydrology_comparison(
                        run_records, args.output_dir / f"{run_name}_comparison.png"
                    )
            except Exception as error:
                if not args.continue_on_error:
                    raise
                failures.append({"basin": basin, "period": period.label, "error": repr(error)})
                print(f"FAILED {run_name}: {error}", flush=True)

    basin_results = summarize_basin_results(records)
    payload = {
        "schema_version": 1,
        "experiment": "hydrology_distill_to_simple",
        "configuration": _configuration(args),
        "basin_results": basin_results,
        "records": [asdict(record) for record in records],
        "failures": failures,
    }
    result_path = args.output_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as output:
        json.dump(_json_safe(payload), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    if not records:
        raise RuntimeError("no basin-period run completed")
    print()
    print(format_basin_results_table(basin_results), flush=True)
    print(f"saved {len(records)} records to {result_path}", flush=True)


if __name__ == "__main__":
    main()
