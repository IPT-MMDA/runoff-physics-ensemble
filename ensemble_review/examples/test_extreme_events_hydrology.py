"""Minimal one-day-ahead high-flow extreme experiment.

This script intentionally leaves the rainfall-runoff training objective unchanged.
It derives a basin-specific discharge threshold from the training split only, then
tests whether ordinary discharge forecasts can identify threshold exceedances.

Run from the repository root; for example::

    python ensemble_review/examples/test_extreme_events_hydrology.py \
        --data-root /path/to/camels_ch --basin 2011 --epochs 20

The output contains daily classification metrics, event-level detection metrics,
and runoff errors for MLP, PINN-Mamba, persistence, and climatology forecasts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import torch

# Keep the file runnable from a source checkout without requiring installation.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from complexity_ensemble.hydrology import SingleComplexHydrologyModel
from complexity_ensemble.hydrology_comparison import train_hydrology_model
from complexity_ensemble.hydrology_data import (
    HydrologyData,
    HydrologySplit,
    load_camels_ch,
    load_ukraine_csv,
    make_hydrology_data,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")

@dataclass(frozen=True)
class ExtremeEventRecord:
    country: str
    basin: str
    model: str
    seed: int
    threshold_quantile: float
    threshold_mm_day: float
    test_days: int
    observed_extreme_days: int
    observed_events: int
    predicted_events: int
    precision: float
    recall: float
    critical_success_index: float
    false_alarm_ratio: float
    average_precision: float
    event_precision: float
    event_recall: float
    rmse_mm_day: float
    extreme_rmse_mm_day: float
    peak_magnitude_mae_mm_day: float
    peak_timing_mae_days: float


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision with tied scores evaluated as a single threshold."""
    scores = scores.detach().flatten().double().cpu()
    labels = labels.detach().flatten().bool().cpu()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # End indices of equal-score groups make AP invariant to ordering within ties.
    group_end = torch.ones(len(scores), dtype=torch.bool)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    true_positive = sorted_labels.cumsum(0)[group_end].double()
    retrieved = torch.arange(1, len(scores) + 1, dtype=torch.double)[group_end]
    precision = true_positive / retrieved
    previous = torch.cat((torch.zeros(1, dtype=torch.double), true_positive[:-1]))
    recall_increment = (true_positive - previous) / positives
    return float((precision * recall_increment).sum())


def _events(labels: torch.Tensor, dates: tuple[date, ...]) -> list[tuple[int, int]]:
    """Return inclusive index ranges for consecutive positive calendar days."""
    flags = labels.detach().flatten().bool().cpu().tolist()
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, positive in enumerate(flags):
        continues = (
            positive
            and start is not None
            and index > 0
            and dates[index] - dates[index - 1] == timedelta(days=1)
        )
        if positive and start is None:
            start = index
        elif positive and not continues:
            result.append((start, index - 1))
            start = index
        elif not positive and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(flags) - 1))
    return result


def _overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def evaluate_extremes(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dates: tuple[date, ...],
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate discharge scores at daily and declustered-event levels."""
    prediction = prediction.detach().flatten().double().cpu()
    target = target.detach().flatten().double().cpu()
    observed = target > threshold
    predicted = prediction > threshold
    true_positive = int((observed & predicted).sum())
    false_positive = int((~observed & predicted).sum())
    false_negative = int((observed & ~predicted).sum())

    observed_events = _events(observed, dates)
    predicted_events = _events(predicted, dates)
    detected_observed = [
        event for event in observed_events
        if any(_overlaps(event, candidate) for candidate in predicted_events)
    ]
    detected_predicted = [
        event for event in predicted_events
        if any(_overlaps(event, candidate) for candidate in observed_events)
    ]

    peak_errors: list[float] = []
    timing_errors: list[float] = []
    for start, stop in detected_observed:
        actual_slice = target[start : stop + 1]
        predicted_slice = prediction[start : stop + 1]
        actual_peak_offset = int(actual_slice.argmax())
        predicted_peak_offset = int(predicted_slice.argmax())
        peak_errors.append(abs(float(predicted_slice.max() - actual_slice.max())))
        timing_errors.append(float(abs(predicted_peak_offset - actual_peak_offset)))

    error = prediction - target
    extreme_error = error[observed]
    return {
        "test_days": len(target),
        "observed_extreme_days": int(observed.sum()),
        "observed_events": len(observed_events),
        "predicted_events": len(predicted_events),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
        "critical_success_index": _safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "false_alarm_ratio": _safe_ratio(false_positive, true_positive + false_positive),
        "average_precision": _average_precision(prediction, observed),
        "event_precision": _safe_ratio(len(detected_predicted), len(predicted_events)),
        "event_recall": _safe_ratio(len(detected_observed), len(observed_events)),
        "rmse_mm_day": float(error.square().mean().sqrt()),
        "extreme_rmse_mm_day": (
            float(extreme_error.square().mean().sqrt()) if len(extreme_error) else float("nan")
        ),
        "peak_magnitude_mae_mm_day": (
            sum(peak_errors) / len(peak_errors) if peak_errors else float("nan")
        ),
        "peak_timing_mae_days": (
            sum(timing_errors) / len(timing_errors) if timing_errors else float("nan")
        ),
    }


@torch.no_grad()
def _model_prediction(
    model: SingleComplexHydrologyModel,
    split: HydrologySplit,
    discharge_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    return (model(split.inputs.to(device)).squeeze(-1).cpu() * discharge_scale.cpu())


def _persistence_prediction(data: HydrologyData, split: HydrologySplit) -> torch.Tensor:
    flow_index = data.feature_names.index("previous_discharge")
    return split.physical_inputs[:, -1, flow_index].cpu()


def _seasonal_climatology_prediction(data: HydrologyData) -> torch.Tensor:
    """Predict the training mean discharge for each calendar day."""
    train_target = data.train.targets.cpu() * data.discharge_scale.cpu()
    grouped: dict[tuple[int, int], list[float]] = {}
    for day, value in zip(data.train.target_dates, train_target):
        grouped.setdefault((day.month, day.day), []).append(float(value))
    climatology = {key: sum(values) / len(values) for key, values in grouped.items()}
    fallback = float(train_target.mean())
    return torch.tensor(
        [climatology.get((day.month, day.day), fallback) for day in data.test.target_dates]
    )


def _make_record(
    data: HydrologyData,
    model: str,
    seed: int,
    quantile: float,
    threshold: float,
    prediction: torch.Tensor,
) -> ExtremeEventRecord:
    target = data.test.targets.cpu() * data.discharge_scale.cpu()
    metrics = evaluate_extremes(prediction, target, data.test.target_dates, threshold)
    return ExtremeEventRecord(
        country=data.country,
        basin=data.basin_id,
        model=model,
        seed=seed,
        threshold_quantile=quantile,
        threshold_mm_day=threshold,
        **metrics,
    )


def run_experiment(
    data: HydrologyData,
    *,
    architectures: tuple[str, ...] = ("mlp", "pinnmamba"),
    seeds: tuple[int, ...] = (0, 1, 2),
    quantile: float = 0.95,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[ExtremeEventRecord]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if not architectures or set(architectures) - {"mlp", "pinnmamba"}:
        raise ValueError("architectures must contain only 'mlp' and/or 'pinnmamba'")
    if not seeds:
        raise ValueError("at least one seed is required")
    device = torch.device(device)
    train_target = data.train.targets.cpu() * data.discharge_scale.cpu()
    threshold = float(torch.quantile(train_target, quantile))

    records = [
        _make_record(
            data,
            model="persistence",
            seed=-1,
            quantile=quantile,
            threshold=threshold,
            prediction=_persistence_prediction(data, data.test),
        ),
        _make_record(
            data,
            model="seasonal_climatology",
            seed=-1,
            quantile=quantile,
            threshold=threshold,
            prediction=_seasonal_climatology_prediction(data),
        ),
    ]
    for architecture in architectures:
        for seed in seeds:
            torch.manual_seed(seed)
            model = SingleComplexHydrologyModel(
                data.feature_names,
                data.train.inputs.shape[1],
                data.discharge_scale,
                complex_kind=architecture,
            ).to(device)
            train_hydrology_model(
                model,
                data.train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                training_noise=0.0,
                seed=seed,
                device=device,
            )
            prediction = _model_prediction(model, data.test, data.discharge_scale, device)
            records.append(
                _make_record(
                    data,
                    model=architecture,
                    seed=seed,
                    quantile=quantile,
                    threshold=threshold,
                    prediction=prediction,
                )
            )
    return records


def save_records(records: list[ExtremeEventRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv_tuple(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_CAMELS_CH_ROOT))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--architectures", default="mlp,pinnmamba")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hydrology/extreme_events/results.csv"),
    )
    args = parser.parse_args()

    architectures = _parse_csv_tuple(args.architectures)
    seeds = _parse_seeds(args.seeds)
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
    records = run_experiment(
        data,
        architectures=architectures,
        seeds=seeds,
        quantile=args.quantile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    save_records(records, args.output)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    configuration.update({"architectures": architectures, "seeds": seeds, "device": str(device)})
    config_path = args.output.with_suffix(".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as output:
        json.dump(configuration, output, indent=2, sort_keys=True)
        output.write("\n")

    print(f"Training-only Q{100 * args.quantile:g} threshold: {records[0].threshold_mm_day:.3f} mm/day")
    print("model            seed  AP     recall  CSI    event recall  extreme RMSE")
    for record in records:
        values = (
            record.average_precision,
            record.recall,
            record.critical_success_index,
            record.event_recall,
            record.extreme_rmse_mm_day,
        )
        rendered = ["nan" if not math.isfinite(value) else f"{value:.3f}" for value in values]
        print(
            f"{record.model:<16} {record.seed:>4}  {rendered[0]:>5}  {rendered[1]:>6}  "
            f"{rendered[2]:>5}  {rendered[3]:>12}  {rendered[4]:>12}"
        )
    print(f"saved {args.output} and {config_path}")


if __name__ == "__main__":
    main()
