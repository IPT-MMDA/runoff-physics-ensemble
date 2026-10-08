from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "artifacts/hydrology/distill_to_simple"
PAIR_FIELDS = (
    "basin",
    "period",
    "training_noise",
    "seed",
    "inference_noise_seed",
)


@dataclass(frozen=True)
class Metric:
    field: str
    title: str
    unit: str
    higher_is_better: bool | None
    percent_drift: bool = False

    def drift(self, clean: float, noisy: float) -> float:
        if self.percent_drift:
            return 100.0 * (noisy - clean) / abs(clean) if clean != 0.0 else math.nan
        if self.higher_is_better is True:
            return clean - noisy
        return noisy - clean


METRICS = (
    Metric("test_nse", "Test NSE degradation", "NSE", True),
    Metric("validation_nse", "Validation NSE degradation", "NSE", True),
    Metric("test_kge", "Test KGE degradation", "KGE", True),
    Metric("test_rmse_mm_day", "Test RMSE increase", "%", False, True),
    Metric("physics_error", "Physics-error increase", "%", False, True),
)


def _label(record: dict[str, object]) -> str:
    model = str(record["model"]).replace("_", " ")
    simple_expert = str(record["simple_expert"]).replace("_", " ")
    complex_expert = str(record["complex_expert"]).replace("_", " ")
    method = str(record["training_method"]).replace("_", " ")
    return f"{model} · {simple_expert}→{complex_expert} · {method}"


def load_records(input_dir: Path) -> tuple[list[dict[str, object]], list[Path]]:
    paths = sorted(input_dir.glob("*/results.json"))
    if not paths:
        raise FileNotFoundError(f"no */results.json files found under {input_dir}")
    records: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_records = payload.get("records")
        if not isinstance(source_records, list) or not source_records:
            raise ValueError(f"{path} has no non-empty 'records' list")
        for record in source_records:
            if not isinstance(record, dict):
                raise ValueError(f"{path} contains a non-object record")
            item = dict(record)
            item["analysis_label"] = _label(item)
            item["source_json"] = str(path)
            records.append(item)
    return records, paths


def pair_noise_drift(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        label = record["analysis_label"]
        key = (label, *(record[field] for field in PAIR_FIELDS))
        grouped[key].append(record)

    paired: list[dict[str, object]] = []
    for key, group in grouped.items():
        by_noise = {float(record["inference_noise"]): record for record in group}
        baseline_noise = min(by_noise)
        clean = by_noise[baseline_noise]
        for noise, noisy in sorted(by_noise.items()):
            row: dict[str, object] = {
                "label": key[0],
                **{field: clean[field] for field in PAIR_FIELDS},
                "baseline_noise": baseline_noise,
                "inference_noise": noise,
            }
            for metric in METRICS:
                clean_value = float(clean[metric.field])
                noisy_value = float(noisy[metric.field])
                row[f"clean_{metric.field}"] = clean_value
                row[f"noisy_{metric.field}"] = noisy_value
                row[f"drift_{metric.field}"] = metric.drift(clean_value, noisy_value)
            paired.append(row)
    return paired


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_ci(
    values: list[float], *, draws: int = 5000, seed: int = 20260917
) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else math.nan
        return value, value
    generator = random.Random(seed)
    estimates = [
        mean(generator.choice(values) for _ in values)
        for _ in range(draws)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def aggregate_drift(paired: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in paired:
        groups[(str(row["label"]), float(row["inference_noise"]))].append(row)

    result: list[dict[str, object]] = []
    for (label, noise), rows in sorted(groups.items()):
        summary: dict[str, object] = {"label": label, "inference_noise": noise, "n": len(rows)}
        for metric_index, metric in enumerate(METRICS):
            values = [float(row[f"drift_{metric.field}"]) for row in rows]
            values = [value for value in values if math.isfinite(value)]
            low, high = _bootstrap_mean_ci(
                values, seed=20260917 + 1009 * metric_index + round(noise * 10000)
            )
            summary[f"clean_mean_{metric.field}"] = mean(
                float(row[f"clean_{metric.field}"]) for row in rows
            )
            summary[f"noisy_mean_{metric.field}"] = mean(
                float(row[f"noisy_{metric.field}"]) for row in rows
            )
            summary[f"drift_mean_{metric.field}"] = mean(values)
            summary[f"drift_median_{metric.field}"] = median(values)
            summary[f"drift_std_{metric.field}"] = stdev(values) if len(values) > 1 else 0.0
            summary[f"drift_ci_low_{metric.field}"] = low
            summary[f"drift_ci_high_{metric.field}"] = high
            summary[f"fraction_positive_{metric.field}"] = mean(value > 0.0 for value in values)
        result.append(summary)
    return result


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(aggregates: list[dict[str, object]], path: Path) -> None:
    labels = sorted({str(row["label"]) for row in aggregates})
    colors = dict(zip(labels, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for label in labels:
            rows = sorted(
                (row for row in aggregates if row["label"] == label),
                key=lambda row: float(row["inference_noise"]),
            )
            x = [float(row["inference_noise"]) for row in rows]
            y = [float(row[f"drift_mean_{metric.field}"]) for row in rows]
            low = [float(row[f"drift_ci_low_{metric.field}"]) for row in rows]
            high = [float(row[f"drift_ci_high_{metric.field}"]) for row in rows]
            color = colors[label]
            axis.plot(x, y, marker="o", linewidth=2, label=label, color=color)
            axis.fill_between(x, low, high, color=color, alpha=0.16)
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_title(metric.title)
        axis.set_xlabel("Inference noise (standardized input units)")
        axis.set_ylabel(metric.unit)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(METRICS) :]:
        axis.set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=max(1, len(labels)))
    figure.suptitle(
        "Inference-noise drift across basins\n"
        "Lines are paired means; bands are basin-bootstrap 95% intervals",
        fontsize=15,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_basin_heatmap(paired: list[dict[str, object]], path: Path) -> None:
    labels = sorted({str(row["label"]) for row in paired})
    max_noise = max(float(row["inference_noise"]) for row in paired)
    basins = sorted({str(row["basin"]) for row in paired})
    metrics = (METRICS[0], METRICS[3])
    figure, axes = plt.subplots(
        len(labels), len(metrics), figsize=(14, 2.2 + 0.5 * len(basins)), squeeze=False,
        constrained_layout=True,
    )
    for label_index, label in enumerate(labels):
        rows = {
            str(row["basin"]): row
            for row in paired
            if row["label"] == label and float(row["inference_noise"]) == max_noise
        }
        for metric_index, metric in enumerate(metrics):
            axis = axes[label_index, metric_index]
            values = [float(rows[basin][f"drift_{metric.field}"]) for basin in basins]
            limit = max(max(abs(value) for value in values), 1e-12)
            image = axis.imshow(
                [[value] for value in values],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            axis.set_xticks([0], [metric.title.replace("Test ", "")])
            if metric_index == 0:
                axis.set_yticks(range(len(basins)), basins, fontsize=8)
                axis.set_ylabel(label)
            else:
                axis.set_yticks([])
            for basin_index, value in enumerate(values):
                axis.text(0, basin_index, f"{value:+.3g}", ha="center", va="center", fontsize=7)
            figure.colorbar(image, ax=axis, shrink=0.65)
    figure.suptitle(f"Basin-level drift at inference noise {max_noise:g}", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    paired: list[dict[str, object]],
    aggregates: list[dict[str, object]],
    sources: list[Path],
    path: Path,
) -> None:
    max_noise = max(float(row["inference_noise"]) for row in paired)
    final = [row for row in aggregates if float(row["inference_noise"]) == max_noise]
    lines = [
        "# Inference-noise drift analysis",
        "",
        f"This analysis pairs every noisy evaluation with the clean evaluation of the same basin, "
        f"period, training condition, seed, and inference-noise draw. The highest analyzed noise is "
        f"`{max_noise:g}`.",
        "",
        "Positive NSE/KGE drift means degradation; positive RMSE and physics-error percentages mean "
        "increased error.",
        "",
        "## Results at maximum noise",
        "",
        "| Model and training | Basins | Clean NSE | NSE degradation | Basins degraded | RMSE increase | Physics-error increase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(final, key=lambda item: str(item["label"])):
        lines.append(
            f"| {row['label']} | {row['n']} | "
            f"{float(row['clean_mean_test_nse']):.3f} | "
            f"{float(row['drift_mean_test_nse']):+.3f} "
            f"[{float(row['drift_ci_low_test_nse']):+.3f}, {float(row['drift_ci_high_test_nse']):+.3f}] | "
            f"{100.0 * float(row['fraction_positive_test_nse']):.0f}% | "
            f"{float(row['drift_mean_test_rmse_mm_day']):+.1f}% | "
            f"{float(row['drift_mean_physics_error']):+.1f}% |"
        )

    nse_ranked = sorted(final, key=lambda row: float(row["drift_mean_test_nse"]))
    most_robust = nse_ranked[0]
    lines.extend(
        [
            "",
            "## Key observations",
            "",
            f"- **{most_robust['label']} has the smallest mean NSE drift** at noise "
            f"`{max_noise:g}` ({float(most_robust['drift_mean_test_nse']):+.3f}).",
        ]
    )
    if len(nse_ranked) > 1:
        less_robust = nse_ranked[-1]
        robust_drift = float(most_robust["drift_mean_test_nse"])
        less_robust_drift = float(less_robust["drift_mean_test_nse"])
        if robust_drift > 0.0:
            comparison = f"about {less_robust_drift / robust_drift:.1f}× larger"
        else:
            comparison = "larger"
        lines.append(
            f"- **{less_robust['label']} degrades more consistently:** its mean NSE loss is "
            f"{less_robust_drift:+.3f}, {comparison}, and "
            f"{100.0 * float(less_robust['fraction_positive_test_nse']):.0f}% of basins degrade."
        )
    for row in sorted(final, key=lambda item: str(item["label"])):
        kge_drift = float(row["drift_mean_test_kge"])
        kge_word = "improves" if kge_drift < 0.0 else "degrades"
        lines.append(
            f"- **{row['label']}:** mean KGE {kge_word} by {abs(kge_drift):.3f}."
        )

    lines.extend(["", "## Most noise-sensitive basins", ""])
    for label in sorted({str(row["label"]) for row in paired}):
        rows = [
            row for row in paired
            if row["label"] == label and float(row["inference_noise"]) == max_noise
        ]
        rows.sort(key=lambda row: float(row["drift_test_nse"]), reverse=True)
        detail = ", ".join(
            f"{row['basin']} ({float(row['drift_test_nse']):+.3f} NSE)" for row in rows[:3]
        )
        lines.append(f"- **{label}:** {detail}")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The result sets use different model/training combinations, so their contrast is descriptive rather than a controlled causal estimate of distillation alone.",
            "- There is one training seed and one inference-noise draw per basin. Bootstrap intervals quantify variation across basins, not seed or noise-realization uncertainty.",
            "- Input noise is expressed in standardized feature units.",
            "",
            "## Inputs",
            "",
            *(f"- `{source}`" for source in sources),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired inference-noise drift in distill-to-simple results"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "noise_drift_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, sources = load_records(args.input_dir)
    paired = pair_noise_drift(records)
    aggregates = aggregate_drift(paired)
    _write_csv(paired, output_dir / "paired_basin_drift.csv")
    _write_csv(aggregates, output_dir / "aggregate_drift.csv")
    plot_overview(aggregates, output_dir / "inference_noise_drift_overview.png")
    plot_basin_heatmap(paired, output_dir / "inference_noise_drift_by_basin.png")
    write_report(paired, aggregates, sources, output_dir / "analysis.md")
    print(f"Analyzed {len(records)} records from {len(sources)} result files.")
    print(f"Wrote paired analytics and visualizations to {output_dir}")


if __name__ == "__main__":
    main()
