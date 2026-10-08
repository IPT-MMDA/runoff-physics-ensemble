"""Persistent, versioned experiment runs shared by CLI and Gradio clients."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .regional import ExperimentResult, write_result_artifacts


SCHEMA_VERSION = 1
DEFAULT_RESULTS_DIRECTORY = Path("artifacts/hydrology/run_registry")

# These are the dimensions that identify a setup in the comparison UI. Values
# are read from resolved configs, so the registry does not need to know which
# front end produced a run.
FILTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("strategy.approach", "Approach"),
    ("model.complex_expert", "Complex expert"),
    ("model.simple_expert", "Simple expert"),
    ("strategy.epochs", "Epochs"),
    ("physics.optimization_mode", "KAN / physics optimization"),
    ("physics.distillation_teacher", "Physics teacher"),
    ("basins.training_scope", "Training scope"),
    ("hyperparameters.sequence_length", "Sequence length"),
    ("hyperparameters.seeds", "Seeds"),
    ("hyperparameters.batch_size", "Batch size"),
    ("hyperparameters.learning_rate", "Learning rate"),
    ("hyperparameters.training_noise", "Training noise"),
    ("extremes.mode", "Extreme-event mode"),
    ("forecast.mode", "Forecast setup"),
    ("forecast.horizon_days", "Rolling horizon"),
)


def parse_bool(value: str | bool) -> bool:
    """Parse explicit CLI booleans such as ``--save-results true``."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected true or false")


def _json_safe(value):
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _nested(config: Mapping[str, object], path: str):
    value: object = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def display_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class SavedRun:
    run_id: str
    created_at: str
    path: Path
    config: dict[str, object]
    aggregate_metrics: dict[str, float]
    per_basin_metrics: dict[str, dict[str, float]]
    routing_diagnostics: dict[str, object]
    loss_traces: dict[str, list[float]]
    extreme_events: list[dict[str, object]]
    failures: list[str]
    parameter_count: int
    runtime_seconds: float
    cancelled: bool
    forecast_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def value(self, field: str) -> str:
        return display_value(_nested(self.config, field))

    @property
    def label(self) -> str:
        approach = self.value("strategy.approach")
        complex_expert = self.value("model.complex_expert")
        simple_expert = self.value("model.simple_expert")
        epochs = self.value("strategy.epochs")
        physics = self.value("physics.optimization_mode")
        stamp = self.created_at.replace("T", " ").replace("Z", " UTC")
        return (
            f"{approach} · {complex_expert}+{simple_expert} · {epochs} epochs · "
            f"KAN {physics} · {stamp} · {self.run_id[-8:]}"
        )


@dataclass(frozen=True)
class RegistrySnapshot:
    runs: tuple[SavedRun, ...]
    warnings: tuple[str, ...] = ()


def save_run(result: ExperimentResult, directory: str | Path = DEFAULT_RESULTS_DIRECTORY) -> SavedRun:
    """Atomically add one experiment result to the shared run registry."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0)
    digest_input = json.dumps(_json_safe(result.resolved_config), sort_keys=True).encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:10]
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{digest}-{os.urandom(3).hex()}"
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=root))
    destination = root / run_id
    try:
        paths = write_result_artifacts(result, staging)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "cancelled": result.cancelled,
            "resolved_config": result.resolved_config,
            "aggregate_metrics": result.aggregate_metrics,
            "per_basin_metrics": result.per_basin_metrics,
            "routing_diagnostics": result.routing_diagnostics,
            "loss_traces": result.loss_traces,
            "extreme_events": result.extreme_events,
            "forecast_metrics": result.forecast_metrics,
            "failures": result.failures,
            "parameter_count": result.parameter_count,
            "runtime_seconds": result.runtime_seconds,
            "files": {key: Path(value).name for key, value in paths.items()},
        }
        (staging / "run.json").write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_run(destination / "run.json")


def load_run(path: str | Path) -> SavedRun:
    manifest = Path(path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run schema {payload.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    required = {"run_id", "created_at", "resolved_config", "aggregate_metrics"}
    missing = required - payload.keys()
    if missing:
        raise ValueError("missing required fields: " + ", ".join(sorted(missing)))
    return SavedRun(
        str(payload["run_id"]), str(payload["created_at"]), manifest.parent,
        dict(payload["resolved_config"]), dict(payload["aggregate_metrics"]),
        {str(key): dict(value) for key, value in payload.get("per_basin_metrics", {}).items()},
        dict(payload.get("routing_diagnostics", {})),
        {str(key): list(value) for key, value in payload.get("loss_traces", {}).items()},
        list(payload.get("extreme_events", [])), list(payload.get("failures", [])),
        int(payload.get("parameter_count", 0)), float(payload.get("runtime_seconds", 0.0)),
        bool(payload.get("cancelled", False)),
        {str(key): dict(value) for key, value in payload.get("forecast_metrics", {}).items()},
    )


def discover_runs(directory: str | Path = DEFAULT_RESULTS_DIRECTORY) -> RegistrySnapshot:
    """Load valid registry entries without allowing one broken run to hide the rest."""
    root = Path(directory)
    if not root.exists():
        return RegistrySnapshot(())
    runs: list[SavedRun] = []
    warnings: list[str] = []
    for manifest in sorted(root.glob("*/run.json")):
        try:
            runs.append(load_run(manifest))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            warnings.append(f"{manifest}: {error}")
    runs.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
    return RegistrySnapshot(tuple(runs), tuple(warnings))


def filter_runs(runs: Iterable[SavedRun], selections: Mapping[str, object]) -> list[SavedRun]:
    """Return runs matching every non-empty setup selection."""
    normalized = {
        field: display_value(value) for field, value in selections.items()
        if value not in (None, "", "Any")
    }
    return [
        run for run in runs
        if all(run.value(field) == expected for field, expected in normalized.items())
    ]


def available_values(
    runs: Iterable[SavedRun], selections: Mapping[str, object], field: str
) -> list[str]:
    """Values for one dropdown that remain possible under all other filters."""
    others = {key: value for key, value in selections.items() if key != field}
    return sorted({run.value(field) for run in filter_runs(runs, others)})


def leaderboard_rows(runs: Iterable[SavedRun]) -> list[list[object]]:
    rows = []
    for run in runs:
        metrics = run.aggregate_metrics
        rows.append([
            run.run_id, run.created_at, run.value("strategy.approach"),
            run.value("model.complex_expert"), run.value("model.simple_expert"),
            run.value("strategy.epochs"), run.value("physics.optimization_mode"),
            metrics.get("nse"), metrics.get("kge"), metrics.get("rmse_mm_day"),
            metrics.get("physics_error"), run.parameter_count, run.runtime_seconds,
        ])
    return rows


def read_predictions(run: SavedRun) -> list[dict[str, str]]:
    path = run.path / "predictions.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


__all__ = [
    "DEFAULT_RESULTS_DIRECTORY", "FILTER_FIELDS", "RegistrySnapshot", "SCHEMA_VERSION",
    "SavedRun", "available_values", "discover_runs", "display_value", "filter_runs",
    "leaderboard_rows", "load_run", "parse_bool", "read_predictions", "save_run",
]
