"""KAN 2.0 discovery of symbolic corrections to the reservoir balance."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Sequence

import torch
from torch import nn


PHYSICS_OPTIMIZATION_MODES = (
    "accuracy", "robustness", "balanced", "distillation",
)
KAN_FEATURES = (
    "precipitation", "pet", "temperature", "previous_discharge", "day_sin", "day_cos",
)
_SYMBOLIC_LIBRARY = ("x", "x^2", "x^3", "abs", "sin", "cos", "tanh")


@dataclass(frozen=True)
class KANPhysicsSettings:
    mode: str
    candidates: int = 3
    steps: int = 50
    grid: int = 3
    spline_order: int = 3
    additive_nodes: int = 2
    multiplicative_nodes: int = 1
    sparsity_weight: float = 1e-3
    noise_levels: tuple[float, ...] = (0.05, 0.1, 0.2)
    robustness_accuracy_tolerance: float = 0.05
    symbolic: bool = True


@dataclass
class KANPhysicsResult:
    correction: "KANPhysicsCorrection"
    formula: str
    candidate_index: int
    candidate_metrics: list[dict[str, float]]


class KANPhysicsCorrection(nn.Module):
    """Normalize physical inputs around a frozen numerical/symbolic MultKAN."""

    def __init__(
        self,
        model: nn.Module,
        feature_indices: torch.Tensor,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        output_mean: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("feature_indices", feature_indices.long())
        self.register_buffer("input_mean", input_mean.float())
        self.register_buffer("input_scale", input_scale.float())
        self.register_buffer("output_mean", output_mean.reshape(()).float())
        self.register_buffer("output_scale", output_scale.reshape(()).float())
        self.requires_grad_(False)

    def selected_inputs(self, physical_inputs: torch.Tensor) -> torch.Tensor:
        values = physical_inputs[:, -1].index_select(1, self.feature_indices)
        return (values - self.input_mean) / self.input_scale

    def forward(self, physical_inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.model(self.selected_inputs(physical_inputs)).squeeze(-1)
        return normalized * self.output_scale + self.output_mean


def _multkan_class():
    try:
        # Import from the defining module. Some pykan releases expose
        # ``kan.MultKAN`` as the module object, which is not callable.
        from kan.MultKAN import MultKAN
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "KAN physics optimization requires pykan; install the project with the 'kan' extra"
        ) from error
    return MultKAN


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    ordered = labels[order].to(torch.float32)
    precision = ordered.cumsum(0) / torch.arange(
        1, len(ordered) + 1, device=ordered.device
    )
    return float((precision * ordered).sum() / positives)


def _macro_average_precision(
    scores: torch.Tensor, labels: torch.Tensor, basin_ids: Sequence[str]
) -> float:
    values = []
    for basin in dict.fromkeys(basin_ids):
        indices = torch.tensor(
            [item == basin for item in basin_ids], device=scores.device, dtype=torch.bool
        )
        if int(labels[indices].sum()):
            values.append(_average_precision(scores[indices], labels[indices]))
    return sum(values) / len(values) if values else 0.0


def _select_candidate(mode: str, metrics: list[dict[str, float]], tolerance: float) -> int:
    if mode == "distillation":
        return min(range(len(metrics)), key=lambda index: metrics[index]["validation_mse"])
    if mode == "accuracy":
        return max(range(len(metrics)), key=lambda index: metrics[index]["clean_ap"])
    best_accuracy = max(item["clean_ap"] for item in metrics)
    if mode == "robustness":
        eligible = [
            index for index, item in enumerate(metrics)
            if item["clean_ap"] >= best_accuracy - tolerance
        ]
        return min(eligible, key=lambda index: metrics[index]["noise_degradation"])
    accuracies = torch.tensor([item["clean_ap"] for item in metrics])
    robustness = -torch.tensor([item["noise_degradation"] for item in metrics])

    def normalize(values: torch.Tensor) -> torch.Tensor:
        span = values.max() - values.min()
        return torch.ones_like(values) if float(span) == 0.0 else (values - values.min()) / span

    return int(torch.argmax(normalize(accuracies) + normalize(robustness)))


def _reservoir_change(
    physical: torch.Tensor,
    feature_names: Sequence[str],
    response: float,
    recession: float,
) -> torch.Tensor:
    last = physical[:, -1]
    precipitation = last[:, feature_names.index("precipitation")]
    pet = last[:, feature_names.index("pet")]
    previous = last[:, feature_names.index("previous_discharge")]
    return response * torch.relu(precipitation - pet) - recession * previous


def standalone_kan_prediction(
    physical: torch.Tensor,
    feature_names: Sequence[str],
    response: float,
    recession: float,
    correction: KANPhysicsCorrection,
) -> torch.Tensor:
    """Evaluate the frozen KAN-augmented reservoir equation by itself."""
    previous = physical[:, -1, feature_names.index("previous_discharge")]
    return (
        previous
        + _reservoir_change(physical, feature_names, response, recession)
        + correction(physical)
    )


def _noisy_physical(
    physical: torch.Tensor,
    feature_indices: torch.Tensor,
    scale: torch.Tensor,
    level: float,
    seed: int,
) -> torch.Tensor:
    result = physical.clone()
    generator = torch.Generator(device=physical.device).manual_seed(seed)
    noise = torch.randn(
        (len(result), len(feature_indices)), generator=generator,
        device=result.device, dtype=result.dtype,
    )
    result[:, -1, feature_indices] += noise * scale * level
    return result


def discover_kan_physics_correction(
    *,
    train_physical: torch.Tensor,
    validation_physical: torch.Tensor,
    train_observed_flow: torch.Tensor,
    validation_observed_flow: torch.Tensor,
    feature_names: Sequence[str],
    response: float,
    recession: float,
    settings: KANPhysicsSettings,
    seed: int,
    device: torch.device,
    train_teacher_flow: torch.Tensor | None = None,
    validation_teacher_flow: torch.Tensor | None = None,
    train_basin_ids: Sequence[str] | None = None,
    validation_basin_ids: Sequence[str] | None = None,
) -> KANPhysicsResult:
    """Fit candidate MultKAN residual laws and select one without touching test data."""
    if settings.mode not in PHYSICS_OPTIMIZATION_MODES:
        raise ValueError(f"unknown KAN physics mode {settings.mode!r}")
    if settings.mode == "distillation" and (
        train_teacher_flow is None or validation_teacher_flow is None
    ):
        raise ValueError("physics distillation requires train and validation teacher predictions")
    MultKAN = _multkan_class()
    selected_names = tuple(name for name in KAN_FEATURES if name in feature_names)
    feature_indices = torch.tensor(
        [feature_names.index(name) for name in selected_names], device=device
    )
    noise_names = tuple(
        name for name in selected_names if name not in {"day_sin", "day_cos"}
    )
    noise_feature_indices = torch.tensor(
        [feature_names.index(name) for name in noise_names], device=device
    )
    train_physical = train_physical.to(device)
    validation_physical = validation_physical.to(device)
    raw_train = train_physical[:, -1].index_select(1, feature_indices)
    input_mean = raw_train.mean(0)
    input_scale = raw_train.std(0, unbiased=False).clamp_min(1e-6)
    train_x = (raw_train - input_mean) / input_scale
    validation_x = (
        validation_physical[:, -1].index_select(1, feature_indices) - input_mean
    ) / input_scale
    train_goal = train_teacher_flow if settings.mode == "distillation" else train_observed_flow
    validation_goal = (
        validation_teacher_flow if settings.mode == "distillation" else validation_observed_flow
    )
    assert train_goal is not None and validation_goal is not None
    train_previous = train_physical[:, -1, feature_names.index("previous_discharge")]
    train_correction = (
        train_goal.to(device) - train_previous
        - _reservoir_change(train_physical, feature_names, response, recession)
    )
    output_mean = train_correction.mean()
    output_scale = train_correction.std(unbiased=False).clamp_min(1e-6)
    train_y = ((train_correction - output_mean) / output_scale)[:, None]
    validation_previous = validation_physical[
        :, -1, feature_names.index("previous_discharge")
    ]
    train_basin_ids = tuple(train_basin_ids or ("all",) * len(train_observed_flow))
    validation_basin_ids = tuple(
        validation_basin_ids or ("all",) * len(validation_observed_flow)
    )
    thresholds = {
        basin: torch.quantile(
            train_observed_flow.to(device)[torch.tensor(
                [item == basin for item in train_basin_ids], device=device, dtype=torch.bool
            )],
            0.95,
        )
        for basin in dict.fromkeys(validation_basin_ids)
    }
    labels = torch.tensor(
        [
            bool(validation_observed_flow[index].to(device) >= thresholds[basin])
            for index, basin in enumerate(validation_basin_ids)
        ],
        device=device,
    )
    dataset = {
        "train_input": train_x,
        "train_label": train_y,
        "test_input": validation_x,
        "test_label": torch.zeros((len(validation_x), 1), device=device),
    }
    candidates: list[KANPhysicsCorrection] = []
    formulas: list[str] = []
    metrics: list[dict[str, float]] = []
    for candidate_index in range(settings.candidates):
        with tempfile.TemporaryDirectory(prefix="kan-physics-") as checkpoint_directory:
            model = MultKAN(
                width=[
                    len(selected_names),
                    [settings.additive_nodes, settings.multiplicative_nodes],
                    1,
                ],
                grid=settings.grid,
                k=settings.spline_order,
                mult_arity=2,
                seed=seed + candidate_index,
                auto_save=False,
                ckpt_path=checkpoint_directory,
                symbolic_enabled=True,
                device=str(device),
            )
            model.fit(
                dataset,
                opt="Adam",
                steps=settings.steps,
                lr=0.02,
                lamb=settings.sparsity_weight,
                update_grid=False,
                log=max(settings.steps + 1, 1000),
            )
            formula = "numeric MultKAN correction"
            if settings.symbolic:
                model = model.prune()
                model.auto_symbolic(lib=list(_SYMBOLIC_LIBRARY), verbose=0)
                symbolic = model.symbolic_formula(
                    var=list(selected_names),
                    normalizer=[input_mean.detach().cpu(), input_scale.detach().cpu()],
                    output_normalizer=[
                        [float(output_mean.detach().cpu())],
                        [float(output_scale.detach().cpu())],
                    ],
                )
                if symbolic is None:
                    raise RuntimeError(
                        "MultKAN could not convert every active edge to a symbolic form"
                    )
                formula = str(symbolic[0][0])
        correction = KANPhysicsCorrection(
            model, feature_indices, input_mean, input_scale, output_mean, output_scale
        ).to(device)
        correction.eval()
        with torch.no_grad():
            clean_prediction = (
                validation_previous
                + _reservoir_change(validation_physical, feature_names, response, recession)
                + correction(validation_physical)
            )
            item = {
                "clean_ap": _macro_average_precision(
                    clean_prediction, labels, validation_basin_ids
                ),
                "validation_mse": float(
                    (clean_prediction - validation_goal.to(device)).square().mean()
                ),
            }
            noisy_aps = []
            for noise_index, level in enumerate(settings.noise_levels):
                noisy = _noisy_physical(
                    validation_physical,
                    noise_feature_indices,
                    torch.stack([
                        input_scale[selected_names.index(name)] for name in noise_names
                    ]),
                    level,
                    seed + candidate_index * 1009 + noise_index,
                )
                noisy_prediction = (
                    noisy[:, -1, feature_names.index("previous_discharge")]
                    + _reservoir_change(noisy, feature_names, response, recession)
                    + correction(noisy)
                )
                noisy_aps.append(_macro_average_precision(
                    noisy_prediction, labels, validation_basin_ids
                ))
            item["noise_degradation"] = (
                sum(max(0.0, item["clean_ap"] - value) for value in noisy_aps)
                / max(1, len(noisy_aps))
            )
        candidates.append(correction)
        formulas.append(formula)
        metrics.append(item)
    selected = _select_candidate(
        settings.mode, metrics, settings.robustness_accuracy_tolerance
    )
    return KANPhysicsResult(candidates[selected], formulas[selected], selected, metrics)


__all__ = [
    "KANPhysicsCorrection", "KANPhysicsResult", "KANPhysicsSettings",
    "PHYSICS_OPTIMIZATION_MODES", "discover_kan_physics_correction",
    "standalone_kan_prediction",
]
