from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .hydrology import (
    LearnedHydrologyMorsePotential,
    SingleComplexHydrologyModel,
    SingleSimpleHydrologyModel,
)
from .hydrology_comparison import train_hydrology_model
from .hydrology_data import HydrologyData, HydrologySplit


@dataclass(frozen=True)
class MorseCalibrationReport:
    samples: int
    folds: int
    pilot_epochs: int
    potential_epochs: int
    positive_utility_fraction: float
    score_utility_spearman: float
    final_ranking_loss: float


@dataclass(frozen=True)
class MorsePotentialAssessment:
    split: str
    samples: int
    score_utility_spearman: float
    top_fraction: float
    top_utility_precision: float
    positive_utility_high_score: float
    positive_utility_low_score: float


def _slice_split(split: HydrologySplit, start: int, stop: int) -> HydrologySplit:
    return HydrologySplit(
        split.inputs[start:stop], split.physical_inputs[start:stop], split.targets[start:stop],
        split.target_dates[start:stop],
        None if split.routing_inputs is None else split.routing_inputs[start:stop],
    )


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    def ranks(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values.flatten())
        result = torch.empty_like(order, dtype=torch.float64)
        result[order] = torch.arange(len(order), dtype=torch.float64, device=values.device)
        return result

    first_rank = ranks(first)
    second_rank = ranks(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = torch.linalg.vector_norm(first_rank) * torch.linalg.vector_norm(second_rank)
    return float((first_rank * second_rank).sum() / denominator.clamp_min(1e-12))


@torch.no_grad()
def _pilot_utility(
    simple: SingleSimpleHydrologyModel,
    complex_model: SingleComplexHydrologyModel,
    split: HydrologySplit,
    *,
    physics_weight: float,
    compute_penalty: float,
    device: torch.device,
) -> torch.Tensor:
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    targets = split.targets.to(device)
    simple_prediction = simple(inputs)
    complex_prediction = complex_model(inputs)
    data_advantage = (
        (simple_prediction.squeeze(-1) - targets).square()
        - (complex_prediction.squeeze(-1) - targets).square()
    )
    physics_advantage = (
        simple.physics_residual(simple_prediction, physical).square()
        - complex_model.physics_residual(complex_prediction, physical).square()
    )
    return (data_advantage + physics_weight * physics_advantage - compute_penalty).cpu()


def collect_cross_fitted_utility(
    data: HydrologyData,
    *,
    simple_kind: str,
    complex_kind: str,
    folds: int = 3,
    pilot_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    physics_weight: float = 0.05,
    compute_penalty: float = 0.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create forward-chained, out-of-sample labels of complex-expert utility."""
    if folds < 2:
        raise ValueError("at least two chronological folds are required")
    device = torch.device(device)
    train = data.train
    boundaries = [round(index * len(train.inputs) / folds) for index in range(folds + 1)]
    calibrated_inputs, utilities = [], []
    for fold in range(1, folds):
        prefix_stop = boundaries[fold]
        validation_stop = boundaries[fold + 1]
        if prefix_stop < 15 or validation_stop <= prefix_stop:
            raise ValueError("folds leave too few samples for pilot calibration")
        pilot_train = _slice_split(train, 0, prefix_stop)
        pilot_validation = _slice_split(train, prefix_stop, validation_stop)
        torch.manual_seed(seed + 1009 * fold)
        simple = SingleSimpleHydrologyModel(
            data.feature_names, train.inputs.shape[1], data.discharge_scale,
            simple_kind=simple_kind,
        ).to(device)
        torch.manual_seed(seed + 1009 * fold)
        complex_model = SingleComplexHydrologyModel(
            data.feature_names, train.inputs.shape[1], data.discharge_scale,
            complex_kind=complex_kind,
        ).to(device)
        for model in (simple, complex_model):
            train_hydrology_model(
                model, pilot_train, epochs=pilot_epochs, batch_size=batch_size,
                learning_rate=learning_rate, training_noise=0.0,
                seed=seed + fold, device=device,
            )
        utilities.append(
            _pilot_utility(
                simple, complex_model, pilot_validation,
                physics_weight=physics_weight, compute_penalty=compute_penalty,
                device=device,
            )
        )
        calibrated_inputs.append(pilot_validation.inputs)
    return torch.cat(calibrated_inputs), torch.cat(utilities)


def train_morse_potential(
    inputs: torch.Tensor,
    utility: torch.Tensor,
    feature_names: tuple[str, ...],
    *,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    hidden_dim: int = 24,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[LearnedHydrologyMorsePotential, float]:
    """Fit gradient-magnitude ordering using stochastic pairwise ranking."""
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs must be positive and batch_size must be at least two")
    device = torch.device(device)
    sequence_length = inputs.shape[1]
    inputs = inputs.flatten(start_dim=1).to(device)
    utility = utility.to(device)
    torch.manual_seed(seed)
    potential = LearnedHydrologyMorsePotential(
        sequence_length, feature_names, hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(potential.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 65537)
    last_loss = torch.tensor(float("nan"), device=device)
    for _ in range(epochs):
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            if len(indices) < 2:
                continue
            batch_inputs = inputs[indices]
            batch_utility = utility[indices]
            partner = torch.randperm(len(indices), generator=generator, device=device)
            utility_difference = batch_utility - batch_utility[partner]
            informative = utility_difference.abs() > 1e-6
            if not bool(informative.any()):
                continue
            score = potential.complexity(batch_inputs, create_graph=True)
            score_difference = score - score[partner]
            direction = utility_difference.sign()
            ranking = F.softplus(-direction[informative] * score_difference[informative]).mean()
            scale_regularization = 1e-4 * score.square().mean()
            last_loss = ranking + scale_regularization
            optimizer.zero_grad()
            last_loss.backward()
            optimizer.step()
    return potential.freeze(), float(last_loss.detach())


def calibrate_learned_morse(
    data: HydrologyData,
    *,
    simple_kind: str = "rbf",
    complex_kind: str = "mlp",
    folds: int = 3,
    pilot_epochs: int = 5,
    potential_epochs: int = 100,
    batch_size: int = 64,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[LearnedHydrologyMorsePotential, MorseCalibrationReport]:
    inputs, utility = collect_cross_fitted_utility(
        data, simple_kind=simple_kind, complex_kind=complex_kind,
        folds=folds, pilot_epochs=pilot_epochs, batch_size=batch_size,
        seed=seed, device=device,
    )
    potential, final_loss = train_morse_potential(
        inputs, utility, data.feature_names, epochs=potential_epochs,
        batch_size=max(2, batch_size), seed=seed, device=device,
    )
    scores = potential.complexity(inputs.flatten(start_dim=1).to(device)).cpu()
    report = MorseCalibrationReport(
        len(inputs), folds, pilot_epochs, potential_epochs,
        float((utility > 0.0).float().mean()),
        _spearman(scores, utility), final_loss,
    )
    return potential, report


def assess_learned_morse(
    data: HydrologyData,
    potential: LearnedHydrologyMorsePotential,
    *,
    simple_kind: str,
    complex_kind: str,
    pilot_epochs: int = 5,
    batch_size: int = 64,
    top_fraction: float = 0.2,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> list[MorsePotentialAssessment]:
    """Measure potential transfer using pilots trained on all training data."""
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be strictly between zero and one")
    device = torch.device(device)
    torch.manual_seed(seed + 99991)
    simple = SingleSimpleHydrologyModel(
        data.feature_names, data.train.inputs.shape[1], data.discharge_scale,
        simple_kind=simple_kind,
    ).to(device)
    torch.manual_seed(seed + 99991)
    complex_model = SingleComplexHydrologyModel(
        data.feature_names, data.train.inputs.shape[1], data.discharge_scale,
        complex_kind=complex_kind,
    ).to(device)
    for model in (simple, complex_model):
        train_hydrology_model(
            model, data.train, epochs=pilot_epochs, batch_size=batch_size,
            learning_rate=1e-3, training_noise=0.0, seed=seed + 17, device=device,
        )
    assessments = []
    for split_name, split in (("validation", data.validation), ("test", data.test)):
        utility = _pilot_utility(
            simple, complex_model, split, physics_weight=0.05,
            compute_penalty=0.0, device=device,
        )
        scores = potential.complexity(
            split.inputs.flatten(start_dim=1).to(device)
        ).cpu()
        top_count = max(1, round(top_fraction * len(scores)))
        score_mask = torch.zeros(len(scores), dtype=torch.bool)
        utility_mask = torch.zeros(len(scores), dtype=torch.bool)
        score_mask[torch.topk(scores, top_count).indices] = True
        utility_mask[torch.topk(utility, top_count).indices] = True
        positive = utility > 0.0
        assessments.append(
            MorsePotentialAssessment(
                split_name, len(scores), _spearman(scores, utility), top_fraction,
                float((score_mask & utility_mask).sum() / top_count),
                float(positive[score_mask].float().mean()),
                float(positive[~score_mask].float().mean()),
            )
        )
    return assessments


def save_calibration_report(report: MorseCalibrationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(report)))
        writer.writeheader()
        writer.writerow(asdict(report))


def save_potential_assessment(
    assessments: list[MorsePotentialAssessment], path: str | Path
) -> None:
    if not assessments:
        raise ValueError("cannot save an empty assessment")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(assessments[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in assessments)
