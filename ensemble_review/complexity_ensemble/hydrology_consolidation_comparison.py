from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .hydrology import (
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
    SingleSimpleHydrologyModel,
)
from .hydrology_comparison import (
    _add_observation_noise,
    _complexity_diagnostics,
    _evaluate,
    _make_model,
    _parameter_count,
    _estimate_complexity_features,
    _prepare_complexity_estimators,
    _select_models,
    hydrology_comparison_summary,
    plot_hydrology_comparison,
    save_hydrology_comparison,
)
from .hydrology_data import HydrologyData, HydrologySplit
from .hydrology_complexity import HydrologyComplexityEstimator


@dataclass(frozen=True)
class HydrologyConsolidationComparisonRecord:
    country: str
    basin: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise: float
    inference_noise: float
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    physics_error: float
    parameters: int
    mean_complex_weight: float
    mean_complexity_score: float
    valid_complexity_fraction: float
    consolidation_weight: float


def train_hydrology_model_with_consolidation(
    model: RoutedHydrologyModel | SingleComplexHydrologyModel | SingleSimpleHydrologyModel,
    split: HydrologySplit,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_noise: float,
    seed: int,
    device: torch.device,
    consolidation_weight: float,
    feature_names: tuple[str, ...],
    routing_features: torch.Tensor | None = None,
    verbose: bool = False,
) -> None:
    """Train without changing the behavior of the standard hydrology trainer."""
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be non-negative")
    if training_noise < 0.0:
        raise ValueError("training_noise must be non-negative")
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    targets = split.targets.to(device)
    routing_features = None if routing_features is None else routing_features.to(device)
    if isinstance(model, RoutedHydrologyModel):
        model.fit_router(inputs, routing_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)
    for epoch in range(epochs):
        epoch_total = 0.0
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names, training_noise, generator
            )
            optimizer.zero_grad()
            if isinstance(model, RoutedHydrologyModel):
                batch_routing = None if routing_features is None else routing_features[indices]
                losses = model.losses(
                    batch_inputs,
                    physical[indices],
                    targets[indices],
                    routing_features=batch_routing,
                    interface_weight=consolidation_weight,
                )
            else:
                losses = model.losses(batch_inputs, physical[indices], targets[indices])
            losses.total.backward()
            optimizer.step()
            epoch_total += float(losses.total.detach()) * len(indices)
        if verbose:
            print(
                f"  training epoch {epoch + 1}/{epochs}: "
                f"loss={epoch_total / len(inputs):.6f}",
                flush=True,
            )


def compare_hydrology_models_with_consolidation(
    data: HydrologyData,
    *,
    models: tuple[str, ...] | None = None,
    simple_kind: str = "rbf",
    complex_kind: str = "mlp",
    seeds: tuple[int, ...] = (0,),
    training_noise_levels: tuple[float, ...] = (0.0,),
    inference_noise_levels: tuple[float, ...] = (0.0, 0.05),
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    consolidation_weight: float = 0.05,
    device: torch.device | str = "cpu",
    learned_morse_potential: LearnedHydrologyMorsePotential | None = None,
    complexity_estimators: Mapping[str, HydrologyComplexityEstimator] | None = None,
    complexity_percentile: float = 80.0,
    gate_temperature: float = 0.15,
) -> list[HydrologyConsolidationComparisonRecord]:
    """Run an isolated comparison with configurable boundary consolidation."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be non-negative")
    if any(noise < 0.0 for noise in (*training_noise_levels, *inference_noise_levels)):
        raise ValueError("noise levels must be non-negative")
    selected_models = _select_models(models, learned_morse_potential)
    prepared_estimators = _prepare_complexity_estimators(
        data, selected_models, complexity_estimators
    )
    feature_cache: dict[tuple[str, str, float, int], torch.Tensor | None] = {}

    def features_for(
        model_name: str,
        split_name: str,
        split: HydrologySplit,
        noise: float,
        score_seed: int,
    ) -> torch.Tensor | None:
        key = (model_name, split_name, noise, score_seed)
        if key not in feature_cache:
            feature_cache[key] = _estimate_complexity_features(
                prepared_estimators.get(model_name),
                split,
                data.feature_names,
                noise=noise,
                seed=score_seed,
            )
        return feature_cache[key]

    device = torch.device(device)
    records = []
    for seed in seeds:
        for training_noise in sorted(set(training_noise_levels)):
            for model_name in selected_models:
                train_features = features_for(
                    model_name, "train", data.train, training_noise, seed + 100003
                )
                torch.manual_seed(seed)
                model = _make_model(
                    model_name,
                    data,
                    simple_kind,
                    complex_kind,
                    learned_morse_potential,
                    complexity_percentile,
                    gate_temperature,
                ).to(device)
                train_hydrology_model_with_consolidation(
                    model,
                    data.train,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    training_noise=training_noise,
                    seed=seed,
                    device=device,
                    consolidation_weight=consolidation_weight,
                    feature_names=data.feature_names,
                    routing_features=train_features,
                )
                for inference_noise in sorted(set(inference_noise_levels)):
                    validation_features = features_for(
                        model_name,
                        "validation",
                        data.validation,
                        inference_noise,
                        seed + 200003,
                    )
                    test_features = features_for(
                        model_name,
                        "test",
                        data.test,
                        inference_noise,
                        seed + 300007,
                    )
                    validation_nse, _, _, _, _ = _evaluate(
                        model,
                        data.validation,
                        data,
                        inference_noise=inference_noise,
                        seed=seed + 200003,
                        device=device,
                        routing_features=validation_features,
                    )
                    test_nse, test_kge, test_rmse, physics, usage = _evaluate(
                        model,
                        data.test,
                        data,
                        inference_noise=inference_noise,
                        seed=seed + 300007,
                        device=device,
                        routing_features=test_features,
                    )
                    mean_score, valid_fraction = _complexity_diagnostics(test_features)
                    records.append(
                        HydrologyConsolidationComparisonRecord(
                            data.country,
                            data.basin_id,
                            model_name,
                            simple_kind if model_name != "single_complex" else "none",
                            complex_kind,
                            seed,
                            training_noise,
                            inference_noise,
                            validation_nse,
                            test_nse,
                            test_kge,
                            test_rmse,
                            physics,
                            _parameter_count(model),
                            usage,
                            mean_score,
                            valid_fraction,
                            consolidation_weight,
                        )
                    )
    return records


__all__ = [
    "HydrologyConsolidationComparisonRecord",
    "compare_hydrology_models_with_consolidation",
    "hydrology_comparison_summary",
    "plot_hydrology_comparison",
    "save_hydrology_comparison",
    "train_hydrology_model_with_consolidation",
]
