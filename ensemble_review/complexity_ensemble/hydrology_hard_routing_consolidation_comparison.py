from __future__ import annotations

from typing import Mapping

import torch

from .hydrology import (
    HydrologyLosses,
    HydrologyRoutedOutput,
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
    SingleSimpleHydrologyModel,
)
from .hydrology_comparison import (
    _add_observation_noise,
    _complexity_diagnostics,
    _estimate_complexity_features,
    _evaluate,
    _make_model,
    _parameter_count,
    _prepare_complexity_estimators,
    _select_models,
    hydrology_comparison_summary,
    plot_hydrology_comparison,
    save_hydrology_comparison,
)
from .hydrology_complexity import HydrologyComplexityEstimator
from .hydrology_consolidation_comparison import HydrologyConsolidationComparisonRecord
from .hydrology_data import HydrologyData, HydrologySplit


def hard_routed_losses(
    model: RoutedHydrologyModel,
    inputs: torch.Tensor,
    physical_inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    routing_features: torch.Tensor | None = None,
    physics_weight: float = 0.05,
    interface_weight: float = 0.05,
    routing_weight: float = 0.05,
    compute_weight: float = 0.0,
) -> HydrologyLosses:
    """Compute prediction losses with disjoint, binary expert assignment.

    The data and physics terms use a hard route, so a sample contributes gradients
    to exactly one expert.  The soft weight is retained only for the router's
    regularization and the boundary-consolidation term.
    """
    soft_routed = model(
        inputs,
        routing_features=routing_features,
        return_details=True,
    )
    assert isinstance(soft_routed, HydrologyRoutedOutput)

    router_features = model._router_features(inputs, routing_features)
    hard_weight = model.router.complex_weight(router_features, hard=True)
    # Straight-through routing preserves a binary forward assignment while letting
    # the trainable learned gate receive the same soft surrogate gradient as before.
    routed_weight = (
        hard_weight
        + soft_routed.complex_weight
        - soft_routed.complex_weight.detach()
    )
    discharge = model._positive_discharge(
        torch.lerp(
            soft_routed.simple_raw,
            soft_routed.complex_raw,
            routed_weight[:, None],
        )
    )

    data = (discharge.squeeze(-1) - targets).square().mean()
    physics = model.physics_residual(discharge, physical_inputs).square().mean()
    transition = 4.0 * soft_routed.complex_weight * (1.0 - soft_routed.complex_weight)
    interface = (
        transition
        * (soft_routed.simple_raw - soft_routed.complex_raw).square().squeeze(-1)
    ).mean()
    routing = model.router.regularization(router_features)
    complex_usage = soft_routed.complex_weight.mean()
    total = (
        data
        + physics_weight * physics
        + interface_weight * interface
        + routing_weight * routing
        + compute_weight * complex_usage
    )
    return HydrologyLosses(total, data, physics, interface, routing, complex_usage)


def train_hydrology_model_with_hard_routing_and_consolidation(
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
    """Train routed experts on disjoint data while preserving other behavior."""
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
                losses = hard_routed_losses(
                    model,
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


def compare_hydrology_models_with_hard_routing_and_consolidation(
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
    """Run the consolidation comparison with disjoint expert training data."""
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
                train_hydrology_model_with_hard_routing_and_consolidation(
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
    "compare_hydrology_models_with_hard_routing_and_consolidation",
    "hard_routed_losses",
    "hydrology_comparison_summary",
    "plot_hydrology_comparison",
    "save_hydrology_comparison",
    "train_hydrology_model_with_hard_routing_and_consolidation",
]
