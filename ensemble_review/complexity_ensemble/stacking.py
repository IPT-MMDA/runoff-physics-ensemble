"""Utilities for small, interpretable validation-fitted ensembles."""

from __future__ import annotations

import torch


@torch.no_grad()
def fit_convex_stacking_weight(
    simple_prediction: torch.Tensor,
    complex_prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    """Fit ``(1-w) * simple + w * complex`` by validation MSE.

    The closed-form least-squares solution is projected onto ``[0, 1]`` so the
    fitted result remains an interpolation rather than an extrapolating meta-model.
    Returns the complex-expert weight and the resulting validation MSE.
    """
    simple = simple_prediction.detach().flatten().double()
    complex_value = complex_prediction.detach().flatten().double()
    observed = target.detach().flatten().double()
    if not len(simple) or simple.shape != complex_value.shape or simple.shape != observed.shape:
        raise ValueError("stacking predictions and targets must be non-empty and equally shaped")
    difference = complex_value - simple
    denominator = difference.square().sum()
    if float(denominator) <= torch.finfo(denominator.dtype).eps:
        weight = torch.tensor(0.5, dtype=denominator.dtype, device=denominator.device)
    else:
        weight = ((observed - simple) * difference).sum() / denominator
        weight = weight.clamp(0.0, 1.0)
    combined = torch.lerp(simple, complex_value, weight)
    validation_mse = (combined - observed).square().mean()
    return float(weight), float(validation_mse)


__all__ = ["fit_convex_stacking_weight"]
