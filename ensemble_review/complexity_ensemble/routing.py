from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

GradientFunction = Callable[[torch.Tensor], torch.Tensor]


class MorseRouter(nn.Module):
    """Routes by the norm of the gradient of a supplied Morse function.

    ``morse_gradient`` must be analytic PyTorch code. This avoids NumPy detaches and
    permits derivatives of the assembled PINN solution to include derivatives of
    the gate itself.
    """

    def __init__(
        self,
        morse_gradient: GradientFunction,
        percentile: float = 80.0,
        temperature: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be strictly between 0 and 100")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.morse_gradient = morse_gradient
        self.percentile = percentile
        self.temperature = temperature
        self.eps = eps
        self.register_buffer("threshold", torch.tensor(float("nan")))
        self.register_buffer("score_scale", torch.tensor(float("nan")))

    @property
    def is_fitted(self) -> bool:
        return bool(torch.isfinite(self.threshold).item())

    def complexity(self, inputs: torch.Tensor) -> torch.Tensor:
        gradient = self.morse_gradient(inputs)
        if gradient.shape != inputs.shape:
            raise ValueError(
                f"morse_gradient returned {tuple(gradient.shape)}, expected {tuple(inputs.shape)}"
            )
        return torch.linalg.vector_norm(gradient, dim=-1)

    @torch.no_grad()
    def fit(self, reference_inputs: torch.Tensor) -> "MorseRouter":
        if reference_inputs.ndim != 2 or len(reference_inputs) == 0:
            raise ValueError("reference_inputs must be a non-empty [batch, features] tensor")
        scores = self.complexity(reference_inputs)
        q = self.percentile / 100.0
        threshold = torch.quantile(scores, q)
        q25, q75 = torch.quantile(scores, torch.tensor([0.25, 0.75], device=scores.device))
        scale = (q75 - q25).clamp_min(self.eps)
        self.threshold.copy_(threshold.to(self.threshold))
        self.score_scale.copy_(scale.to(self.score_scale))
        return self

    def complex_weight(self, inputs: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("fit the MorseRouter on training/reference inputs before routing")
        score = self.complexity(inputs)
        if hard:
            return (score >= self.threshold).to(inputs.dtype)
        width = (self.temperature * self.score_scale).clamp_min(self.eps)
        return torch.sigmoid((score - self.threshold) / width)

    def regularization(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=inputs.dtype, device=inputs.device)


class LearnedRouter(nn.Module):
    """Conventional trainable input gate used as the non-complexity baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        target_complex_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_complex_fraction = target_complex_fraction

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, reference_inputs: torch.Tensor) -> "LearnedRouter":
        return self

    def complexity(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return gate logits for compatibility with routing diagnostics."""
        return self.net(inputs).squeeze(-1)

    def complex_weight(self, inputs: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        weight = torch.sigmoid(self.complexity(inputs))
        return (weight >= 0.5).to(inputs.dtype) if hard else weight

    def regularization(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.complex_weight(inputs)
        balance = (weight.mean() - self.target_complex_fraction).square()
        decisiveness = (weight * (1.0 - weight)).mean()
        return balance + 0.01 * decisiveness


class ScoreRouter(nn.Module):
    """Convert externally estimated scalar complexity and confidence into a soft gate."""

    def __init__(
        self,
        percentile: float = 80.0,
        temperature: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be strictly between 0 and 100")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.percentile = percentile
        self.temperature = temperature
        self.eps = eps
        self.register_buffer("threshold", torch.tensor(float("nan")))
        self.register_buffer("score_scale", torch.tensor(float("nan")))

    @property
    def is_fitted(self) -> bool:
        return bool(torch.isfinite(self.threshold).item())

    @staticmethod
    def _parts(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 1:
            return features, torch.ones_like(features)
        if features.ndim != 2 or features.shape[1] not in (1, 2):
            raise ValueError("score routing features must have shape [batch], [batch, 1], or [batch, 2]")
        score = features[:, 0]
        confidence = features[:, 1].clamp(0.0, 1.0) if features.shape[1] == 2 else torch.ones_like(score)
        return score, confidence

    def complexity(self, features: torch.Tensor) -> torch.Tensor:
        return self._parts(features)[0]

    @torch.no_grad()
    def fit(self, reference_features: torch.Tensor) -> "ScoreRouter":
        scores, confidence = self._parts(reference_features)
        valid = torch.isfinite(scores) & (confidence > 0.0)
        if not bool(valid.any()):
            # Keep the route operational and conservative when an estimator cannot
            # establish confidence on this training period: every resulting weight
            # is multiplied by zero confidence.
            valid = torch.isfinite(scores)
        if not bool(valid.any()):
            raise ValueError("no finite complexity estimates are available to fit the router")
        scores = scores[valid]
        threshold = torch.quantile(scores, self.percentile / 100.0)
        q25, q75 = torch.quantile(scores, torch.tensor([0.25, 0.75], device=scores.device))
        self.threshold.copy_(threshold.to(self.threshold))
        self.score_scale.copy_((q75 - q25).clamp_min(self.eps).to(self.score_scale))
        return self

    def complex_weight(self, features: torch.Tensor, *, hard: bool = False) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("fit the ScoreRouter on training estimates before routing")
        score, confidence = self._parts(features)
        valid_score = torch.where(torch.isfinite(score), score, self.threshold)
        width = (self.temperature * self.score_scale).clamp_min(self.eps)
        weight = torch.sigmoid((valid_score - self.threshold) / width) * confidence
        return (weight >= 0.5).to(features.dtype) if hard else weight

    def regularization(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=features.dtype, device=features.device)


def heat_morse_gradient(tx: torch.Tensor) -> torch.Tensor:
    """Gradient of phi(t, x)=(1-t) sin(pi*x), derived from the heat IC."""
    t, x = tx[:, 0], tx[:, 1]
    grad_t = -torch.sin(torch.pi * x)
    grad_x = (1.0 - t) * torch.pi * torch.cos(torch.pi * x)
    return torch.stack((grad_t, grad_x), dim=-1)


def pendulum_morse_gradient(tqv: torch.Tensor) -> torch.Tensor:
    """Gradient of pendulum energy phi(q,v)=1-cos(q)+v^2/2.

    Time is included in the model input but the potential is autonomous.
    """
    q, velocity = tqv[:, 1], tqv[:, 2]
    return torch.stack((torch.zeros_like(q), torch.sin(q), velocity), dim=-1)
