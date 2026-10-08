from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch


ROUTING_METHODS = ("morse", "learned_morse", "learned", "tda", "lyapunov")
MODEL_NAMES = (*ROUTING_METHODS, "single_complex")
SCORE_ROUTING_METHODS = frozenset(("tda", "lyapunov"))


def takens_delay_embedding(
    series: torch.Tensor,
    *,
    embedding_dim: int,
    delay: int,
) -> torch.Tensor:
    """Return delay coordinates ordered from oldest to newest observation."""
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if embedding_dim < 2 or delay < 1:
        raise ValueError("embedding_dim must be at least two and delay must be positive")
    span = (embedding_dim - 1) * delay + 1
    if len(series) < span + 1:
        raise ValueError(
            f"time series length {len(series)} is too short for dimension "
            f"{embedding_dim} and delay {delay}"
        )
    return series.unfold(0, span, 1)[:, ::delay]


class HydrologyComplexityEstimator(ABC):
    """Stateless or training-fitted transform from history windows to score/confidence pairs."""

    def fit(self, routing_windows: torch.Tensor) -> "HydrologyComplexityEstimator":
        self._validate_windows(routing_windows)
        return self

    def _validate_windows(self, routing_windows: torch.Tensor) -> None:
        if routing_windows.ndim != 3 or len(routing_windows) == 0:
            raise ValueError("routing_windows must be a non-empty [batch, time, features] tensor")

    @abstractmethod
    def estimate(self, routing_windows: torch.Tensor) -> torch.Tensor:
        """Return ``[score, confidence]`` for every history window."""


class TakensPersistenceEstimator(HydrologyComplexityEstimator):
    """Persistent-homology complexity of a delay reconstruction of discharge."""

    def __init__(
        self,
        feature_index: int,
        *,
        embedding_dim: int = 3,
        delay: int = 7,
        min_persistence: float = 0.1,
        score: str = "total_persistence",
        max_points: int = 48,
        context_length: int | None = None,
    ) -> None:
        if feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if min_persistence < 0.0:
            raise ValueError("min_persistence must be non-negative")
        if score not in {"total_persistence", "persistent_holes", "persistence_entropy"}:
            raise ValueError(
                "score must be 'total_persistence', 'persistent_holes', or "
                "'persistence_entropy'"
            )
        if max_points < 8:
            raise ValueError("max_points must be at least eight")
        if context_length is not None and context_length < 2:
            raise ValueError("context_length must be at least two")
        self.feature_index = feature_index
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.min_persistence = min_persistence
        self.score = score
        self.max_points = max_points
        self.context_length = context_length

    def _score_window(self, series: torch.Tensor) -> tuple[float, float]:
        centered = series - series.mean()
        scale = centered.std(unbiased=False)
        if not torch.isfinite(scale) or float(scale) < 1e-6:
            return 0.0, 0.0
        points = takens_delay_embedding(
            centered / scale,
            embedding_dim=self.embedding_dim,
            delay=self.delay,
        )
        if len(points) > self.max_points:
            indices = torch.linspace(0, len(points) - 1, self.max_points).round().long()
            points = points[indices]
        try:
            from ripser import ripser
        except ImportError as error:
            raise RuntimeError(
                "TDA routing requires the optional 'tda' dependencies; "
                "install the project with `pip install -e '.[tda]'`"
            ) from error
        diagrams = ripser(points.detach().cpu().numpy(), maxdim=1)["dgms"]
        one_dimensional = diagrams[1]
        if len(one_dimensional) == 0:
            return 0.0, 1.0
        persistence = torch.as_tensor(
            one_dimensional[:, 1] - one_dimensional[:, 0], dtype=torch.float64
        )
        persistence = persistence[torch.isfinite(persistence)]
        significant = persistence[persistence >= self.min_persistence]
        if len(significant) == 0:
            return 0.0, 1.0
        if self.score == "persistent_holes":
            value = float(len(significant))
        elif self.score == "persistence_entropy":
            probabilities = significant / significant.sum().clamp_min(1e-12)
            value = float(-(probabilities * probabilities.log()).sum())
        else:
            value = float(significant.sum())
        return value, 1.0

    @torch.no_grad()
    def estimate(self, routing_windows: torch.Tensor) -> torch.Tensor:
        self._validate_windows(routing_windows)
        if self.feature_index >= routing_windows.shape[-1]:
            raise ValueError("feature_index is outside the routing feature dimension")
        estimates = [
            self._score_window(
                window[-self.context_length :, self.feature_index].detach().cpu()
                if self.context_length is not None
                else window[:, self.feature_index].detach().cpu()
            )
            for window in routing_windows
        ]
        return torch.tensor(estimates, dtype=routing_windows.dtype, device=routing_windows.device)


class LyapunovComplexityEstimator(HydrologyComplexityEstimator):
    """Confidence-aware Rosenstein estimate of the finite-time largest exponent."""

    def __init__(
        self,
        feature_index: int,
        *,
        embedding_dim: int = 3,
        delay: int = 7,
        theiler_window: int = 30,
        fit_horizon: int = 20,
        min_r2: float = 0.8,
        max_points: int = 256,
        context_length: int | None = None,
    ) -> None:
        if feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if theiler_window < 1 or fit_horizon < 3:
            raise ValueError("theiler_window must be positive and fit_horizon at least three")
        if not 0.0 <= min_r2 < 1.0:
            raise ValueError("min_r2 must be in [0, 1)")
        if max_points < fit_horizon + 2:
            raise ValueError("max_points must exceed fit_horizon")
        if context_length is not None and context_length < 2:
            raise ValueError("context_length must be at least two")
        self.feature_index = feature_index
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.theiler_window = theiler_window
        self.fit_horizon = fit_horizon
        self.min_r2 = min_r2
        self.max_points = max_points
        self.context_length = context_length

    def _score_window(self, series: torch.Tensor) -> tuple[float, float]:
        centered = series.double() - series.double().mean()
        scale = centered.std(unbiased=False)
        if not torch.isfinite(scale) or float(scale) < 1e-6:
            return 0.0, 0.0
        points = takens_delay_embedding(
            centered / scale,
            embedding_dim=self.embedding_dim,
            delay=self.delay,
        )
        stride = max(1, math.ceil(len(points) / self.max_points))
        points = points[::stride]
        count = len(points)
        if count <= self.fit_horizon + 2:
            return 0.0, 0.0
        distances = torch.cdist(points, points)
        positions = torch.arange(count)
        excluded = (positions[:, None] - positions[None, :]).abs() * stride <= self.theiler_window
        distances[excluded] = torch.inf
        nearest_distance, nearest = distances.min(dim=1)
        valid_neighbour = torch.isfinite(nearest_distance) & (nearest_distance > 1e-9)
        divergence = []
        for step in range(self.fit_horizon):
            valid = (
                valid_neighbour
                & (positions + step < count)
                & (nearest + step < count)
            )
            if int(valid.sum()) < 4:
                break
            distance = torch.linalg.vector_norm(
                points[positions[valid] + step] - points[nearest[valid] + step], dim=1
            ).clamp_min(1e-9)
            divergence.append(distance.log().mean())
        if len(divergence) < 3:
            return 0.0, 0.0
        y = torch.stack(divergence)
        selected: tuple[torch.Tensor, float] | None = None
        fallback = torch.tensor(0.0, dtype=y.dtype)
        for stop in range(4, len(y) + 1):
            candidate = y[:stop]
            x = torch.arange(stop, dtype=y.dtype) * stride
            x_centered = x - x.mean()
            slope = (
                (x_centered * (candidate - candidate.mean())).sum()
                / x_centered.square().sum().clamp_min(1e-12)
            )
            fitted = candidate.mean() + slope * x_centered
            residual = (candidate - fitted).square().sum()
            total = (candidate - candidate.mean()).square().sum().clamp_min(1e-12)
            r2 = float((1.0 - residual / total).clamp(0.0, 1.0))
            fallback = slope
            if torch.isfinite(slope) and float(slope) > 0.0 and r2 >= self.min_r2:
                # Prefer the longest credible initial divergence region.
                selected = slope, r2
        if selected is None:
            return float(fallback) if torch.isfinite(fallback) else 0.0, 0.0
        return float(selected[0]), selected[1]

    @torch.no_grad()
    def estimate(self, routing_windows: torch.Tensor) -> torch.Tensor:
        self._validate_windows(routing_windows)
        if self.feature_index >= routing_windows.shape[-1]:
            raise ValueError("feature_index is outside the routing feature dimension")
        estimates = [
            self._score_window(
                window[-self.context_length :, self.feature_index].detach().cpu()
                if self.context_length is not None
                else window[:, self.feature_index].detach().cpu()
            )
            for window in routing_windows
        ]
        return torch.tensor(estimates, dtype=routing_windows.dtype, device=routing_windows.device)


def build_hydrology_complexity_estimator(
    kind: str,
    feature_names: tuple[str, ...],
    **kwargs: object,
) -> HydrologyComplexityEstimator:
    if "previous_discharge" not in feature_names:
        raise ValueError("previous_discharge is required for dynamical complexity routing")
    feature_index = feature_names.index("previous_discharge")
    if kind == "tda":
        return TakensPersistenceEstimator(feature_index, **kwargs)
    if kind == "lyapunov":
        return LyapunovComplexityEstimator(feature_index, **kwargs)
    raise ValueError(f"no time-series complexity estimator is registered for {kind!r}")


__all__ = [
    "HydrologyComplexityEstimator",
    "LyapunovComplexityEstimator",
    "MODEL_NAMES",
    "ROUTING_METHODS",
    "SCORE_ROUTING_METHODS",
    "TakensPersistenceEstimator",
    "build_hydrology_complexity_estimator",
    "takens_delay_embedding",
]
