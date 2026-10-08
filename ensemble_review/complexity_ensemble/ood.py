"""Small, dependency-free detectors used by fallback ensemble baselines."""

from __future__ import annotations

import torch


class MahalanobisOODDetector:
    """Empirical Mahalanobis detector with diagonal covariance shrinkage."""

    def __init__(self, quantile: float = 0.99, shrinkage: float = 0.1) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("OOD quantile must be strictly between zero and one")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("OOD shrinkage must be between zero and one")
        self.quantile = quantile
        self.shrinkage = shrinkage
        self.mean: torch.Tensor | None = None
        self.precision: torch.Tensor | None = None
        self.threshold: torch.Tensor | None = None

    @staticmethod
    def features(
        inputs: torch.Tensor,
        static_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim < 2 or len(inputs) == 0:
            raise ValueError("OOD inputs must be a non-empty batch")
        flattened = inputs.flatten(start_dim=1)
        if static_inputs is not None and static_inputs.shape[1]:
            if len(static_inputs) != len(inputs):
                raise ValueError("static OOD inputs must match the dynamic input batch")
            flattened = torch.cat((flattened, static_inputs), dim=1)
        return flattened

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.precision is not None and self.threshold is not None

    @torch.no_grad()
    def fit(
        self,
        inputs: torch.Tensor,
        static_inputs: torch.Tensor | None = None,
    ) -> "MahalanobisOODDetector":
        values = self.features(inputs, static_inputs).double()
        self.mean = values.mean(dim=0)
        centered = values - self.mean
        denominator = max(len(values) - 1, 1)
        covariance = centered.T @ centered / denominator
        diagonal = torch.diag(torch.diagonal(covariance).clamp_min(1e-6))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        covariance = covariance + torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        ) * 1e-6
        self.precision = torch.linalg.pinv(covariance)
        training_scores = self._score_features(values)
        self.threshold = torch.quantile(training_scores, self.quantile)
        return self

    def _score_features(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.precision is None:
            raise RuntimeError("fit the OOD detector before scoring inputs")
        centered = values.double() - self.mean.to(values.device)
        squared = torch.einsum(
            "bi,ij,bj->b", centered, self.precision.to(values.device), centered
        )
        return squared.clamp_min(0.0).sqrt().to(values.dtype)

    @torch.no_grad()
    def score(
        self,
        inputs: torch.Tensor,
        static_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._score_features(self.features(inputs, static_inputs))

    @torch.no_grad()
    def is_ood(
        self,
        inputs: torch.Tensor,
        static_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("fit the OOD detector before classifying inputs")
        scores = self.score(inputs, static_inputs)
        return scores > self.threshold.to(scores.device)


__all__ = ["MahalanobisOODDetector"]
