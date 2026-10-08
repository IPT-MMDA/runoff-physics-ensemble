from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn


class MLPExpert(nn.Module):
    """The high-capacity expert. Its interface can later be backed by an SSM."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.Tanh()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class RBFExpert(nn.Module):
    """Vectorized, fully differentiable radial-basis expert."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_centers: int = 64,
        domain_low: Sequence[float] | None = None,
        domain_high: Sequence[float] | None = None,
        min_width: float = 1e-3,
        initial_width: float = 1.0,
    ) -> None:
        super().__init__()
        if initial_width <= 0.0:
            raise ValueError("initial_width must be positive")
        low = torch.as_tensor(domain_low or [-1.0] * input_dim, dtype=torch.float32)
        high = torch.as_tensor(domain_high or [1.0] * input_dim, dtype=torch.float32)
        if low.shape != (input_dim,) or high.shape != (input_dim,):
            raise ValueError("domain bounds must have one value per input dimension")
        centers = low + (high - low) * torch.rand(n_centers, input_dim)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(torch.full((n_centers,), math.log(initial_width)))
        self.linear = nn.Linear(n_centers, output_dim)
        self.min_width = min_width

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        squared_distance = (inputs[:, None, :] - self.centers[None, :, :]).square().sum(-1)
        widths = self.log_widths.exp().clamp_min(self.min_width)
        return torch.exp(-0.5 * squared_distance / widths.square()[None, :])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(inputs))


class FourierExpert(nn.Module):
    """Shallow Fourier-feature expert with fixed or trainable frequencies."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_frequencies: int = 64,
        frequency_scale: float = 2.0,
        learnable_frequencies: bool = True,
    ) -> None:
        super().__init__()
        frequencies = torch.randn(input_dim, n_frequencies) * frequency_scale
        if learnable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.linear = nn.Linear(2 * n_frequencies, output_dim)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        phase = inputs @ self.frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(inputs))


def build_expert(kind: str, input_dim: int, output_dim: int, **kwargs: object) -> nn.Module:
    """Factory kept deliberately small so an ``ssm`` kind can be added later."""
    expert_types = {"mlp": MLPExpert, "rbf": RBFExpert, "fourier": FourierExpert}
    try:
        expert_type = expert_types[kind.lower()]
    except KeyError as error:
        raise ValueError(f"unknown expert kind {kind!r}; choose from {sorted(expert_types)}") from error
    return expert_type(input_dim=input_dim, output_dim=output_dim, **kwargs)
