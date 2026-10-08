from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .experts import MLPExpert, build_expert
from .pinnmamba import PINNMamba
from .hydrology_complexity import SCORE_ROUTING_METHODS
from .routing import LearnedRouter, MorseRouter, ScoreRouter


@dataclass
class HydrologyLosses:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    interface: torch.Tensor
    routing: torch.Tensor
    complex_usage: torch.Tensor


@dataclass
class HydrologyRoutedOutput:
    discharge: torch.Tensor
    complex_weight: torch.Tensor
    simple_raw: torch.Tensor
    complex_raw: torch.Tensor


def hydrology_morse_gradient(
    flat_inputs: torch.Tensor,
    *,
    sequence_length: int,
    feature_names: Sequence[str],
    difference_weight: float = 0.5,
) -> torch.Tensor:
    """Gradient of a wetness/flow potential over a forcing-history window.

    The potential combines squared effective precipitation, antecedent flow,
    and first differences of both signals. Inputs are training-standardized,
    so neither variable dominates solely because of its physical units.
    """
    feature_count = len(feature_names)
    if flat_inputs.ndim != 2 or flat_inputs.shape[1] != sequence_length * feature_count:
        raise ValueError("flat hydrology inputs have an unexpected shape")
    required = {"precipitation", "pet", "previous_discharge"}
    if not required.issubset(feature_names):
        raise ValueError(f"hydrology features must contain {sorted(required)}")
    values = flat_inputs.reshape(-1, sequence_length, feature_count)
    gradient = torch.zeros_like(values)
    precipitation_index = feature_names.index("precipitation")
    pet_index = feature_names.index("pet")
    flow_index = feature_names.index("previous_discharge")
    recency = torch.linspace(0.25, 1.0, sequence_length, device=values.device, dtype=values.dtype)

    wetness = values[:, :, precipitation_index] - values[:, :, pet_index]
    gradient[:, :, precipitation_index] += recency * wetness
    gradient[:, :, pet_index] -= recency * wetness
    gradient[:, :, flow_index] += recency * values[:, :, flow_index]
    for index in (precipitation_index, flow_index):
        differences = values[:, 1:, index] - values[:, :-1, index]
        gradient[:, 1:, index] += difference_weight * differences
        gradient[:, :-1, index] -= difference_weight * differences
    return gradient.reshape_as(flat_inputs)


class HydrologyPINNMambaExpert(nn.Module):
    """Sequence adapter retaining the existing pure-PyTorch PINNMamba core."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_dim: int = 16,
        num_layers: int = 1,
        hidden_d_ff: int = 64,
        heads: int = 2,
        static_dim: int = 0,
        static_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.static_embedding = (
            nn.Sequential(nn.Linear(static_dim, static_embedding_dim), nn.Tanh())
            if static_dim else None
        )
        self.model = PINNMamba(
            in_dim=feature_count + (static_embedding_dim if static_dim else 0),
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hidden_d_ff=hidden_d_ff,
            heads=heads,
        )

    def forward(self, inputs: torch.Tensor, static_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if self.static_embedding is not None:
            if static_inputs is None:
                raise ValueError("static_inputs are required by this regional expert")
            context = self.static_embedding(static_inputs)[:, None, :].expand(
                -1, inputs.shape[1], -1
            )
            inputs = torch.cat((inputs, context), dim=-1)
        return self.model(inputs)[:, -1]


class LearnedHydrologyMorsePotential(nn.Module):
    """Small scalar potential learned before the final ensemble is trained."""

    def __init__(self, sequence_length: int, feature_names: Sequence[str], hidden_dim: int = 24) -> None:
        super().__init__()
        if sequence_length < 2 or hidden_dim < 1:
            raise ValueError("sequence_length must be at least two and hidden_dim must be positive")
        self.sequence_length = sequence_length
        self.feature_names = tuple(feature_names)
        summary_dim = 6 * len(feature_names)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def summaries(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        expected = self.sequence_length * len(self.feature_names)
        if flat_inputs.ndim != 2 or flat_inputs.shape[1] != expected:
            raise ValueError(f"expected flattened hydrology windows with {expected} features")
        values = flat_inputs.reshape(-1, self.sequence_length, len(self.feature_names))
        last_three = min(3, self.sequence_length)
        last_seven = min(7, self.sequence_length)
        return torch.cat(
            (
                values[:, -1],
                values.mean(dim=1),
                ((values - values.mean(dim=1, keepdim=True)).square().mean(dim=1) + 1e-6).sqrt(),
                values[:, -1] - values[:, 0],
                values[:, -last_three:].mean(dim=1),
                values[:, -last_seven:].mean(dim=1),
            ),
            dim=-1,
        )

    def forward(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        return self.net(self.summaries(flat_inputs)).squeeze(-1)

    def gradient(self, flat_inputs: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        with torch.enable_grad():
            differentiable = flat_inputs
            if not differentiable.requires_grad:
                differentiable = flat_inputs.detach().clone().requires_grad_(True)
            potential = self(differentiable)
            gradient = torch.autograd.grad(
                potential.sum(), differentiable, create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
        return gradient

    def complexity(self, flat_inputs: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self.gradient(flat_inputs, create_graph=create_graph), dim=-1
        )

    def freeze(self) -> "LearnedHydrologyMorsePotential":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class LearnedHydrologyMorseRouter(MorseRouter):
    """Standard Morse router backed by a pretrained, frozen scalar potential."""

    def __init__(
        self,
        potential: LearnedHydrologyMorsePotential,
        percentile: float = 80.0,
        temperature: float = 0.15,
    ) -> None:
        super().__init__(lambda inputs: inputs, percentile, temperature)
        self.potential = potential.freeze()
        self.morse_gradient = self.potential.gradient


class _FlattenedExpert(nn.Module):
    def __init__(self, expert: nn.Module, static_dim: int = 0) -> None:
        super().__init__()
        self.expert = expert
        self.static_dim = static_dim

    def forward(self, inputs: torch.Tensor, static_inputs: torch.Tensor | None = None) -> torch.Tensor:
        flattened = inputs.flatten(start_dim=1)
        if self.static_dim:
            if static_inputs is None:
                raise ValueError("static_inputs are required by this regional expert")
            flattened = torch.cat((flattened, static_inputs), dim=-1)
        return self.expert(flattened)


def build_hydrology_complex_expert(
    kind: str,
    sequence_length: int,
    feature_count: int,
    kwargs: dict[str, object] | None = None,
    static_dim: int = 0,
) -> nn.Module:
    kwargs = dict(kwargs or {})
    if kind == "mlp":
        kwargs.setdefault("hidden_dims", (64, 64))
        return _FlattenedExpert(
            MLPExpert(sequence_length * feature_count + static_dim, 1, **kwargs), static_dim
        )
    if kind == "pinnmamba":
        return HydrologyPINNMambaExpert(feature_count, static_dim=static_dim, **kwargs)
    raise ValueError("complex_kind must be 'mlp' or 'pinnmamba'")


class _HydrologyModelBase(nn.Module):
    def __init__(self, discharge_scale: float | torch.Tensor, feature_names: Sequence[str]) -> None:
        super().__init__()
        self.feature_names = tuple(feature_names)
        self.register_buffer("discharge_scale", torch.as_tensor(discharge_scale).reshape(()).float())
        self.raw_response = nn.Parameter(torch.tensor(0.0))
        self.raw_recession = nn.Parameter(torch.tensor(-2.0))
        self.physics_correction: nn.Module | None = None

    def set_physics_correction(self, correction: nn.Module | None) -> None:
        """Attach a frozen, data-derived correction to the reservoir balance."""
        self.physics_correction = correction
        if correction is not None:
            correction.requires_grad_(False)

    def _positive_discharge(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def reservoir_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return 2.0 * torch.sigmoid(self.raw_response), torch.sigmoid(self.raw_recession)

    def physics_residual(self, prediction: torch.Tensor, physical_inputs: torch.Tensor) -> torch.Tensor:
        precipitation = physical_inputs[:, -1, self.feature_names.index("precipitation")]
        pet = physical_inputs[:, -1, self.feature_names.index("pet")]
        previous_flow = physical_inputs[:, -1, self.feature_names.index("previous_discharge")]
        response, recession = self.reservoir_parameters()
        effective_precipitation = F.relu(precipitation - pet)
        expected_change = response * effective_precipitation - recession * previous_flow
        if self.physics_correction is not None:
            expected_change = expected_change + self.physics_correction(physical_inputs)
        predicted_flow = prediction.squeeze(-1) * self.discharge_scale
        return (predicted_flow - previous_flow - expected_change) / self.discharge_scale


class RoutedHydrologyModel(_HydrologyModelBase):
    """RBF/Fourier + MLP/PINNMamba rainfall-runoff ensemble."""

    def __init__(
        self,
        feature_names: Sequence[str],
        sequence_length: int,
        discharge_scale: float | torch.Tensor,
        *,
        simple_kind: str = "rbf",
        complex_kind: str = "mlp",
        routing: str = "morse",
        complexity_percentile: float = 80.0,
        gate_temperature: float = 0.15,
        simple_kwargs: dict[str, object] | None = None,
        complex_kwargs: dict[str, object] | None = None,
        learned_morse_potential: LearnedHydrologyMorsePotential | None = None,
        static_dim: int = 0,
    ) -> None:
        super().__init__(discharge_scale, feature_names)
        feature_count = len(feature_names)
        flat_dim = sequence_length * feature_count
        expert_dim = flat_dim + static_dim
        simple_kwargs = dict(simple_kwargs or {})
        if simple_kind == "rbf":
            simple_kwargs.setdefault("n_centers", 32)
            simple_kwargs.setdefault("domain_low", [-2.0] * expert_dim)
            simple_kwargs.setdefault("domain_high", [2.0] * expert_dim)
            simple_kwargs.setdefault("initial_width", math.sqrt(expert_dim))
        elif simple_kind == "fourier":
            simple_kwargs.setdefault("n_frequencies", 32)
            simple_kwargs.setdefault("frequency_scale", 1.0 / math.sqrt(expert_dim))
        self.simple_expert = _FlattenedExpert(
            build_expert(simple_kind, expert_dim, 1, **simple_kwargs), static_dim
        )
        self.complex_expert = build_hydrology_complex_expert(
            complex_kind, sequence_length, feature_count, complex_kwargs, static_dim
        )
        if routing == "morse":
            gradient = partial(
                hydrology_morse_gradient,
                sequence_length=sequence_length,
                feature_names=self.feature_names,
            )
            self.router: MorseRouter | LearnedRouter = MorseRouter(
                gradient, complexity_percentile, gate_temperature
            )
        elif routing == "learned":
            self.router = LearnedRouter(
                flat_dim, target_complex_fraction=1.0 - complexity_percentile / 100.0
            )
        elif routing == "learned_morse":
            if learned_morse_potential is None:
                raise ValueError("routing='learned_morse' requires a pretrained potential")
            if (
                learned_morse_potential.sequence_length != sequence_length
                or learned_morse_potential.feature_names != self.feature_names
            ):
                raise ValueError("learned Morse potential does not match the hydrology inputs")
            self.router = LearnedHydrologyMorseRouter(
                learned_morse_potential, complexity_percentile, gate_temperature
            )
        elif routing in SCORE_ROUTING_METHODS:
            self.router = ScoreRouter(complexity_percentile, gate_temperature)
        else:
            choices = ", ".join(("morse", "learned_morse", "learned", *sorted(SCORE_ROUTING_METHODS)))
            raise ValueError(f"routing must be one of: {choices}")
        self.simple_kind = simple_kind
        self.complex_kind = complex_kind
        self.routing_kind = routing
        self.static_dim = static_dim

    def _router_features(
        self,
        inputs: torch.Tensor,
        routing_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.routing_kind in SCORE_ROUTING_METHODS:
            if routing_features is None:
                raise ValueError(f"routing_features are required for {self.routing_kind} routing")
            return routing_features
        return inputs.flatten(start_dim=1)

    def fit_router(
        self,
        inputs: torch.Tensor,
        routing_features: torch.Tensor | None = None,
    ) -> "RoutedHydrologyModel":
        self.router.fit(self._router_features(inputs, routing_features))
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        routing_features: torch.Tensor | None = None,
        hard: bool = False,
        return_details: bool = False,
        static_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor | HydrologyRoutedOutput:
        router_features = self._router_features(inputs, routing_features)
        weight = self.router.complex_weight(router_features, hard=hard)
        simple = self.simple_expert(inputs, static_inputs)
        complex_value = self.complex_expert(inputs, static_inputs)
        discharge = self._positive_discharge(torch.lerp(simple, complex_value, weight[:, None]))
        if return_details:
            return HydrologyRoutedOutput(discharge, weight, simple, complex_value)
        return discharge

    def losses(
        self,
        inputs: torch.Tensor,
        physical_inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        routing_features: torch.Tensor | None = None,
        physics_weight: float = 0.05,
        interface_weight: float = 0.05,
        routing_weight: float = 0.05,
        compute_weight: float = 0.0,
        static_inputs: torch.Tensor | None = None,
    ) -> HydrologyLosses:
        routed = self(
            inputs, routing_features=routing_features, return_details=True,
            static_inputs=static_inputs,
        )
        assert isinstance(routed, HydrologyRoutedOutput)
        data = (routed.discharge.squeeze(-1) - targets).square().mean()
        physics = self.physics_residual(routed.discharge, physical_inputs).square().mean()
        transition = 4.0 * routed.complex_weight * (1.0 - routed.complex_weight)
        interface = (transition * (routed.simple_raw - routed.complex_raw).square().squeeze(-1)).mean()
        router_features = self._router_features(inputs, routing_features)
        routing = self.router.regularization(router_features)
        complex_usage = routed.complex_weight.mean()
        total = (
            data + physics_weight * physics + interface_weight * interface
            + routing_weight * routing + compute_weight * complex_usage
        )
        return HydrologyLosses(total, data, physics, interface, routing, complex_usage)


class SingleComplexHydrologyModel(_HydrologyModelBase):
    """Unitary baseline using exactly the selected complex architecture."""

    def __init__(
        self,
        feature_names: Sequence[str],
        sequence_length: int,
        discharge_scale: float | torch.Tensor,
        *,
        complex_kind: str = "mlp",
        complex_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(discharge_scale, feature_names)
        self.complex_expert = build_hydrology_complex_expert(
            complex_kind, sequence_length, len(feature_names), complex_kwargs
        )
        self.complex_kind = complex_kind

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._positive_discharge(self.complex_expert(inputs))

    def losses(
        self,
        inputs: torch.Tensor,
        physical_inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        physics_weight: float = 0.05,
    ) -> HydrologyLosses:
        prediction = self(inputs)
        data = (prediction.squeeze(-1) - targets).square().mean()
        physics = self.physics_residual(prediction, physical_inputs).square().mean()
        zero = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        return HydrologyLosses(data + physics_weight * physics, data, physics, zero, zero, zero + 1.0)


class SingleSimpleHydrologyModel(_HydrologyModelBase):
    """Standalone simple expert used only to create cross-fitted routing labels."""

    def __init__(
        self,
        feature_names: Sequence[str],
        sequence_length: int,
        discharge_scale: float | torch.Tensor,
        *,
        simple_kind: str = "rbf",
        simple_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(discharge_scale, feature_names)
        flat_dim = sequence_length * len(feature_names)
        simple_kwargs = dict(simple_kwargs or {})
        if simple_kind == "rbf":
            simple_kwargs.setdefault("n_centers", 32)
            simple_kwargs.setdefault("domain_low", [-2.0] * flat_dim)
            simple_kwargs.setdefault("domain_high", [2.0] * flat_dim)
            simple_kwargs.setdefault("initial_width", math.sqrt(flat_dim))
        elif simple_kind == "fourier":
            simple_kwargs.setdefault("n_frequencies", 32)
            simple_kwargs.setdefault("frequency_scale", 1.0 / math.sqrt(flat_dim))
        self.simple_expert = _FlattenedExpert(
            build_expert(simple_kind, flat_dim, 1, **simple_kwargs)
        )
        self.simple_kind = simple_kind

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._positive_discharge(self.simple_expert(inputs))

    def losses(
        self,
        inputs: torch.Tensor,
        physical_inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        physics_weight: float = 0.05,
    ) -> HydrologyLosses:
        prediction = self(inputs)
        data = (prediction.squeeze(-1) - targets).square().mean()
        physics = self.physics_residual(prediction, physical_inputs).square().mean()
        zero = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        return HydrologyLosses(data + physics_weight * physics, data, physics, zero, zero, zero)
