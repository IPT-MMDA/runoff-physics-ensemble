from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class WaveActivation(nn.Module):
    """Trainable sine/cosine activation used by the released PINNMamba."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w1 * torch.sin(inputs) + self.w2 * torch.cos(inputs)


class SelectiveSSM(nn.Module):
    """Small, pure-PyTorch selective state-space layer.

    The parameterization and sequential scan match the official ICML 2025
    implementation.  The scan is deliberately explicit: sequence length is
    only seven in the paper, and this version works on CPU without custom CUDA
    kernels while remaining differentiable with respect to PDE coordinates.
    """

    def __init__(self, in_features: int, dt_rank: int, dim_inner: int, d_state: int) -> None:
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.deltaBC_layer = nn.Linear(in_features, dt_rank + 2 * d_state)
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner)
        initial_a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(dim_inner, 1)
        self.A_log = nn.Parameter(torch.log(initial_a))
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(self.A_log.float())
        delta_bc = self.deltaBC_layer(inputs)
        delta, b, c = torch.split(
            delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        bx = delta.unsqueeze(-1) * b.unsqueeze(2) * inputs.unsqueeze(-1)

        state = torch.zeros(
            inputs.shape[0], self.dim_inner, self.d_state,
            device=inputs.device, dtype=inputs.dtype,
        )
        states = []
        for index in range(inputs.shape[1]):
            state = delta_a[:, index] * state + bx[:, index]
            states.append(state)
        stacked = torch.stack(states, dim=1)
        return (stacked @ c.unsqueeze(-1)).squeeze(-1) + self.D.float() * inputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int = 2) -> None:
        super().__init__()
        # `heads` is retained for checkpoint/API compatibility; the released
        # SSM layer does not use attention heads.
        del heads
        self.ssm = SelectiveSSM(d_model, dt_rank=8, dim_inner=d_model, d_state=8)
        self.act1 = WaveActivation()
        self.act2 = WaveActivation()
        self.z_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip = inputs
        z = self.act1(self.z_proj(inputs))
        x = self.x_proj(inputs).transpose(1, 2)
        x = self.softplus(self.conv1d(x)).transpose(1, 2)
        x = self.ssm(self.act2(x))
        return skip + self.out_proj(x * z)


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, heads: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d_model, heads) for _ in range(n_layers))
        self.act = WaveActivation()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return self.act(inputs)


class PINNMamba(nn.Module):
    """PINNMamba sequence model with the paper's default dimensions."""

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        hidden_d_ff: int = 512,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.linear_emb = nn.Linear(in_dim, hidden_dim)
        self.encoder = Encoder(hidden_dim, num_layers, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_d_ff),
            WaveActivation(),
            nn.Linear(hidden_d_ff, hidden_d_ff),
            WaveActivation(),
            nn.Linear(hidden_d_ff, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the initialization function in the released experiments."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(self, coordinates: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        if time is not None:
            coordinates = torch.cat((coordinates, time), dim=-1)
        expected_features = self.linear_emb.in_features
        if coordinates.ndim != 3 or coordinates.shape[-1] != expected_features:
            raise ValueError(
                "PINNMamba expects coordinates shaped "
                f"[batch, sequence, {expected_features}]"
            )
        encoded = self.encoder(self.linear_emb(coordinates))
        return self.linear_out(encoded)

    def load_official_checkpoint(self, path: str | Path) -> None:
        """Load a state dict released by miniHuiHui/PINNMamba."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PINNMambaExpert(nn.Module):
    """Adapt PINNMamba to the ensemble's ``(t, x) -> u`` expert boundary.

    Pointwise calls return the anchor prediction so existing evaluators remain
    usable. Sequence-aware PINN training calls :meth:`forward_sequence`.
    """

    def __init__(
        self,
        sequence_length: int = 7,
        sequence_step: float = 0.01,
        **model_kwargs: object,
    ) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if sequence_step <= 0.0:
            raise ValueError("sequence_step must be positive")
        self.sequence_length = sequence_length
        self.sequence_step = sequence_step
        self.model = PINNMamba(**model_kwargs)

    def make_tx_sequence(self, tx: torch.Tensor) -> torch.Tensor:
        if tx.ndim != 2 or tx.shape[-1] != 2:
            raise ValueError("tx must have shape [batch, 2]")
        sequence = tx[:, None, :].repeat(1, self.sequence_length, 1)
        offsets = (
            torch.arange(self.sequence_length, device=tx.device, dtype=tx.dtype)
            * self.sequence_step
        )
        sequence[:, :, 0] = sequence[:, :, 0] + offsets[None, :]
        return sequence

    def forward_tx_sequence(self, tx_sequence: torch.Tensor) -> torch.Tensor:
        if tx_sequence.ndim != 3 or tx_sequence.shape[-1] != 2:
            raise ValueError("tx_sequence must have shape [batch, sequence, 2]")
        return self.model(tx_sequence[..., [1, 0]])

    def forward_sequence(self, tx: torch.Tensor) -> torch.Tensor:
        return self.forward_tx_sequence(self.make_tx_sequence(tx))

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(tx)[:, 0]


def make_time_subsequences(coordinates: torch.Tensor, length: int = 7, step: float = 0.01) -> torch.Tensor:
    """Expand point coordinates ``(x,t)`` into forward time subsequences."""
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [n_points, 2]")
    sequence = coordinates[:, None, :].repeat(1, length, 1)
    offsets = torch.arange(length, device=coordinates.device, dtype=coordinates.dtype) * step
    sequence[:, :, 1] = sequence[:, :, 1] + offsets[None, :]
    return sequence


def reaction_initial_condition(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x - torch.pi) ** 2) / (2.0 * (torch.pi / 4.0) ** 2))


def reaction_solution(coordinates: torch.Tensor) -> torch.Tensor:
    x, t = coordinates[..., 0:1], coordinates[..., 1:2]
    initial = reaction_initial_condition(x)
    growth = torch.exp(5.0 * t)
    return initial * growth / (initial * growth + 1.0 - initial)


@dataclass
class ReactionGrid:
    coordinates: torch.Tensor
    sequences: torch.Tensor
    n_time: int
    n_space: int
    time_step: float
    sequence_step: float


def make_reaction_grid(
    n_space: int = 101,
    n_time: int = 101,
    sequence_length: int = 7,
    sequence_step: float = 0.01,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ReactionGrid:
    x = torch.linspace(0.0, 2.0 * torch.pi, n_space, device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, n_time, device=device, dtype=dtype)
    t_grid, x_grid = torch.meshgrid(t, x, indexing="ij")
    coordinates = torch.stack((x_grid.flatten(), t_grid.flatten()), dim=-1)
    return ReactionGrid(
        coordinates=coordinates,
        sequences=make_time_subsequences(coordinates, sequence_length, sequence_step),
        n_time=n_time,
        n_space=n_space,
        time_step=1.0 / (n_time - 1),
        sequence_step=sequence_step,
    )


@dataclass
class PINNMambaLosses:
    total: torch.Tensor
    physics: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor
    alignment: torch.Tensor


def reaction_losses(
    model: PINNMamba,
    grid: ReactionGrid,
    *,
    physics_weight: float = 1.0,
    initial_weight: float = 1.0,
    boundary_weight: float = 1.0,
    alignment_weight: float = 1000.0,
) -> PINNMambaLosses:
    sequences = grid.sequences.detach().clone().requires_grad_(True)
    x, t = sequences[..., 0:1], sequences[..., 1:2]
    prediction = model(x, t)
    u_t = torch.autograd.grad(
        prediction, t, torch.ones_like(prediction), retain_graph=True, create_graph=True
    )[0]
    physics = (u_t - 5.0 * prediction * (1.0 - prediction)).square().mean()

    shaped = prediction.reshape(grid.n_time, grid.n_space, prediction.shape[1], 1)
    initial_x = sequences[: grid.n_space, 0, 0:1]
    initial = (shaped[0, :, 0] - reaction_initial_condition(initial_x)).square().mean()
    boundary = (shaped[:, -1] - shaped[:, 0]).square().mean()

    shift_float = grid.time_step / grid.sequence_step
    shift = round(shift_float)
    if shift < 1 or abs(shift_float - shift) > 1e-6 or shift >= prediction.shape[1]:
        raise ValueError("grid and sequence steps must have an integer overlap within the sequence")
    alignment = (shaped[1:, :, :-shift] - shaped[:-1, :, shift:]).square().mean()
    total = (
        physics_weight * physics
        + initial_weight * initial
        + boundary_weight * boundary
        + alignment_weight * alignment
    )
    return PINNMambaLosses(total, physics, initial, boundary, alignment)


@dataclass
class PINNMambaMetrics:
    relative_mae: float
    relative_rmse: float
    mse: float
    mae: float
    max_absolute_error: float
    parameter_count: int
    paper_relative_mae: float = 0.0094
    paper_relative_rmse: float = 0.0217


@dataclass
class PINNMambaEvaluation:
    metrics: PINNMambaMetrics
    x_grid: torch.Tensor
    t_grid: torch.Tensor
    prediction: torch.Tensor
    reference: torch.Tensor
    absolute_error: torch.Tensor


def evaluate_reaction_pinnmamba(
    model: PINNMamba, n_space: int = 101, n_time: int = 101, sequence_step: float = 0.01
) -> PINNMambaEvaluation:
    parameter = next(model.parameters())
    grid = make_reaction_grid(
        n_space, n_time, sequence_length=7, sequence_step=sequence_step,
        device=parameter.device, dtype=parameter.dtype,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(grid.sequences)[:, 0]
        reference = reaction_solution(grid.coordinates)
        difference = prediction - reference
        absolute = difference.abs()
        relative_mae = absolute.sum() / reference.abs().sum().clamp_min(1e-12)
        relative_rmse = torch.sqrt(difference.square().sum() / reference.square().sum().clamp_min(1e-12))
    shape = (n_time, n_space)
    x = grid.coordinates[:, 0].reshape(shape)
    t = grid.coordinates[:, 1].reshape(shape)
    metrics = PINNMambaMetrics(
        relative_mae=float(relative_mae),
        relative_rmse=float(relative_rmse),
        mse=float(difference.square().mean()),
        mae=float(absolute.mean()),
        max_absolute_error=float(absolute.max()),
        parameter_count=model.parameter_count,
    )
    return PINNMambaEvaluation(
        metrics,
        x.detach().cpu(),
        t.detach().cpu(),
        prediction.reshape(shape).detach().cpu(),
        reference.reshape(shape).detach().cpu(),
        absolute.reshape(shape).detach().cpu(),
    )
