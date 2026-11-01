"""Source-only symbolic flux distillation and low-data catchment adaptation.

This module deliberately does not use regional experiment KAN corrections.
States are simulated continuously; measured discharge is only a training label.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
import math
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .devices import DEVICE_CHOICES, resolve_device

COMPARISON_MODELS = tuple(
    f"{architecture}_{scope}_target"
    for architecture in ("pinnmamba", "mlp", "lstm")
    for scope in ("source", "target", "mixed")
) + ("kan_pure", "kan_tuned")
DEFAULT_COMPARISON_MODELS = COMPARISON_MODELS
COMPARISON_MODELS += ("kan_complex",)
BASE_SYMBOLIC_LIBRARY = ("x", "x^2", "tanh")
COMPLEX_SYMBOLIC_LIBRARY = (*BASE_SYMBOLIC_LIBRARY, "x^3", "abs")
LOCALLY_CALIBRATED_KAN_MODELS = ("kan_tuned", "kan_complex")


@dataclass(frozen=True)
class CaseStudyConfig:
    source_basins: tuple[str, ...] = ()
    validation_basins: tuple[str, ...] = ()
    target_basins: tuple[str, ...] = ()
    train_start: str = "2010-01-01"
    train_end: str = "2016-12-31"
    validation_start: str = "2017-01-01"
    validation_end: str = "2018-12-31"
    test_start: str = "2019-01-01"
    test_end: str = "2020-12-31"
    observation_budgets: tuple[int, ...] = (30, 90, 180)
    target_observation_budgets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    seeds: tuple[int, ...] = (0,)
    warmup_days: int = 365
    minimum_source_days: int = 3
    teacher_epochs: int = 20
    adaptation_epochs: int = 30
    teacher_width: int = 12
    learning_rate: float = 0.01
    kan_candidates: int = 3
    kan_steps: int = 50
    kan_grid: int = 3
    kan_hidden: int = 2
    max_formula_ops: int = 150
    max_distillation_samples_per_basin: int = 512
    sparsity_weight: float = 0.001
    device: str = "auto"
    selected_models: tuple[str, ...] = DEFAULT_COMPARISON_MODELS
    neural_epochs: int = 20
    sequence_length: int = 30
    neural_batch_size: int = 64
    mlp_widths: tuple[int, ...] = (64, 64)
    mamba_hidden_size: int = 16
    mamba_layers: int = 1
    mamba_feed_forward_size: int = 64
    lstm_hidden_size: int = 32
    lstm_layers: int = 1

    def validate(self, *, require_cohorts=True):
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"Device must be one of {DEVICE_CHOICES}")
        if (
            not self.selected_models
            or len(set(self.selected_models)) != len(self.selected_models)
            or set(self.selected_models) - set(COMPARISON_MODELS)
        ):
            raise ValueError(
                "Select at least one unique comparison model from the supported variants"
            )
        cohorts = [self.source_basins, self.validation_basins, self.target_basins]
        required = (self.needs_sources(), self.needs_kan(), True)
        if require_cohorts and any(
            (needed and not c) or len(c) != len(set(c)) for c, needed in zip(cohorts, required)
        ):
            raise ValueError(
                "Choose nonempty required cohorts without duplicates: targets always, sources for source/mixed/KAN models, source-validation for KAN"
            )
        if require_cohorts and any(
            set(cohorts[i]) & set(cohorts[j]) for i in range(3) for j in range(i)
        ):
            raise ValueError("Source, source-validation and target basins must be disjoint")
        days = [
            date.fromisoformat(getattr(self, name))
            for name in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        ]
        if not (days[0] <= days[1] < days[2] <= days[3] < days[4] <= days[5]):
            raise ValueError(
                "Training, validation and test periods must be chronological and disjoint"
            )
        positive = (
            "warmup_days",
            "minimum_source_days",
            "teacher_epochs",
            "adaptation_epochs",
            "teacher_width",
            "kan_candidates",
            "kan_steps",
            "kan_grid",
            "kan_hidden",
            "max_formula_ops",
            "max_distillation_samples_per_basin",
            "neural_epochs",
            "sequence_length",
            "neural_batch_size",
            "mamba_hidden_size",
            "mamba_layers",
            "mamba_feed_forward_size",
            "lstm_hidden_size",
            "lstm_layers",
        )
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in positive):
            raise ValueError(
                "Step counts, widths, sample limits and warm-up must be positive integers"
            )
        if not self.mlp_widths or any(type(v) is not int or v < 1 for v in self.mlp_widths):
            raise ValueError("MLP widths must be positive integers")
        if self.needs_neural() and self.warmup_days < self.sequence_length - 1:
            raise ValueError("Warm-up must cover sequence_length minus one weather days")
        if not self.observation_budgets or any(
            type(v) is not int or v < 3 for v in self.observation_budgets
        ):
            raise ValueError("Observation budgets must contain integers of at least three days")
        if tuple(sorted(set(self.observation_budgets))) != self.observation_budgets:
            raise ValueError("Observation budgets must be unique and increasing")
        if self.minimum_source_days < 3:
            raise ValueError("Source fitting requires at least three observations")
        if require_cohorts and set(self.target_observation_budgets) - set(self.target_basins):
            raise ValueError("Custom observation budgets must refer to selected target catchments")
        for budgets in self.target_observation_budgets.values():
            if not budgets or any(type(v) is not int or v < 3 for v in budgets):
                raise ValueError(
                    "Each target budget must contain integers of at least three observations"
                )
            if tuple(sorted(set(budgets))) != tuple(budgets):
                raise ValueError("Each target's budgets must be unique and increasing")
        if (
            not self.seeds
            or any(type(v) is not int or v < 0 for v in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("Choose unique nonnegative integer seeds")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if not math.isfinite(self.sparsity_weight) or self.sparsity_weight < 0:
            raise ValueError("Sparsity weight must be finite and nonnegative")

    def budgets_for(self, basin):
        return self.target_observation_budgets.get(basin, self.observation_budgets)

    def needs_kan(self):
        return any(m.startswith("kan_") for m in self.selected_models)

    def needs_neural(self):
        return any(not m.startswith("kan_") for m in self.selected_models)

    def needs_sources(self):
        return self.needs_kan() or any(
            "_source_" in m or "_mixed_" in m for m in self.selected_models
        )

    def needs_target_labels(self):
        return any(
            m in LOCALLY_CALIBRATED_KAN_MODELS or "_target_target" in m or "_mixed_" in m
            for m in self.selected_models
        )

    def budget_scenarios(self):
        count = max(len(self.budgets_for(b)) for b in self.target_basins)
        return [
            {
                b: self.budgets_for(b)[min(i, len(self.budgets_for(b)) - 1)]
                for b in self.target_basins
            }
            for i in range(count)
        ]


def config_from_dict(value, *, require_cohorts=True):
    value = dict(value)
    for key in (
        "source_basins",
        "validation_basins",
        "target_basins",
        "observation_budgets",
        "seeds",
        "selected_models",
        "mlp_widths",
    ):
        if key in value:
            value[key] = tuple(value[key])
    if "target_observation_budgets" in value:
        value["target_observation_budgets"] = {
            b: tuple(v) for b, v in value["target_observation_budgets"].items()
        }
    config = CaseStudyConfig(**value)
    config.validate(require_cohorts=require_cohorts)
    return config


class StudyCancelled(RuntimeError):
    pass


def _check_cancel(token):
    if token is not None and token.is_cancelled():
        raise StudyCancelled("Case study cancelled")


class BasinParameters(nn.Module):
    """Exactly three locally trainable coefficients, with physical bounds."""

    def __init__(self, initial=None):
        super().__init__()
        self.raw = nn.Parameter(
            torch.tensor(initial if initial is not None else [0.0, -1.0, 0.0], dtype=torch.float32)
        )

    def values(self):
        capacity = 10.0 + 990.0 * torch.sigmoid(self.raw[0])
        melt = 0.1 + 9.9 * torch.sigmoid(self.raw[1])
        offset = 4.0 * torch.tanh(self.raw[2])
        return capacity, melt, offset

    def report(self):
        return dict(
            zip(
                ("capacity_mm", "melt_mm_day_c", "runoff_logit_offset"),
                (float(v.detach()) for v in self.values()),
            )
        )


class NeuralFlux(nn.Module):
    """Learn the logit of the daily water release fraction, not discharge."""

    def __init__(self, width=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, 1))
        with torch.no_grad():
            output_layer = self.net[-1]
            assert isinstance(output_layer, nn.Linear) and output_layer.bias is not None
            output_layer.bias.fill_(-2.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReservoirFlux(nn.Module):
    def forward(self, x):
        return x[..., 0] * 0.0 - 2.0


class SymbolicFlux(nn.Module):
    """Execute the exported SymPy tree with torch, retaining state gradients.

    Expressions are generated internally by PyKAN, never evaluated from API text.
    No Python eval/lambdify is used for execution.
    """

    def __init__(self, expression):
        super().__init__()
        import sympy as sp

        self.expression = expression
        self.symbols = sp.symbols("storage liquid pet")
        variables = {symbol: i for i, symbol in enumerate(self.symbols)}

        def compile_node(node):
            if node in variables:
                index = variables[node]
                return lambda x: x[..., index]
            if node.is_Number:
                value = float(node)
                return lambda x: x.new_tensor(value)
            args = [compile_node(arg) for arg in node.args]
            if node.func == sp.Add:
                return lambda x: sum(arg(x) for arg in args)
            if node.func == sp.Mul:

                def multiply(x):
                    result = x.new_tensor(1.0)
                    for arg in args:
                        result = result * arg(x)
                    return result

                return multiply
            if node.func == sp.Pow:
                return lambda x: args[0](x) ** args[1](x)
            if node.func == sp.tanh:
                return lambda x: torch.tanh(args[0](x))
            if node.func == sp.Abs:
                return lambda x: torch.abs(args[0](x))
            raise ValueError(f"Unsupported symbolic operation: {node.func}")

        self._compiled = compile_node(expression)

    def forward(self, x):
        return self._compiled(x) + x[..., 0] * 0.0


def simulate(forcing, flux, parameters, *, initial_state=None, token=None):
    """Daily conservative snow/water bucket; forcing columns are P, T, PET.

    Q = overflow + retained_water * sigmoid(F(S/C, liquid/C, PET/C) + offset).
    Evaporation and overflow precede release. State is never inferred from Qobs.
    """
    capacity, melt_factor, offset = parameters.values()
    snow, water = (
        initial_state
        if initial_state is not None
        else (forcing.new_zeros(()), forcing.new_zeros(()))
    )
    rows, features = [], []
    for i, (rain, temperature, pet) in enumerate(forcing):
        if i % 64 == 0:
            _check_cancel(token)
        old_total = snow + water
        snowfall = rain * (temperature < 0).to(rain.dtype)
        snow = snow + snowfall
        melt = torch.minimum(snow, melt_factor * torch.relu(temperature))
        snow = snow - melt
        liquid = rain - snowfall + melt
        x = torch.stack((water / capacity, liquid / capacity, pet / capacity))
        available = water + liquid
        evaporation = torch.minimum(available, pet)
        available = available - evaporation
        overflow = torch.relu(available - capacity)
        retained = torch.minimum(available, capacity)
        fraction = torch.sigmoid(flux(x[None]).reshape(()) + offset)
        runoff = overflow + retained * fraction
        water = retained * (1.0 - fraction)
        balance = old_total + rain - evaporation - runoff - snow - water
        rows.append(torch.stack((runoff, evaporation, snow, water, balance)))
        features.append(x)
    if not rows:
        raise ValueError("No forcing records to simulate")
    return torch.stack(rows), torch.stack(features), (snow, water)


@dataclass
class BasinData:
    dates: tuple[date, ...]
    forcing: torch.Tensor
    observed: torch.Tensor
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor

    def to(self, device):
        return replace(
            self,
            **{
                name: getattr(self, name).to(device)
                for name in ("forcing", "observed", "train", "validation", "test")
            },
        )


def case_study_hardware(requested):
    device = resolve_device(requested)
    return {
        "requested": requested,
        "resolved": str(device),
        "teacher": str(device),
        "calibration": str(device),
        "simulation": str(device),
        "kan_fitting": str(device) if device.type == "cuda" else "cpu",
        "symbolic_extraction": "cpu",
        "metrics": "cpu",
    }


def _median_parameters(parameters):
    # Small coefficient summaries use CPU, including on Metal where median(dim)
    # support varies across PyTorch versions.
    return torch.stack([p.detach().cpu() for p in parameters]).median(0).values.tolist()


def prepare_data(config, catalog=None, series_by_basin=None):
    config.validate()
    start = date.fromisoformat(config.train_start) - timedelta(days=config.warmup_days)
    end = date.fromisoformat(config.test_end)
    expected = tuple(start + timedelta(days=i) for i in range((end - start).days + 1))
    data = {}
    sources = config.source_basins if config.needs_sources() else ()
    validations = config.validation_basins if config.needs_kan() else ()
    for basin in (*sources, *validations, *config.target_basins):
        series = (
            series_by_basin[basin]
            if series_by_basin is not None
            else catalog.load_case_study_series(basin, start, end)
        )
        indices = [i for i, day in enumerate(series.dates) if start <= day <= end]
        dates = tuple(series.dates[i] for i in indices)
        if dates != expected:
            raise ValueError(
                f"{basin}: complete contiguous records are required from {start} to {end}, including spin-up"
            )
        try:
            columns = [
                series.forcing_names.index(name) for name in ("precipitation", "temperature", "pet")
            ]
        except ValueError as exc:
            raise ValueError(f"{basin}: precipitation, temperature and PET are required") from exc
        forcing = series.forcings[indices][:, columns].detach().cpu().float()
        observed = series.discharge[indices].detach().cpu().float()
        available = torch.isfinite(observed) & (observed >= 0)
        if not torch.isfinite(forcing).all() or (forcing[:, [0, 2]] < 0).any():
            raise ValueError(
                f"{basin}: forcing must be finite with nonnegative precipitation and PET"
            )
        masks = []
        for prefix in ("train", "validation", "test"):
            a, b = (
                date.fromisoformat(getattr(config, prefix + "_start")),
                date.fromisoformat(getattr(config, prefix + "_end")),
            )
            masks.append(torch.tensor([a <= d <= b for d in dates]) & available)
        data[basin] = BasinData(dates, forcing, observed, *masks)
        if basin in config.source_basins and int(masks[0].sum()) < config.minimum_source_days:
            raise ValueError(f"{basin}: insufficient source training observations")
        if basin in config.validation_basins and int(masks[1].sum()) < 3:
            raise ValueError(f"{basin}: at least three source-validation observations are required")
        if (
            config.needs_target_labels()
            and basin in config.target_basins
            and int(masks[0].sum()) < max(config.budgets_for(basin))
        ):
            raise ValueError(
                f"{basin}: available training observations are fewer than the largest observation budget"
            )
        if basin in config.target_basins and int(masks[2].sum()) < 3:
            raise ValueError(
                f"{basin}: at least three target test observations are required for scoring; "
                f"found {int(masks[2].sum())} in {config.test_start} through {config.test_end}"
            )
    return data


def _rollout(data, flux, parameters, end=None, token=None):
    return simulate(data.forcing[:end], flux, parameters, token=token)


def _fit(flux, params_by_basin, data, masks, epochs, lr, token, callback=None):
    """Equal basin loss; truncate gradients every 90 days, never reset storage."""
    trainable = [p for p in flux.parameters() if p.requires_grad]
    trainable += [p for params in params_by_basin.values() for p in params.parameters()]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    trace = []
    scales = {b: data[b].observed[masks[b]].std(unbiased=False).clamp_min(0.1) for b in masks}
    for epoch in range(epochs):
        _check_cancel(token)
        optimizer.zero_grad()
        total = 0.0
        for basin, mask in masks.items():
            last = int(mask.nonzero()[-1]) + 1
            state = None
            for first in range(0, last, 90):
                submask = mask[first : first + 90][: last - first]
                with torch.set_grad_enabled(bool(submask.any())):
                    rows, _, state = simulate(
                        data[basin].forcing[first : min(first + 90, last)],
                        flux,
                        params_by_basin[basin],
                        initial_state=state,
                        token=token,
                    )
                    if submask.any():
                        loss = (
                            (
                                (
                                    rows[submask, 0]
                                    - data[basin].observed[first : first + len(rows)][submask]
                                )
                                / scales[basin]
                            )
                            .square()
                            .sum()
                        )
                        loss = loss / int(mask.sum()) / len(masks)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite simulation during calibration")
                        loss.backward()
                        total += float(loss.detach())
                state = tuple(v.detach() for v in state)
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        trace.append(total)
        if callback:
            callback(epoch, total)
    return trace


def _metrics(observed, prediction):
    error = prediction - observed
    denominator = (observed - observed.mean()).square().sum()
    nse = float(1 - error.square().sum() / denominator) if float(denominator) > 0 else None
    return {
        "rmse_mm_day": float(error.square().mean().sqrt()),
        "nse": nse,
        "bias_mm_day": float(error.mean()),
        "days": len(observed),
    }


def _source_samples(data, basins, masks, teacher, parameters, maximum, token):
    xs, ys = [], []
    # Every basin contributes the same number of states to KAN fitting/scoring.
    count = min(maximum, min(int(masks[b].sum()) for b in basins))
    with torch.no_grad():
        for basin in basins:
            _, states, _ = _rollout(
                data[basin],
                teacher,
                parameters[basin],
                end=int(masks[basin].nonzero()[-1]) + 1,
                token=token,
            )
            states = states[masks[basin][: len(states)]]
            index = torch.linspace(0, len(states) - 1, count, device=states.device).long()
            x = states[index]
            xs.append(x)
            ys.append(teacher(x)[:, None])
    return torch.cat(xs), torch.cat(ys)


def distil_symbolic(
    train_x,
    train_y,
    val_x,
    val_y,
    config,
    seed,
    token=None,
    progress=None,
    simulation_score=None,
    device="cpu",
    symbolic_library=BASE_SYMBOLIC_LIBRARY,
):
    import sympy as sp

    try:
        from kan.MultKAN import MultKAN
    except ImportError as exc:
        raise ValueError(
            "The KAN case study requires pykan; install the project's kan extra"
        ) from exc
    # PyKAN spline fitting requires least-squares operations unavailable on MPS.
    # CUDA can fit numerically; pruning and SymPy extraction stay on CPU.
    device = torch.device(device)
    kan_device = device if device.type == "cuda" else torch.device("cpu")
    train_x, train_y, val_x, val_y = (
        v.detach().to(kan_device) for v in (train_x, train_y, val_x, val_y)
    )
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    dataset = {
        "train_input": (train_x - mean) / scale,
        "train_label": train_y,
        "test_input": (val_x - mean) / scale,
        "test_label": val_y,
    }
    candidates, usable = [], []
    for candidate in range(config.kan_candidates):
        _check_cancel(token)
        if progress:
            progress(candidate)
        try:
            with tempfile.TemporaryDirectory(prefix="kan-transfer-") as folder:
                model = MultKAN(
                    width=[3, config.kan_hidden, 1],
                    grid=config.kan_grid,
                    k=3,
                    grid_range=[-3, 3],
                    seed=seed + candidate,
                    device=str(kan_device),
                    auto_save=False,
                    ckpt_path=folder,
                    symbolic_enabled=True,
                )

                # Keep Adam moments across steps and check cancellation in the loss.
                def loss_fn(prediction, target):
                    _check_cancel(token)
                    return (prediction - target).square().mean()

                model.fit(
                    dataset,
                    opt="Adam",
                    steps=config.kan_steps,
                    lr=0.02,
                    lamb=config.sparsity_weight,
                    update_grid=False,
                    loss_fn=loss_fn,
                    log=max(1000, config.kan_steps + 1),
                )
                with torch.no_grad():
                    numeric = model(dataset["test_input"]).detach().cpu().reshape(-1)
                model = model.to("cpu")
                # Refresh cached activations on CPU before pruning. Moving only
                # parameters leaves PyKAN's cached training tensors on CUDA.
                with torch.no_grad():
                    model(dataset["test_input"].cpu())
                model = model.prune()
                model.auto_symbolic(lib=list(symbolic_library), verbose=0)
                symbolic = model.symbolic_formula(
                    var=["storage", "liquid", "pet"], normalizer=[mean.cpu(), scale.cpu()]
                )
                if symbolic is None:
                    raise ValueError("Incomplete symbolic conversion")
                expression = symbolic[0][0]
                ops = int(sp.count_ops(expression))
                if ops > config.max_formula_ops:
                    raise ValueError(
                        f"Expression has {ops} operations; limit is {config.max_formula_ops}"
                    )
                flux = SymbolicFlux(expression)
                with torch.no_grad():
                    symbolic_prediction = flux(val_x.cpu())
                    reference = model(dataset["test_input"].cpu()).reshape(-1)
                    if not torch.isfinite(symbolic_prediction).all():
                        raise ValueError("Nonfinite symbolic validation predictions")
                    if not torch.allclose(symbolic_prediction, reference, atol=2e-3, rtol=2e-3):
                        raise ValueError("Exported expression disagrees with the converted KAN")
                    # Score bounded release fractions, the quantity used in storage equations.
                    mse = float(
                        (symbolic_prediction.sigmoid() - val_y[:, 0].cpu().sigmoid())
                        .square()
                        .mean()
                    )
                    conversion_mse = float(
                        (symbolic_prediction.sigmoid() - numeric.sigmoid()).square().mean()
                    )
                item = {
                    "candidate": candidate,
                    "validation_fraction_mse": mse,
                    "conversion_fraction_mse": conversion_mse,
                    "operations": ops,
                    "expression": str(expression),
                    "status": "ok",
                }
                rollout_score = simulation_score(flux) if simulation_score else mse
                if not math.isfinite(rollout_score):
                    raise ValueError("Nonfinite recursive source-validation simulation")
                item["validation_simulation_teacher_nmse"] = rollout_score
                usable.append((rollout_score + mse + 1e-6 * ops, flux, candidate))
        except StudyCancelled:
            raise
        except (ValueError, RuntimeError, TypeError, IndexError) as exc:
            item = {"candidate": candidate, "status": "rejected", "reason": str(exc)}
        candidates.append(item)
    if not usable:
        raise ValueError(
            "No valid symbolic candidate: "
            + "; ".join(str(c.get("reason", "")) for c in candidates)
        )
    _, selected, index = min(usable, key=lambda item: item[0])
    return selected, {
        "selected_candidate": index,
        "candidates": candidates,
        "kan_fitting_device": str(kan_device),
        "symbolic_extraction_device": "cpu",
        "input_mean": mean.cpu().tolist(),
        "input_scale": scale.cpu().tolist(),
        "training_state_min": train_x.min(0).values.cpu().tolist(),
        "training_state_max": train_x.max(0).values.cpu().tolist(),
    }


def _discover_source_equation(
    config, data, seed, device, token, progress, symbolic_library=BASE_SYMBOLIC_LIBRARY
):
    train_masks = {b: data[b].train for b in config.source_basins}
    val_masks = {b: data[b].validation for b in config.validation_basins}
    torch.manual_seed(seed)
    teacher = NeuralFlux(config.teacher_width).to(device)
    parameters = {b: BasinParameters().to(device) for b in config.source_basins}
    best_loss, best = float("inf"), None

    def teacher_epoch(epoch, loss):
        nonlocal best_loss, best
        initial = _median_parameters(p.raw for p in parameters.values())
        validation_parameters = BasinParameters(initial).to(device)
        with torch.no_grad():
            errors = []
            for basin, mask in val_masks.items():
                rows, _, _ = _rollout(
                    data[basin],
                    teacher,
                    validation_parameters,
                    end=int(mask.nonzero()[-1]) + 1,
                    token=token,
                )
                target = data[basin].observed[: len(rows)][mask[: len(rows)]]
                scale = target.std(unbiased=False).clamp_min(0.1)
                errors.append(
                    float(((rows[mask[: len(rows)], 0] - target) / scale).square().mean())
                )
            validation_loss = sum(errors) / len(errors)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best = (
                copy.deepcopy(teacher.state_dict()),
                {b: p.raw.detach().clone() for b, p in parameters.items()},
            )
        progress(
            0.5 * (epoch + 1) / config.teacher_epochs,
            f"Seed {seed}: source teacher epoch {epoch + 1}/{config.teacher_epochs}",
        )

    trace = _fit(
        teacher,
        parameters,
        data,
        train_masks,
        config.teacher_epochs,
        config.learning_rate,
        token,
        teacher_epoch,
    )
    if best is None:
        raise ValueError("Teacher validation failed")
    teacher.load_state_dict(best[0])
    for basin in parameters:
        parameters[basin].raw.data.copy_(best[1][basin])
    teacher.requires_grad_(False).eval()
    initial = _median_parameters(best[1].values())
    validation_parameters = {
        b: BasinParameters(initial).to(device) for b in config.validation_basins
    }
    train_x, train_y = _source_samples(
        data,
        config.source_basins,
        train_masks,
        teacher,
        parameters,
        config.max_distillation_samples_per_basin,
        token,
    )
    val_x, val_y = _source_samples(
        data,
        config.validation_basins,
        val_masks,
        teacher,
        validation_parameters,
        config.max_distillation_samples_per_basin,
        token,
    )
    with torch.no_grad():
        validation_teacher = {
            b: _rollout(
                data[b],
                teacher,
                validation_parameters[b],
                end=int(val_masks[b].nonzero()[-1]) + 1,
                token=token,
            )[0][:, 0]
            for b in val_masks
        }

    def simulation_score(flux):
        errors = []
        with torch.no_grad():
            for basin, mask in val_masks.items():
                reference = validation_teacher[basin]
                selected = mask[: len(reference)]
                prediction = _rollout(
                    data[basin], flux, validation_parameters[basin], end=len(reference), token=token
                )[0][:, 0]
                scale = reference[selected].std(unbiased=False).clamp_min(0.1)
                errors.append(
                    float(((prediction[selected] - reference[selected]) / scale).square().mean())
                )
        return sum(errors) / len(errors)

    symbolic, discovery = distil_symbolic(
        train_x,
        train_y,
        val_x,
        val_y,
        config,
        seed,
        token,
        lambda i: progress(
            0.5 + 0.5 * i / config.kan_candidates,
            f"Seed {seed}: distilling source flux candidate {i + 1}/{config.kan_candidates}",
        ),
        simulation_score=simulation_score,
        device=device,
        symbolic_library=symbolic_library,
    )
    equation = {
        "seed": seed,
        "logit_expression": str(symbolic.expression),
        "source_validation_teacher_nmse": best_loss,
        "teacher_parameter_count": sum(p.numel() for p in teacher.parameters()),
        "teacher_state_dict": {
            k: v.detach().cpu().tolist() for k, v in teacher.state_dict().items()
        },
        "initial_local_parameters": BasinParameters(initial).report(),
        "initial_raw_parameters": initial,
        "source_parameters": {b: p.report() for b, p in parameters.items()},
        **discovery,
    }
    return symbolic, discovery, initial, equation, trace


def kan_discovery_profiles(config):
    """Independent discovery settings; existing variants keep their original pipeline."""
    profiles: list[tuple[str, tuple[str, ...], CaseStudyConfig, tuple[str, ...]]] = []
    standard = tuple(m for m in config.selected_models if m in ("kan_pure", "kan_tuned"))
    if standard:
        profiles.append(("standard", standard, config, BASE_SYMBOLIC_LIBRARY))
    if "kan_complex" in config.selected_models:
        profiles.append(
            (
                "complex",
                ("kan_complex",),
                replace(
                    config,
                    teacher_epochs=max(50, config.teacher_epochs),
                    teacher_width=max(24, config.teacher_width),
                    kan_candidates=max(6, config.kan_candidates),
                ),
                COMPLEX_SYMBOLIC_LIBRARY,
            )
        )
    return profiles


def _record_comparison(
    result,
    config,
    item,
    basin,
    seed,
    method,
    scenario,
    budget,
    mask,
    prediction,
    *,
    rows=None,
    states=None,
    discovery=None,
    local=None,
    parameter_count=0,
    local_parameter_count=0,
    fit_id=None,
):
    observed, test, mask = item.observed.cpu(), item.test.cpu(), mask.cpu()
    prediction = prediction.detach().cpu()
    if not torch.isfinite(prediction[test]).all():
        raise ValueError(f"{basin}: nonfinite {method} test prediction")
    scores = _metrics(observed[test], prediction[test])
    if int(mask.sum()) >= 3:
        low, high = torch.quantile(observed[mask], torch.tensor([0.1, 0.9]))
        for label, event in (
            ("low_flow", observed[test] <= low),
            ("high_flow", observed[test] >= high),
        ):
            scores[label + "_rmse"] = (
                float((prediction[test][event] - observed[test][event]).square().mean().sqrt())
                if event.any()
                else None
            )
    else:
        scores.update(low_flow_rmse=None, high_flow_rmse=None)
    locally_fitted = (
        method in LOCALLY_CALIBRATED_KAN_MODELS or "_target_target" in method or "_mixed_" in method
    )
    chosen = mask.nonzero().flatten() if locally_fitted else torch.empty(0, dtype=torch.long)
    outside = None
    if states is not None:
        lower, upper = (
            torch.tensor(discovery["training_state_min"]),
            torch.tensor(discovery["training_state_max"]),
        )
        outside = float(((states[test] < lower) | (states[test] > upper)).any(1).float().mean())
    result["metrics"].append(
        {
            "seed": seed,
            "basin_id": basin,
            "scenario": scenario,
            "budget": budget,
            "method": method,
            **scores,
            "local_parameters": local.report() if local is not None else {},
            "trainable_parameters": parameter_count,
            "local_trainable_parameters": local_parameter_count,
            "fit_id": fit_id,
            "max_abs_water_balance_mm": float(rows[:, 4].abs().max()) if rows is not None else None,
            "state_outside_source_range_fraction": outside,
            "calibration_first": item.dates[int(chosen[0])].isoformat() if len(chosen) else None,
            "calibration_last": item.dates[int(chosen[-1])].isoformat() if len(chosen) else None,
        }
    )
    start, end = date.fromisoformat(config.test_start), date.fromisoformat(config.test_end)
    for index, day in enumerate(item.dates):
        if start <= day <= end:
            result["predictions"].append(
                {
                    "seed": seed,
                    "basin_id": basin,
                    "scenario": scenario,
                    "budget": budget,
                    "method": method,
                    "date": day.isoformat(),
                    "observed_mm_day": float(observed[index]) if test[index] else None,
                    "predicted_mm_day": float(prediction[index]),
                    "evaporation_mm": float(rows[index, 1]) if rows is not None else None,
                    "snow_mm": float(rows[index, 2]) if rows is not None else None,
                    "water_storage_mm": float(rows[index, 3]) if rows is not None else None,
                    "balance_error_mm": float(rows[index, 4]) if rows is not None else None,
                }
            )


def run_case_study(
    config, *, catalog=None, series_by_basin=None, progress=None, cancellation_token=None
):
    """Compare selected architecture/scope variants and source-derived KAN laws."""
    from .kan_comparisons import (
        budget_mask,
        comparison_masks,
        fit_comparison,
        fit_report,
        predict_comparison,
    )

    config.validate()
    token, progress = cancellation_token, progress or (lambda fraction, message: None)
    _check_cancel(token)
    hardware = case_study_hardware(config.device)
    device = torch.device(hardware["resolved"])
    if config.needs_kan() and importlib.util.find_spec("kan") is None:
        raise ValueError("Selected KAN comparisons require pykan; install the project's kan extra")
    if not config.needs_kan():
        hardware.update(kan_fitting="not_used", symbolic_extraction="not_used")
    progress(0.0, f"Compute: {device}; checking selected models and daily coverage")
    data = {b: d.to(device) for b, d in prepare_data(config, catalog, series_by_basin).items()}
    scenarios = config.budget_scenarios()
    result = {
        "schema_version": 2,
        "kind": "kan_case_study",
        "config": asdict(config),
        "hardware": hardware,
        "interpretation": "Architecture/scope comparison against a source-derived symbolic process approximation.",
        "forecast_interpretation": "Observed-weather hindcast; no measured discharge enters model inputs or states.",
        "parameter_names": ["capacity_mm", "melt_mm_day_c", "runoff_logit_offset"],
        "local_parameter_count": 3
        if any(m in LOCALLY_CALIBRATED_KAN_MODELS for m in config.selected_models)
        else 0,
        "equations": [],
        "model_fits": [],
        "metrics": [],
        "predictions": [],
        "traces": {},
        "audit": {
            "teacher_training_basins": list(config.source_basins) if config.needs_kan() else [],
            "equation_selection_basins": list(config.validation_basins)
            if config.needs_kan()
            else [],
            "target_basins": list(config.target_basins),
            "selected_models": list(config.selected_models),
            "hardware": hardware,
            "internal_process_teacher_used": config.needs_kan(),
            "target_validation_used": False,
            "target_test_used_for_fitting": False,
            "discharge_assimilation": False,
            "budget_scenarios": [
                {"scenario": i + 1, "target_budgets": plan} for i, plan in enumerate(scenarios)
            ],
            "budget_policy": "Budget lists paired by position; shorter lists retain their last value. Target/mixed models pool all selected targets at each scenario's budgets.",
            "observation_policy": "First N available training labels per target. Source models use source labels only; no held-out labels enter neural fitting or normalization.",
            "neural_features": ["precipitation", "temperature", "pet", "day_sin", "day_cos"],
            "neural_validation_policy": "Fixed training epochs; no validation/test labels select neural baseline checkpoints.",
            "coverage": {
                b: {
                    "forcing_days": len(d.dates),
                    "training_labels": int(d.train.sum()),
                    "validation_labels": int(d.validation.sum()),
                    "test_labels": int(d.test.sum()),
                }
                for b, d in data.items()
            },
            "units": "Daily water quantities in mm; temperature in Celsius",
            "warmup_days": config.warmup_days,
            "storage_equation": "S_next = S + liquid - ET - Q; snow_next = snow + snowfall - melt",
            "runoff_equation": "Q = overflow + retained * sigmoid(F(S/C, liquid/C, PET/C) + offset)",
        },
    }
    profiles = kan_discovery_profiles(config)
    units = len(config.seeds) * (len(profiles) + len(scenarios) * len(config.selected_models))
    completed = 0
    for seed in config.seeds:
        discoveries = {}
        for profile, methods, discovery_config, library in profiles:
            symbolic, discovery, initial, equation, trace = _discover_source_equation(
                discovery_config,
                data,
                seed,
                device,
                token,
                lambda f, m: progress((completed + f) / units, f"{profile}: {m}"),
                symbolic_library=library,
            )
            equation.update(
                models=list(methods),
                discovery_profile=profile,
                discovery_settings={
                    "teacher_epochs": discovery_config.teacher_epochs,
                    "teacher_width": discovery_config.teacher_width,
                    "kan_candidates": discovery_config.kan_candidates,
                    "symbolic_library": list(library),
                },
            )
            result["equations"].append(equation)
            result["traces"][
                f"teacher_{seed}" if profile == "standard" else f"teacher_complex_{seed}"
            ] = trace
            for method in methods:
                discoveries[method] = symbolic, discovery, initial
            completed += 1
        source_fits: dict[str, Any] = {}
        for scenario_index, plan in enumerate(scenarios, 1):
            masks = {b: budget_mask(data[b], plan[b]) for b in config.target_basins}
            for method in config.selected_models:
                _check_cancel(token)
                fitted, fit_id = None, None
                if not method.startswith("kan_"):
                    architecture, scope, _ = method.split("_")
                    fit_id = f"{seed}_{method}_" + (
                        "shared" if scope == "source" else str(scenario_index)
                    )
                    if method in source_fits:
                        fitted = source_fits[method]
                    else:
                        training = comparison_masks(config, data, scope, plan)
                        fitted = fit_comparison(
                            architecture,
                            data,
                            training,
                            config,
                            seed,
                            device,
                            token,
                            lambda epoch: progress(
                                (completed + 0.8 * epoch / config.neural_epochs) / units,
                                f"Seed {seed}: {method}, scenario {scenario_index}, epoch {epoch}/{config.neural_epochs}",
                            ),
                        )
                        result["traces"][fit_id] = fitted["trace"]
                        result["model_fits"].append(
                            {
                                "fit_id": fit_id,
                                "seed": seed,
                                "method": method,
                                "scenario": None if scope == "source" else scenario_index,
                                "training_scope": scope,
                                "target_budget_plan": {} if scope == "source" else dict(plan),
                                **fit_report(fitted),
                            }
                        )
                        if scope == "source":
                            source_fits[method] = fitted
                for basin in config.target_basins:
                    item, budget, mask = data[basin], plan[basin], masks[basin]
                    if fitted is not None:
                        prediction = predict_comparison(fitted, item, config, token)
                        local_count = (
                            fitted["parameter_count"] if scope in {"target", "mixed"} else 0
                        )
                        _record_comparison(
                            result,
                            config,
                            item,
                            basin,
                            seed,
                            method,
                            scenario_index,
                            budget,
                            mask,
                            prediction,
                            parameter_count=fitted["parameter_count"],
                            local_parameter_count=local_count,
                            fit_id=fit_id,
                        )
                    else:
                        symbolic, discovery, initial = discoveries[method]
                        local = BasinParameters(initial).to(device)
                        if method in LOCALLY_CALIBRATED_KAN_MODELS:
                            trace = _fit(
                                symbolic,
                                {basin: local},
                                data,
                                {basin: mask},
                                config.adaptation_epochs,
                                config.learning_rate,
                                token,
                            )
                            result["traces"][f"{seed}_{basin}_{scenario_index}_{method}"] = trace
                        with torch.no_grad():
                            rows, states, _ = _rollout(item, symbolic, local, token=token)
                        rows, states = rows.cpu(), states.cpu()
                        _record_comparison(
                            result,
                            config,
                            item,
                            basin,
                            seed,
                            method,
                            scenario_index,
                            budget,
                            mask,
                            rows[:, 0],
                            rows=rows,
                            states=states,
                            discovery=discovery,
                            local=local,
                            parameter_count=3 if method in LOCALLY_CALIBRATED_KAN_MODELS else 0,
                            local_parameter_count=3
                            if method in LOCALLY_CALIBRATED_KAN_MODELS
                            else 0,
                        )
                completed += 1
                progress(
                    completed / units, f"Seed {seed}: completed {method}, scenario {scenario_index}"
                )
    return result


def _csv(rows):
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def save_case_study(result, folder):
    """Portable artifacts: actual executed formula, parameters, audit and fluxes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2, allow_nan=False)
    (folder / "case_study.json").write_text(report)
    with zipfile.ZipFile(folder / "kan_case_study.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("case_study.json", report)
        bundle.writestr("config.json", json.dumps(result["config"], indent=2))
        bundle.writestr("equations.json", json.dumps(result["equations"], indent=2))
        bundle.writestr("model_fits.json", json.dumps(result.get("model_fits", []), indent=2))
        bundle.writestr("audit.json", json.dumps(result["audit"], indent=2))
        bundle.writestr("predictions.csv", _csv(result["predictions"]))
        bundle.writestr(
            "metrics.csv",
            _csv(
                [
                    {**r, "local_parameters": json.dumps(r["local_parameters"])}
                    for r in result["metrics"]
                ]
            ),
        )
    return folder / "kan_case_study.zip"


def main():
    import argparse

    from .regional import CAMELSCHCatalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = config_from_dict(json.loads(args.config.read_text()))
    result = run_case_study(
        config,
        catalog=CAMELSCHCatalog(args.data_root),
        progress=lambda f, text: print(f"{f:.0%}: {text}", flush=True),
    )
    print(save_case_study(result, args.output))


if __name__ == "__main__":
    main()
