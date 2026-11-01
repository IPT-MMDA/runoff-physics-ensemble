"""Existing discharge architectures under explicit source/target training scopes."""
from __future__ import annotations

import math
import torch
from torch import nn

from .devices import randperm, seeded_generator
from .hydrology import build_hydrology_complex_expert
from .kan_case_study import _check_cancel


def weather_windows(item, length):
    """Same-day weather plus past weather; measured discharge never enters inputs."""
    angle = item.forcing.new_tensor([d.timetuple().tm_yday for d in item.dates]) * (2 * math.pi / 365.25)
    features = torch.cat((item.forcing, angle.sin()[:, None], angle.cos()[:, None]), 1)
    return features.unfold(0, length, 1).permute(0, 2, 1)


def budget_mask(item, budget):
    mask = torch.zeros_like(item.train)
    mask[item.train.nonzero().flatten()[:budget]] = True
    return mask


def comparison_masks(config, data, scope, plan):
    masks = {}
    if scope in {"source", "mixed"}:
        masks.update({b: data[b].train for b in config.source_basins})
    if scope in {"target", "mixed"}:
        masks.update({b: budget_mask(data[b], plan[b]) for b in config.target_basins})
    return masks


def fit_comparison(architecture, data, masks, config, seed, device, token=None, progress=None):
    """Fixed epochs, equal basin contribution, training-only normalization."""
    torch.manual_seed(seed + ("pinnmamba", "mlp", "lstm").index(architecture) * 100003)
    kwargs = ({"hidden_dims": config.mlp_widths} if architecture == "mlp" else
              {"hidden_dim": config.mamba_hidden_size, "num_layers": config.mamba_layers,
               "hidden_d_ff": config.mamba_feed_forward_size} if architecture == "pinnmamba" else
              {"hidden_dim": config.lstm_hidden_size, "num_layers": config.lstm_layers})
    model = build_hydrology_complex_expert(architecture, config.sequence_length, 5, kwargs).to(device)
    batches = {}
    for basin, mask in masks.items():
        indices = mask.nonzero().flatten()
        if not len(indices):
            raise ValueError(f"{basin}: no allowed training labels for {architecture}")
        if int(indices[0]) < config.sequence_length - 1:
            raise ValueError("Insufficient historical weather for neural input windows")
        x = weather_windows(data[basin], config.sequence_length)[indices - config.sequence_length + 1]
        batches[basin] = (x, data[basin].observed[indices])
    all_x = torch.cat([x for x, _ in batches.values()])
    mean = all_x.mean((0, 1))
    scale = all_x.reshape(-1, 5).std(0, unbiased=False).clamp_min(1e-5)
    output_scale = torch.cat([y for _, y in batches.values()]).std(unbiased=False).clamp_min(.1)
    batches = {b: ((x - mean) / scale, y / output_scale) for b, (x, y) in batches.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = seeded_generator(device, seed + 7919)
    trace = []
    for epoch in range(config.neural_epochs):
        _check_cancel(token)
        optimizer.zero_grad()
        total = 0.
        for x, y in batches.values():
            order = randperm(len(y), generator=generator, device=device)
            for start in range(0, len(y), config.neural_batch_size):
                _check_cancel(token)
                selected = order[start:start + config.neural_batch_size]
                prediction = nn.functional.softplus(model(x[selected])).squeeze(-1)
                loss = (prediction - y[selected]).square().sum() / len(y) / len(batches)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {architecture} training loss")
                loss.backward()
                total += float(loss.detach().cpu())
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        trace.append(total)
        if progress:
            progress(epoch + 1)
    model.eval()
    return {"model": model, "input_mean": mean, "input_scale": scale, "output_scale": output_scale,
            "trace": trace, "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_labels": {b: int(mask.sum()) for b, mask in masks.items()},
            "training_first": {b: data[b].dates[int(mask.nonzero()[0])].isoformat() for b, mask in masks.items()},
            "training_last": {b: data[b].dates[int(mask.nonzero()[-1])].isoformat() for b, mask in masks.items()}}


def predict_comparison(fitted, item, config, token=None):
    windows = weather_windows(item, config.sequence_length)
    output = []
    with torch.no_grad():
        for start in range(0, len(windows), config.neural_batch_size):
            _check_cancel(token)
            x = (windows[start:start + config.neural_batch_size] - fitted["input_mean"]) / fitted["input_scale"]
            output.append(nn.functional.softplus(fitted["model"](x)).squeeze(-1) * fitted["output_scale"])
    values = torch.cat(output).detach().cpu()
    return torch.cat((torch.full((config.sequence_length - 1,), float("nan")), values))


def fit_report(fitted):
    return {"training_labels": fitted["training_labels"], "training_first": fitted["training_first"],
            "training_last": fitted["training_last"], "parameter_count": fitted["parameter_count"],
            "feature_names": ["precipitation", "temperature", "pet", "day_sin", "day_cos"],
            "input_mean": fitted["input_mean"].detach().cpu().tolist(),
            "input_scale": fitted["input_scale"].detach().cpu().tolist(),
            "output_scale": float(fitted["output_scale"].detach().cpu()),
            "state_dict": {k: v.detach().cpu().tolist() for k, v in fitted["model"].state_dict().items()}}
