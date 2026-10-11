"""Device selection and portable seeded sampling for CPU, CUDA, and Metal."""

from __future__ import annotations

import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    if str(requested) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type not in DEVICE_CHOICES[1:]:
        raise ValueError(f"unsupported device {requested!r}; choose from {DEVICE_CHOICES}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available; use an Apple Silicon Mac with an MPS-enabled PyTorch build, or select cpu")
    return device


def seeded_generator(device: str | torch.device, seed: int) -> torch.Generator:
    # Use independent CPU generators for Metal rather than backend-global RNG state.
    device = torch.device(device)
    return torch.Generator(device="cpu" if device.type == "mps" else device).manual_seed(seed)


def randperm(n: int, *, generator: torch.Generator, device: str | torch.device) -> torch.Tensor:
    return torch.randperm(n, generator=generator, device=generator.device).to(device)


def randn(shape: tuple[int, ...] | torch.Size, *, generator: torch.Generator,
          device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=generator.device, dtype=dtype).to(device)


def as_double(values: torch.Tensor) -> torch.Tensor:
    """Preserve double precision by doing Metal-incompatible analysis on CPU."""
    if values.device.type == "mps":
        values = values.cpu()
    return values.double()
