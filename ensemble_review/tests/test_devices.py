"""Device selection and cross-device analysis regressions."""

import pytest
import torch

from complexity_ensemble.devices import as_double, randn, randperm, resolve_device, seeded_generator
from complexity_ensemble.hydrology_comparison import _add_observation_noise, _metrics
from complexity_ensemble.hydrology_extreme_comparison import _average_precision
from complexity_ensemble.hydrology_morse_learning import _spearman
from complexity_ensemble.ood import MahalanobisOODDetector
from complexity_ensemble.stacking import fit_convex_stacking_weight


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert str(resolve_device()) == expected
    assert str(resolve_device("cpu")) == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_fails_clearly(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match=f"{device.upper()} was requested but is not available"):
        resolve_device(device)


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


def test_seeded_sampling_is_repeatable_and_independent_of_global_rng(device):
    first = seeded_generator(device, 42)
    permutation = randperm(20, generator=first, device=device)
    noise = randn((4, 3), generator=first, device=device)
    torch.manual_seed(999)
    second = seeded_generator(device, 42)
    assert torch.equal(permutation, randperm(20, generator=second, device=device))
    assert torch.equal(noise, randn((4, 3), generator=second, device=device))
    assert permutation.device.type == noise.device.type == device.type
    assert sorted(permutation.cpu().tolist()) == list(range(20))


def test_analysis_preserves_precision_and_returns_ood_scores_to_input_device(device):
    values = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    detector = MahalanobisOODDetector().fit(values.to(device))
    expected = MahalanobisOODDetector().fit(values).score(values)
    scores = detector.score(values.to(device))
    assert scores.device.type == device.type
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores.cpu(), expected, rtol=1e-4, atol=1e-5)
    assert detector.is_ood(values.to(device)).device.type == device.type
    assert as_double(values.to(device)).dtype == torch.float64


def test_metrics_and_stacking_accept_accelerator_tensors(device):
    simple = torch.tensor([0.0, 2.0, 4.0], device=device)
    complex_prediction = simple + 2
    target = 0.75 * simple + 0.25 * complex_prediction
    weight, mse = fit_convex_stacking_weight(simple, complex_prediction, target)
    assert weight == pytest.approx(0.25)
    assert mse == pytest.approx(0.0)
    assert _metrics(target, target) == pytest.approx((1.0, 1.0, 0.0))
    assert _spearman(target, target) == pytest.approx(1.0)
    assert _average_precision(target, target > 2) == pytest.approx(1.0)


def test_observation_noise_stays_on_device_and_only_changes_dynamic_features(device):
    inputs = torch.zeros((4, 3, 2), device=device)
    noisy = _add_observation_noise(
        inputs, ("precipitation", "day_sin"), 0.1, seeded_generator(device, 42)
    )
    assert noisy.device.type == device.type
    assert torch.count_nonzero(noisy[..., 0]) > 0
    assert torch.count_nonzero(noisy[..., 1]) == 0
