import pytest
import torch

from complexity_ensemble.kan_physics import (
    KANPhysicsCorrection, _multkan_class, _select_candidate,
    standalone_kan_prediction,
)


class _SumModel(torch.nn.Module):
    def forward(self, values):
        return values.sum(dim=1, keepdim=True)


def test_physics_correction_normalizes_inputs_and_restores_units():
    correction = KANPhysicsCorrection(
        _SumModel(), torch.tensor([0, 2]),
        torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]),
        torch.tensor(3.0), torch.tensor(5.0),
    )
    physical = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 9.0, 6.0]]])

    # Normalized inputs are (1, 1), so the restored correction is 2*5 + 3.
    assert torch.allclose(correction(physical), torch.tensor([13.0]))
    assert not any(parameter.requires_grad for parameter in correction.parameters())


def test_standalone_prediction_combines_reservoir_and_kan_correction():
    correction = KANPhysicsCorrection(
        _SumModel(), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]),
        torch.tensor(0.0), torch.tensor(1.0),
    )
    # precipitation=3, pet=1, previous discharge=2, and KAN correction=3.
    physical = torch.tensor([[[3.0, 1.0, 2.0]]])

    prediction = standalone_kan_prediction(
        physical, ("precipitation", "pet", "previous_discharge"),
        response=0.5, recession=0.25, correction=correction,
    )

    assert torch.allclose(prediction, torch.tensor([5.5]))


def test_candidate_selection_implements_all_four_modes():
    metrics = [
        {"clean_ap": 0.90, "noise_degradation": 0.20, "validation_mse": 0.30},
        {"clean_ap": 0.88, "noise_degradation": 0.02, "validation_mse": 0.10},
        {"clean_ap": 0.60, "noise_degradation": 0.00, "validation_mse": 0.20},
    ]

    assert _select_candidate("accuracy", metrics, 0.05) == 0
    assert _select_candidate("robustness", metrics, 0.05) == 1
    assert _select_candidate("distillation", metrics, 0.05) == 1
    assert _select_candidate("balanced", metrics, 0.05) in {0, 1}


def test_pykan_loader_returns_the_multkan_class_when_installed():
    pytest.importorskip("kan")

    assert callable(_multkan_class())
