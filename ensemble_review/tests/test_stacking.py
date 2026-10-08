import pytest
import torch

from complexity_ensemble.stacking import fit_convex_stacking_weight


def test_convex_stacking_recovers_an_interior_weight():
    simple = torch.tensor([0.0, 2.0, 4.0])
    complex_prediction = torch.tensor([2.0, 4.0, 6.0])
    target = 0.75 * simple + 0.25 * complex_prediction

    weight, mse = fit_convex_stacking_weight(simple, complex_prediction, target)

    assert weight == pytest.approx(0.25)
    assert mse == pytest.approx(0.0)


def test_convex_stacking_clamps_extrapolating_solution():
    simple = torch.tensor([0.0, 1.0])
    complex_prediction = torch.tensor([1.0, 2.0])
    target = torch.tensor([3.0, 4.0])

    weight, _ = fit_convex_stacking_weight(simple, complex_prediction, target)

    assert weight == 1.0
