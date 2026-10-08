from __future__ import annotations

import unittest

import torch

from complexity_ensemble.hydrology import RoutedHydrologyModel
from complexity_ensemble.hydrology_hard_routing_consolidation_comparison import (
    hard_routed_losses,
)


def _gradient_size(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class HydrologyHardRoutingConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = RoutedHydrologyModel(
            ("precipitation", "pet", "previous_discharge"),
            2,
            1.0,
            routing="tda",
            complexity_percentile=50.0,
            simple_kwargs={"n_centers": 4},
            complex_kwargs={"hidden_dims": (4,)},
        )
        self.inputs = torch.randn(2, 2, 3)
        self.physical = torch.randn(2, 2, 3)
        self.targets = torch.randn(2)
        self.routing_features = torch.tensor([[0.0], [1.0]])
        self.model.fit_router(self.inputs, self.routing_features)

    def _backward_for_sample(self, index: int) -> tuple[float, float]:
        self.model.zero_grad(set_to_none=True)
        losses = hard_routed_losses(
            self.model,
            self.inputs[index : index + 1],
            self.physical[index : index + 1],
            self.targets[index : index + 1],
            routing_features=self.routing_features[index : index + 1],
            interface_weight=0.0,
        )
        losses.total.backward()
        return (
            _gradient_size(self.model.simple_expert),
            _gradient_size(self.model.complex_expert),
        )

    def test_simple_sample_only_updates_simple_expert(self) -> None:
        simple_gradient, complex_gradient = self._backward_for_sample(0)
        self.assertGreater(simple_gradient, 0.0)
        self.assertEqual(complex_gradient, 0.0)

    def test_complex_sample_only_updates_complex_expert(self) -> None:
        simple_gradient, complex_gradient = self._backward_for_sample(1)
        self.assertEqual(simple_gradient, 0.0)
        self.assertGreater(complex_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
