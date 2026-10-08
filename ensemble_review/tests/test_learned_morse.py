from __future__ import annotations

import unittest

import torch

from complexity_ensemble.hydrology import (
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
)


class LearnedMorseTests(unittest.TestCase):
    def test_potential_gradient_and_frozen_router(self) -> None:
        names = ("precipitation", "temperature", "pet", "previous_discharge", "day_sin", "day_cos")
        potential = LearnedHydrologyMorsePotential(7, names, hidden_dim=8)
        inputs = torch.randn(12, 7, len(names))
        flat = inputs.flatten(start_dim=1)
        score = potential.complexity(flat, create_graph=True)
        self.assertEqual(score.shape, (12,))
        score.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in potential.parameters()))

        model = RoutedHydrologyModel(
            names, 7, torch.tensor(1.0), routing="learned_morse",
            learned_morse_potential=potential,
        )
        model.fit_router(inputs)
        self.assertEqual(model(inputs).shape, (12, 1))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.router.potential.parameters()))

    def test_learned_morse_requires_calibration(self) -> None:
        names = ("precipitation", "temperature", "pet", "previous_discharge")
        with self.assertRaises(ValueError):
            RoutedHydrologyModel(names, 7, 1.0, routing="learned_morse")


if __name__ == "__main__":
    unittest.main()
