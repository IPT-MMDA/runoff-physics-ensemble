from __future__ import annotations

import csv
from datetime import date, timedelta
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch

from complexity_ensemble.hydrology_complexity import (
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
    takens_delay_embedding,
)
from complexity_ensemble.hydrology_consolidation_comparison import (
    compare_hydrology_models_with_consolidation,
)
from complexity_ensemble.hydrology_data import load_ukraine_csv, make_hydrology_data
from complexity_ensemble.routing import ScoreRouter


def _logistic_series(rate: float, length: int = 600) -> torch.Tensor:
    value = 0.234567
    result = []
    for index in range(length + 100):
        value = rate * value * (1.0 - value)
        if index >= 100:
            result.append(value)
    return torch.tensor(result)


class ComplexityEstimatorTests(unittest.TestCase):
    def test_delay_embedding_contract(self) -> None:
        embedding = takens_delay_embedding(
            torch.arange(8.0), embedding_dim=3, delay=2
        )
        self.assertEqual(embedding.shape, (4, 3))
        self.assertTrue(torch.equal(embedding[0], torch.tensor([0.0, 2.0, 4.0])))

    @unittest.skipUnless(importlib.util.find_spec("ripser"), "ripser is optional")
    def test_tda_detects_persistent_cycle_in_sinusoid(self) -> None:
        time = torch.linspace(0.0, 12.0 * math.pi, 256)
        windows = torch.stack((torch.zeros_like(time), torch.sin(time)))[:, :, None]
        estimator = TakensPersistenceEstimator(
            0, embedding_dim=3, delay=4, min_persistence=0.05, max_points=48
        )
        estimates = estimator.estimate(windows)
        self.assertEqual(float(estimates[0, 1]), 0.0)
        self.assertGreater(float(estimates[1, 0]), 0.5)
        self.assertEqual(float(estimates[1, 1]), 1.0)

    def test_lyapunov_accepts_chaotic_logistic_divergence(self) -> None:
        windows = torch.stack(
            (_logistic_series(3.2), _logistic_series(4.0))
        )[:, :, None]
        estimator = LyapunovComplexityEstimator(
            0,
            embedding_dim=3,
            delay=1,
            theiler_window=20,
            fit_horizon=12,
            min_r2=0.8,
            max_points=256,
        )
        estimates = estimator.estimate(windows)
        self.assertEqual(float(estimates[0, 1]), 0.0)
        self.assertGreater(float(estimates[1, 0]), 0.0)
        self.assertGreaterEqual(float(estimates[1, 1]), 0.8)

    def test_score_router_uses_confidence_as_conservative_fallback(self) -> None:
        features = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
        router = ScoreRouter(percentile=50.0).fit(features)
        weights = router.complex_weight(features)
        self.assertEqual(float(weights[-1]), 0.0)
        self.assertGreater(float(weights[1]), float(weights[0]))


class ComplexityPipelineTests(unittest.TestCase):
    def _data(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "daily.csv"
        fields = (
            "date",
            "discharge_mm_day",
            "precipitation_mm_day",
            "temperature_c",
            "pet_mm_day",
        )
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for index in range(90):
                writer.writerow(
                    {
                        "date": date(2000, 1, 1) + timedelta(days=index),
                        "discharge_mm_day": 1.0 + math.sin(index / 3.0),
                        "precipitation_mm_day": 1.0 + index % 4,
                        "temperature_c": 5.0 + math.sin(index / 20.0),
                        "pet_mm_day": 0.5,
                    }
                )
        return make_hydrology_data(
            load_ukraine_csv(path, basin_id="dynamic"),
            sequence_length=7,
            routing_context_length=21,
        )

    def test_prediction_and_routing_contexts_are_aligned(self) -> None:
        data = self._data()
        self.assertEqual(data.train.inputs.shape[1], 7)
        self.assertEqual(data.train.routing_inputs.shape[1], 21)
        self.assertTrue(
            torch.equal(data.train.inputs, data.train.routing_inputs[:, -7:])
        )

    @unittest.skipUnless(importlib.util.find_spec("ripser"), "ripser is optional")
    def test_consolidation_pipeline_trains_tda_route(self) -> None:
        data = self._data()
        estimator = TakensPersistenceEstimator(
            data.feature_names.index("previous_discharge"),
            embedding_dim=2,
            delay=2,
            min_persistence=0.01,
            max_points=16,
            context_length=21,
        )
        records = compare_hydrology_models_with_consolidation(
            data,
            models=("tda",),
            epochs=1,
            batch_size=128,
            training_noise_levels=(0.0,),
            inference_noise_levels=(0.0,),
            complexity_estimators={"tda": estimator},
        )
        self.assertEqual(records[0].model, "tda")
        self.assertEqual(records[0].valid_complexity_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
