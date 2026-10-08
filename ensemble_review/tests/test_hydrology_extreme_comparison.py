from __future__ import annotations

import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import torch

from complexity_ensemble.hydrology_data import HydrologySeries, make_hydrology_data
from complexity_ensemble.hydrology_extreme_comparison import (
    ExtremeComparisonConfig,
    compare_extreme_event_ensembles,
    extreme_metrics,
    save_extreme_ensemble_records,
)
from examples.compare_extreme_event_ensembles import build_parser


def _synthetic_data():
    sample_count = 140
    dates = tuple(date(2000, 1, 1) + timedelta(days=index) for index in range(sample_count))
    rain = torch.tensor(
        [8.0 if index % 17 in (0, 1) else 0.2 for index in range(sample_count)]
    )
    discharge = torch.tensor(
        [
            1.0 + 0.04 * (index % 17) + (4.0 if index % 17 in (1, 2) else 0.0)
            for index in range(sample_count)
        ]
    )
    forcings = torch.stack(
        (rain, torch.full((sample_count,), 10.0), torch.full((sample_count,), 0.5)),
        dim=1,
    )
    series = HydrologySeries(
        dates,
        discharge,
        forcings,
        ("precipitation", "temperature", "pet"),
        "synthetic",
        "test",
    )
    return make_hydrology_data(series, sequence_length=7)


class ExtremeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _synthetic_data()

    def test_all_three_approaches_share_the_same_event_definition(self) -> None:
        config = ExtremeComparisonConfig(
            seeds=(0,),
            complex_kind="mlp",
            epochs=1,
            complex_epochs=1,
            distillation_epochs=1,
            consolidation_epochs=1,
            batch_size=256,
            include_baselines=False,
        )
        records = compare_extreme_event_ensembles(self.data, config)
        self.assertEqual(
            [record.approach for record in records],
            ["soft_routing", "hard_routing", "distillation"],
        )
        self.assertEqual(len({record.threshold_mm_day for record in records}), 1)
        self.assertEqual(len({record.observed_extreme_days for record in records}), 1)
        self.assertTrue(all(record.parameters > 0 for record in records))
        self.assertTrue(0.0 < records[-1].simple_training_fraction < 1.0)

    def test_independent_baseline_approaches_are_selectable(self) -> None:
        config = ExtremeComparisonConfig(
            approaches=("no_routing", "static_50_50", "ood_fallback", "stacking"),
            seeds=(0,), complex_kind="mlp", epochs=1, batch_size=256,
            include_baselines=False, inference_noise=0.2,
        )
        records = compare_extreme_event_ensembles(self.data, config)
        self.assertEqual(
            [record.approach for record in records],
            ["no_routing", "static_50_50", "ood_fallback", "stacking"],
        )
        self.assertEqual(records[0].mean_complex_weight, 1.0)
        self.assertEqual(records[1].mean_complex_weight, 0.5)
        self.assertTrue(math.isfinite(records[2].ood_threshold))
        self.assertTrue(0.0 <= records[3].mean_complex_weight <= 1.0)
        self.assertTrue(math.isfinite(records[3].stacking_validation_mse_mm2_day2))
        self.assertTrue(all(record.inference_noise == 0.2 for record in records))

    def test_approaches_are_independently_selectable(self) -> None:
        config = ExtremeComparisonConfig(
            approaches=("hard_routing",),
            seeds=(3,),
            complex_kind="mlp",
            epochs=1,
            batch_size=256,
            hard_inference=True,
            include_baselines=False,
        )
        records = compare_extreme_event_ensembles(self.data, config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].approach, "hard_routing")
        self.assertEqual(records[0].inference_routing, "hard")

    def test_distillation_rejects_jointly_learned_router(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed 'morse' router"):
            ExtremeComparisonConfig(
                approaches=("distillation",), routing="learned"
            ).validate()

    def test_event_metrics_decluster_consecutive_days(self) -> None:
        dates = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(6))
        target = torch.tensor([0.0, 2.0, 3.0, 0.0, 4.0, 0.0])
        prediction = torch.tensor([0.0, 2.5, 0.0, 0.0, 5.0, 0.0])
        metrics = extreme_metrics(prediction, target, dates, threshold=1.0)
        self.assertEqual(metrics["observed_events"], 2)
        self.assertEqual(metrics["predicted_events"], 2)
        self.assertEqual(metrics["event_recall"], 1.0)

    def test_records_are_csv_serializable(self) -> None:
        config = ExtremeComparisonConfig(
            approaches=("soft_routing",),
            seeds=(0,),
            complex_kind="mlp",
            epochs=1,
            batch_size=256,
        )
        records = compare_extreme_event_ensembles(self.data, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            save_extreme_ensemble_records(records, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("approach", text)
        self.assertIn("soft_routing", text)

    def test_cli_accepts_an_explicit_comparison(self) -> None:
        args = build_parser().parse_args(
            [
                "--approaches", "no_routing,static_50_50,ood_fallback,stacking",
                "--simple", "fourier",
                "--complex", "pinnmamba",
                "--seeds", "0,42",
                "--inference-noise", "0.25",
            ]
        )
        self.assertEqual(
            args.approaches,
            ("no_routing", "static_50_50", "ood_fallback", "stacking"),
        )
        self.assertEqual(args.seeds, (0, 42))
        self.assertEqual(args.inference_noise, 0.25)

    def test_negative_inference_noise_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise"):
            ExtremeComparisonConfig(inference_noise=-0.1).validate()


if __name__ == "__main__":
    unittest.main()
