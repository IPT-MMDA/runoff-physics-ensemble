from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import torch

from complexity_ensemble.hydrology import HydrologyRoutedOutput, RoutedHydrologyModel
from complexity_ensemble.hydrology_data import load_ukraine_csv, make_hydrology_data
from complexity_ensemble.hydrology_distillation import (
    HydrologyDistillationRecord,
    build_parser,
    format_basin_results_table,
    simple_space_distillation_error,
    summarize_basin_results,
    train_hydrology_distill_to_simple,
)


def _write_daily_csv(path: Path, days: int = 80) -> None:
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
        for index in range(days):
            writer.writerow(
                {
                    "date": date(2000, 1, 1) + timedelta(days=index),
                    "discharge_mm_day": 1.0 + 0.01 * index,
                    "precipitation_mm_day": 2.0 if index % 5 == 0 else 0.2,
                    "temperature_c": 5.0 + 0.1 * index,
                    "pet_mm_day": 0.5,
                }
            )


class HydrologyDistillToSimpleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "ua.csv"
        _write_daily_csv(path)
        self.data = make_hydrology_data(
            load_ukraine_csv(path, basin_id="UA-test"), sequence_length=7
        )

    def _model(self) -> RoutedHydrologyModel:
        torch.manual_seed(7)
        return RoutedHydrologyModel(
            self.data.feature_names,
            7,
            self.data.discharge_scale,
            simple_kind="rbf",
            complex_kind="mlp",
            routing="morse",
        )

    def test_staged_training_builds_routed_ensemble(self) -> None:
        model = self._model()
        optimizer_parameter_ids = []
        adam = torch.optim.Adam

        def record_optimizer(parameters, *args, **kwargs):
            parameters = list(parameters)
            optimizer_parameter_ids.append({id(parameter) for parameter in parameters})
            return adam(parameters, *args, **kwargs)

        with patch(
            "complexity_ensemble.hydrology_distillation.torch.optim.Adam",
            side_effect=record_optimizer,
        ):
            report = train_hydrology_distill_to_simple(
                model,
                self.data.train,
                complex_epochs=1,
                distillation_epochs=1,
                consolidation_epochs=1,
                batch_size=128,
                learning_rate=1e-3,
                training_noise=0.0,
                seed=0,
                device="cpu",
                feature_names=self.data.feature_names,
            )

        self.assertTrue(model.router.is_fitted)
        self.assertEqual(len(report.complex_losses), 1)
        self.assertEqual(len(report.distillation_losses), 1)
        self.assertEqual(len(report.consolidation_losses), 1)
        self.assertGreater(report.simple_sample_count, 0)
        self.assertLess(report.simple_sample_count, report.total_sample_count)
        self.assertTrue(0.0 < report.simple_fraction < 1.0)
        simple_parameters = {id(parameter) for parameter in model.simple_expert.parameters()}
        complex_parameters = {id(parameter) for parameter in model.complex_expert.parameters()}
        self.assertEqual(len(optimizer_parameter_ids), 3)
        self.assertTrue(optimizer_parameter_ids[0].isdisjoint(simple_parameters))
        self.assertTrue(complex_parameters.issubset(optimizer_parameter_ids[0]))
        self.assertEqual(optimizer_parameter_ids[1], simple_parameters)
        self.assertTrue(simple_parameters | complex_parameters <= optimizer_parameter_ids[2])
        output = model(self.data.validation.inputs, return_details=True)
        self.assertIsInstance(output, HydrologyRoutedOutput)
        self.assertEqual(output.discharge.shape, (len(self.data.validation.inputs), 1))

    def test_distillation_reports_finite_simple_space_error(self) -> None:
        model = self._model()
        train_hydrology_distill_to_simple(
            model,
            self.data.train,
            complex_epochs=1,
            distillation_epochs=1,
            consolidation_epochs=1,
            batch_size=128,
            learning_rate=1e-3,
            training_noise=0.0,
            seed=0,
            device="cpu",
            feature_names=self.data.feature_names,
        )
        error = simple_space_distillation_error(model, self.data.train.inputs)
        self.assertTrue(torch.isfinite(torch.tensor(error)))

    def test_invalid_phase_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_hydrology_distill_to_simple(
                self._model(),
                self.data.train,
                complex_epochs=1,
                distillation_epochs=0,
                consolidation_epochs=1,
                batch_size=128,
                learning_rate=1e-3,
                training_noise=0.0,
                seed=0,
                device="cpu",
                feature_names=self.data.feature_names,
            )

    def test_unfitted_learned_gate_is_rejected(self) -> None:
        model = RoutedHydrologyModel(
            self.data.feature_names,
            7,
            self.data.discharge_scale,
            routing="learned",
        )
        with self.assertRaisesRegex(ValueError, "fixed complexity-space router"):
            train_hydrology_distill_to_simple(
                model,
                self.data.train,
                complex_epochs=1,
                distillation_epochs=1,
                consolidation_epochs=1,
                batch_size=128,
                learning_rate=1e-3,
                training_noise=0.0,
                seed=0,
                device="cpu",
                feature_names=self.data.feature_names,
            )

    def test_robustness_grid_cli_is_accepted(self) -> None:
        args = build_parser().parse_args(
            [
                "--basins", "2247,2210", "--period", "early:2000-01-01:2009-12-31",
                "--period", "late:2010-01-01:2020-12-31",
                "--models", "morse,learned,single_complex", "--seeds", "0,42",
                "--training-noise-levels", "0,0.1", "--inference-noise-levels", "0,0.25",
                "--inference-noise-draws", "1", "--epochs", "100",
            ]
        )
        self.assertEqual(args.basins, "2247,2210")
        self.assertEqual(len(args.period), 2)
        self.assertEqual(args.complex_epochs, 100)

    def test_final_results_are_aggregated_and_formatted_per_basin(self) -> None:
        first = HydrologyDistillationRecord(
            country="Test", basin="2247", period="early",
            period_start="2000-01-01", period_end="2009-12-31",
            model="morse", training_method="distill_to_simple",
            simple_expert="rbf", complex_expert="mlp", seed=0,
            training_noise=0.1, inference_noise=0.25,
            inference_noise_seed=300007, validation_nse=0.6, test_nse=0.5,
            test_kge=0.4, test_rmse_mm_day=1.0, physics_error=0.2,
            parameters=100, mean_complex_weight=0.2,
            mean_complexity_score=1.0, valid_complexity_fraction=1.0,
            training_seconds=2.0, complex_epochs=1,
            distillation_epochs=1, consolidation_epochs=1,
        )
        rows = summarize_basin_results(
            [first, replace(first, seed=42, test_nse=0.7, training_seconds=4.0)]
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["test_nse_mean"], 0.6)
        self.assertEqual(rows[0]["seed_count"], 2)
        table = format_basin_results_table(rows)
        self.assertIn("2247", table)
        self.assertIn("morse", table)
        self.assertIn("0.6000", table)


if __name__ == "__main__":
    unittest.main()
