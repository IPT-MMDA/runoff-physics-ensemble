from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import torch

from complexity_ensemble.hydrology import (
    HydrologyPINNMambaExpert,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
    hydrology_morse_gradient,
)
from complexity_ensemble.hydrology_comparison import (
    _add_observation_noise,
    compare_hydrology_models,
)
from complexity_ensemble.hydrology_data import load_ukraine_csv, make_hydrology_data
from complexity_ensemble.pinnmamba import PINNMamba


def write_daily_csv(path: Path, days: int = 80) -> None:
    fields = (
        "date", "discharge_mm_day", "precipitation_mm_day",
        "temperature_c", "pet_mm_day",
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


class HydrologyDataTests(unittest.TestCase):
    def test_ukraine_contract_and_chronological_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ua.csv"
            write_daily_csv(path)
            series = load_ukraine_csv(path, basin_id="UA-test")
            data = make_hydrology_data(series, sequence_length=7)
        self.assertEqual(series.country, "Ukraine")
        self.assertEqual(data.feature_names[:3], ("precipitation", "temperature", "pet"))
        self.assertLess(data.train.target_dates[-1], data.validation.target_dates[0])
        self.assertLess(data.validation.target_dates[-1], data.test.target_dates[0])
        self.assertEqual(data.train.inputs.shape[1:], (7, 6))

    def test_gaps_are_not_bridged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ua.csv"
            write_daily_csv(path, 80)
            rows = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(rows[:30] + rows[31:]) + "\n", encoding="utf-8")
            data = make_hydrology_data(load_ukraine_csv(path, basin_id="gap"), sequence_length=7)
        for split in (data.train, data.validation, data.test):
            self.assertTrue(all(a < b for a, b in zip(split.target_dates, split.target_dates[1:])))


class HydrologyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "ua.csv"
        write_daily_csv(path)
        self.data = make_hydrology_data(load_ukraine_csv(path, basin_id="UA-test"), sequence_length=7)

    def test_pinnmamba_accepts_hydrology_feature_dimension(self) -> None:
        model = PINNMamba(in_dim=6, out_dim=1, hidden_dim=8, hidden_d_ff=16)
        self.assertEqual(model(torch.randn(2, 7, 6)).shape, (2, 7, 1))

    def test_morse_gradient_has_matching_shape(self) -> None:
        inputs = self.data.train.inputs[:3].flatten(start_dim=1)
        gradient = hydrology_morse_gradient(
            inputs, sequence_length=7, feature_names=self.data.feature_names
        )
        self.assertEqual(gradient.shape, inputs.shape)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_observation_noise_only_changes_dynamic_features(self) -> None:
        inputs = torch.zeros(2, 7, len(self.data.feature_names))
        noisy = _add_observation_noise(
            inputs,
            self.data.feature_names,
            0.1,
            torch.Generator().manual_seed(123),
        )
        dynamic = {"precipitation", "temperature", "pet", "previous_discharge"}
        for index, name in enumerate(self.data.feature_names):
            if name in dynamic:
                self.assertGreater(float(noisy[:, :, index].abs().sum()), 0.0)
            else:
                self.assertEqual(float(noisy[:, :, index].abs().sum()), 0.0)

    def test_all_candidates_use_selected_pinnmamba(self) -> None:
        morse = RoutedHydrologyModel(
            self.data.feature_names, 7, self.data.discharge_scale,
            complex_kind="pinnmamba", routing="morse",
        )
        learned = RoutedHydrologyModel(
            self.data.feature_names, 7, self.data.discharge_scale,
            complex_kind="pinnmamba", routing="learned",
        )
        single = SingleComplexHydrologyModel(
            self.data.feature_names, 7, self.data.discharge_scale,
            complex_kind="pinnmamba",
        )
        self.assertIsInstance(morse.complex_expert, HydrologyPINNMambaExpert)
        self.assertIsInstance(learned.complex_expert, HydrologyPINNMambaExpert)
        self.assertIsInstance(single.complex_expert, HydrologyPINNMambaExpert)

    def test_comparison_smoke(self) -> None:
        records = compare_hydrology_models(
            self.data, seeds=(0,), epochs=1, batch_size=128,
            training_noise_levels=(0.0,), inference_noise_levels=(0.0,),
        )
        self.assertEqual([record.model for record in records], ["morse", "learned", "single_complex"])
        self.assertTrue(all(record.complex_expert == "mlp" for record in records))

    def test_comparison_can_train_only_morse(self) -> None:
        records = compare_hydrology_models(
            self.data, models=("morse",), seeds=(0, 1), epochs=1, batch_size=128,
            training_noise_levels=(0.0,), inference_noise_levels=(0.0,),
        )
        self.assertEqual([record.model for record in records], ["morse", "morse"])

    def test_comparison_builds_full_noise_grid(self) -> None:
        records = compare_hydrology_models(
            self.data, models=("morse",), seeds=(0,), epochs=1, batch_size=128,
            training_noise_levels=(0.0, 0.1), inference_noise_levels=(0.0, 0.1),
        )
        self.assertEqual(
            {(record.training_noise, record.inference_noise) for record in records},
            {(0.0, 0.0), (0.0, 0.1), (0.1, 0.0), (0.1, 0.1)},
        )

if __name__ == "__main__":
    unittest.main()
