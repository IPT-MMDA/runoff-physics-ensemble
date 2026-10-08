from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from complexity_ensemble.hydrology_consolidation_comparison import (
    compare_hydrology_models_with_consolidation,
)
from complexity_ensemble.hydrology_data import load_ukraine_csv, make_hydrology_data


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


class HydrologyConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "ua.csv"
        _write_daily_csv(path)
        self.data = make_hydrology_data(
            load_ukraine_csv(path, basin_id="UA-test"), sequence_length=7
        )

    def test_optional_weight_is_recorded(self) -> None:
        records = compare_hydrology_models_with_consolidation(
            self.data,
            models=("morse",),
            epochs=1,
            batch_size=128,
            training_noise_levels=(0.0,),
            inference_noise_levels=(0.0,),
            consolidation_weight=0.0,
        )
        self.assertEqual(records[0].consolidation_weight, 0.0)

    def test_negative_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_hydrology_models_with_consolidation(
                self.data,
                models=("morse",),
                epochs=1,
                consolidation_weight=-0.1,
            )

    def test_consolidation_comparison_builds_full_noise_grid(self) -> None:
        records = compare_hydrology_models_with_consolidation(
            self.data,
            models=("morse",),
            epochs=1,
            batch_size=128,
            training_noise_levels=(0.0, 0.1),
            inference_noise_levels=(0.0, 0.1),
        )
        self.assertEqual(
            {(record.training_noise, record.inference_noise) for record in records},
            {(0.0, 0.0), (0.0, 0.1), (0.1, 0.0), (0.1, 0.1)},
        )


if __name__ == "__main__":
    unittest.main()
