"""Numerical and cohort checks for the daily hydropower case study."""
import csv
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import geopandas as gpd
from shapely.geometry import Point, box

from complexity_ensemble.hydropower_comparison import (
    EnergyScenario, compare_hydropower, read_comparison_predictions,
)
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog


class HydropowerComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(
            "basin", "Test basin", {"area": 86.4}, eligible=True, geometry=box(0, 0, 2, 2))])
        self.sites = gpd.GeoDataFrame(
            {"_site_id": ["one", "two", "outside"], "_site_name": ["Plant 1", "Plant 2", "Outside"]},
            geometry=[Point(1, 1), Point(1, 1.5), Point(10, 10)], crs=4326)
        self.a = self.write("a", [(1, "basin", "2020-01-01", 10, 10), (1, "basin", "2020-01-02", 20, 20)])
        self.b = self.write("b", [(1, "basin", "2020-01-01", 10, 11), (1, "basin", "2020-01-02", 20, 18)])

    def write(self, name, rows, metadata=False):
        path = self.root / (name + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seed", "basin_id", "target_date", "observed_mm_day", "predicted_mm_day"]
                            + (["issue_date", "lead_day"] if metadata else []))
            writer.writerows(rows)
        return path

    def compare(self, **kwargs):
        return compare_hydropower(self.a, self.b, self.catalog, self.sites, **kwargs)

    def test_energy_conserves_daily_volume_and_has_correct_units(self):
        scenario = EnergyScenario()
        self.assertAlmostEqual(scenario.energy(10), 211.896)
        self.assertAlmostEqual(scenario.energy(10), 96 * .25 * .00981 * .9 * 100 * 10)
        self.assertEqual(scenario.energy(-1), 0)

    def test_flow_limits_and_asymmetric_penalties(self):
        scenario = EnergyScenario(environmental_flow_m3_s=2, max_turbine_flow_m3_s=5)
        self.assertAlmostEqual(scenario.energy(20), EnergyScenario().energy(5))
        self.assertEqual(scenario.energy(1), 0)
        self.assertEqual(scenario.cost(12, 10), 100)
        self.assertEqual(scenario.cost(8, 10), 40)

    def test_paired_accuracy_costs_and_shared_map_scale(self):
        result = self.compare()
        self.assertEqual((result["samples"], result["basins"]), (2, 1))
        self.assertEqual(result["metrics_a"]["mean_basin_nse"], 1)
        self.assertAlmostEqual(result["metrics_b"]["mean_basin_nse"], .9)
        self.assertEqual(result["metrics_b"]["mae_m3_s"], 1.5)
        self.assertEqual(result["accuracy_change"]["mae_reduction_pct"], 100)
        self.assertAlmostEqual(result["accuracy_change"]["mean_basin_nse_gain"], .1)
        # Two inventory plants in one basin do not double the cost.
        self.assertAlmostEqual(result["economics"]["cost_b_eur"], 21.1896 * (50 + 2 * 20))
        self.assertEqual(result["economics"]["saving_pct"], 100)
        self.assertEqual(result["economics"]["cost_index_a"], 0)
        props = result["map"]["features"][0]["properties"]
        self.assertEqual(props["error_b_m3_s"], -2)
        self.assertEqual(result["map"]["error_scale_m3_s"], 2)
        self.assertIsNone(result["map"]["features"][2]["properties"]["a"])
        self.assertIsNone(result["map"]["features"][2]["properties"]["basin_id"])

    def test_reversed_runs_preserve_negative_savings(self):
        result = compare_hydropower(self.b, self.a, self.catalog, self.sites)
        self.assertLess(result["economics"]["saving_eur"], 0)
        self.assertIsNone(result["economics"]["saving_pct"])
        self.assertIsNone(result["economics"]["cost_index_b"])

    def test_zero_penalties_and_undefined_nse(self):
        result = self.compare(start=date(2020, 1, 2), scenario=EnergyScenario(shortfall_eur_mwh=0, surplus_eur_mwh=0))
        self.assertIsNone(result["metrics_a"]["mean_basin_nse"])
        self.assertIsNone(result["economics"]["saving_pct"])
        self.assertEqual(result["samples"], 1)

    def test_period_and_map_date_validation(self):
        with self.assertRaisesRegex(ValueError, "Start"):
            self.compare(start=date(2021, 1, 1), end=date(2020, 1, 1))
        with self.assertRaisesRegex(ValueError, "Map date"):
            self.compare(target_date=date(2022, 1, 1))

    def test_missing_and_conflicting_observations_are_excluded(self):
        self.b = self.write("b", [(1, "basin", "2020-01-01", 999, 11),
            (1, "basin", "2020-01-02", 20, 18), (1, "basin", "2020-01-03", "nan", 2)])
        result = self.compare()
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["audit"]["observation_conflicts"], 1)
        self.assertEqual(result["audit"]["run_b"]["invalid_rows"], 1)

    def test_seeds_are_averaged_and_duplicates_rejected(self):
        path = self.write("seeds", [(1, "basin", "2020-01-01", 10, 8), (2, "basin", "2020-01-01", 10, 12)])
        result, _ = read_comparison_predictions(path)
        self.assertEqual(next(iter(result.values())), (10, 10))
        path = self.write("duplicates", [(1, "basin", "2020-01-01", 10, 8)] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_comparison_predictions(path)

    def test_seed_observation_conflict_rejected(self):
        path = self.write("seeds", [(1, "basin", "2020-01-01", 10, 8), (2, "basin", "2020-01-01", 11, 12)])
        with self.assertRaisesRegex(ValueError, "Observations differ"):
            read_comparison_predictions(path)

    def test_selected_lead_is_never_mixed_with_other_leads(self):
        path = self.write("rolling", [(1, "basin", "2020-01-02", 20, 18, "2020-01-01", 1),
            (1, "basin", "2020-01-02", 20, 100, "2019-12-31", 2)], metadata=True)
        result, audit = read_comparison_predictions(path, lead_day=1, rolling=True)
        self.assertEqual(next(iter(result.values()))[1], 18)
        self.assertEqual(audit["other_lead_rows"], 1)
        result, audit = read_comparison_predictions(self.a, rolling=True)
        self.assertEqual(result, {})
        self.assertEqual(audit["invalid_rows"], 2)

    def test_different_issue_lead_forecasts_do_not_pair(self):
        self.b = self.write("rolling", [(1, "basin", "2020-01-02", 20, 18, "2019-12-31", 2)], metadata=True)
        with self.assertRaisesRegex(ValueError, "No paired"):
            self.compare()

    def test_incompatible_observation_definitions_rejected(self):
        with self.assertRaisesRegex(ValueError, "observation source"):
            self.compare(config_b={"benchmark": {"mode": "bafu_only"}})

    def test_daily_maximum_cannot_be_treated_as_daily_mean_energy(self):
        config = {"benchmark": {"target": "daily_max"}}
        with self.assertRaisesRegex(ValueError, "daily_mean"):
            self.compare(config_a=config, config_b=config)

    def test_missing_observation_column_rejected(self):
        self.b.write_text("seed,basin_id,target_date,predicted_mm_day\n1,basin,2020-01-01,1\n")
        with self.assertRaisesRegex(ValueError, "observed_mm_day"):
            self.compare()

    def test_invalid_scenario_rejected(self):
        for kwargs in ({"head_m": 0}, {"efficiency": 1.1}, {"shortfall_eur_mwh": -1},
                       {"head_m": float("nan")}, {"max_turbine_flow_m3_s": 0}):
            with self.assertRaises(ValueError):
                EnergyScenario(**kwargs)

    def test_same_run_is_a_valid_zero_savings_sanity_check(self):
        result = compare_hydropower(self.b, self.b, self.catalog, self.sites)
        self.assertEqual(result["economics"]["saving_pct"], 0)
        self.assertEqual(result["economics"]["cost_index_a"], 100)


if __name__ == "__main__":
    unittest.main()
