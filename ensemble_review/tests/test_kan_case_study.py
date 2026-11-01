"""Conservation, symbolic execution, leakage and queued workflow checks."""
from dataclasses import asdict, replace
from datetime import date, timedelta
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import sympy as sp
import torch

from complexity_ensemble.hydrology_data import HydrologySeries, load_camels_ch
from complexity_ensemble.kan_case_study import (
    BasinParameters, CaseStudyConfig, NeuralFlux, ReservoirFlux, StudyCancelled,
    SymbolicFlux, distil_symbolic, prepare_data, run_case_study, save_case_study, simulate,
    case_study_hardware, config_from_dict,
    COMPARISON_MODELS, DEFAULT_COMPARISON_MODELS, kan_discovery_profiles,
    BASE_SYMBOLIC_LIBRARY, COMPLEX_SYMBOLIC_LIBRARY,
)


def small_config():
    return CaseStudyConfig(source_basins=("source",), validation_basins=("validation",),
        target_basins=("target",), train_start="2001-01-09", train_end="2001-01-30",
        validation_start="2001-01-31", validation_end="2001-02-10",
        test_start="2001-02-11", test_end="2001-02-25", warmup_days=8,
        minimum_source_days=10, observation_budgets=(3, 6), teacher_epochs=2,
        adaptation_epochs=2, teacher_width=3, kan_candidates=1, kan_steps=3,
        kan_hidden=1, max_formula_ops=1000, max_distillation_samples_per_basin=15, device="cpu", selected_models=("kan_pure", "kan_tuned"),
        sequence_length=3, neural_epochs=1, neural_batch_size=128, mlp_widths=(4,),
        mamba_hidden_size=4, mamba_feed_forward_size=8, lstm_hidden_size=4)


def synthetic_series():
    dates = tuple(date(2001, 1, 1) + timedelta(days=i) for i in range(56))
    t = torch.arange(56.)
    forcing = torch.stack((torch.where(t.remainder(6) < 2, 5., .2),
                           3. * torch.sin(t / 5.), torch.ones(56) * .3), 1)
    result = {}
    for i, basin in enumerate(("source", "validation", "target")):
        with torch.no_grad():
            rows, _, _ = simulate(forcing, ReservoirFlux(), BasinParameters([-.2 * i, -1., .05 * i]))
        result[basin] = HydrologySeries(dates, rows[:, 0], forcing,
                                        ("precipitation", "temperature", "pet"), basin, "CH")
    return result


class ProcessTests(unittest.TestCase):
    def test_comparison_model_selection_and_required_cohorts(self):
        self.assertEqual(len(COMPARISON_MODELS), 12)
        self.assertNotIn("kan_complex", DEFAULT_COMPARISON_MODELS)
        for selection in ((), ("mlp_bad_scope",), ("kan_pure", "kan_pure")):
            with self.assertRaisesRegex(ValueError, "comparison model"):
                replace(small_config(), selected_models=selection).validate()
        replace(small_config(), selected_models=("mlp_target_target",), source_basins=(), validation_basins=()).validate()
        replace(small_config(), selected_models=("mlp_source_target",), validation_basins=()).validate()

    def test_complex_discovery_is_separate_and_preserves_kan_architecture(self):
        config = replace(small_config(), selected_models=("kan_pure", "kan_tuned", "kan_complex"))
        standard, complex_profile = kan_discovery_profiles(config)
        self.assertIs(standard[2], config)
        self.assertEqual(standard[1], ("kan_pure", "kan_tuned"))
        self.assertEqual(standard[3], BASE_SYMBOLIC_LIBRARY)
        enhanced = complex_profile[2]
        self.assertEqual((enhanced.teacher_epochs, enhanced.teacher_width, enhanced.kan_candidates), (50, 24, 6))
        self.assertEqual((enhanced.kan_hidden, enhanced.kan_grid, enhanced.kan_steps),
                         (config.kan_hidden, config.kan_grid, config.kan_steps))
        self.assertEqual(complex_profile[3], COMPLEX_SYMBOLIC_LIBRARY)
        larger = replace(config, teacher_epochs=60, teacher_width=32, kan_candidates=8)
        self.assertEqual(kan_discovery_profiles(larger)[1][2], larger)
        complex_only = replace(config, selected_models=("kan_complex",))
        self.assertEqual(len(kan_discovery_profiles(complex_only)), 1)
        self.assertTrue(complex_only.needs_target_labels())
        complex_only.validate()
        series = synthetic_series()
        observed = series["target"].discharge.clone()
        observed[8:30] = float("nan")
        series["target"] = replace(series["target"], discharge=observed)
        with self.assertRaisesRegex(ValueError, "largest observation budget"):
            prepare_data(complex_only, series_by_basin=series)

    def test_complex_symbolic_functions_execute_with_gradients(self):
        storage, liquid, pet = sp.symbols("storage liquid pet")
        flux = SymbolicFlux(storage ** 3 + sp.Abs(liquid) + sp.tanh(pet))
        x = torch.tensor([[.5, -.2, .3]], requires_grad=True)
        expected = x[:, 0] ** 3 + x[:, 1].abs() + x[:, 2].tanh()
        self.assertTrue(torch.allclose(flux(x), expected))
        flux(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_hardware_choices_follow_shared_resolver_and_report_phase_fallbacks(self):
        self.assertEqual(case_study_hardware("cpu")["resolved"], "cpu")
        for selected, expected_kan in (("mps", "cpu"), ("cuda", "cuda")):
            with patch("complexity_ensemble.kan_case_study.resolve_device", return_value=torch.device(selected)):
                hardware = case_study_hardware(selected)
            self.assertEqual(hardware["teacher"], selected)
            self.assertEqual(hardware["calibration"], selected)
            self.assertEqual(hardware["simulation"], selected)
            self.assertEqual(hardware["kan_fitting"], expected_kan)
            self.assertEqual(hardware["symbolic_extraction"], "cpu")
        with self.assertRaisesRegex(ValueError, "Device must be"):
            replace(small_config(), device="invalid").validate()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA was requested but is not available"):
                case_study_hardware("cuda")
        config = config_from_dict({**asdict(small_config()), "device": "cpu"})
        self.assertEqual(config.device, "cpu")

    def test_unknown_job_kind_fails_before_reading_run_id(self):
        from types import SimpleNamespace
        from complexity_ensemble.worker import run_job
        store = SimpleNamespace(get=lambda job_id: {"cancel_requested": False})
        with self.assertRaisesRegex(ValueError, "Unsupported job kind.*restart"):
            run_job(None, store, {"id": "unknown", "kind": "future_study", "payload": {}}, "owner")

    def test_daily_conservation_nonnegative_states_and_gradients(self):
        forcing = torch.tensor([[10., -3., .5], [0., 4., .2], [1200., 9., 1.], [0., 1., 1000.]])
        params = BasinParameters()
        flux = NeuralFlux(3)
        rows, _, _ = simulate(forcing, flux, params)
        self.assertTrue((rows[:, :4] >= 0).all())
        self.assertLess(float(rows[:, 4].abs().max()), 2e-4)
        self.assertAlmostEqual(float(forcing[:, 0].sum()),
            float(rows[:, 0].sum() + rows[:, 1].sum() + rows[-1, 2] + rows[-1, 3]), places=3)
        rows[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(params.raw.grad).all())
        self.assertEqual(params.raw.numel(), 3)

    def test_symbolic_expression_is_executed_with_gradients(self):
        storage, liquid, pet = sp.symbols("storage liquid pet")
        flux = SymbolicFlux(2 * storage + sp.tanh(liquid) - pet ** 2)
        x = torch.tensor([[.2, .5, .1]], requires_grad=True)
        expected = 2 * x[:, 0] + torch.tanh(x[:, 1]) - x[:, 2] ** 2
        self.assertTrue(torch.allclose(flux(x), expected))
        flux(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(sum(p.numel() for p in flux.parameters()), 0)

    def test_disjoint_cohorts_gaps_and_budget_validation(self):
        config = small_config()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            replace(config, target_basins=config.source_basins).validate()
        with self.assertRaisesRegex(ValueError, "increasing"):
            replace(config, observation_budgets=(6, 3)).validate()
        with self.assertRaisesRegex(ValueError, "largest observation budget"):
            prepare_data(replace(config, observation_budgets=(40,)), series_by_basin=synthetic_series())
        data = synthetic_series()
        item = data["target"]
        data["target"] = replace(item, dates=item.dates[1:], discharge=item.discharge[1:], forcings=item.forcings[1:])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            prepare_data(config, series_by_basin=data)

    def test_cancellation_before_data_access(self):
        class Cancelled:
            def is_cancelled(self): return True
        with self.assertRaises(StudyCancelled):
            run_case_study(small_config(), cancellation_token=Cancelled())

    def test_case_study_loader_keeps_weather_dates_with_missing_discharge(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "timeseries"
            (root / "observation_based").mkdir(parents=True)
            (root / "simulation_based").mkdir()
            observed = "date,discharge_spec(mm/d)\n"
            weather = "date,precipitation_sim(mm/d),temperature_sim(degC),pet_sim(mm/d)\n"
            for i in range(1, 6):
                observed += f"2001-01-{i:02d},{'' if i == 3 else '1'}\n"
                weather += f"2001-01-{i:02d},3,5,1\n"
            (root / "observation_based" / "CAMELS_CH_obs_based_target.csv").write_text(observed)
            (root / "simulation_based" / "CAMELS_CH_sim_based_target.csv").write_text(weather)
            kwargs = dict(start="2001-01-01", end="2001-01-05", include_landcover=False)
            normal = load_camels_ch(root.parent, "target", **kwargs)
            low_data = load_camels_ch(root.parent, "target", allow_missing_discharge=True, **kwargs)
            self.assertEqual(len(normal.dates), 4)
            self.assertEqual(len(low_data.dates), 5)
            self.assertTrue(torch.isnan(low_data.discharge[2]))

    def test_retired_regional_mode_fails_with_migration_guidance(self):
        from complexity_ensemble.configuration import config_from_mapping
        from complexity_ensemble.regional import BasinScopeConfig, DateRange, DateSplitConfig, RegionalExperimentConfig
        config = RegionalExperimentConfig(".", BasinScopeConfig(("target",), "all_basins"),
            DateSplitConfig(DateRange("2001-01-01", "2001-01-30"),
                            DateRange("2001-01-31", "2001-02-10"),
                            DateRange("2001-02-11", "2001-02-25")))
        old = asdict(config)
        old["physics"].update(optimization_mode="distillation", kan_steps=50, distillation_teacher="hard_routing")
        with self.assertRaisesRegex(ValueError, "separate KAN case study"):
            config_from_mapping(old).validate()


class NeuralComparisonTests(unittest.TestCase):
    def test_scopes_pool_all_target_budgets_and_reuse_source_fit(self):
        series = synthetic_series()
        series["second_target"] = replace(series["target"], basin_id="second_target")
        config = replace(small_config(), target_basins=("target", "second_target"),
            validation_basins=(), selected_models=("mlp_source_target", "mlp_target_target", "mlp_mixed_target"),
            target_observation_budgets={"second_target": (4,)})
        with patch("complexity_ensemble.kan_case_study._discover_source_equation", side_effect=AssertionError("KAN should be skipped")):
            result = run_case_study(config, series_by_basin=series)
        self.assertEqual(len(result["metrics"]), 12)
        self.assertEqual(len(result["model_fits"]), 5)
        self.assertEqual(result["equations"], [])
        self.assertEqual(result["hardware"]["kan_fitting"], "not_used")
        self.assertEqual(result["audit"]["budget_scenarios"][1]["target_budgets"], {"target": 6, "second_target": 4})
        source = [r for r in result["model_fits"] if r["training_scope"] == "source"]
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0]["training_labels"], {"source": 22})
        for fitted in result["model_fits"]:
            if fitted["training_scope"] != "source":
                target_budget = 3 if fitted["scenario"] == 1 else 6
                self.assertEqual(fitted["training_labels"]["target"], target_budget)
                self.assertEqual(fitted["training_labels"]["second_target"], 4)
                self.assertEqual("source" in fitted["training_labels"], fitted["training_scope"] == "mixed")

    def test_source_model_can_predict_targets_without_local_training_labels(self):
        series = synthetic_series()
        q = series["target"].discharge.clone(); q[8:30] = float("nan")
        series["target"] = replace(series["target"], discharge=q)
        config = replace(small_config(), validation_basins=(), selected_models=("mlp_source_target",), observation_budgets=(3,))
        result = run_case_study(config, series_by_basin=series)
        self.assertEqual(result["model_fits"][0]["training_labels"], {"source": 22})
        self.assertIsNone(result["metrics"][0]["calibration_first"])
        self.assertEqual(result["metrics"][0]["days"], 15)

    def test_target_only_model_ignores_source_and_held_out_target_labels(self):
        config = replace(small_config(), source_basins=(), validation_basins=(),
                         selected_models=("mlp_target_target",), observation_budgets=(3,))
        series = synthetic_series()
        first = run_case_study(config, series_by_basin=series)
        q = series["target"].discharge.clone(); q[11:] = q[11:] * 8 + 20
        series["target"] = replace(series["target"], discharge=q)
        series["source"] = replace(series["source"], discharge=series["source"].discharge + 200)
        second = run_case_study(config, series_by_basin=series)
        self.assertEqual(first["model_fits"], second["model_fits"])
        self.assertEqual([r["predicted_mm_day"] for r in first["predictions"]], [r["predicted_mm_day"] for r in second["predictions"]])
        self.assertNotEqual(first["metrics"][0]["rmse_mm_day"], second["metrics"][0]["rmse_mm_day"])


@unittest.skipUnless(importlib.util.find_spec("kan"), "Install pykan for real symbolic integration")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.series = synthetic_series()
        cls.result = run_case_study(cls.config, series_by_basin=cls.series)

    def test_known_flux_relation_can_be_approximated_on_unseen_states(self):
        x = torch.rand((180, 3), generator=torch.Generator().manual_seed(23))
        logits = (2 * x[:, 0] + .5 * x[:, 1] - x[:, 2] - 2)[:, None]
        config = replace(self.config, kan_steps=100, kan_hidden=2, sparsity_weight=.0001)
        flux, diagnostics = distil_symbolic(x[:120], logits[:120], x[120:], logits[120:], config, 0)
        error = (flux(x[120:]).sigmoid() - logits[120:, 0].sigmoid()).square().mean()
        self.assertLess(float(error), .003)
        self.assertLess(diagnostics["candidates"][0]["validation_fraction_mse"], .003)

    def test_real_pykan_pipeline_records_executed_expression_and_budgets(self):
        result = self.result
        self.assertEqual(result["hardware"]["resolved"], "cpu")
        self.assertEqual(result["hardware"]["teacher"], "cpu")
        self.assertEqual(result["equations"][0]["kan_fitting_device"], "cpu")
        self.assertEqual(len(result["metrics"]), 4)
        self.assertEqual(len(result["predictions"]), 4 * 15)
        self.assertFalse(result["audit"]["target_test_used_for_fitting"])
        self.assertEqual(result["audit"]["equation_selection_basins"], ["validation"])
        self.assertTrue(result["equations"][0]["logit_expression"])
        self.assertEqual(result["equations"][0]["candidates"][0]["status"], "ok")
        for row in result["metrics"]:
            self.assertLess(row["max_abs_water_balance_mm"], 1e-3)
            self.assertEqual(row["trainable_parameters"], 0 if row["method"] == "kan_pure" else 3)
            if row["method"] == "kan_tuned":
                self.assertEqual(row["calibration_first"], "2001-01-09")
                self.assertEqual(row["calibration_last"], "2001-01-11" if row["budget"] == 3 else "2001-01-14")
            else:
                self.assertIsNone(row["calibration_first"])

    def _check_accelerator_pipeline(self, selected):
        seen = set()
        def observe(forcing, *args, **kwargs):
            seen.add(forcing.device.type)
            return simulate(forcing, *args, **kwargs)
        with patch("complexity_ensemble.kan_case_study.simulate", side_effect=observe):
            result = run_case_study(replace(self.config, device=selected), series_by_basin=self.series)
        self.assertEqual(seen, {selected})
        self.assertEqual(result["hardware"]["resolved"], selected)
        self.assertEqual(result["equations"][0]["kan_fitting_device"], "cuda" if selected == "cuda" else "cpu")
        self.assertTrue(all(math.isfinite(row["rmse_mm_day"]) for row in result["metrics"]))
        json.dumps(result, allow_nan=False)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable on this test host")
    def test_mps_pipeline_uses_metal_for_training_and_cpu_for_kan(self):
        self._check_accelerator_pipeline("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this test host")
    def test_cuda_pipeline_uses_cuda_for_training_and_numeric_kan(self):
        self._check_accelerator_pipeline("cuda")

    def test_target_budget_override_is_used_by_calibration_and_saved_results(self):
        config = replace(self.config, observation_budgets=(40,),
                         target_observation_budgets={"target": (3, 6)})
        result = run_case_study(config, series_by_basin=self.series)
        self.assertEqual(len(result["metrics"]), 4)
        self.assertEqual({row["budget"] for row in result["metrics"]}, {3, 6})
        self.assertEqual(result["config"]["target_observation_budgets"], {"target": (3, 6)})

    def test_all_selected_models_run_and_record_exact_training_cohorts(self):
        result = run_case_study(replace(self.config, selected_models=COMPARISON_MODELS, observation_budgets=(3,)),
                                series_by_basin=self.series)
        self.assertEqual({r["method"] for r in result["metrics"]}, set(COMPARISON_MODELS))
        self.assertEqual(len(result["metrics"]), 12)
        self.assertEqual(len(result["predictions"]), 12 * 15)
        self.assertEqual(len(result["equations"]), 2)
        standard, complex_equation = result["equations"]
        self.assertEqual(standard["logit_expression"], self.result["equations"][0]["logit_expression"])
        for method in ("kan_pure", "kan_tuned"):
            self.assertEqual(next(r for r in result["metrics"] if r["method"] == method),
                             next(r for r in self.result["metrics"] if r["method"] == method and r["budget"] == 3))
        self.assertEqual(complex_equation["models"], ["kan_complex"])
        self.assertEqual(complex_equation["discovery_settings"]["teacher_epochs"], 50)
        self.assertEqual(len(complex_equation["candidates"]), 6)
        complex_rows = [r for r in result["metrics"] if r["method"] == "kan_complex"]
        self.assertTrue(all(r["local_trainable_parameters"] == 3 for r in complex_rows))
        self.assertTrue(all(r["calibration_first"] == "2001-01-09" for r in complex_rows))
        self.assertEqual(len(result["model_fits"]), 9)
        for fitted in result["model_fits"]:
            expected = ({"source": 22} if fitted["training_scope"] == "source" else
                        {"target": 3} if fitted["training_scope"] == "target" else {"source": 22, "target": 3})
            self.assertEqual(fitted["training_labels"], expected)
            self.assertNotIn("validation", fitted["training_labels"])
        neural_rows = [r for r in result["metrics"] if not r["method"].startswith("kan_")]
        self.assertTrue(all(r["max_abs_water_balance_mm"] is None for r in neural_rows))
        json.dumps(result, allow_nan=False)

    def test_hidden_target_labels_cannot_change_equation_adaptation_or_predictions(self):
        modified = copy.deepcopy(self.series)
        target = modified["target"]
        observed = target.discharge.clone()
        # After the largest allowed budget: training remainder, validation and test.
        observed[14:] = observed[14:] * 7 + 20
        modified["target"] = replace(target, discharge=observed)
        result = run_case_study(self.config, series_by_basin=modified)
        self.assertEqual(result["equations"], self.result["equations"])
        self.assertEqual(result["traces"], self.result["traces"])
        for old, new in zip(self.result["predictions"], result["predictions"]):
            self.assertEqual(old["predicted_mm_day"], new["predicted_mm_day"])
        self.assertNotEqual(result["metrics"][0]["rmse_mm_day"], self.result["metrics"][0]["rmse_mm_day"])

    def test_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = save_case_study(self.result, folder)
            with zipfile.ZipFile(archive) as bundle:
                restored = json.loads(bundle.read("case_study.json"))
                self.assertEqual(restored["equations"], self.result["equations"])
                self.assertIn("balance_error_mm", bundle.read("predictions.csv").decode())
                self.assertIn("audit.json", bundle.namelist())

    def test_missing_target_discharge_does_not_remove_weather_or_simulated_days(self):
        series = copy.deepcopy(self.series)
        observed = series["target"].discharge.clone()
        observed[:8] = float("nan")
        observed[[9, 11, 13, 15, 42]] = float("nan")
        series["target"] = replace(series["target"], discharge=observed)
        result = run_case_study(self.config, series_by_basin=series)
        self.assertEqual(result["equations"], self.result["equations"])
        self.assertEqual(len(result["predictions"]), len(self.result["predictions"]))
        self.assertEqual(sum(r["observed_mm_day"] is None for r in result["predictions"]), 4)
        self.assertTrue(all(r["days"] == 14 for r in result["metrics"]))
        for row in result["metrics"]:
            if row["budget"] == 6 and row["method"] == "kan_tuned":
                self.assertEqual(row["calibration_last"], "2001-01-18")
        json.dumps(result, allow_nan=False)

    def test_authenticated_api_queue_worker_result_and_archive(self):
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.application import Application
        from complexity_ensemble.jobs import JobStore
        from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog
        from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings
        from complexity_ensemble.worker import run_job
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = ServerSettings(root / "state", {"ch": DatasetRegistration(root / "data")}, token="secret")
            application = Application(settings)
            catalog = CAMELSCHCatalog(root / "data", [BasinCatalogRecord(b) for b in self.series])
            catalog.load_series = lambda basin, start, end: self.series[basin]
            catalog.load_case_study_series = catalog.load_series
            application.catalog = lambda dataset: catalog
            store = JobStore(settings.jobs_path)
            with TestClient(create_app(settings, application=application, store=store)) as client:
                self.assertEqual(client.get("/api/v1/case-studies/kan").status_code, 401)
                client.headers["Authorization"] = "Bearer secret"
                payload = {"dataset_id": "ch", "config": asdict(self.config)}
                self.assertEqual(client.post("/api/v1/case-studies/kan/validate", json=payload).status_code, 200)
                response = client.post("/api/v1/case-studies/kan", json=payload)
                self.assertEqual(response.status_code, 202, response.text)
                store.acquire_worker("test", stale_after=0)
                job = store.claim("test")
                with patch("complexity_ensemble.kan_case_study.run_case_study", return_value=copy.deepcopy(self.result)):
                    result = run_job(application, store, job, "test")
                store.finish(job["id"], "test", result=result)
                store.release_worker("test")
                report = client.get(f"/api/v1/jobs/{job['id']}/result")
                self.assertEqual(report.status_code, 200)
                self.assertEqual(report.json()["kind"], "kan_case_study")
                self.assertEqual(client.get(f"/api/v1/case-studies/kan/jobs/{job['id']}/archive").status_code, 200)
                self.assertEqual(len(client.get("/api/v1/case-studies/kan").json()["items"]), 1)
                payload["config"]["target_basins"] = ["source"]
                self.assertEqual(client.post("/api/v1/case-studies/kan", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
