"""Independent catalogue matching, spatial assignment and saved-run API integration."""
from dataclasses import replace
from datetime import date, timedelta
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from complexity_ensemble.hanze import (CatalogueEvent, evaluate, evaluate_run, gauge_regions,
                                       load_events, match_catalogue_events)
from complexity_ensemble.regional import (BasinCatalogRecord, CAMELSCHCatalog, ExperimentResult,
                                          PredictionRecord)
from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings

DAY = timedelta(days=1)
START = date(2005, 8, 1)


def configuration():
    return {"benchmark": {"mode": "bafu_only"}, "extremes": {"mode": "statistical"},
            "benchmark_manifest": {"thresholds_mm_day": {"A": 5, "B": 5}},
            "dates": {"test": {"start": START.isoformat(), "end": (START + 11 * DAY).isoformat()}},
            "basins": {"target_basins": ["A", "B"]}, "hyperparameters": {"seeds": [0]},
            "forecast": {"mode": "one_day"}, "dataset_id": "ch"}


def predictions(basin="A", seed=0, lead=1, peaks=(3, 4, 8)):
    return [PredictionRecord(seed, basin, START + i * DAY, 1, 10 if i in peaks else 1, .5,
                             issue_date=START + (i - lead) * DAY, lead_day=lead) for i in range(12)]


def events():
    return [CatalogueEvent("flood", START + 3 * DAY, START + 4 * DAY, ("CH021",)),
            CatalogueEvent("outside-region", START + 8 * DAY, START + 8 * DAY, ("CH040",))]


class HanzeMatchingTests(unittest.TestCase):
    def test_known_case_and_unlinked_prediction_have_distinct_meanings(self):
        report = evaluate(predictions(), configuration(), events(), {"A": {"region": "CH021"}, "B": {"region": "CH040"}})
        a, b = report["metrics"]
        self.assertEqual((a["eligible_events"], a["predicted_events"], a["matched_events"]), (1, 2, 1))
        self.assertEqual(a["catalogue_precision"], .5)
        self.assertEqual(a["catalogue_recall"], 1)
        self.assertEqual(a["unlinked_predicted_events"], 1)
        self.assertEqual(b["status"], "unavailable")
        self.assertIsNone(b["catalogue_recall"])
        self.assertEqual(report["events"][0]["hanze_id"], "flood")
        self.assertTrue(report["events"][0]["matched"])
        self.assertEqual([p["catalogue_linked"] for p in report["predicted_events"]], [True, False])
        self.assertNotIn("false_alarm_ratio", a)

    def test_missing_event_days_are_excluded_not_scored_as_misses(self):
        rows = [r for r in predictions() if r.target_date != START + 4 * DAY]
        report = evaluate(rows, configuration(), events(), {"A": {"region": "CH021"}})
        a = report["metrics"][0]
        self.assertEqual((a["eligible_events"], a["excluded_events"]), (0, 1))
        self.assertIsNone(a["catalogue_recall"])
        self.assertEqual(report["events"][0]["exclusion_reason"], "incomplete_prediction_coverage")
        self.assertIsNone(report["events"][0]["matched"])

    def test_event_crossing_test_boundary_and_censored_forecast_are_excluded(self):
        event = CatalogueEvent("partial", START - DAY, START + DAY, ("CH021",))
        report = evaluate(predictions(peaks=(0, 1)), configuration(), [event], {"A": {"region": "CH021"}})
        a = report["metrics"][0]
        self.assertEqual(a["excluded_events"], 1)
        self.assertEqual(a["excluded_predicted_events"], 1)
        self.assertIsNone(a["catalogue_precision"])
        self.assertIsNone(a["catalogue_recall"])

    def test_one_to_one_link_does_not_count_one_prediction_twice(self):
        known = [CatalogueEvent("one", START + 3 * DAY, START + 3 * DAY, ("CH021",)),
                 CatalogueEvent("two", START + 4 * DAY, START + 4 * DAY, ("CH021",))]
        report = evaluate(predictions(peaks=(3, 4)), configuration(), known, {"A": {"region": "CH021"}})
        a = report["metrics"][0]
        self.assertEqual(a["catalogue_precision"], 1)
        self.assertEqual(a["catalogue_recall"], .5)
        self.assertEqual(a["matched_events"], 1)

    def test_overlapping_catalogue_windows_find_maximum_cardinality(self):
        known = [CatalogueEvent("wide", START, START + 9 * DAY, ("CH021",)),
                 CatalogueEvent("narrow", START + 2 * DAY, START + 2 * DAY, ("CH021",))]
        predicted = [CatalogueEvent("p1", START + 2 * DAY, START + 2 * DAY, ()),
                     CatalogueEvent("p2", START + 8 * DAY, START + 8 * DAY, ())]
        self.assertEqual(set(match_catalogue_events(known, predicted)), {(0, 1), (1, 0)})

    def test_seeds_and_forecast_leads_are_never_pooled(self):
        config = configuration()
        config["hyperparameters"]["seeds"] = [0, 1]
        config["forecast"] = {"mode": "rolling", "horizon_days": 2}
        rows = predictions(seed=0, lead=1) + predictions(seed=1, lead=1, peaks=()) + predictions(seed=0, lead=2, peaks=())
        report = evaluate(rows, config, events(), {"A": {"region": "CH021"}})
        a = {(r["seed"], r["lead_day"]): r for r in report["metrics"] if r["basin_id"] == "A"}
        self.assertEqual(a[0, 1]["catalogue_recall"], 1)
        self.assertEqual(a[1, 1]["catalogue_recall"], 0)
        self.assertEqual(a[0, 2]["catalogue_recall"], 0)
        self.assertIsNone(a[1, 2]["catalogue_recall"])
        self.assertRaises(ValueError, evaluate, rows + [rows[0]], config, events(), {"A": {"region": "CH021"}})

    def test_weekly_rolling_blocks_cover_events_collectively_without_daily_fixed_leads(self):
        config = configuration()
        config['forecast'] = {'mode': 'rolling', 'horizon_days': 7}
        rows = [replace(row, lead_day=i % 7 + 1,
                        issue_date=START + (i // 7 * 7 - 1) * DAY)
                for i, row in enumerate(predictions(peaks=tuple(range(3, 10))))]
        known = [CatalogueEvent('across-blocks', START + 2 * DAY, START + 10 * DAY, ('CH021',))]
        report = evaluate(rows, config, known, {'A': {'region': 'CH021'}})
        self.assertEqual(report['default_scope'], 'all')
        scores = {r['lead_day']: r for r in report['metrics'] if r['basin_id'] == 'A'}
        self.assertEqual(scores['all']['covered_days'], 12)
        self.assertEqual(scores['all']['eligible_events'], 1)
        self.assertEqual(scores['all']['catalogue_recall'], 1)
        self.assertEqual(scores['all']['catalogue_precision'], 1)
        for lead in range(1, 8):
            self.assertEqual(scores[lead]['eligible_events'], 0)
            self.assertEqual(scores[lead]['excluded_events'], 1)
            self.assertIsNone(scores[lead]['catalogue_recall'])
        daily_missing = [r for r in rows if r.target_date != START + 4 * DAY]
        missing = evaluate(daily_missing, config, known, {'A': {'region': 'CH021'}})
        score = next(r for r in missing['metrics'] if r['basin_id'] == 'A' and r['lead_day'] == 'all')
        self.assertEqual(score['eligible_events'], 0)
        self.assertIsNone(score['catalogue_recall'])

    def test_overlapping_rolling_blocks_select_shortest_lead_without_averaging(self):
        config = configuration()
        config['forecast'] = {'mode': 'rolling', 'horizon_days': 2}
        rows = predictions(lead=1, peaks=(3, 4)) + predictions(lead=2, peaks=())
        report = evaluate(rows, config, events(), {'A': {'region': 'CH021'}})
        scores = {r['lead_day']: r for r in report['metrics'] if r['basin_id'] == 'A'}
        self.assertEqual(scores['all']['catalogue_recall'], 1)
        self.assertEqual(scores['all']['covered_days'], 12)
        self.assertEqual(scores[2]['catalogue_recall'], 0)

    def test_unmapped_gauges_and_missing_thresholds_never_get_zero_scores(self):
        for region in ({"region": None, "reason": "missing_gauge_coordinates"}, {"region": "CH021"}):
            config = configuration()
            config["benchmark_manifest"]["thresholds_mm_day"] = {}
            report = evaluate(predictions(), config, events(), {"A": region})
            a = report["metrics"][0]
            self.assertEqual(a["status"], "unavailable")
            self.assertIsNone(a["catalogue_precision"])
            self.assertIsNone(a["catalogue_recall"])


class HanzeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hanze = self.root / "hanze"
        self.hanze.mkdir()
        with (self.hanze / "HANZE_events_v3_0_1b.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Country code", "Type", "Start date", "End date", "Regions affected (NUTS 3)", "Sources"])
            writer.writeheader()
            for identifier, country, kind in (("river", "CH", "River"), ("flash", "CH", "Flash"), ("foreign", "DE", "River")):
                writer.writerow({"ID": identifier, "Country code": country, "Type": kind, "Start date": "2005-08-04", "End date": "2005-08-05", "Regions affected (NUTS 3)": "CH021", "Sources": "Independent report"})
        import geopandas as gpd
        from shapely.geometry import box
        directory = self.hanze / "Regions_v2024_simplified"
        directory.mkdir()
        frame = gpd.GeoDataFrame({"Code": ["CH021"], "Name": ["Bern"]}, geometry=[box(7, 46, 8, 48)], crs=4326)
        frame.to_crs(3035).to_file(directory / "NUTS3_regions_v2024_simplified.shp")
        records = [BasinCatalogRecord("A", static_attributes={"gauge_lat": 47, "gauge_lon": 7.5}),
                   BasinCatalogRecord("B")]
        self.catalog = CAMELSCHCatalog(self.root / "camels", records)
        self.settings = ServerSettings(self.root / "state", {"ch": DatasetRegistration(self.root / "camels", hanze_root=self.hanze)}, token="secret")

    def result(self):
        return ExperimentResult(predictions(), {}, {}, {}, {}, [], [], configuration())

    def test_load_filters_and_spatial_assignment_reprojects(self):
        self.assertEqual([e.event_id for e in load_events(self.hanze)], ["river"])
        mapping = gauge_regions(self.hanze, self.catalog.records)
        self.assertEqual(mapping["A"]["region"], "CH021")
        self.assertEqual(mapping["B"]["reason"], "missing_gauge_coordinates")
        report = evaluate_run(self.hanze, predictions(), configuration(), self.catalog)
        self.assertEqual(report["metrics"][0]["catalogue_recall"], 1)
        self.assertEqual(len(report["provenance"]["files"]), 5)
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_existing_run_api_evaluates_without_training_and_downloads_report(self):
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.application import Application
        application = Application(self.settings)
        application.catalog = lambda dataset_id: self.catalog
        saved = application.runs.save(self.result())
        with TestClient(create_app(self.settings, application=application)) as client:
            path = f"/api/v1/runs/{saved.run_id}/hanze"
            self.assertEqual(client.get(path).status_code, 401)
            client.headers["Authorization"] = "Bearer secret"
            response = client.get(path)
            self.assertEqual(response.status_code, 200, response.text)
            report = response.json()
            self.assertEqual(report["metrics"][0]["catalogue_precision"], .5)
            self.assertEqual(report["metrics"][0]["catalogue_recall"], 1)
            download = client.get(f"/api/v1/runs/{saved.run_id}/artifacts/hanze")
            self.assertEqual(download.json(), report)
            self.assertIn("attachment", download.headers["Content-Disposition"])
            self.assertEqual(client.get('/api/v1/runs/missing/hanze').status_code, 404)
            import io, zipfile
            archive = client.get(f"/api/v1/runs/{saved.run_id}/artifacts/archive")
            self.assertEqual(archive.status_code, 200, archive.text if archive.status_code != 200 else "")
            with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
                self.assertEqual(json.loads(bundle.read("hanze_evaluation.json")), report)

    def test_worker_persists_report_and_round_trips_it(self):
        from complexity_ensemble.application import Application
        from complexity_ensemble.jobs import JobStore
        from complexity_ensemble.worker import run_job
        application = Application(self.settings)
        application.catalog = lambda dataset_id: self.catalog
        application.experiment_config = lambda payload: configuration()
        result = self.result()
        store = JobStore(self.settings.jobs_path)
        owner = "hanze-test"
        store.acquire_worker(owner, stale_after=0)
        job = {"id": "synthetic", "kind": "experiment", "payload": {"dataset_id": "ch", "save_to_leaderboard": False}}
        with patch('complexity_ensemble.worker.JobCancellation') as cancellation, patch('complexity_ensemble.services.execute_experiment', return_value=result):
            cancellation.return_value.cancelled = False
            with patch.object(store, "progress"):
                output = run_job(application, store, job, owner)
        saved = application.runs.get(output["run_id"])
        self.assertEqual(saved.hanze_details["status"], "available")
        self.assertEqual(saved.hanze_details["metrics"][0]["catalogue_recall"], 1)
        self.assertTrue((saved.path / "hanze_evaluation.json").is_file())
        self.assertEqual(application.runs.result(saved.run_id).hanze_details, saved.hanze_details)
        with patch.object(application, "evaluate_hanze", side_effect=AssertionError("Saved report must be reused")):
            self.assertEqual(application.hanze_for_run(saved.run_id), saved.hanze_details)
        store.release_worker(owner)

    def test_previous_verifier_report_is_recomputed_from_saved_predictions(self):
        from complexity_ensemble.application import Application
        application = Application(self.settings)
        application.catalog = lambda dataset_id: self.catalog
        result = self.result()
        result.hanze_details = {'status': 'available', 'verification': 'hanze_swiss_river_gauge_region_v1'}
        saved = application.runs.save(result)
        report = application.hanze_for_run(saved.run_id)
        self.assertEqual(report['verification'], 'hanze_swiss_river_gauge_region_v2')
        self.assertEqual(report['metrics'][0]['catalogue_recall'], 1)

    def test_worker_keeps_discharge_results_if_catalogue_evaluation_fails(self):
        from complexity_ensemble.application import Application
        from complexity_ensemble.jobs import JobStore
        from complexity_ensemble.worker import run_job
        application = Application(self.settings)
        application.experiment_config = lambda payload: configuration()
        result = self.result()
        store = JobStore(self.settings.jobs_path)
        job = {"id": "synthetic", "kind": "experiment", "payload": {"dataset_id": "ch", "save_to_leaderboard": False}}
        with patch('complexity_ensemble.worker.JobCancellation') as cancellation, patch('complexity_ensemble.services.execute_experiment', return_value=result), patch.object(application, 'evaluate_hanze', side_effect=ValueError("Broken catalogue")), patch.object(store, 'progress'):
            cancellation.return_value.cancelled = False
            with self.assertLogs('complexity_ensemble.worker', level='ERROR'):
                output = run_job(application, store, job, 'worker')
        saved = application.runs.get(output['run_id'])
        self.assertFalse(output['execution_failed'])
        self.assertEqual(saved.hanze_details['status'], 'error')
        self.assertTrue(saved.failures)
        self.assertEqual(len(list(application.runs.predictions(saved.run_id))), 12)

    def test_settings_resolve_hanze_path_and_unconfigured_datasets_work(self):
        settings_path = self.root / 'server.json'
        settings_path.write_text(json.dumps({"datasets": {"ch": {"data_root": "camels", "hanze_root": "hanze"}}}))
        settings = ServerSettings.load(settings_path)
        self.assertEqual(settings.datasets["ch"].hanze_root, self.hanze.resolve())
        from complexity_ensemble.application import Application
        settings = replace(self.settings, datasets={"ch": DatasetRegistration(self.root / "camels")})
        report = Application(settings).evaluate_hanze("ch", predictions(), configuration())
        self.assertEqual(report["status"], "not_configured")
