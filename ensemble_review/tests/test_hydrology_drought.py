"""Drought software checks only: no models are trained or experiments launched."""
from dataclasses import replace
from datetime import date
from pathlib import Path
import csv
import json
import io
import math
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from complexity_ensemble.hydrology_drought import (
    DAY, DroughtEvent, extract_events, fit_thresholds, match_events, smooth, verify_drought,
)
from complexity_ensemble.hydrology_events import Event, _event_scores
from complexity_ensemble.regional import (
    BasinScopeConfig, DateRange, DateSplitConfig, ExtremeEventConfig, ForecastConfig,
    RegionalExperimentConfig, ExperimentResult, write_result_artifacts,
)
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.serialization import json_safe
from complexity_ensemble.result_registry import save_run, load_run
from complexity_ensemble.services import result_summary


def sample_predictions(smoothing='trailing', missing=False):
    start, issue = date(2000, 1, 1), date(2001, 1, 10)
    days = [start + i * DAY for i in range((issue + 14 * DAY - start).days + 1)]
    flow = {day: 10.0 if day <= issue else 2.0 for day in days}
    if missing:
        flow.pop(issue + 4 * DAY)
    series = SimpleNamespace(dates=list(flow), discharge=list(flow.values()))
    rows = [SimpleNamespace(seed=0, basin_id='a', issue_date=issue,
                            target_date=issue + i * DAY, lead_day=i,
                            predicted_mm_day=2.0, observed_mm_day=flow.get(issue + i * DAY, float('nan')))
            for i in range(1, 15)]
    return rows, {'a': series}, start, date(2000, 12, 31)


class DroughtChecks(unittest.TestCase):
    def test_trailing_and_centered_windows_and_gaps(self):
        day = date(2000, 1, 10)
        flow = {day + i * DAY: float(i + 10) for i in range(-6, 4)}
        self.assertEqual(smooth(flow, day, 'trailing'), 7)
        self.assertEqual(smooth(flow, day, 'centered'), 10)
        del flow[day + DAY]
        self.assertEqual(smooth(flow, day, 'trailing'), 7)
        self.assertIsNone(smooth(flow, day, 'centered'))
        del flow[day - 5 * DAY]
        self.assertIsNone(smooth(flow, day, 'trailing'))

    def test_training_only_reference_and_smoothing_edges(self):
        start = date(2000, 1, 1)
        flow = {start + i * DAY: 10.0 for i in range(20)}
        a = fit_thresholds(flow, start, start + 9 * DAY, smoothing='centered')
        flow.update({start + i * DAY: 9999 for i in range(10, 20)})
        b = fit_thresholds(flow, start, start + 9 * DAY, smoothing='centered')
        self.assertEqual(a, b)
        self.assertEqual(a.at(start), 10)

    def test_seasonal_calendar_wrap_and_leap_alignment(self):
        flow = {date(2000, 12, 20) + i * DAY: 3.0 for i in range(12)}
        fitted = fit_thresholds(flow, date(2000, 12, 20), date(2000, 12, 31), calendar_window=31)
        self.assertEqual(fitted.at(date(2001, 1, 1)), 3)
        self.assertIsNone(fitted.at(date(2001, 7, 1)))
        flow = {date(2000, 2, 20) + i * DAY: 8.0 for i in range(16)}
        fitted = fit_thresholds(flow, min(flow), max(flow), calendar_window=1)
        self.assertEqual(fitted.at(date(2000, 3, 1)), fitted.at(date(2001, 3, 1)))

    def test_q347_uses_raw_daily_discharge(self):
        start = date(2000, 1, 1)
        flow = {start + i * DAY: float(i) for i in range(366)}
        fitted = fit_thresholds(flow, start, max(flow), definition='q347')
        self.assertAlmostEqual(fitted.fixed, 18)

    def test_centered_excludes_final_three_leads(self):
        records, series, start, end = sample_predictions()
        scores, details = verify_drought(records, series, start, end, definition='q347', smoothing='centered')
        issue_score = scores[0]
        self.assertEqual(issue_score['evaluated_days'], 11)
        self.assertEqual(issue_score['excluded_days'], 3)
        self.assertEqual({row['lead_day'] for row in details['days'] if row['eligible']}, set(range(1, 12)))
        self.assertTrue(all(row['observed_m7_mm_day'] == row['predicted_m7_mm_day'] for row in details['days'] if row['eligible']))

    def test_post_issue_observations_never_enter_predicted_smoothing(self):
        records, series, start, end = sample_predictions()
        _, before = verify_drought(records, series, start, end, definition='q347', smoothing='centered')
        issue = records[0].issue_date
        changed = SimpleNamespace(dates=series['a'].dates,
            discharge=[value if day <= issue else 999 for day, value in zip(series['a'].dates, series['a'].discharge)])
        _, after = verify_drought(records, {'a': changed}, start, end, definition='q347', smoothing='centered')
        self.assertEqual([row['predicted_m7_mm_day'] for row in before['days']],
                         [row['predicted_m7_mm_day'] for row in after['days']])
        self.assertNotEqual(before['days'][0]['observed_m7_mm_day'], after['days'][0]['observed_m7_mm_day'])

    def test_issues_do_not_borrow_predictions_from_later_issues(self):
        records, series, start, end = sample_predictions()
        # Two seven-day blocks. Each loses its own final three centred leads.
        second_issue = records[0].issue_date + 7 * DAY
        records = [SimpleNamespace(**{**vars(row), 'issue_date': second_issue,
                   'lead_day': row.lead_day - 7}) if row.lead_day > 7 else row for row in records]
        scores, details = verify_drought(records, series, start, end, definition='q347', smoothing='centered')
        self.assertEqual([row['evaluated_days'] for row in scores if row['evaluation_scope'] == 'issue'], [4, 4])
        self.assertEqual(sum(row['eligible'] for row in details['days']), 8)

    def test_missing_observation_excludes_whole_affected_window(self):
        records, series, start, end = sample_predictions(missing=True)
        scores, details = verify_drought(records, series, start, end, definition='q347')
        self.assertEqual(scores[0]['excluded_days'], 7)
        self.assertEqual(sum(row['eligible'] for row in details['days']), 7)

    def test_one_to_one_matching_prevents_merged_event_inflation(self):
        day = date(2000, 1, 1)
        observed = [DroughtEvent(day, day + DAY, 2, 2, False, False),
                    DroughtEvent(day + 4 * DAY, day + 5 * DAY, 2, 2, False, False)]
        predicted = [DroughtEvent(day, day + 5 * DAY, 6, 6, False, False)]
        self.assertEqual(len(match_events(observed, predicted)), 1)
        legacy = _event_scores([Event(x.start, x.end, x.duration, x.deficit) for x in observed],
                               [Event(x.start, x.end, x.duration, x.deficit) for x in predicted])
        self.assertEqual(legacy['event_recall'], 1)
        self.assertEqual(legacy['event_precision'], 1)

    def test_matching_optimizes_total_overlap_then_number_of_pairs(self):
        day = date(2000, 1, 1)
        event = lambda a, b: DroughtEvent(day + a * DAY, day + b * DAY, b - a + 1, 0, False, False)
        self.assertEqual(match_events([event(0, 4), event(6, 7)], [event(0, 0), event(2, 7)]), [(0, 0), (1, 1)])

    def test_censored_boundaries_are_not_complete_events(self):
        records, series, start, end = sample_predictions()
        scores, details = verify_drought(records, series, start, end, definition='q347')
        score = scores[0]
        self.assertEqual(score['matched_events'], 1)
        self.assertEqual(score['duration_pairs'], 0)
        self.assertTrue(math.isnan(score['duration_mae_days']))
        self.assertTrue(all(row['right_censored'] for row in details['events']))

    def test_complete_event_timing_and_duration_are_scored(self):
        records, series, start, end = sample_predictions()
        issue = records[0].issue_date
        series['a'].discharge = [10.0 if day > issue + 3 * DAY else value
            for day, value in zip(series['a'].dates, series['a'].discharge)]
        for row in records:
            if row.lead_day > 3:
                row.predicted_mm_day = 10.0
        scores, details = verify_drought(records, series, start, end, definition='q347')
        score = scores[0]
        self.assertEqual(score['onset_pairs'], 1)
        self.assertEqual(score['recovery_pairs'], 1)
        self.assertEqual(score['duration_pairs'], 1)
        self.assertEqual(score['duration_mae_days'], 0)
        self.assertEqual(score['deficit_mae_mm'], 0)
        self.assertFalse(any(row['left_censored'] or row['right_censored'] for row in details['events']))

    def test_minimum_duration_and_missing_days_split_events(self):
        day = date(2000, 1, 1)
        rows = [{'date': day + i * DAY, 'threshold_mm_day': 2, 'q': 1} for i in [0, 1, 3, 4, 5]]
        events = extract_events(rows, 'q', 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].duration, 3)
        self.assertEqual(events[0].deficit, 3)
        self.assertTrue(events[0].left_censored and events[0].right_censored)


class DroughtIntegrationChecks(unittest.TestCase):
    def config(self, **drought):
        return RegionalExperimentConfig('.', BasinScopeConfig(('a',), 'targets_only'),
            DateSplitConfig(DateRange('2000-01-01', '2000-12-31'),
                            DateRange('2001-01-01', '2001-12-31'), DateRange('2002-01-01', '2002-12-31')),
            extremes=ExtremeEventConfig(mode='drought', **drought))

    def test_config_roundtrip_and_api_without_fixed_basin_or_dataset(self):
        config = replace(self.config(drought_smoothing='centered', drought_quantile=.1), forecast=ForecastConfig('rolling', 14))
        config.validate()
        restored = config_from_mapping(json_safe(config))
        self.assertEqual(restored.extremes, config.extremes)
        self.assertEqual(restored.forecast, config.forecast)
        self.assertIsNone(restored.hyd_responses_root)
        from complexity_ensemble.api_models import Extremes
        api = Extremes.model_validate(json_safe(config.extremes))
        self.assertEqual(api.drought_smoothing, 'centered')

    def test_invalid_centered_forecast_rejected_without_training(self):
        config = self.config(drought_smoothing='centered')
        with self.assertRaisesRegex(ValueError, 'at least 4 days'):
            config.validate()
        with self.assertRaisesRegex(ValueError, 'at least 4 days'):
            replace(config, forecast=ForecastConfig('rolling', 3)).validate()
        replace(config, forecast=ForecastConfig('rolling', 4)).validate()

    def test_invalid_drought_fields_do_not_change_other_tasks(self):
        for fields in ({'drought_quantile': 0}, {'drought_smoothing': 'bad'},
                       {'drought_calendar_window': 30}, {'drought_minimum_days': 0}):
            with self.subTest(fields=fields):
                config = self.config(**fields)
                with self.assertRaises(ValueError):
                    config.validate()
                replace(config, extremes=replace(config.extremes, mode='event_based')).validate()

    def test_old_config_uses_defaults(self):
        payload = json_safe(self.config())
        payload['extremes'] = {'mode': 'event_based'}
        restored = config_from_mapping(payload)
        restored.validate()
        self.assertEqual(restored.extremes.mode, 'event_based')
        self.assertEqual(restored.extremes.low_quantile, .2)

    def test_exports_registry_and_service_preserve_audit_tables(self):
        records, series, start, end = sample_predictions()
        scores, details = verify_drought(records, series, start, end, definition='q347')
        result = ExperimentResult([], {}, {}, {}, {}, scores, [], json_safe(self.config()), drought_details=details)
        with tempfile.TemporaryDirectory() as root:
            paths = write_result_artifacts(result, root)
            with zipfile.ZipFile(paths['archive']) as bundle:
                self.assertTrue({'drought_metrics.csv', 'drought_days.csv', 'drought_events.csv'} <= set(bundle.namelist()))
            with open(paths['drought_days']) as source:
                self.assertEqual(len(list(csv.DictReader(source))), 14)
            saved = save_run(result, Path(root) / 'registry')
            self.assertEqual(saved.drought_details, details)
            self.assertEqual(result_summary(saved)['drought_details'], details)
            # Existing manifests without this optional field remain readable.
            manifest = saved.path / 'run.json'
            payload = json.loads(manifest.read_text())
            payload.pop('drought_details')
            manifest.write_text(json.dumps(payload))
            self.assertEqual(load_run(manifest).drought_details, {})

    def test_api_download_and_comparison_include_drought_results_without_jobs(self):
        from fastapi.testclient import TestClient
        from complexity_ensemble.api import create_app
        from complexity_ensemble.application import Application
        from complexity_ensemble.server_settings import ServerSettings
        records, series, start, end = sample_predictions()
        scores, details = verify_drought(records, series, start, end, definition='q347')
        result = ExperimentResult([], {}, {}, {}, {}, scores, [], json_safe(self.config()), drought_details=details)
        with tempfile.TemporaryDirectory() as root:
            settings = ServerSettings(Path(root) / 'state', {}, Path(root) / 'leaderboard')
            application = Application(settings)
            first = application.runs.save(result)
            second = application.runs.save(result)
            with TestClient(create_app(settings, application=application)) as client:
                response = client.get(f'/api/v1/runs/{first.run_id}/artifacts/archive')
                self.assertEqual(response.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    self.assertIn('drought_days.csv', archive.namelist())
                    self.assertNotIn('data_root', json.loads(archive.read('config.json')))
                response = client.get(f'/api/v1/runs/{first.run_id}/artifacts/drought_events')
                self.assertEqual(response.status_code, 200)
                comparison = client.post('/api/v1/comparisons', json={'run_a': first.run_id, 'run_b': second.run_id})
                self.assertEqual(comparison.status_code, 200)
                self.assertEqual(comparison.json()['drought_delta_b_minus_a']['lead_1.csi'], 0)
                with client.app.state.jobs.connect() as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)

    def test_ui_values_preserve_new_fields_and_old_positions(self):
        from complexity_ensemble.app import _config_from_ui_values
        values = ['2000-01-01', '2000-12-31', '2001-01-01', '2001-12-31',
                  '2002-01-01', '2002-12-31', 'no_routing', 'mlp', 'rbf',
                  1, 30, '0', 16, .001, 0, 80, .15, .05, .05, 1, 1, 1,
                  '8', 16, 1, 64, 32, 32, 'drought', .05, 0, 'cpu', 'automatic_q95', .95,
                  .99, .1, 'none', 'hard_routing', 3, 50, 3, 3, 2, 1, .001, '.05', .05, True,
                  False, '', [], False, '', [], 'rolling', 14, ['low_flow_spell'],
                  .8, .2, .95, 3, 7, 2, .95, 'morse', 365, 3, 7, .1, 'total_persistence', 30, 20, .8,
                  [], False, '', [], False, 'q347', 'centered', .1, 91, 5]
        self.assertEqual(len(values), 83)
        config = _config_from_ui_values('.', ['arbitrary'], 'targets_only', values)
        config.validate()
        self.assertEqual(config.extremes.drought_smoothing, 'centered')
        self.assertEqual(config.extremes.drought_definition, 'q347')
        self.assertEqual(config.extremes.drought_calendar_window, 91)
        self.assertEqual(config.extremes.drought_minimum_days, 5)
        self.assertEqual(config.forecast.horizon_days, 14)
        old = _config_from_ui_values('.', ['arbitrary'], 'targets_only', values[:78])
        self.assertEqual(old.extremes.drought_smoothing, 'trailing')

    def test_gradio_build_exposes_drought_task_and_controls(self):
        from complexity_ensemble.app import build_app
        from complexity_ensemble.regional import CAMELSCHCatalog, BasinCatalogRecord
        catalog = CAMELSCHCatalog('.', [BasinCatalogRecord('a')])
        with tempfile.TemporaryDirectory() as root, patch('complexity_ensemble.app.CAMELSCHCatalog', return_value=catalog):
            app = build_app('.', results_dir=root)
        blocks = app.get_config_file()['components']
        props = [row.get('props', {}) for row in blocks]
        task = next(p for p in props if p.get('label') == 'Analysis task')
        self.assertIn('drought', [choice[1] for choice in task['choices']])
        self.assertTrue(any(p.get('label') == 'Discharge smoothing' for p in props))
        self.assertTrue(any(p.get('label') == 'Minimum drought spell days' for p in props))


if __name__ == '__main__':
    unittest.main()
