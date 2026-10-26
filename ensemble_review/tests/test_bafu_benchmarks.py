"""Synthetic BAFU read, contract and verification checks with a tiny CAMELS training regression."""
import csv
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch
from fastapi.testclient import TestClient

from complexity_ensemble.api import create_app
from complexity_ensemble.api_models import ExperimentRequest
from complexity_ensemble.application import Application
from complexity_ensemble.bafu import basin_compatibility, load_observations, load_return_levels, statistics_index
from complexity_ensemble.benchmarks import BenchmarkConfig, manifest, paired_comparison, preflight
from complexity_ensemble.configuration import config_from_mapping
from complexity_ensemble.flood_verification import flood_metrics
from complexity_ensemble.hydrology import RoutedHydrologyModel, HydrologyLSTMExpert, hydrology_morse_gradient
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.regional import (BasinCatalogRecord, CAMELSCHCatalog, PredictionRecord,
    ExperimentResult, _baseline_predictions, _resolved, prepare_regional_data, write_result_artifacts)
from complexity_ensemble.server_settings import DatasetRegistration, ServerSettings
from complexity_ensemble.services import compare_runs
from complexity_ensemble.serialization import public_config


class BAFUBenchmarks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.bafu = self.root/'bafu'
        self.partition = self.bafu/'observations/live/data/2001/water_observations_2001_data_1day_mean_200101.csv.gz'
        self.partition.parent.mkdir(parents=True)
        self.days = tuple(date(2001, 1, 1)+timedelta(days=i) for i in range(40))
        self.rows = []
        for i, day in enumerate(self.days):
            stamp = datetime.combine(day, datetime.min.time(), ZoneInfo('Europe/Zurich')).astimezone(timezone.utc).replace(tzinfo=None)
            self.rows.append({'station_no':'2011', 'parameter_name':'Q', 'unit_symbol':'m³/s',
                'timestamp':str(stamp), 'value':str(1+i%9), 'release_state':'3'})
        self.write_rows()
        self.catalog = CAMELSCHCatalog(self.root/'camels', [BasinCatalogRecord('2011', static_attributes={'area':86.4, 'elev_mean':1000.})])
        self.forcings = torch.stack((torch.arange(40).float(), torch.ones(40), torch.ones(40)), 1)
        self.catalog.load_series = lambda basin, start, end: HydrologySeries(self.days, torch.ones(40)*999,
            self.forcings, ('precipitation','temperature','pet'), basin, 'CH')
        self.mapping = {'basins':{'target_basins':['2011'],'training_scope':'targets_only'},
            'dates':{'train':{'start':'2001-01-04','end':'2001-01-18'},
                     'validation':{'start':'2001-01-19','end':'2001-01-25'},
                     'test':{'start':'2001-01-26','end':'2001-02-09'}},
            'benchmark':{'mode':'bafu_only'}, 'hyperparameters':{'sequence_length':3,'physics_weight':0},
            'extremes':{'mode':'statistical'}}
        self.config = config_from_mapping(self.mapping, data_root=self.catalog.root, bafu_root=self.bafu)

    def write_rows(self):
        with gzip.open(self.partition,'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)

    def test_timezone_units_and_quality_filter(self):
        result, provenance=load_observations(self.bafu,['2011'],self.days[0],self.days[-1],areas={'2011':86.4})
        self.assertEqual(result['2011'].dates,self.days)
        self.assertEqual(result['2011'].discharge[0],1)
        self.assertEqual(result['2011'].forcings.shape,(40,0))
        self.assertEqual(len(provenance['files'][0]['sha256']),64)
        self.rows[0]['release_state']='1';self.write_rows()
        result,_=load_observations(self.bafu,['2011'],self.days[0],self.days[-1],areas={'2011':86.4})
        self.assertEqual(result['2011'].dates[0],self.days[1])

    def test_reject_conflicting_duplicates_and_missing_area(self):
        self.rows.append({**self.rows[0],'value':'999'});self.write_rows()
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            load_observations(self.bafu,['2011'],self.days[0],self.days[-1],areas={'2011':86.4})
        with self.assertRaisesRegex(ValueError,'area'):
            load_observations(self.bafu,['2011'],self.days[0],self.days[-1])

    def test_missing_observations_are_gaps_for_daily_mean_and_max(self):
        for target, aggregation in [('daily_mean', '1day_mean'), ('daily_max', '1day_max')]:
            with self.subTest(target=target):
                self.partition = self.partition.with_name(f'water_observations_2001_data_{aggregation}_200101.csv.gz')
                original = self.rows[:]
                self.rows = original[:]
                for i, missing in enumerate([
                    {'release_state': '', 'value': ''},
                    {'release_state': '3', 'value': ''},
                    {'release_state': '  ', 'value': '5'},
                ]):
                    self.rows[i] = {**original[i], **missing}
                self.write_rows()
                result, _ = load_observations(self.bafu, ['2011'], self.days[0], self.days[-1],
                                              target=target, areas={'2011': 86.4})
                self.assertEqual(result['2011'].dates, self.days[3:])
                self.assertEqual(result['2011'].discharge.tolist(), [float(row['value']) for row in original[3:]])
                self.rows = original

    def test_submission_with_missing_observation_rows_in_both_training_scopes(self):
        self.rows.extend([
            {**self.rows[0], 'release_state': '', 'value': ''},
            {**self.rows[0], 'release_state': '3', 'value': ''},
        ])
        self.write_rows()
        settings = ServerSettings(self.root/'state', {'ch': DatasetRegistration(self.catalog.root, bafu_root=self.bafu)})
        application = Application(settings)
        application.catalog = lambda _: self.catalog
        with TestClient(create_app(settings, application=application)) as client:
            for scope in ('targets_only', 'all_basins'):
                with self.subTest(scope=scope):
                    body = {'dataset_id': 'ch', 'config': {
                        **self.mapping, 'basins': {'target_basins': ['2011'], 'training_scope': scope}}}
                    response = client.post('/api/v1/experiments', json=body)
                    self.assertEqual(response.status_code, 202, response.text)

    def test_only_vs_combined_keep_bafu_target_and_exclude_external_features(self):
        _,only,series,prov=prepare_regional_data(self.config,catalog=self.catalog)
        combined=replace(self.config,benchmark=replace(self.config.benchmark,mode='bafu_combined'))
        _,both,joined,other=prepare_regional_data(combined,catalog=self.catalog)
        self.assertEqual(only.feature_names,('previous_discharge','day_sin','day_cos'))
        self.assertEqual(only.static_feature_names,())
        self.assertIn('precipitation',both.feature_names)
        self.assertTrue(torch.equal(only.test.targets,both.test.targets))
        self.assertLess(float(joined['2011'].discharge.max()),999)
        first=manifest(self.config,only,series,prov);second=manifest(combined,both,joined,other)
        self.assertEqual(first['evaluation_fingerprint'],second['evaluation_fingerprint'])
        self.assertNotEqual(first['training_fingerprint'],second['training_fingerprint'])

    def test_preflight_never_constructs_or_fits_models_and_pin_is_enforced(self):
        with patch('complexity_ensemble.regional.RoutedHydrologyModel',side_effect=AssertionError('No models in preflight')):
            report=preflight(self.config,self.catalog)
        self.assertTrue(report['cohort_complete'])
        _, data, series, provenance = prepare_regional_data(self.config, catalog=self.catalog)
        resolved = manifest(self.config, data, series, provenance)
        self.assertEqual(report['evaluation_fingerprint'], resolved['evaluation_fingerprint'])
        self.assertEqual(report['training_fingerprint'], resolved['training_fingerprint'])
        pinned=replace(self.config,benchmark=replace(self.config.benchmark,expected_evaluation_fingerprint=report['evaluation_fingerprint']))
        self.assertEqual(preflight(pinned,self.catalog)['evaluation_fingerprint'],report['evaluation_fingerprint'])
        self.rows[-1]['value']='999';self.write_rows()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            preflight(pinned,self.catalog)

    def test_train_threshold_does_not_use_test_values(self):
        a=preflight(self.config,self.catalog)
        self.rows[-1]['value']='999';self.write_rows()
        b=preflight(self.config,self.catalog)
        self.assertEqual(a['thresholds_mm_day'],b['thresholds_mm_day'])
        self.assertNotEqual(a['evaluation_fingerprint'],b['evaluation_fingerprint'])
        self.assertEqual(a['training_fingerprint'],b['training_fingerprint'])

    def test_strict_cohort_fails_for_missing_station(self):
        self.catalog=CAMELSCHCatalog(self.catalog.root,[*self.catalog.records,BasinCatalogRecord('2012',static_attributes={'area':5})])
        config=replace(self.config,basins=replace(self.config.basins,target_basins=('2011','2012')))
        with self.assertRaisesRegex(ValueError,'Strict benchmark cohort'):
            prepare_regional_data(config,catalog=self.catalog)
        audit=preflight(config,self.catalog)
        self.assertFalse(audit['cohort_complete'])
        self.assertIn('2012',audit['provenance']['missing_stations'])

    def test_flow_only_router_and_lstm_forward(self):
        model=RoutedHydrologyModel(('previous_discharge','day_sin','day_cos'),3,1.,complex_kind='lstm',complex_kwargs={'hidden_dim':4})
        x=torch.randn(2,3,3)
        model.router.fit(x.flatten(1))
        self.assertEqual(model(x).shape,(2,1))
        self.assertFalse(model.physics_available)
        self.assertEqual(float(model.physics_residual(torch.ones(2,1),x).sum()),0)
        self.assertEqual(hydrology_morse_gradient(x.flatten(1),sequence_length=3,feature_names=model.feature_names).shape,(2,9))

    def drought_config(self, **settings):
        return replace(self.config, extremes=replace(self.config.extremes, mode='drought', **settings))

    def test_bafu_drought_validation_keeps_high_flow_and_camels_defaults(self):
        self.config.validate()
        self.drought_config().validate()
        replace(self.drought_config(), benchmark=replace(self.config.benchmark, mode='bafu_combined')).validate()
        with self.assertRaisesRegex(ValueError, 'daily_mean'):
            replace(self.drought_config(), benchmark=replace(self.config.benchmark, target='daily_max')).validate()
        with self.assertRaisesRegex(ValueError, 'flood return levels'):
            replace(self.drought_config(), benchmark=replace(self.config.benchmark, threshold_source='official_return_level')).validate()
        normal = replace(self.config, benchmark=BenchmarkConfig())
        normal.validate()
        self.assertEqual(normal.extremes.mode, 'statistical')
        with patch('complexity_ensemble.bafu.load_observations', side_effect=AssertionError('CAMELS must not read BAFU')):
            _, data, series, provenance = prepare_regional_data(normal, catalog=self.catalog)
        self.assertIn('precipitation', data.feature_names)
        self.assertTrue(data.static_feature_names)
        self.assertEqual(float(series['2011'].discharge[0]), 999)
        self.assertNotIn('adapter', provenance)

    def test_drought_preflight_thresholds_identity_and_no_test_leakage(self):
        config = self.drought_config(drought_definition='q347')
        with patch('complexity_ensemble.regional.RoutedHydrologyModel', side_effect=AssertionError('No training')):
            before = preflight(config, self.catalog)
        self.assertIn('drought', before['evaluation']['verifier'])
        self.assertGreater(before['drought_coverage']['2011']['eligible_days'], 0)
        self.assertEqual(before['threshold_provenance']['source'], 'training_observations_only')
        flood = preflight(self.config, self.catalog)
        self.assertNotEqual(before['evaluation_fingerprint'], flood['evaluation_fingerprint'])
        self.rows[-1]['value'] = '999'; self.write_rows()
        after = preflight(config, self.catalog)
        self.assertEqual(before['thresholds_mm_day'], after['thresholds_mm_day'])
        self.assertEqual(before['training_fingerprint'], after['training_fingerprint'])
        self.assertNotEqual(before['evaluation_fingerprint'], after['evaluation_fingerprint'])
        rolling = replace(config, forecast=replace(config.forecast, mode='rolling', horizon_days=7))
        reference = preflight(rolling, self.catalog)
        for settings in ({'drought_minimum_days': 2}, {'drought_smoothing': 'centered'},
                         {'drought_definition': 'seasonal_quantile'}, {'drought_quantile': .2},
                         {'drought_calendar_window': 33}):
            changed = replace(rolling, extremes=replace(rolling.extremes, **settings))
            self.assertNotEqual(reference['evaluation_fingerprint'], preflight(changed, self.catalog)['evaluation_fingerprint'])

    def test_drought_identity_includes_pre_test_smoothing_context(self):
        config = self.drought_config(drought_definition='q347')
        before = preflight(config, self.catalog)
        self.rows[24]['value'] = '999'; self.write_rows()  # Validation day before the test begins.
        after = preflight(config, self.catalog)
        self.assertEqual(before['thresholds_mm_day'], after['thresholds_mm_day'])
        self.assertEqual(before['training_fingerprint'], after['training_fingerprint'])
        self.assertNotEqual(before['evaluation_fingerprint'], after['evaluation_fingerprint'])
        pinned = replace(config, benchmark=replace(config.benchmark, expected_evaluation_fingerprint=before['evaluation_fingerprint']))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            preflight(pinned, self.catalog)

    def test_drought_preflight_rejects_uncovered_training_calendar(self):
        with self.assertRaisesRegex(ValueError, 'no evaluable seven-day test windows'):
            preflight(self.drought_config(drought_calendar_window=1), self.catalog)

    def test_camels_rainfall_runoff_training_still_uses_existing_path(self):
        from complexity_ensemble.regional import run_regional_experiment
        config = replace(self.config, benchmark=BenchmarkConfig(),
            model=replace(self.config.model, complex_expert='mlp', mlp_widths=(4,)),
            strategy=replace(self.config.strategy, approach='no_routing', epochs=1),
            hyperparameters=replace(self.config.hyperparameters, batch_size=64))
        with patch('complexity_ensemble.bafu.load_observations', side_effect=AssertionError('CAMELS must not read BAFU')):
            result = run_regional_experiment(config, catalog=self.catalog)
        self.assertTrue(result.predictions)
        self.assertFalse(result.failures)
        self.assertGreater(result.parameter_count, 0)
        self.assertIn('precipitation', result.resolved_config['feature_names'])
        self.assertNotIn('benchmark_manifest', result.resolved_config)
        self.assertFalse(result.drought_details)
        self.assertTrue(all(row['event_type'] == 'statistical_high_flow' for row in result.extreme_events))

    def test_drought_preflight_matches_execution_coverage_and_exports(self):
        from complexity_ensemble.regional import run_regional_experiment
        for mode in ('bafu_only', 'bafu_combined'):
            for smoothing in ('trailing', 'centered'):
                with self.subTest(mode=mode, smoothing=smoothing):
                    config = self.drought_config(drought_definition='q347', drought_smoothing=smoothing)
                    config = replace(config, benchmark=replace(config.benchmark, mode=mode),
                        strategy=replace(config.strategy, approach='persistence'),
                        forecast=replace(config.forecast, mode='rolling', horizon_days=7))
                    audit = preflight(config, self.catalog)
                    with patch('complexity_ensemble.regional._train', side_effect=AssertionError('No training')):
                        result = run_regional_experiment(config, catalog=self.catalog)
                    self.assertFalse(result.failures)
                    self.assertEqual(result.resolved_config['benchmark_manifest']['evaluation_fingerprint'], audit['evaluation_fingerprint'])
                    days = [r for r in result.drought_details['days'] if r['seed'] == config.hyperparameters.seeds[0]]
                    self.assertEqual(sum(r['eligible'] for r in days), audit['drought_coverage']['2011']['eligible_days'])
                    self.assertTrue(all(r['event_type'] == 'hydrological_drought' for r in result.extreme_events))
                    if mode == 'bafu_only':
                        self.assertEqual(result.resolved_config['drought_verification']['weather'], 'no_weather_inputs')
                        self.assertIn('without weather', audit['forecast_interpretation'])
                    paths = write_result_artifacts(result, self.root/f'{mode}-{smoothing}')
                    for name in ('drought_metrics', 'drought_days', 'drought_events', 'benchmark_manifest'):
                        self.assertTrue(Path(paths[name]).is_file())
                    self.assertNotIn('flood_metrics', paths)

    def test_drought_api_preflight_submission_results_and_downloads(self):
        from complexity_ensemble.regional import run_regional_experiment
        from complexity_ensemble.result_registry import save_run
        settings = ServerSettings(self.root/'state', {'ch': DatasetRegistration(self.catalog.root, bafu_root=self.bafu)})
        application = Application(settings); application.catalog = lambda _: self.catalog
        config = replace(self.drought_config(drought_definition='q347'), strategy=replace(self.config.strategy, approach='persistence'))
        body = {'dataset_id': 'ch', 'config': public_config(config)}
        result = run_regional_experiment(config, catalog=self.catalog)
        saved = save_run(result, settings.runs_dir)
        with TestClient(create_app(settings, application=application)) as client:
            for endpoint, status in [('preflight', 200), ('validate', 200), ('', 202)]:
                response = client.post('/api/v1/experiments' + ('/' + endpoint if endpoint else ''), json=body)
                self.assertEqual(response.status_code, status, response.text)
            response = client.get('/api/v1/runs/' + saved.run_id)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['drought_details']['days'])
            self.assertTrue(response.json()['drought_summary'])
            for name in ('drought_metrics', 'drought_days', 'drought_events', 'benchmark_manifest'):
                response = client.get('/api/v1/runs/' + saved.run_id + '/artifacts/' + name)
                self.assertEqual(response.status_code, 200, response.text)

    def test_baselines_use_available_history_and_train_only(self):
        _,data,series,_=prepare_regional_data(self.config,catalog=self.catalog)
        config=replace(self.config,strategy=replace(self.config.strategy,approach='persistence'),forecast=replace(self.config.forecast,mode='rolling',horizon_days=3))
        rows=_baseline_predictions(config,data,series,0)
        self.assertEqual([r.lead_day for r in rows[:4]],[1,2,3,1])
        self.assertEqual(rows[0].predicted_mm_day,rows[2].predicted_mm_day)
        self.assertEqual(rows[0].issue_date,rows[2].issue_date)
        climate=replace(self.config,strategy=replace(self.config.strategy,approach='seasonal_climatology'))
        before=_baseline_predictions(climate,data,series,0)
        modified={b:replace(s,discharge=s.discharge.clone()) for b,s in series.items()}
        modified['2011'].discharge[-1]=999
        after=_baseline_predictions(climate,data,modified,0)
        self.assertEqual([r.predicted_mm_day for r in before],[r.predicted_mm_day for r in after])

    def test_synthetic_baseline_orchestration_never_fits_a_model_and_exports(self):
        from complexity_ensemble.regional import run_regional_experiment
        config=replace(self.config,strategy=replace(self.config.strategy,approach='persistence'))
        with patch('complexity_ensemble.regional._train',side_effect=AssertionError('No training')),             patch('complexity_ensemble.regional.RoutedHydrologyModel',side_effect=AssertionError('No model construction')):
            result=run_regional_experiment(config,catalog=self.catalog)
        self.assertTrue(result.predictions)
        self.assertFalse(result.failures)
        self.assertEqual(result.parameter_count,0)
        self.assertIn('verifier',result.extreme_events[0])
        self.assertEqual(result.resolved_config['fitted_diagnostics']['mean_complex_weight'],None)
        self.assertTrue(Path(write_result_artifacts(result,self.root/'baseline-artifacts')['flood_metrics']).is_file())

    def test_validation_rejects_weather_physics_and_external_predictors(self):
        with self.assertRaisesRegex(ValueError,'physics_weight'):
            replace(self.config,hyperparameters=replace(self.config.hyperparameters,physics_weight=.1)).validate()
        with self.assertRaisesRegex(ValueError,'external predictors'):
            replace(self.config,hyd_responses_root=self.root).validate()
        cfg=replace(self.config,benchmark=replace(self.config.benchmark,mode='bafu_combined',target='daily_max'),hyperparameters=replace(self.config.hyperparameters,physics_weight=.1))
        with self.assertRaisesRegex(ValueError,'water balance'):
            cfg.validate()

    def test_official_threshold_import_and_provenance(self):
        path=self.root/'hq.csv'
        path.write_text('station_id,return_period_years,discharge_m3_s,source_url,reference_period\n2011,2,5,https://example.org/report.pdf,1900-2020\n')
        cfg=replace(self.config,bafu_return_levels=path,benchmark=replace(self.config.benchmark,threshold_source='official_return_level'))
        report=preflight(cfg,self.catalog)
        self.assertAlmostEqual(report['thresholds_mm_day']['2011'],5)
        self.assertIn('Retrospective',report['threshold_provenance']['interpretation'])
        path.write_text('station_id,return_period_years,discharge_m3_s,source_url,reference_period\n2011,2,-5,,\n')
        with self.assertRaises(ValueError):load_return_levels(path)

    def test_api_preflight_queue_untouched_and_contracts_reject_paths(self):
        settings=ServerSettings(self.root/'state',{'ch':DatasetRegistration(self.catalog.root,bafu_root=self.bafu)},token='secret')
        app=Application(settings);app.catalog=lambda _:self.catalog
        with TestClient(create_app(settings,application=app)) as client:
            self.assertEqual(client.get('/api/v1/benchmarks').status_code,401)
            client.headers['Authorization']='Bearer secret'
            body={'dataset_id':'ch','config':self.mapping}
            with patch('complexity_ensemble.services.run_regional_experiment',side_effect=AssertionError('No experiment')):
                result=client.post('/api/v1/experiments/preflight',json=body)
            self.assertEqual(result.status_code,200,result.text)
            self.assertFalse(result.json()['training_started'])
            self.assertNotIn(str(self.root),result.text)
            self.assertIn('benchmark',client.get('/api/v1/ui/bootstrap').json()['config'])
            self.assertTrue(client.get('/api/v1/datasets').json()['items'][0]['bafu_configured'])
            self.assertFalse(client.get('/api/v1/benchmarks').json()['items'][1]['runnable'])
            body['config']['bafu_root']='/arbitrary/client/path'
            self.assertEqual(client.post('/api/v1/experiments/preflight',json=body).status_code,422)
        import sqlite3
        connection = sqlite3.connect(settings.jobs_path)
        try:
            self.assertEqual(connection.execute('select count(*) from jobs').fetchone()[0],0)
        finally:
            connection.close()

    def test_compatibility_distinguishes_areas_stations_and_predictors(self):
        directory = self.bafu/'observations/live/stations'
        directory.mkdir()
        with gzip.open(directory/'water_observations_stations.csv.gz', 'wt') as source:
            source.write('no,name\n2011,One\n2063,Two\n')
        catalog = CAMELSCHCatalog(self.catalog.root, [
            BasinCatalogRecord('2011', static_attributes={'area': 86.4}, eligible=False),
            BasinCatalogRecord('2063'),
            BasinCatalogRecord('2067', static_attributes={'area': 100}),
        ])
        rows = basin_compatibility(catalog, self.bafu)
        self.assertTrue(rows['2011']['bafu_only']['eligible'])
        self.assertEqual(rows['2011']['bafu_combined']['reasons'], ['CAMELS predictors unavailable'])
        self.assertIn('Missing positive catchment area', rows['2063']['bafu_only']['reasons'])
        self.assertIn('Not listed in BAFU station inventory', rows['2067']['bafu_only']['reasons'])
        self.assertFalse(basin_compatibility(catalog, None)['2011']['bafu_only']['eligible'])

    def test_api_filters_training_pool_and_rejects_incompatible_targets(self):
        settings = ServerSettings(self.root/'state', {'ch': DatasetRegistration(self.catalog.root, bafu_root=self.bafu)})
        application = Application(settings)
        catalog = CAMELSCHCatalog(self.catalog.root, [*self.catalog.records, BasinCatalogRecord('2063')])
        application.catalog = lambda _: catalog
        body = {'dataset_id': 'ch', 'config': self.mapping}
        config = application.experiment_config(body)
        self.assertEqual(config.basins.eligible_basins, ('2011',))
        with TestClient(create_app(settings, application=application)) as client:
            rows = client.get('/api/v1/datasets/ch/basins').json()['items']
            self.assertFalse(rows[1]['compatibility']['bafu_only']['eligible'])
            body = {**body, 'config': {**self.mapping, 'basins': {'target_basins': ['2063']}}}
            response = client.post('/api/v1/experiments', json=body)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn('Missing positive catchment area', response.text)

    def test_submission_checks_date_coverage_before_queueing(self):
        settings = ServerSettings(self.root/'state', {'ch': DatasetRegistration(self.catalog.root, bafu_root=self.bafu)})
        application = Application(settings)
        application.catalog = lambda _: self.catalog
        body = {'dataset_id': 'ch', 'config': self.mapping}
        with TestClient(create_app(settings, application=application)) as client:
            # No validation/test observations: configuration remains valid,
            # but the request must fail before a worker receives a job.
            self.rows = self.rows[:18]; self.write_rows()
            response = client.post('/api/v1/experiments', json=body)
            self.assertEqual(response.status_code, 422, response.text)
            import sqlite3
            with sqlite3.connect(settings.jobs_path) as connection:
                self.assertEqual(connection.execute('select count(*) from jobs').fetchone()[0], 0)
            self.rows = [
                {**self.rows[0], 'timestamp': str(datetime.combine(day, datetime.min.time(), ZoneInfo('Europe/Zurich')).astimezone(timezone.utc)), 'value': str(1+i%9)}
                for i, day in enumerate(self.days)
            ]
            self.write_rows()
            checked = client.post('/api/v1/experiments/validate', json=body)
            self.assertEqual(checked.status_code, 200, checked.text)
            submitted = client.post('/api/v1/experiments', json=body)
            self.assertEqual(submitted.status_code, 202, submitted.text)

    def test_audited_station_selection_can_be_submitted_with_strict_coverage(self):
        settings = ServerSettings(self.root/'state', {'ch': DatasetRegistration(self.catalog.root, bafu_root=self.bafu)})
        application = Application(settings)
        application.catalog = lambda _: CAMELSCHCatalog(self.catalog.root, [
            *self.catalog.records, BasinCatalogRecord('2012', static_attributes={'area': 86.4})])
        with TestClient(create_app(settings, application=application)) as client:
            for scope in ('targets_only', 'all_basins'):
                with self.subTest(scope=scope):
                    body = {'dataset_id': 'ch', 'config': {
                        **self.mapping, 'basins': {'target_basins': ['2011', '2012'], 'training_scope': scope}}}
                    rejected = client.post('/api/v1/experiments', json=body)
                    self.assertEqual(rejected.status_code, 422, rejected.text)
                    self.assertIn('Use stations with complete coverage', rejected.text)
                    audited = client.post('/api/v1/experiments/preflight', json=body)
                    self.assertEqual(audited.status_code, 200, audited.text)
                    report = audited.json()['manifest']
                    self.assertFalse(report['cohort_complete'])
                    self.assertEqual(report['effective_targets'], ['2011'])
                    body['config']['basins'].update(
                        target_basins=report['effective_targets'],
                        eligible_basins=sorted(set(report['training_basins']) | set(report['effective_targets'])))
                    self.assertTrue(application.experiment_config(body).benchmark.strict_cohort)
                    submitted = client.post('/api/v1/experiments', json=body)
                    self.assertEqual(submitted.status_code, 202, submitted.text)

    def test_configuration_roundtrip_and_manifest_export(self):
        _,data,series,prov=prepare_regional_data(self.config,catalog=self.catalog)
        cfg=_resolved(self.config,data);cfg['benchmark_manifest']=manifest(self.config,data,series,prov)
        restored=config_from_mapping(json.loads(json.dumps(cfg)))
        self.assertEqual(restored.benchmark,self.config.benchmark)
        self.assertNotIn('bafu_root',public_config(self.config))
        result=ExperimentResult([],{}, {},{}, {},[],[],cfg)
        paths=write_result_artifacts(result,self.root/'artifacts')
        self.assertTrue(Path(paths['benchmark_manifest']).is_file())
        import zipfile
        with zipfile.ZipFile(paths['archive']) as archive:self.assertIn('benchmark_manifest.json',archive.namelist())

    def test_saved_benchmark_http_results_and_comparison_without_execution(self):
        from complexity_ensemble.result_registry import save_run
        _,data,series,provenance=prepare_regional_data(self.config,catalog=self.catalog)
        cfg=_resolved(self.config,data);cfg['benchmark_manifest']=manifest(self.config,data,series,provenance)
        rows=_baseline_predictions(replace(self.config,strategy=replace(self.config.strategy,approach='persistence')),data,series,0)
        metrics={'nse':0.,'kge':0.,'rmse_mm_day':1.,'physics_error':float('nan')}
        event={'basin_id':'2011','seed':0,'event_type':'statistical_high_flow','verifier':'high_flow_one_to_one_v1','event_recall':.5}
        result=ExperimentResult(rows,metrics,{'2011':metrics},{},{},[event],[],cfg)
        settings=ServerSettings(self.root/'saved-state',{'ch':DatasetRegistration(self.catalog.root,bafu_root=self.bafu)})
        first=save_run(result,settings.runs_dir);second=save_run(result,settings.runs_dir)
        with TestClient(create_app(settings)) as client:
            response=client.get('/api/v1/runs/'+first.run_id)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['flood_summary']['event_recall'],.5)
            self.assertNotIn(str(self.root),response.text)
            comparison=client.post('/api/v1/comparisons',json={'run_a':first.run_id,'run_b':second.run_id})
            self.assertEqual(comparison.status_code,200,comparison.text)
            self.assertTrue(comparison.json()['benchmark_comparison']['compatible'])
            self.assertTrue(comparison.json()['paired_comparison']['compatible'])
            self.assertEqual(comparison.json()['flood_delta_b_minus_a']['event_recall'],0)
            for artifact in ('benchmark_manifest','flood_metrics','archive'):
                self.assertEqual(client.get(f'/api/v1/runs/{first.run_id}/artifacts/{artifact}').status_code,200)
        old=replace(first,config={'benchmark_manifest':{'version':'unverified'}})
        self.assertFalse(compare_runs(old,second)['benchmark_comparison']['compatible'])

    def test_statistics_zip_is_station_index(self):
        # Build a minimal synthetic DBF without any geospatial dependency.
        import struct,zipfile
        fields=[('EDV',4),('NAME',8),('URL_HQPDF',32)];header=32+32*len(fields)+1;size=1+sum(n for _,n in fields)
        data=bytearray(header+size);struct.pack_into('<IHH',data,4,1,header,size)
        for i,(name,width) in enumerate(fields):
            offset=32+32*i;data[offset:offset+len(name)]=name.encode();data[offset+11]=ord('C');data[offset+16]=width
        data[header-1]=13;data[header:]=b' '+b'2011'+b'Gauge   '+b'https://example.org/a.pdf'.ljust(32)
        path=self.root/'hq.zip'
        with zipfile.ZipFile(path,'w') as z:z.writestr('HQStat.dbf',data)
        index=statistics_index(path)
        self.assertFalse(index['contains_return_levels']);self.assertEqual(index['records'][0]['station_id'],'2011')


class FloodVerification(unittest.TestCase):
    def records(self, observed, predicted, start=date(2001,1,1)):
        return [PredictionRecord(0,'2011',start+timedelta(days=i),a,b,0,0,0,0,start+timedelta(days=i-1),1) for i,(a,b) in enumerate(zip(observed,predicted))]

    def test_one_long_prediction_cannot_match_two_observed_events(self):
        rows=self.records([0,5,0,5,0],[0,5,5,5,0])
        result=flood_metrics(rows,3)
        self.assertEqual(result['observed_events'],2);self.assertEqual(result['matched_events'],1)
        self.assertEqual(result['event_recall'],.5)

    def test_unmatched_boundary_and_missing_dates_are_explicit(self):
        rows=self.records([5,5,0,0],[5,5,0,0]);result=flood_metrics(rows,3)
        self.assertEqual(result['censored_observed_events'],1);self.assertEqual(result['complete_peak_pairs'],0)
        self.assertTrue(torch.isnan(torch.tensor(result['peak_timing_mae_days'])))
        self.assertTrue(torch.isnan(torch.tensor(flood_metrics(self.records([0,0],[0,0]),3)['event_recall'])))

    def test_rolling_events_do_not_cross_issue_boundaries(self):
        rows=self.records([0,5,5,5,5,0],[0,5,5,5,5,0])
        rows=[replace(r,issue_date=date(2001,1,1) if i<3 else date(2001,1,4),lead_day=i%3+1) for i,r in enumerate(rows)]
        result=flood_metrics(rows,3)
        self.assertEqual(result['observed_events'],2);self.assertEqual(result['complete_peak_pairs'],0)

    def test_paired_comparison_rejects_different_targets_and_bootstraps_years(self):
        a=self.records([1,2],[2,3])+self.records([1,2],[2,3],date(2002,1,1))
        b=[replace(r,predicted_mm_day=r.observed_mm_day) for r in a]
        report=paired_comparison(a,b)
        self.assertTrue(report['compatible']);self.assertEqual(report['rmse_delta_b_minus_a'],-1)
        self.assertEqual(report['confidence_interval_95'],[-1,-1])
        self.assertFalse(paired_comparison(a,b[:-1])['compatible'])
        b[0]=replace(b[0],observed_mm_day=99)
        self.assertFalse(paired_comparison(a,b)['compatible'])
