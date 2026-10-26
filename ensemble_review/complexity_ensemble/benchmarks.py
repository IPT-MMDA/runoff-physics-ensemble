"""Versioned forecast tasks, read-only preflight, and model-independent provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
from typing import Literal

from .serialization import json_safe


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: Literal['none', 'bafu_only', 'bafu_combined'] = 'none'
    version: Literal['bafu_forecast_v1'] = 'bafu_forecast_v1'
    target: Literal['daily_mean', 'daily_max'] = 'daily_mean'
    release_states: tuple[int, ...] = (2, 3)
    day_timezone: Literal['Europe/Zurich', 'UTC'] = 'Europe/Zurich'
    threshold_source: Literal['training_quantile', 'official_return_level'] = 'training_quantile'
    return_period_years: int = 2
    strict_cohort: bool = True
    expected_evaluation_fingerprint: str = ""


def profiles():
    return [
        {'id': 'bafu_forecast_v1', 'runnable': True,
         'description': 'BAFU discharge forecasts with drought or high-flow verification. Drought thresholds use training observations; verified flood return levels apply only to high flow. Combined rolling runs are perfect-weather hindcasts.',
         'published_reproduction': False},
        {'id': 'swiss_probabilistic_lstm_2026', 'runnable': False,
         'title': 'The need for uncertainty: why probabilistic LSTMs are key to improving flood predictions and enabling learned warning rules',
         'paper': 'https://hess.copernicus.org/articles/30/6339/2026/',
         'code': 'https://doi.org/10.5281/zenodo.18385505',
         'archive_version': 'https://zenodo.org/records/22959931',
         'archive_file': 'sb2025.zip', 'archive_md5': '1278a51dbb48491ad94d53f5eb9cadce',
         'reported_cohort': 196, 'sequence_length': 365,
         'reported_splits': {'train': ['1980-10-01', '1995-09-30'],
                             'validation': ['1995-10-01', '2000-09-30'],
                             'test': ['2000-10-01', '2020-09-30']},
         'status': 'Reference only: archived cohort, 22 attributes, target-day meteorology, warm-up, threshold versions and released code have not been reconciled. Do not compare published scores to bafu_forecast_v1.'}]


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_benchmark(config):
    cfg = config.benchmark
    if cfg.mode not in {'none', 'bafu_only', 'bafu_combined'}:
        raise ValueError('Unknown benchmark mode')
    if cfg.mode == 'none':
        return
    if cfg.expected_evaluation_fingerprint and (len(cfg.expected_evaluation_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in cfg.expected_evaluation_fingerprint)):
        raise ValueError("Expected evaluation fingerprint must be a SHA-256 hex digest")
    if cfg.version != 'bafu_forecast_v1' or cfg.target not in {'daily_mean', 'daily_max'}:
        raise ValueError('Unsupported benchmark version/target')
    if cfg.day_timezone not in {'UTC', 'Europe/Zurich'} or not cfg.release_states or set(cfg.release_states) - {2, 3}:
        raise ValueError('Use UTC/Europe/Zurich and validated/definitive BAFU release states (2, 3)')
    if config.bafu_root is None:
        raise ValueError('BAFU observations are not configured for this dataset')
    drought = config.extremes.mode == 'drought'
    if config.extremes.mode not in {'statistical', 'drought'}:
        raise ValueError('BAFU benchmark requires statistical high-flow or drought evaluation')
    if drought and cfg.target != 'daily_mean':
        raise ValueError('BAFU drought evaluation requires daily_mean discharge')
    if drought and cfg.threshold_source != 'training_quantile':
        raise ValueError('BAFU drought thresholds are fitted to training observations; flood return levels are not drought thresholds')
    if cfg.threshold_source not in {'training_quantile', 'official_return_level'}:
        raise ValueError('Unknown benchmark threshold source')
    if not drought and cfg.threshold_source == 'training_quantile' and config.extremes.definition == 'absolute':
        raise ValueError('Training-quantile benchmark requires automatic_q95 or quantile thresholds')
    if cfg.threshold_source == 'official_return_level':
        if config.bafu_return_levels is None or cfg.return_period_years not in {2, 10, 30, 100, 300}:
            raise ValueError('Configure a verified return-level CSV and supported return period')
    if cfg.mode == 'bafu_only':
        if any((config.estreams_root, config.camels_chem_root, config.camels_state_features,
                config.hyd_responses_root, config.hyd_responses_static)):
            raise ValueError('BAFU-only mode excludes optional external predictors')
        if config.hyperparameters.physics_weight != 0 or config.physics.optimization_mode != 'none':
            raise ValueError('BAFU-only mode requires physics_weight=0 and physics optimization=none; weather is unavailable')
    if cfg.target == 'daily_max' and (config.hyperparameters.physics_weight != 0 or config.physics.optimization_mode != 'none'):
        raise ValueError('Daily-maximum discharge is not a daily water balance: disable physics loss/optimization')


def prepare_bafu(config, catalog, relevant, start, end):
    from .bafu import load_observations, merge_predictors, station_areas
    areas = station_areas(catalog)
    observed, provenance = load_observations(config.bafu_root, relevant, start, end,
        target=config.benchmark.target, areas=areas, release_states=config.benchmark.release_states,
        day_timezone=config.benchmark.day_timezone)
    if config.benchmark.mode == 'bafu_only':
        return observed, provenance
    predictors = {}; failures = {}
    for basin in relevant:
        try:
            predictors[basin] = catalog.load_series(basin, start, end)
        except (ValueError, FileNotFoundError) as error:
            failures[basin] = 'Predictor series unavailable or invalid'
    provenance['predictor_load_failures'] = failures
    return merge_predictors(observed, predictors), provenance


def threshold_values(config, series, targets, areas):
    from .regional import _thresholds
    if config.extremes.mode == 'drought':
        from .hydrology_drought import fit_thresholds
        period = config.dates.train.resolved()
        cfg = config.extremes
        thresholds = {}
        for basin in targets:
            flow = dict(zip(series[basin].dates, map(float, series[basin].discharge)))
            thresholds[basin] = asdict(fit_thresholds(flow, period.start, period.end,
                definition=cfg.drought_definition, smoothing=cfg.drought_smoothing,
                probability=cfg.drought_quantile, calendar_window=cfg.drought_calendar_window))
        return thresholds, {'source': 'training_observations_only', 'averaging_days': 7,
                            **drought_settings(config)}
    if config.benchmark.threshold_source == 'training_quantile':
        return _thresholds(config.extremes, targets, series, config.dates.train), {'source': 'training_quantile'}
    from .bafu import load_return_levels, file_digest
    levels = load_return_levels(config.bafu_return_levels)
    values = {}; sources = {}
    for basin in targets:
        key = (basin, config.benchmark.return_period_years)
        if key not in levels:
            raise ValueError(f'Missing verified HQ{key[1]} for station {basin}')
        row = levels[key]; area = areas[basin]
        values[basin] = row['discharge_m3_s'] * 86.4 / area
        sources[basin] = row
    return values, {'source': 'official_return_level', 'sha256': file_digest(config.bafu_return_levels),
                    'return_period_years': config.benchmark.return_period_years, 'records': sources,
                    'interpretation': 'Retrospective official peak-flow thresholds applied to selected daily statistic; not an operational historical threshold reconstruction.'}


def drought_settings(config):
    cfg = config.extremes
    return {'definition': cfg.drought_definition, 'smoothing': cfg.drought_smoothing,
            'probability': cfg.drought_quantile, 'calendar_window': cfg.drought_calendar_window,
            'minimum_days': cfg.drought_minimum_days}


def drought_coverage(config, data, series):
    """Use the production verifier to audit windows without constructing a model.

    Observations stand in for predictions solely to check which forecast windows
    can be scored. These placeholder forecast scores are never reported.
    """
    from types import SimpleNamespace
    from .hydrology_drought import verify_drought
    period = config.dates.train.resolved()
    horizon = config.forecast.horizon_days if config.forecast.mode == 'rolling' else 1
    coverage = {}; observed_digest = hashlib.sha256()
    for basin in data.target_basins:
        flow = dict(zip(series[basin].dates, map(float, series[basin].discharge)))
        days = sorted(day for day, station in zip(data.test.target_dates, data.test.basin_ids) if station == basin)
        records = []; previous = None; lead = 0
        for day in days:
            if previous is None or day != previous + timedelta(days=1) or lead >= horizon:
                issue = day - timedelta(days=1); lead = 0
            lead += 1; previous = day
            records.append(SimpleNamespace(seed=0, basin_id=basin, issue_date=issue,
                target_date=day, lead_day=lead, predicted_mm_day=flow[day]))
        _, details = verify_drought(records, {basin: series[basin]}, period.start, period.end,
                                    **drought_settings(config))
        eligible = sum(row['eligible'] for row in details['days'])
        if not eligible:
            raise ValueError(f'BAFU drought station {basin}: no evaluable seven-day test windows. '
                             'Check training calendar coverage, observation gaps, smoothing and forecast horizon.')
        coverage[basin] = {'eligible_days': eligible, 'excluded_days': len(days) - eligible}
        # Include smoothing context and observed event boundaries in the task
        # identity, including observations before the first test target date.
        for row in details['days']:
            observed_digest.update(digest({k: v for k, v in row.items() if k != 'predicted_m7_mm_day'}).encode())
        for row in details['events']:
            if row['kind'] == 'observed':
                observed_digest.update(digest({k: v for k, v in row.items() if k != 'matched_event_index'}).encode())
    return coverage, observed_digest.hexdigest()


def training_digest(config, data, series, provenance):
    """Hash source training records without expanding every overlapping window."""
    train = config.dates.train.resolved()
    content = hashlib.sha256()
    for basin in data.training_basins:
        s = series[basin]
        content.update(basin.encode())
        for i, day in enumerate(s.dates):
            if day <= train.end:
                content.update(json.dumps([day.isoformat(), float(s.discharge[i]), s.forcings[i].tolist()], separators=(',', ':')).encode())
    attrs = provenance.get('training_static_attributes', {})
    return digest({'version': config.benchmark.version, 'train_dates': asdict(config.dates.train),
        'training_basins': data.training_basins, 'sequence_length': config.hyperparameters.sequence_length,
        'effective_context_length': provenance.get('context_length'),
        'feature_names': data.feature_names, 'record_digest': content.hexdigest(),
        'static_attributes': {b: {name: attrs.get(b, {}).get(name) for name in data.static_feature_names} for b in data.training_basins}})


def manifest(config, data, series, provenance):
    from .regional import effective_training_basins
    if config.benchmark.strict_cohort:
        requested = effective_training_basins(config.basins, config.basins.eligible_basins or series)
        if set(data.target_basins) != set(config.basins.target_basins) or set(data.training_basins) != set(requested):
            raise ValueError('Strict benchmark cohort lost stations; inspect preflight exclusions before changing the cohort')
    thresholds, threshold_source = threshold_values(config, series, data.target_basins, provenance.get("areas_km2", {}))
    train = config.dates.train.resolved()
    for basin in data.target_basins:
        if not any(train.contains(d) for d in series[basin].dates):
            raise ValueError(f'No training reference observations for {basin}')
    # Identity includes actual values, not just paths or requested date bounds.
    observed = {b: {day: float(q) for day, q in zip(series[b].dates, series[b].discharge)} for b in data.target_basins}
    evaluation = {'version': config.benchmark.version, 'target': config.benchmark.target,
        'units': 'mm/day basin-equivalent', 'day_timezone': config.benchmark.day_timezone,
        'forecast': asdict(config.forecast), 'thresholds': thresholds, 'threshold_source': threshold_source,
        'verifier': 'high_flow_one_to_one_v1',
        'observations': [(b, d, observed[b][d]) for b, d in zip(data.test.basin_ids, data.test.target_dates)]}
    coverage = {}
    if config.extremes.mode == 'drought':
        from .hydrology_drought import VERIFICATION
        coverage, observation_digest = drought_coverage(config, data, series)
        evaluation.update(verifier=VERIFICATION, drought=drought_settings(config),
                          drought_observation_digest=observation_digest)
    test_fingerprint = digest(evaluation)
    if config.benchmark.expected_evaluation_fingerprint and test_fingerprint != config.benchmark.expected_evaluation_fingerprint:
        raise ValueError("Evaluation fingerprint differs from the pinned benchmark cohort/task")
    report = {'version': config.benchmark.version, 'mode': config.benchmark.mode,
        'published_reproduction': False, 'evaluation_fingerprint': test_fingerprint,
        'training_fingerprint': training_digest(config, data, series, provenance),
        'requested_targets': list(config.basins.target_basins), 'effective_targets': list(data.target_basins),
        'training_basins': list(data.training_basins), 'excluded_basins': dict(data.skipped_basins),
        'samples': {name: len(getattr(data, name).targets) for name in ('train', 'validation', 'test')},
        'thresholds_mm_day': thresholds, 'threshold_provenance': threshold_source,
        'features': list(data.feature_names), 'static_features': list(data.static_feature_names),
        'physics_available': config.benchmark.mode != 'bafu_only' and config.benchmark.target == 'daily_mean',
        'forecast_interpretation': ('one-day forecast using preceding observations' if config.forecast.mode == 'one_day'
                                    else 'autoregressive discharge forecast without weather inputs' if config.benchmark.mode == 'bafu_only'
                                    else 'perfect-weather hindcast; historical future meteorology'),
        'provenance': {k: v for k, v in provenance.items() if k != 'training_static_attributes'}, 'evaluation': {k: v for k, v in evaluation.items() if k != 'observations'}}
    if config.extremes.mode == 'drought':
        report['drought_coverage'] = coverage
    return report


def preflight(config, catalog):
    """Uses the same preparation as execution; never constructs or fits a model."""
    from .regional import prepare_regional_data
    from dataclasses import replace
    audit_config = replace(config, benchmark=replace(config.benchmark, strict_cohort=False))
    _, data, series, provenance = prepare_regional_data(audit_config, catalog=catalog, audit_only=True)
    report = manifest(audit_config, data, series, provenance)
    expected = set(config.basins.target_basins)
    report['strict_cohort_requested'] = config.benchmark.strict_cohort
    report['cohort_complete'] = not data.skipped_basins and set(data.target_basins) == expected and not provenance.get('missing_stations')
    return report


def paired_comparison(rows_a, rows_b, *, samples=1000):
    """Paired RMSE difference with year blocks shared across stations and seeds."""
    import math
    import random
    def key(r):
        return (r.seed, r.basin_id, r.target_date, r.issue_date, r.lead_day)
    a = {key(r): r for r in rows_a}; b = {key(r): r for r in rows_b}
    if a.keys() != b.keys() or not a:
        return {'compatible': False, 'reason': 'Prediction keys differ or are empty'}
    years = {}
    for k, first in a.items():
        second = b[k]
        values = (first.observed_mm_day, second.observed_mm_day, first.predicted_mm_day, second.predicted_mm_day)
        if not all(math.isfinite(x) for x in values) or abs(values[0] - values[1]) > 1e-5:
            return {'compatible': False, 'reason': 'Observed targets differ or predictions contain invalid values'}
        acc = years.setdefault(first.target_date.year, [0., 0., 0])
        acc[0] += (values[2] - values[0]) ** 2; acc[1] += (values[3] - values[1]) ** 2; acc[2] += 1
    def delta(blocks):
        count = sum(x[2] for x in blocks)
        return math.sqrt(sum(x[1] for x in blocks) / count) - math.sqrt(sum(x[0] for x in blocks) / count)
    result = {'compatible': True, 'pairs': len(a), 'year_blocks': len(years), 'rmse_delta_b_minus_a': delta(list(years.values())),
              'method': 'paired calendar-year block bootstrap; stations and seeds retained within each year', 'confidence_interval_95': None}
    if len(years) >= 2:
        rng = random.Random(0); blocks = list(years.values())
        estimates = sorted(delta(rng.choices(blocks, k=len(blocks))) for _ in range(samples))
        result['confidence_interval_95'] = [estimates[int(samples*.025)], estimates[int(samples*.975)]]
    return result
