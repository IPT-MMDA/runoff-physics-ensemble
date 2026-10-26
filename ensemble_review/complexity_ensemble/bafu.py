"""Local FOEN observation adapter. No network requests or training on import/read."""
from __future__ import annotations

import csv
import gzip
import hashlib
import math
import struct
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .hydrology_data import HydrologySeries


def observation_root(root):
    root = Path(root)
    for candidate in (root / 'observations/live', root):
        if (candidate / 'data').is_dir():
            return candidate
    raise ValueError('BAFU root must contain observations/live/data or data')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def station_areas(catalog):
    result = {}
    for record in catalog.records:
        for key in ('area', 'area_km2', 'area(km2)', 'catchment_area', 'catchment_area(km2)'):
            value = record.static_attributes.get(key)
            if value is not None and math.isfinite(float(value)) and float(value) > 0:
                result[record.basin_id] = float(value)
                break
    return result


def basin_compatibility(catalog, root):
    """Cheap station/area checks; daily Q and window coverage require preflight."""
    areas = station_areas(catalog)
    stations = None
    available = False
    if root is not None:
        try:
            base = observation_root(root)
            available = True
            path = base / 'stations/water_observations_stations.csv.gz'
            if path.is_file():
                with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as source:
                    stations = {row['no'] for row in csv.DictReader(source)}
        except (OSError, ValueError, KeyError):
            available = False
    result = {}
    for record in catalog.records:
        reasons = []
        if not available:
            reasons.append('BAFU observations unavailable')
        if record.basin_id not in areas:
            reasons.append('Missing positive catchment area')
        if stations is not None and record.basin_id not in stations:
            reasons.append('Not listed in BAFU station inventory')
        combined = reasons + ([] if record.eligible else ['CAMELS predictors unavailable'])
        result[record.basin_id] = {
            'bafu_only': {'eligible': not reasons, 'reasons': reasons},
            'bafu_combined': {'eligible': not combined, 'reasons': combined},
        }
    return result


def load_observations(root, basin_ids, start, end, *, target='daily_mean', areas=None,
                      release_states=(2, 3), day_timezone='Europe/Zurich'):
    """Read monthly partitions once for the whole cohort, selecting only Q.

    Export timestamps represent UTC interval starts. Convert to the configured
    local calendar before taking a date; truncating 23:00 UTC shifts Swiss days.
    Conflicting duplicates fail rather than silently selecting one revision.
    """
    if target not in {'daily_mean', 'daily_max'}:
        raise ValueError('BAFU target must be daily_mean or daily_max')
    root = observation_root(root)
    aggregation = '1day_mean' if target == 'daily_mean' else '1day_max'
    wanted = set(map(str, basin_ids)); areas = areas or {}
    missing = wanted - areas.keys()
    if missing:
        raise ValueError('Positive catchment area (km2) required for BAFU stations: ' + ', '.join(sorted(missing)))
    zone = ZoneInfo(day_timezone)
    rows = {basin: {} for basin in wanted}; sources = []
    for path in sorted((root / 'data').glob(f'*/water_observations_*_data_{aggregation}_*.csv.gz')):
        # Include adjacent calendar years because timestamps are UTC.
        try:
            year = int(path.parent.name)
        except ValueError:
            continue
        if not start.year - 1 <= year <= end.year + 1:
            continue
        used = False
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as source:
            for row in csv.DictReader(source):
                basin = row['station_no']
                if basin not in wanted or row['parameter_name'] != 'Q':
                    continue
                # BAFU exports include missing observations with blank quality
                # flags and/or discharge. They are gaps, not numeric values.
                release_state = (row['release_state'] or '').strip()
                value_text = (row['value'] or '').strip()
                if not release_state or not value_text:
                    continue
                if int(release_state) not in release_states:
                    continue
                if row['unit_symbol'] not in {'m³/s', 'm3/s'}:
                    raise ValueError(f'Unexpected BAFU Q unit for {basin}')
                stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                day = stamp.replace(tzinfo=timezone.utc).astimezone(zone).date() if stamp.tzinfo is None else stamp.astimezone(zone).date()
                if not start <= day <= end:
                    continue
                value = float(value_text)
                if not math.isfinite(value) or value < 0:
                    continue
                value *= 86.4 / areas[basin]
                if day in rows[basin] and rows[basin][day] != value:
                    raise ValueError(f'Conflicting BAFU Q values for {basin} on {day}')
                rows[basin][day] = value; used = True
        if used:
            sources.append({'file': str(path.relative_to(root)), 'sha256': file_digest(path)})
    series = {}
    for basin, values in rows.items():
        days = tuple(sorted(values))
        if days:
            series[basin] = HydrologySeries(days, torch.tensor([values[d] for d in days], dtype=torch.float32),
                torch.empty(len(days), 0), (), basin, 'CH')
    return series, {'adapter': 'bafu_q_v1', 'target': target, 'day_timezone': day_timezone,
                    'release_states': list(release_states), 'area_source': 'CAMELS-CH metadata; not a model predictor in BAFU-only mode',
                    'areas_km2': {b: areas[b] for b in sorted(wanted)}, 'files': sources}


def merge_predictors(observed, predictors):
    """Join on station/day, retaining BAFU as the sole target source."""
    result = {}
    for basin, obs in observed.items():
        if basin not in predictors:
            continue
        met = predictors[basin]; lookup = {day: i for i, day in enumerate(met.dates)}
        pairs = [(i, lookup[day]) for i, day in enumerate(obs.dates) if day in lookup]
        if pairs:
            oi, mi = zip(*pairs)
            result[basin] = replace(obs, dates=tuple(obs.dates[i] for i in oi),
                discharge=obs.discharge[list(oi)], forcings=met.forcings[list(mi)], forcing_names=met.forcing_names)
    return result


def statistics_index(path):
    """Read the supplied HQStat DBF directly inside its ZIP, without extraction."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('hqstat.dbf')]
        if len(names) != 1:
            raise ValueError('Expected one HQStat.dbf in the statistics ZIP')
        data = archive.read(names[0])
    count = struct.unpack_from('<I', data, 4)[0]
    header, size = struct.unpack_from('<HH', data, 8)
    fields = []; pos = 32
    while data[pos] != 13:
        field = data[pos:pos+32]
        fields.append((field[:11].split(b'\0')[0].decode(), field[16])); pos += 32
    records = []
    for index in range(count):
        offset = header + index * size
        if data[offset:offset+1] == b'*':
            continue
        offset += 1; row = {}
        for name, width in fields:
            row[name] = data[offset:offset+width].decode('utf-8', errors='replace').strip(); offset += width
        records.append({'station_id': row.get('EDV'), 'name': row.get('NAME'), 'report_url': row.get('URL_HQPDF')})
    return {'records': records, 'sha256': file_digest(path), 'contains_return_levels': False,
            'identifier_type': 'HQStat EDV; verify mapping to observation station_no',
            'note': 'Station/report index only. PDF return levels and station mapping must be verified and imported separately.'}


def load_return_levels(path):
    """Validated explicit import; never infer thresholds from station map records."""
    levels = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            basin = row['station_id']; period = int(row['return_period_years']); value = float(row['discharge_m3_s'])
            if period not in {2, 10, 30, 100, 300} or not math.isfinite(value) or value <= 0:
                raise ValueError('Return levels require positive flow and a supported return period')
            if not row.get('source_url') or not row.get('reference_period'):
                raise ValueError('Return levels require source_url and reference_period')
            if (basin, period) in levels:
                raise ValueError('Duplicate station/return-period threshold')
            levels[basin, period] = {**row, 'discharge_m3_s': value}
    for basin in {b for b, _ in levels}:
        ordered = [levels[b, p]['discharge_m3_s'] for b, p in sorted(levels) if b == basin]
        if any(a >= b for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Return levels must increase with return period')
    return levels
