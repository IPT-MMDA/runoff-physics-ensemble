"""Shared flood-event verification; independent of model and data adapters."""
from dataclasses import dataclass
from datetime import timedelta
import math

from .hydrology_drought import match_events


@dataclass(frozen=True)
class FloodEvent:
    start: object
    end: object
    peak_day: object
    peak: float
    censored: bool


def _events(rows, field, threshold):
    spells = []; current = []
    for index, row in enumerate(rows):
        value = getattr(row, field)
        if current and (row.target_date != current[-1].target_date + timedelta(days=1) or not math.isfinite(value) or value <= threshold):
            spells.append(current); current = []
        if math.isfinite(value) and value > threshold:
            current.append(row)
    if current:
        spells.append(current)
    lookup = {row.target_date: getattr(row, field) for row in rows}
    events = []
    for spell in spells:
        peak = max(spell, key=lambda row: getattr(row, field))
        before = lookup.get(spell[0].target_date - timedelta(days=1), float('nan'))
        after = lookup.get(spell[-1].target_date + timedelta(days=1), float('nan'))
        events.append(FloodEvent(spell[0].target_date, spell[-1].target_date, peak.target_date,
            getattr(peak, field), not (math.isfinite(before) and math.isfinite(after))))
    return events


def flood_metrics(rows, threshold):
    """One-to-one matching within issue trajectories for rolling forecasts.

    Day-1 forecasts form a continuous verification series. Rolling blocks remain
    separate; complete-peak errors exclude events touching a boundary/gap.
    """
    groups = {}
    rolling = any(row.lead_day > 1 for row in rows)
    for row in rows:
        groups.setdefault(row.issue_date if rolling else 'one_day', []).append(row)
    observed_count = predicted_count = matched_count = censored = 0
    errors = []; timing = []
    for group in groups.values():
        group.sort(key=lambda row: row.target_date)
        obs = _events(group, 'observed_mm_day', threshold)
        pred = _events(group, 'predicted_mm_day', threshold)
        pairs = match_events(obs, pred)
        observed_count += len(obs); predicted_count += len(pred); matched_count += len(pairs)
        censored += sum(e.censored for e in obs)
        for i, j in pairs:
            if not obs[i].censored and not pred[j].censored:
                errors.append(abs(pred[j].peak - obs[i].peak))
                timing.append(abs((pred[j].peak_day - obs[i].peak_day).days))
    return {'verifier': 'high_flow_one_to_one_v1',
        'observed_events': observed_count, 'predicted_events': predicted_count, 'matched_events': matched_count,
        'censored_observed_events': censored, 'complete_peak_pairs': len(errors),
        'event_precision': matched_count / predicted_count if predicted_count else float('nan'),
        'event_recall': matched_count / observed_count if observed_count else float('nan'),
        'peak_magnitude_mae_mm_day': sum(errors)/len(errors) if errors else float('nan'),
        'peak_timing_mae_days': sum(timing)/len(timing) if timing else float('nan')}


def flood_summary(rows):
    """Unweighted finite basin/seed means with explicit contributing counts."""
    selected = [r for r in rows if r.get('event_type') == 'statistical_high_flow'
                and r.get('verifier') == 'high_flow_one_to_one_v1']
    if not selected:
        return {}
    result = {}
    for name in ('precision', 'recall', 'critical_success_index', 'false_alarm_ratio',
                 'average_precision', 'event_precision', 'event_recall',
                 'extreme_rmse_mm_day', 'peak_magnitude_mae_mm_day', 'peak_timing_mae_days'):
        values = [r[name] for r in selected if isinstance(r.get(name), (int, float)) and math.isfinite(r[name])]
        result[name] = sum(values) / len(values) if values else None
        result[name + '_count'] = len(values)
    return result
