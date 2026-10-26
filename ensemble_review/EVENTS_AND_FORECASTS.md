# Hydrological events and forecast setup

## Rainfall–runoff replay

After a regional run, open **Rainfall–runoff replay**, choose a target basin,
find rainfall episodes, and show one. The animation uses the run's predictions
and the basin's CAMELS-CH simulation-based daily precipitation and PET plus
observed gauge discharge. Each predicted day is based on the preceding input
window (30 days by default), with predicted discharge averaged across seeds.
The day slider steps through 14 test days across four separate views: weather,
gauge discharge, teaching bucket, and a dated input-to-prediction diagram.
The diagram summarizes the preceding model-input window and the target day's
predicted and observed discharge. In rolling mode it labels predicted discharge
fed back as the preceding day's flow. Target-day weather is outside that day's
input window. Generate GIF
shows the animation in the app and provides a download with the same four
views; install the
`animations` extra to enable GIF export (`pip install -e '.[animations]'`).

The bucket chart is a separate, uncalibrated teaching simulation. Rain enters
storage; evaporation is limited by PET and available water; 30% of the
remainder is released each day. Its storage, evaporation, and release are not
internal states or outputs of the trained ensemble. PET is
potential evaporation demand, not observed actual evaporation. The replay
requires the original CAMELS-CH data directory; saved prediction CSVs alone
do not contain meteorological inputs.

The Task tab offers daily discharge, statistical extremes, hydrological events,
and a dedicated hydrological drought task. Daily discharge forecasting reports
continuous discharge metrics. Statistical extremes retain the existing
single-day high-flow threshold evaluation. Hydrological event evaluation adds
the definitions below. It does not claim that a discharge event is a confirmed
flood, landslide, or other observed impact.

## Event definitions

Thresholds are calculated separately for each target basin from observations
in the training date range. The default high-flow, low-flow, and daily-rise
quantiles are 0.80, 0.20, and 0.95. Daily falls use their own 0.95
training-period quantile. These are configurable in the Task tab.

| Event | Definition | Severity reported |
| --- | --- | --- |
| Sustained high flow | Consecutive days above the high-flow threshold, for at least the configured minimum duration | Sum of discharge above the threshold, in mm |
| Low-flow drought | Consecutive days below the low-flow threshold, for at least the configured minimum duration | Sum of discharge deficits, in mm |
| Rapid daily rise | Consecutive daily changes greater than the training-period rise threshold | Sum of changes above the threshold, in mm/day |
| Rapid daily fall | Consecutive daily decreases greater than the training-period fall threshold | Sum of decreases above the threshold, in mm/day |
| Repeated high-flow pulses | Two or more high-flow spells separated by no more than the configured gap | Total excess flow of the component spells, in mm |
| Regional concurrence | At least the configured number of selected basins above their own high-flow thresholds on the same days, for the minimum duration | Sum of affected basin-days |
| Regional low flow | At least the configured number of selected basins below their own low-flow thresholds on the same days, for the minimum duration | Sum of affected basin-days |

Events are split at missing dates. Results are reported per seed and basin, with
an `all` row for regional concurrence. They include observed and predicted
event counts, event-day counts, total severity, and overlap-based event
precision and recall. A precision or recall with no corresponding events is
reported as undefined. Rapid changes start on the second test
day because a preceding test-day prediction is needed to calculate the rise.

These thresholds are exploratory. A percentile is not a local flood danger
level or ecological minimum flow. Persisted events and impacts need their own
observations and independently selected thresholds.

## One-day and rolling forecasts

The default one-day setup preserves the existing evaluation: each test-day
window uses observed discharge up to the preceding day. Rolling mode produces
non-overlapping blocks of the configured horizon, seven days by default. The
first day of each block uses observed discharge available at its issue date.
For subsequent days, predicted discharge replaces observed discharge in all
input-window positions after that issue date. Observed discharge becomes
available again when the next block begins. `predictions.csv` includes
`issue_date` and `lead_day` to distinguish the two modes and compare lead times.
The Results metrics view and saved manifest report NSE, KGE, and RMSE by lead
day; `metrics.csv` also includes `lead_1`, `lead_2`, and so on when rolling mode
is used.
Event scores in rolling mode are calculated from the stitched sequence of
non-overlapping blocks. An event that crosses a block boundary is therefore
assessed using predictions from two issue dates, and the next block may use
newly observed discharge available at that date.

Rolling mode uses **historically observed meteorological and optional predictor
values** for days after issue, except CAMELS-CH and HYD-RESPONSES water-state
features, which remain at their issue-day values and availability flags.
It is therefore a *perfect-weather hindcast* of
the discharge rollout, not an operational forecast issued without future
weather information. It reuses the trained one-day model; the model is not
separately trained for each lead time. To make an operational forecast, supply
weather forecasts available at each issue date and evaluate the whole
weather-to-discharge chain by lead time and event.

## Paved-site runoff and retention scenario

Open **Impact scenario**, set the controls, and click **Run experiment with
scenario**. The comparison appears there when training finishes. Alternatively,
check **Also apply scenario when using Run experiment in Results** and run from
Results. To compare different
parameters against the same completed run, choose its target basin in Impact
scenario and click **Compare with current run**; this does not retrain the model.
The controls define a site's fraction of the upstream
basin, its original and proposed paved fractions, daily runoff coefficients
for paved and pervious surfaces, the fraction draining to the gauge, and a
retention capacity and maximum daily release. Capacity and release are
millimetres over the full site area. The default 20 mm capacity and 2 mm/day
release are illustrative assumptions, not calibrated design values.

For each day, the scenario converts historical basin-average precipitation
to site runoff using the area-weighted coefficients. Proposed runoff enters a
simple bucket: water above capacity overflows on the same day, and stored
water releases at up to the configured daily limit. The original site is
assumed to have no retention. The difference between proposed output and
original runoff is multiplied by connected site area as a fraction of basin
area, then added to each baseline discharge prediction. Negative results are
clipped to zero and counted in the scenario summary. Storage starts empty at
the chosen start date and resets after gaps in the rainfall series.

The plot averages forecast seeds; the event table reports each seed. Event
thresholds are fixed from training-period observed discharge. The daily
high-flow row uses Q95 by default, or the configured statistical threshold
when that task is selected. Other rows follow the configured hydrological
event definitions, or their defaults for a discharge-only experiment. The
comparison reports changes in predicted events, not observed impacts or a
new measure of forecast skill. The scenario is an interactive, unsaved
sensitivity analysis; it does not retrain the model or alter leaderboard
metrics. Historical daily rainfall is supplied even for rolling hindcasts,
and adjusted discharge is not fed back into later model predictions. This
calculation cannot resolve short storm peaks, drainage hydraulics, or flooding.

## Hydrological drought task

Select **Hydrological drought (HYD-RESPONSES-style definitions)** in Task.
Users retain their selected basins, data paths, date splits, models and seeds;
there is no fixed benchmark cohort, release or reference period. This task
keeps daily discharge as the model target and adds drought verification.
The optional HYD-RESPONSES water-state predictors in Data and features are
independent of selecting the task. Discharge observations come from the
experiment's existing discharge series (CAMELS-CH in the app), not from
HYD-RESPONSES `Q` or its supplied event-number files.

### Definitions and reference data

The drought controls are saved under `extremes`:

| Field | Values / default | Meaning |
| --- | --- | --- |
| `mode` | `drought` | Activate this task only |
| `drought_definition` | `seasonal_quantile` (default), `q347` | Seasonal threshold or fixed training estimate |
| `drought_smoothing` | `trailing` (default), `centered` | Seven-day discharge averaging alignment |
| `drought_quantile` | 0.15 | Seasonal low-flow quantile |
| `drought_calendar_window` | 31 | Odd calendar window, 1–365 days |
| `drought_minimum_days` | 1 | Minimum duration for extracted events |

Seasonal thresholds are fitted separately for each basin using **only the
selected training period**. Seven-day means whose full window is inside that
period are pooled around each target calendar day and the selected quantile
is calculated using linear interpolation. The calendar window wraps at New
Year; a common leap-year calendar aligns month/day across years. Missing
reference coverage is reported as an excluded day. Entirely unsupported
training/test calendar selections fail before training.

The fixed Q347 option estimates the 347/365 exceedance flow from the raw
training daily discharge distribution, using lower-tail quantile 18/365.
This is a pooled training-period estimate, not an official FOEN statistic or
a reconstruction of its statutory reference period. The seasonal quantile
and calendar-window controls do not affect Q347.

These definitions follow the [HYD-RESPONSES methodological framework](https://doi.org/10.5194/essd-18-4113-2026).
The task deliberately refits references to the user's training period instead
of importing the dataset's retrospective 1991–2020 climatology. Consequently,
this is not an exact reproduction of its supplied event catalogue. Predictors
continue to exclude supplied centred averages, event IDs and retrospective
indices; enabling drought verification does not introduce them as inputs.

### Trailing and centred smoothing

Trailing M7 on day `t` averages `t-6` through `t`. Its predicted version combines
observed discharge on/before the issue date with predictions after issue.
This is a causal adaptation of the dataset's centred M7Q.

Centred M7 averages `t-3` through `t+3`, matching the dataset's averaging
alignment. It requires rolling mode with a horizon of at least four days.
Future values must be predictions **from that same issue**. The final three
leads of every block cannot be scored; incomplete final blocks may contribute
no eligible days. Neither later observations nor forecasts from the next block
are substituted to fill these windows. Observations after issue are used only
for verification of the observed target.

Rolling drought evaluation inherits the existing perfect-weather hindcast:
future weather comes from historical data; selected water-state features
remain held at issue-day values. This task adds neither weather forecasts nor
an operational availability model for retrospective predictor products.

### Verification and boundaries

All model approaches use the same drought verifier, identified by
`drought_one_to_one_overlap_v1`. Existing statistical and hydrological-event
tasks retain their original scoring. Drought day classification is simply a
strict crossing below the threshold; `drought_minimum_days` filters extracted
events, not individual day-classification scores.

Day metrics include hits, misses, false alarms, precision, recall, CSI,
false-alarm ratio, M7 MAE and observed-low-flow M7 bias/MAE. They are reported
per issue and by lead within each basin/seed. Requested, excluded and evaluated
counts remain explicit. Undefined rates (for example precision with no
predicted positives) remain undefined, not perfect or zero scores.

Events are extracted separately within each issue trajectory. One-to-one,
chronologically ordered matching maximizes total overlapping days and then
the number of matched pairs. It prevents one long prediction from receiving
credit for several observed droughts. The earlier any-overlap precision and
recall are retained as supplementary `overlap_event_*` columns **only in this
new task**, alongside the stricter `event_precision` and `event_recall`.

Missing/invalid days split events. Events touching an evaluation boundary are
marked left/right censored; a known preceding non-drought state can establish
onset at the first scored day. Onset errors require identifiable starts;
recovery errors require identifiable ends; duration and deficit errors require
complete matched events. Each error reports its contributing pair count.
Unmatched events still affect precision/recall. In a short horizon, long droughts
can remain censored, so complete-event timing may be undefined. One-day
forecasts primarily assess drought state, not full event duration.

Deficits are sums of threshold minus M7 during qualifying spells, in
basin-equivalent mm. These are **window deficits**, not HYD-RESPONSES cumulative
streamflow deficit (CQD) phases extending until compensation. Events repeated
across issue dates are issue-specific forecast segments, not independent
physical droughts or stitched whole-event reconstructions.

### Results and reruns

The ordinary discharge predictions and metrics remain available. Drought
results appear in the Extreme-event results table. The metrics JSON and
saved-run comparisons additionally show finite lead-score means across
available basin/seed rows and each score's contributing count. These are
unweighted descriptive macro means, not significance tests or confidence
intervals. Compare definitions, dates and coverage before interpreting deltas.
Drought definition/smoothing controls are also available in comparison filters.

The run archive adds:

- `drought_metrics.csv`: per-issue event/day scores and per-lead day scores.
- `drought_days.csv`: every requested day, issue, lead, smoothed values,
  threshold, eligibility and exclusion reason.
- `drought_events.csv`: observed/predicted event intervals, window deficits,
  censor flags and matched counterpart indices within that issue and kind.

Saved runs, the CLI, the HTTP API and both UIs preserve all drought settings
and audit tables. HTTP artifacts are available as `drought_metrics`,
`drought_days` and `drought_events`. Old configurations and saved manifests
without drought fields retain their earlier defaults and behavior. No
experiments are run merely by selecting or configuring the task.
