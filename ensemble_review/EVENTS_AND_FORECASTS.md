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

The Task tab offers three analysis choices. Daily discharge forecasting reports
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
values** for days after issue. It is therefore a *perfect-weather hindcast* of
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
