# BAFU forecast benchmarks

The API and custom browser workspace support `bafu_only` and `bafu_combined`
under the versioned `bafu_forecast_v1` task. Merely configuring a task, inspecting
statistics, or requesting preflight never starts training or submits a job.

## Local setup and browser controls

Register these server-owned paths under a dataset in `api-server.json`:

```json
{
  "data_root": "../datasets/camels_ch",
  "bafu_root": "../datasets/bafu_water",
  "bafu_statistics_zip": "../datasets/hydrologie-hochwasserstatistik_2056.shp.zip"
}
```

The workspace's local settings already register the downloaded files. Restart
the API/Studio and worker to load changed settings and code. No processes are
restarted by this implementation. The existing CAMELS-CH catalog supplies
station identity and positive catchment area for conversion from m³/s to
basin-equivalent mm/day. BAFU-only supports this catalog's station intersection;
it does not yet discover arbitrary BAFU stations outside CAMELS-CH.

In the custom workspace, open **BAFU benchmark**, select a mode, choose target
stations, training scope and dates, and click **Check benchmark coverage**.
Choose **Event task → Drought events** or **High-flow events**. Existing setups
retain high-flow evaluation unless drought is explicitly selected. Selecting
BAFU-only disables physics and clears external-feature selections. API clients
must set these options explicitly.

For drought, the workspace selects daily mean discharge and training-fitted
thresholds. Choose **Forecasting → Rolling** and a horizon long enough to study
the event of interest. One-day forecasts support next-day drought warnings;
duration, recovery and deficit of a complete event require a trajectory covering
that event. Forecast boundaries are marked as censored, not complete events.

Preflight displays split counts, thresholds, exclusions, and an audit download.
It shares observation loading and window eligibility with execution, but counts
windows without materializing the training sequence tensors or fitting models.
For drought it also checks seven-day observation windows, training calendar
references and the selected issue/horizon policy using the production verifier.
The audit reports eligible and excluded drought days by station. Stations with
no evaluable drought days fail before queueing. **Use stations with complete
coverage** applies the audit's usable targets and training pool without disabling
strict cohort checks.
**Pin this evaluation for future runs** stores its SHA-256 fingerprint. Export
that setup, change architecture or data mode, and retain the fingerprint to
require the same evaluation target observations, dates, thresholds, day alignment,
and forecast policy. Changing coverage then fails before model training.

Example requests are in [examples/benchmarks](examples/benchmarks). Import them
through the UI or post them to preflight. Their dates and station are examples,
not a published study's cohort. No example has been submitted or executed.

The main API browser supports the new controls. The optional legacy native
Gradio form retains its earlier CAMELS-CH controls; CLI/remote clients can use
the shared JSON configuration.

## Data settings and task definition

- `bafu_only`: BAFU Q history, calendar features, and the selected BAFU Q target.
  CAMELS meteorology, static descriptors, HYD-RESPONSES, chemistry and EStreams
  are excluded from model inputs. Catchment area remains conversion metadata.
  Weather-dependent physics requires `physics_weight: 0` and optimization `none`.
- `bafu_combined`: the same BAFU target and Q history, joined to the existing
  CAMELS simulation-based forcings and selected optional datasets/static inputs.
  This is feature augmentation at matched stations, not concatenating duplicate
  BAFU/CAMELS discharge observations or pooling additional training datasets.
- `target`: `daily_mean` or `daily_max`, selected from their corresponding BAFU
  files. These remain separate tasks. Daily maximum flow is represented as an
  equivalent flow rate in mm/day, not an integrated daily runoff volume; disable
  the daily water-balance physics term for this target in both data modes.

Only Q observations with explicitly accepted release states (default 2 and 3)
are selected. Provisional state 1, nonfinite values and negative flows are
excluded. Unexpected units and conflicting duplicates fail. Missing dates break
windows and events. UTC timestamps are converted to `Europe/Zurich` before
assigning calendar days by default; `UTC` is an explicit alternative contract.
Monthly source partitions are read once for the whole selected station cohort.
This version does not add BAFU W/WT as optional predictors.

One-day inputs stop on the preceding day. Rolling mode recursively substitutes
predicted discharge after issue time. Combined rolling runs retain the existing
perfect-weather historical-meteorology assumption and issue-held water states;
they are hindcasts, not operational weather forecasts. BAFU-only rolling uses
hydrological history and known calendar features, with no weather driver.
Static and retrospective predictors do not have a new operational release model.

Keep the same cohort, dates, sequence length, thresholds and forecast setup for
architecture comparisons. Keep input budgets and tuning budgets comparable.
Adding predictors, adding training stations, and extending the training period
are different experiments. Registry filters expose benchmark mode, target,
threshold source, and evaluation fingerprint.

## Thresholds and the statistics ZIP

### Drought

Set `extremes.mode: "drought"`, `benchmark.target: "daily_mean"` and
`benchmark.threshold_source: "training_quantile"`. Daily maxima and official
flood return levels are rejected for this task. Other `extremes.definition`
settings belong to high-flow scoring and do not change drought thresholds.

The shared drought verifier applies either a seasonal low-flow quantile
(`drought_definition: "seasonal_quantile"`, default probability 0.15) or a
fixed Q347 estimate (`"q347"`). Both references use training observations only.
It compares observed and predicted seven-day mean discharge against the fitted
threshold. Trailing smoothing uses the current and six preceding days. Centred
smoothing requires a rolling horizon of at least four days and cannot score the
last three leads in each issue. Post-issue observed discharge never enters the
predicted smoothing window.

Results show a drought threshold chart and day detection scores for the selected
station and lead. The event table and exports include onset, recovery, duration
and deficit errors where boundaries permit evaluation. Downloads are
`drought_metrics.csv`, `drought_days.csv`, `drought_events.csv`, and
`benchmark_manifest.json`; drought runs do not produce a flood-score file.
The manifest stores the low-flow thresholds and drought settings. Its evaluation
fingerprint includes smoothing context and event policy, so drought and high-flow
runs cannot be treated as the same benchmark. See
[bafu_drought.json](examples/benchmarks/bafu_drought.json) for an example request.

These additions apply when a BAFU source and drought task are selected. The
CAMELS-CH rainfall–runoff input, training, physics and event defaults are unchanged.

### High flow

Default high-flow thresholds are training-period station quantiles, Q95 by
default. Threshold fitting is independent of validation/test values and models.

`HQStat.shp.zip` is a station/report index, not a numeric threshold or event
catalogue. **Inspect flood statistics** displays its stations and linked reports.
Its EDV identifiers must be reconciled with BAFU observation station numbers;
the loader does not guess an identifier crosswalk or PDF table values.

To use checked official thresholds, register `bafu_return_levels` pointing to a
UTF-8 CSV with these columns:

```csv
station_id,return_period_years,discharge_m3_s,source_url,reference_period
```

Use the observation `station_no` for `station_id`; supported return periods are
2, 10, 30, 100 and 300 years. Each row requires a positive value, source URL and
reference period. Duplicate station/period rows and non-increasing return levels
fail validation. Select `threshold_source: official_return_level` and the desired
`return_period_years`. Every evaluation station must have the selected threshold.
No verified numeric threshold CSV has been fabricated from the local ZIP.

Official peak-flow return levels applied to daily means/maxima are explicitly
reported as retrospective threshold-based classifications. They do not turn
daily mean predictions into instantaneous peak forecasts. The source reference
period may include test years; this is not an operational reconstruction of
thresholds known at a historical issue date.

## Baselines, verification and artifacts

Select `model.complex_expert: lstm` and `strategy.approach: no_routing` for the
reference deterministic LSTM. It uses the same windows/features/evaluator as
our architectures. `lstm_hidden_size` and `lstm_layers` control its dimensions.
It is not a replication of the probabilistic published LSTMs. For a conventional
unregularized reference comparison, keep physics weight zero across models.

`strategy.approach: persistence` and `seasonal_climatology` bypass model creation
and training. Persistence holds the last observed discharge throughout a rolling
block. Climatology uses monthly training means; unseen months fall back to the
training mean. Targets excluded from model training use pooled training-basin
climatology, rather than a target-specific fitted baseline. Repeated seeds for
these deterministic baselines are identical, not independent uncertainty samples.

Benchmark event verification is `high_flow_one_to_one_v1`: maximum-overlap
one-to-one event matching, explicit boundary censoring, and peak errors only
for complete matched events. One long predicted event cannot match several
observed events. Rolling event matching occurs separately within each issue
trajectory; one-day predictions form a continuous verification series. Day
precision/recall, CSI, false-alarm ratio, average precision and extreme RMSE are
also reported by lead. Non-benchmark event tasks retain their existing verifier.
Undefined event rates and unavailable physics errors remain undefined, not zero.

Runs include `benchmark_manifest.json` and `flood_metrics.csv` in the full archive,
with direct downloads in the UI/API. The manifest records source checksums,
release/day/unit policies, conversion areas, thresholds, actual evaluation and
training fingerprints, selected features, effective cohorts, and split counts.
Training identity hashes source records and the window policy without expanding
overlapping windows into JSON. A common evaluation fingerprint permits comparing
BAFU-only with combined predictors; different training fingerprints are expected.

Comparison responses identify incompatible evaluation contracts. Compatible saved
prediction sets additionally receive paired RMSE differences and a deterministic
95% calendar-year block-bootstrap interval. Years are sampled jointly across
stations/seeds to preserve shared regional events. Fewer than two years yields
no interval. These are pooled descriptive intervals; overlapping watersheds,
small samples, unequal seeds, and tuning budgets still require interpretation.
No calibrated probabilistic scores are claimed for deterministic model outputs.

## API operations

- `GET /api/v1/benchmarks`: runnable task and published reference metadata.
- `GET /api/v1/datasets/{id}/bafu/statistics`: ZIP index and registered numeric imports.
- `POST /api/v1/experiments/preflight`: full experiment request; read-only audit,
  `training_started: false`, cohort status and manifest. No job is queued.
- `POST /api/v1/experiments/validate`: existing configuration/layout validation;
  it does not replace date-coverage preflight.
- `POST /api/v1/experiments`: existing explicit job submission; unchanged worker path.
- `GET /api/v1/runs/{id}/benchmark`: resolved benchmark manifest.
- `GET /api/v1/runs/{id}/artifacts/benchmark_manifest` and `/flood_metrics`: downloads.
- `POST /api/v1/comparisons`: contract compatibility and optional paired RMSE analysis.

All routes use existing API authentication. Clients cannot supply filesystem
paths; registration remains server-owned. BAFU benchmark replay and paved-site
scenarios are rejected instead of accidentally substituting CAMELS target data.

## Published comparison boundary

The reference profile links [Baste et al. (2026)](https://hess.copernicus.org/articles/30/6339/2026/)
and its [versioned code/results archive](https://zenodo.org/records/22959931)
(v3, `sb2025.zip`, MD5 `1278a51dbb48491ad94d53f5eb9cadce`). The profile records
the paper's reported splits, cohort size and sequence length, but is not runnable
as an exact-reproduction preset. Its exact station list, 22 static attributes,
input/target alignment, target-day meteorology, data revision, warm-up, HQ tables,
probabilistic heads, and warning-policy selection have not been reconciled.
The reported training start also precedes stated CAMELS-CH coverage.

Current supported comparisons rerun a reference architecture and our models on
the identical locally defined BAFU task. They do not establish superiority over
the paper's published scores. Exact literature-score reproduction remains a
separate, explicitly unverified integration step.
