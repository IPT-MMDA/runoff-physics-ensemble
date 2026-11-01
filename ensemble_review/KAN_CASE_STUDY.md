# KAN transfer to low-data catchments

This independent study asks whether a compact symbolic approximation of a
source-trained neural runoff process transfers to a new catchment with fewer
local observations. It does not identify a universal physical law from discharge.

## Run it

In Studio, open **KAN case study**. In the native Gradio interface, open
**KAN low-data case study**. Choose separate source, source-validation and target
cohorts, set the chronological periods and observation budgets, and check coverage
before running. Studio uses the existing training worker and shared cancellation
panel. Finished Studio studies can be reopened from the study's saved-results
selector. Gradio supports local execution and the existing remote executor.

### Assign roles and local budgets yourself

Tick catchment checkboxes under **Source catchments**, **Source-validation
catchments** and **Target catchments**. Untick a checkbox to remove a catchment.
Choices persist when you switch role groups, filter the Studio lists, or refresh
coverage; no Ctrl/Cmd selection is required. You define what counts as high or low data for your study.
The platform does not categorize catchments or assign roles based on counts.
Each catchment must have one role, and all three cohorts must be nonempty.

Set the training dates and click **Scan catchment availability before choosing
cohorts** when you want supporting evidence. The scan is optional: it shows usable
available observations for each selected train, validation and test date range,
alongside gaps, first/last training dates, monthly training coverage and weather
completeness. Training counts alone determine inventory order; held-out
availability is displayed for scoring readiness and never enters fitting.
Scanning preserves your selections. The table can be filtered by your assigned
role, and cohort lists show train/validation/test counts without high/low labels.

The table shows **available** and **planned used** counts separately for each
split. Sources use training labels only. Source-validation catchments use
validation labels for checkpoint and equation selection. Targets use their local
training budget and available test labels for scoring; target validation labels
are unused. Multiple budget values mean separate runs, rather than summed labels.
The Readiness column flags insufficient required splits before launching a job.
For example, Verbania 6010 has 1,096 training labels but zero validation and test
labels under the default 2010–2016 / 2017–2018 / 2019–2020 dates; it cannot be
scored as a target with those splits.

Run submission shows a submitting state, then the queued job identifier. Progress,
cancellation and worker failures are reflected in the case-study tab as well as
the shared job panel. Invalid fields in collapsed settings are opened and their
validation errors shown beside Run instead of silently blocking form submission.

**Default target observation budgets** apply to every target unless overridden.
After selecting targets, set their budgets individually. For example, expose
30 labels for one target and 90 for another, or run several budgets (`30, 60, 90`)
for a target. Leave a target's budget field blank to inherit the defaults.
Studio provides a field per selected target; Gradio provides an editable target
budget table. Configurations use `target_observation_budgets`, for example:

```json
{
  "observation_budgets": [30, 90, 180],
  "target_observation_budgets": {"2488": [30, 60], "4009": [90]}
}
```

All budgets must be increasing positive integer counts of at least three labels.
Preflight checks the chosen budget for each target, rather than requiring the
largest global default when that target has an override. A catchment with a long
record can therefore be your target with only 30 labels exposed during fitting.

Source fitting requires at least three observations by default. The advanced
**Required source observations** setting lets you impose a stricter minimum;
it is a fitting requirement, not a definition of high data. Full preflight also
checks independent validation/test labels and continuous weather coverage.

The previous local 2010–2016 inventory is retained at
`artifacts/kan_case_study/availability_2010_2016.csv` and `.json`. It found that
catchments 4009, 4008 and 2488 each have 2,557 usable observations. Its old threshold
categories are historical; current UI selection uses your assigned roles.

The same pipeline can run without either UI:

```sh
PYTHONPATH=ensemble_review python -m complexity_ensemble.kan_case_study \
  --config ensemble_review/examples/kan_case_study.json \
  --data-root ../datasets/camels_ch \
  --output artifacts/kan_case_study/example
```

The example is a small execution example, not a validated scientific cohort.
For a study, use multiple source and source-validation catchments, several seeds,
multi-year evaluation periods and representative hydrological regimes. The case study has its own **Compute device** selector in both UIs, independent
of the main regional experiment. Choices match the main page: `auto`, `cpu`,
`cuda` and `mps`. Auto chooses CUDA, then Apple Metal, then CPU. Install `pykan`
with the package's `kan` extra.

Teacher training, target calibration and continuous simulation use the selected
device. For CUDA, PyKAN's numerical spline fitting also uses that CUDA device.
For MPS, PyKAN fitting runs on CPU because its spline least-squares operations
are not available on Metal. Pruning, symbolic extraction and scoring run on CPU
for all choices. Extracted expressions execute on the selected device during
calibration and simulation. Data tensors, masks and parameters are moved together;
metrics and exported values return to CPU.

The request and saved configuration carry `device` (default `auto`). Reports and
audit exports record both the requested/resolved device and the hardware used by
each phase. Explicit unavailable devices are rejected before queueing, rather
than silently changing the training device. An example configuration entry is
`"device": "mps"` for an Apple GPU.

## Optional kan_complex model

`kan_complex` is an additional locally calibrated symbolic variant, unchecked by
default. It discovers an independent equation per seed with an MLP process
teacher trained for `max(50, teacher_epochs)` epochs and
`max(24, teacher_width)` hidden units. It searches
`max(6, kan_candidates)` initializations and uses the symbolic library
`x`, `x^2`, `tanh`, `x^3`, `abs`. The added functions are supported by the
differentiable expression executor without introducing divisions or logarithms.

KAN hidden width, spline grid, fitting steps, sparsity, sample limits and formula
complexity limits retain the configured values. Source-validation checkpoint and
candidate selection, export checks and recursive simulation checks still apply.
The selected equation is frozen; each target budget fits the same three local
coefficients as `kan_tuned`, using the same adaptation epochs and observations.

Selecting it alongside `kan_pure`/`kan_tuned` runs two independent discovery
profiles. Existing variants retain their original training and equation.
Saved equations identify their models and record effective teacher settings,
candidate count and symbolic library; teacher traces are stored separately.
Existing saved runs without these fields remain viewable.

This increases search and teacher training cost; improved accuracy is not
guaranteed. Keeping KAN size small reduces capacity but does not establish
robustness. Assess source-validation discharge performance, candidate conversion
errors, recursive stability and held-out target performance across seeds.

## Storage and learned flux

All water quantities are daily catchment-equivalent mm; temperature is Celsius.
Snowfall is precipitation below 0 °C. A bounded degree-day melt process transfers
snow into liquid input. Actual evaporation is bounded by PET and available water.
Water exceeding capacity is routed directly to overflow. Remaining water releases
a bounded fraction, predicted by the teacher or its symbolic approximation:

```
x = [previous_water / C, liquid_input / C, PET / C]
fraction = sigmoid(F(x) + local_offset)
Q = overflow + retained_water * fraction
water_next = retained_water * (1 - fraction)
snow_next = snow_previous + snowfall - melt
```

The water and snow stores start at zero and are spun up with weather alone.
Measured discharge is never used to initialize or update states. The implementation
records each day's mass-balance residual. Conservation comes from the explicit
model equations, not from an assumption that the learned symbolic relationship
must be physically correct.

The shared teacher is a small MLP that predicts a release-fraction logit.
Each training source also has three calibrated parameters. Every source basin
contributes equal normalized loss, with its own training-only discharge scale.
Gradients are truncated every 90 days while simulated storage remains continuous.
Teacher checkpoints are selected using discharge errors on independent
source-validation basins, whose parameters are initialized from source medians.

## Source-only distillation

PyKAN receives teacher logits at source-training simulated states. Each source
contributes the same number of evenly spaced states, up to the configured limit.
Normalization is fitted on these states alone and then frozen. The three inputs
are dimensionless storage, liquid input and PET, each scaled by water capacity.

Candidate splines are fitted, pruned and converted with a small symbolic library
(`x`, `x²`, `tanh`). Symbolic expressions exceeding the operation budget or
disagreeing with the converted KAN are rejected. The exported SymPy expression
is executed directly through a restricted torch evaluator during adaptation and
simulation. It is not merely a label attached to a numerical KAN.

Candidates are selected using independent source-validation catchments: normalized
recursive discharge error against the teacher plus release-fraction MSE against
the teacher, with an operation penalty of `1e-6` per symbolic operation. Both
fraction and closed-loop discharge fidelity are exported, along with conversion
error and rejected-candidate reasons. No target catchment selects the teacher,
the equation, the normalization, or the initial parameter prior.

## Local calibration and comparisons

For each target and seed, its selected budgets use the first N available daily discharge labels in the
training period. These budgets are nested. No other target discharge is read by
the fitting loss or normalization. The target validation labels are unused.
Full weather records still drive all simulated states.

Each adapted model fits exactly three bounded coefficients:

| Coefficient | Bounds | Meaning |
|---|---|---|
| Water capacity | 10–1000 mm | Bucket storage capacity and input scaling |
| Melt factor | 0.1–10 mm/day/°C | Degree-day snowmelt |
| Runoff offset | −4 to 4 | Adjustment to the release-fraction logit |

Four methods share the same state equations and weather:

| Method | Shared relationship | Local fit |
|---|---|---|
| Symbolic transfer | Frozen KAN-derived expression | Three coefficients |
| Neural transfer | Frozen source teacher | Three coefficients |
| Local reservoir | Constant base release logit | Three coefficients, generic initialization |
| Symbolic unadapted | Frozen expression | None; source-median coefficients |

The symbolic and neural adaptations start from the same source-median prior.
Expression coefficients stay frozen; local parameters change the capacity-scaled
inputs and release offset. The unadapted result is repeated across budgets to
provide a common reference line.

All methods are evaluated on the same untouched target test period. The UI shows
RMSE learning curves by catchment and seed. Tables include NSE, bias, low/high-flow
RMSE, local parameter values, maximum daily conservation error and the fraction
of test states outside the source-training range. Low/high-flow thresholds are
Q10/Q90 estimated only from the allowed local calibration labels; comparisons
across budgets therefore use different extreme thresholds. Overall RMSE and NSE
use the same test observations across budgets.

## Artifacts and HTTP endpoints

The ZIP contains the complete report, configuration, executed expressions,
normalization, teacher weights, source and local coefficients, candidate diagnostics,
audit, metric CSV and daily test discharge/storage/evaporation/balance CSV.
It contains enough information to reconstruct the selected teacher and symbolic
model; preprocessing and initial raw coefficients are included.

Authenticated API routes use registered dataset IDs, not client filesystem paths:

- `GET /api/v1/case-studies/kan/defaults`: defaults and request schema.
- `POST /api/v1/case-studies/kan/coverage`: all-catchment date-aware availability;
  accepts empty cohorts and reports training availability without assigning roles.
- `POST /api/v1/case-studies/kan/validate`: cohorts and contiguous coverage check.
- `POST /api/v1/case-studies/kan`: queue a study using `{dataset_id, config}`.
- `GET /api/v1/case-studies/kan`: latest 100 study jobs, including saved results.
- Existing `/api/v1/jobs/{id}` and cancellation routes track execution.
- `GET /api/v1/jobs/{id}/result`: complete JSON report.
- `GET /api/v1/case-studies/kan/jobs/{id}/archive`: completed study ZIP.

## Interpretation and limits

The study supports both masked-data experiments and genuinely missing discharge
labels in CAMELS-CH files. Its dedicated loader preserves weather dates when
discharge is missing, without changing the regional loader's defaults. Calibration
uses the first N finite, nonnegative training-period labels; these may be scattered.
Source-validation and target test scoring require at least three valid labels.
Metrics report the actual scored day count. Predictions and fluxes include all
test dates, with null observations where unavailable. Weather must still be
complete and contiguous, including spin-up. Missing forcing days are rejected,
rather than silently compressed or interpolated.

Observed same-day meteorology drives continuous simulation: this is a weather-driven
hindcast, not an operational weather forecast. Water capacity, latent storage and
learned flux are not uniquely identifiable from discharge alone. A short calibration
period may not identify snowmelt or seasonal response. Three coefficients reduce
the adaptation burden but do not remove identifiability problems or negative
transfer between different hydrological regimes.

The teacher is process-oriented and deliberately separate from the existing
regional ensemble. An arbitrary sequence predictor does not expose identifiable
runoff fluxes that can simply be extracted. This study therefore tests a neural
process teacher with explicit, conserved states. Its symbolic approximation may
retain teacher errors, and its simple snow/evaporation modules omit many processes.
The source-range diagnostic flags extrapolation, but does not guarantee reliability.

Numerical conservation, symbolic execution and absence of data leakage are software
properties. A scientific claim of improved transfer still requires experiments
with multiple cohorts, calibration seasons, seeds and independent target basins.

## Retired regional mode

The former regional KAN residual-correction implementation and UI controls were
removed. New regional configurations accept only `physics.optimization_mode =
"none"`; an enabled legacy mode raises an explicit migration error. Old result
manifests and their recorded symbolic formulas remain readable. The Gradio adapter
retains constant internal positional slots so unrelated settings do not shift.
