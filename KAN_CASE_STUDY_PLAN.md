# KAN low-data transfer case study

## Agreed objective

Distil a source-trained neural runoff relationship into a symbolic approximation,
freeze its structure, and calibrate only a few physical parameters on a separate
low-data target catchment. The expression approximates the teacher; it is not a
claim to have discovered a universal physical law.

## Execution plan

1. [x] Replace the regional KAN correction workflow with a clear migration error for
   old enabled configurations. Keep historical result files readable.
2. [x] Implement an independent CPU pipeline with explicit snow and water storage,
   bounded runoff and evaporation, and auditable daily mass balance. A shared
   neural runoff-fraction teacher is trained with equal basin contributions.
3. [x] Fit PyKAN to source teacher fluxes, convert to an executable symbolic
   expression, and select by source validation fidelity and expression size.
   Freeze expression structure, preprocessing and teacher before local adaptation.
4. [x] Calibrate three parameters per target (water capacity, melt factor, runoff
   logit offset) using nested observation budgets. Compare symbolic transfer,
   frozen neural transfer, unadapted symbolic transfer and a local reservoir.
   No target observations enter teacher training, symbolic selection or spin-up.
5. [x] Add independent case-study tabs to Studio and Gradio, an authenticated queued
   API workflow, cancellation, saved reports and downloadable reproducibility
   artifacts. Remove old KAN controls from the main experiment interface.
6. [x] Verify conservation, source/target separation, observation budgets, symbolic
   execution, target-test independence, API/worker behavior and UI construction;
   document the protocol and practical limits.

## Protocol decisions

- Source, source-validation and target basin cohorts are explicit and disjoint.
- Date ranges are chronological training, validation and test periods. Targets
  use the first N observations in the training period for calibration only.
- Simulated states start from zero and use a forcing-only spin-up. Discharge is
  never assimilated. Evaluation is continuous simulation with observed weather
  (a weather-driven hindcast, not an operational weather forecast).
- Complete contiguous daily forcing records are required. A dedicated opt-in
  CAMELS loader preserves missing discharge labels. Budgets use the first N
  available training observations; masked-data experiments are also supported.
- Runoff law inputs are dimensionless water storage, liquid input and PET,
  scaled by capacity. Snowmelt is an explicit degree-day process. Capacity is
  a calibrated model parameter; storage is simulated, not an observed state.
- Target validation labels are unused. Source-validation basins select teacher
  checkpoints and expressions. Target test labels are scoring-only.
- CPU execution avoids PyKAN Metal incompatibilities. Parameters and equation
  coefficients are exported alongside predictions, fluxes and metrics.

## Completion record

Implemented all six steps. See `ensemble_review/KAN_CASE_STUDY.md` for the full
protocol, endpoint contracts, example invocation and limitations.

Validation completed:

- 12 new Python tests: explicit conservation, symbolic gradients, known analytical
  release-fraction approximation on unseen states, real PyKAN integration,
  independence from hidden target labels, genuine missing-discharge support,
  migration guidance, archive roundtrip, authentication and queued API/worker flow.
- 83 existing Python tests passed across API, Studio, complexity routing,
  hydrology, job queue, CAMELS state integration, HYD-RESPONSES and drought.
- 23 browser-logic tests passed, including separate cohort requests, dataset
  loading races and visible study errors. Python compilation, JavaScript syntax
  and whitespace checks passed. Native Gradio construction is covered by the
  existing API and routing tests; an interactive browser session was not run.
- The complete pytest suite could not run: pytest is absent, and installing a
  temporary copy failed because package-index DNS/network access is unavailable.
  The above Python tests ran through unittest without altering the environment.
- A CAMELS-CH execution smoke used source 4009, source-validation 4008 and target
  2488, with 30/90-label budgets. It generated eight metric rows and 248 daily
  predictions. Artifacts are in `artifacts/kan_case_study/smoke/`. The maximum
  absolute daily water-balance residual was approximately 2.7e-5 mm.

The deliberately short smoke used two teacher epochs, eight KAN steps and three
adaptation epochs. Symbolic transfer was worse than neural transfer in this run;
this validates execution, not the scientific hypothesis. The known-function test
achieved release-fraction validation MSE about 5.0e-4, but did not establish exact
recovery of the original algebraic form. Longer, multi-cohort experiments remain
research work rather than an unfulfilled implementation step.

## Follow-up: selecting cohorts by data availability

Added an authenticated, source-cutoff availability inventory and matching Studio
and Gradio controls. Before choosing cohorts, users can scan all eligible
catchments for training counts, gaps, monthly coverage, weather completeness and
budget readiness. Classification uses only training dates and explicit source/budget
thresholds. Changing the dates invalidates stale UI results. Five additional
Python coverage/API/native-UI tests and two browser race/invalidation tests cover
the follow-up. The original case-study and API tests also passed after this change.

The local 2010–2016 inventory contains 281 high-data and 50 unavailable catchments
at the default thresholds, with no naturally low-data records in that window.
The illustrative target 2488 has 2,557 observations and is made low-data by the
calibration budget. CSV and JSON inventories were saved under
`artifacts/kan_case_study/availability_2010_2016.*`.


## Follow-up: manual cohort roles and individual target budgets

The UI now uses neutral Source, Source-validation and Target selectors. Roles
are manually assigned, scans preserve assignments, and availability tables filter
by the user's roles. Automatic high/low/intermediate categorization was removed
from the coverage service and both front ends. Counts, gaps and seasonal coverage
remain advisory.

Each target can now have its own observation budgets through UI fields/table and
`target_observation_budgets` in saved/API configurations. Preflight and calibration
use each target's effective budget. The default source minimum is three fitting
observations; a stricter user minimum remains an advanced setting. No default
365-observation threshold determines cohort membership.


## Fix: persistent checkbox selection

Replaced Studio's native multi-select lists with labelled checkbox groups backed
by independent selection sets. Each role has its own search filter. Filtering,
changing focus, checking another role and refreshing coverage preserve selections.
Gradio now uses CheckboxGroup components for all three cohort selectors.
Regression tests exercise source selection followed by validation/target selection,
explicit unticking, hidden selections during filtering and coverage refresh.


## Fix: split-aware coverage and Run submission

Availability now counts valid discharge/weather pairs separately in the selected
training, validation and test periods. Both UIs display available and planned-used
counts according to assigned roles and target budgets. Readiness flags empty
validation/test splits; error messages include the actual count and date range.
Verified 6010 against local files: train 1,096, validation 0, test 0 under defaults.

Fixed the silent Run failure: numeric integer fields had a fractional minimum
with step=1, so browser constraint validation rejected valid whole-number defaults
before dispatching submit. Integer constraints now align, and explicit validation
opens hidden invalid controls and shows errors in the study tab. Run has a pending
state, shows the queued job ID, scrolls to progress, and receives subsequent job
status/errors inline. Coverage checking remains read-only.

Validation: 28 Python API/coverage/Studio tests and 14 JavaScript case-study tests
passed, including queued submission, surfaced errors, integer-step compatibility,
role-specific usage and empty target-test readiness. Syntax and diff checks passed.


## Hardware selection

Added an independent Compute device control to Studio and Gradio using the main
pipeline's `auto`, `cpu`, `cuda`, `mps` choices and resolver. Teacher training,
local calibration and simulation move data, masks and parameters to that device.
CUDA supports numerical KAN fitting; MPS uses an explicit CPU fitting fallback.
Pruning, symbolic extraction, coefficient summaries, metrics and serialization
use CPU. Portable symbolic expressions execute on the selected training device.
Reports and audit exports record requested/resolved and phase-specific devices.
Unavailable explicit hardware is rejected before jobs are queued.

Validation: 43 Python tests ran (41 passed; two accelerator execution tests skipped
because neither CUDA nor MPS is available in this execution environment), and all
15 case-study JavaScript tests passed. CPU runs include real PyKAN and symbolic
extraction; phase routing and API rejection are tested independently. Actual CUDA
and MPS execution remain conditional tests to run on hardware-enabled hosts.
