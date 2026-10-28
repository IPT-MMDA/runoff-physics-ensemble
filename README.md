# Regional rainfall–runoff ensemble

## License and permission

Copyright (c) 2026 Alexander Y. All rights reserved. This project uses a
[custom proprietary license](LICENSE). **Any use requires prior written
notice and express written permission**, including personal, academic,
research, and other noncommercial use. **Commercial use requires a separate
written agreement before use begins.** Copying, modification, and
redistribution also require express written authorization.

Submit notices and permission requests through the repository owner's
published GitHub contact methods or the
[repository issue tracker](https://github.com/IPT-MMDA/runoff-physics-ensemble/issues).
Notice alone is not permission; wait for express written authorization.
Earlier versions distributed under MIT retain their existing permissions.
Third-party dependencies and datasets retain their own licenses.

## Overview

This project predicts **daily discharge at a basin gauge** (mm/day) from the
preceding days of meteorological forcing and discharge. The Gradio app runs
regional experiments across CAMELS-CH basins, compares model approaches, and
shows observed versus predicted flows, event scores, and a rainfall–runoff
replay.

The [custom research workspace](#custom-research-workspace) provides the main
browser interface, with training handled by a separate API worker and the native
Gradio interface available alongside it.

The model uses CAMELS-CH simulation-based precipitation, temperature, and
potential evapotranspiration (PET), previous observed discharge, seasonal day
encoding, and available basin descriptors. Optional EStreams vegetation and
snow-cover data and CAMELS-CH-Chem catchment predictors can be added. The
ensemble can route between a complex Mamba or MLP expert and a simpler RBF or
Fourier expert; a linear-reservoir relation provides a physics constraint.

In **Data and features**, **Optional CAMELS-CH water states** lets you select
snow water equivalent (mm) and simulated canopy interception storage (mm)
independently. Both are read from your existing CAMELS-CH files and are off by
default. Missing daily values use a zero placeholder with an availability
flag, without interpolation or carrying values forward. Only historical input
dates feed each forecast. Saved configurations, CLI reruns, and API requests
preserve the selection in `camels_state_features`, using the names
`snow_water_equivalent` and `canopy_interception_storage`.

**HYD-RESPONSES catchment water states** in the same tab adds independently
selectable ERA5-Land soil moisture at four depths (m³/m³), OSHD snowmelt,
and OSHD/SPASS snow water equivalent. Enable the dataset and choose features;
the app pre-fills a sibling `HYD_RESPONSES` directory when it exists. Use
**Validate HYD-RESPONSES path and basin coverage** to check matching basin IDs
and partial Swiss coverage. An additional checkbox enables physical catchment
descriptors (terrain, soils and hydrogeology). The loader reads only base
variables, units metadata and selected descriptors; all dataset files stay
in place. Missing dates and unmatched basins use availability flags. Swiss
snow features include the Swiss-area fraction for border catchments.

Launch with `--hyd-responses-root ../datasets/HYD_RESPONSES` or set
`HYD_RESPONSES_ROOT` to pre-enable the integration. Saved configurations use
`hyd_responses_root`, `hyd_responses_features`, and `hyd_responses_static`.
For rolling forecasts, CAMELS-CH and HYD-RESPONSES water states are held at
their issue-day values rather than reading later state observations.

The Task tab also offers **Hydrological drought (HYD-RESPONSES-style definitions)**.
Choose trailing or centred seven-day discharge means, seasonal low-flow
percentiles or a fixed training-period Q347 estimate, and minimum event duration.
Users retain their own basins, date splits and dataset selections. Thresholds
are fitted only to training observations. Centred smoothing requires rolling
mode with at least four forecast days; the last three leads of each issue are
excluded rather than filled with later observations. This task has separate
issue-aware drought verification and audit CSVs in the run archive; existing
event tasks keep their scoring. HYD-RESPONSES water-state predictors remain
optional and independent of the task. See [drought definitions and verification](ensemble_review/EVENTS_AND_FORECASTS.md#hydrological-drought-task).

The **Complexity routing method** setting in **Training hyperparameters** selects
Morse (the default), PH, Takens, or Lyapunov for soft/hard routing and distillation.
PH scores persistent H₁ loops in delay-embedded discharge. Takens scores the
embedding's effective covariance dimension (a geometric proxy). Lyapunov scores
the divergence of similar historical discharge trajectories. These alternatives
use a separate, past-only discharge history (365 days by default), with adjustable
embedding dimension and delay. Rolling forecasts replace future discharge in
both model and routing histories with predictions. PH requires the `tda` extra
(`ripser`), already included by the root requirements. Invalid estimates fall
back to the simple expert. Other training approaches ignore the routing choice,
except when a routed physics-distillation teacher is selected.

The replay's **input-to-prediction diagram** shows which dates feed each
forecast and which day's discharge is predicted. Its separate teaching bucket
is uncalibrated and does not show the trained model's internal water storage.
Rolling forecasts use historical weather after the issue date, so they are
perfect-weather hindcasts rather than operational weather forecasts.

## Install

Use Python **3.10 or newer**. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The root [requirements.txt](requirements.txt) installs the package in editable
mode, the app and API dependencies, GIF export (`Pillow`), optional TDA and KAN methods,
and `pytest`. The package dependency definitions live in
[ensemble_review/pyproject.toml](ensemble_review/pyproject.toml).

For a smaller installation with the app and GIF export, omit the optional
modeling and test dependencies:

```bash
python -m pip install -e './ensemble_review[animations]'
```

CPU execution is supported; a compatible PyTorch/CUDA installation is needed
only if you select a CUDA device. GIF export does not require FFmpeg.

### Apple Silicon / macOS

Use a native ARM64 Python installation (Python 3.10 or newer), rather than
Apple's older system Python or an interpreter running under Rosetta. Install
with the same pip commands above; no CUDA toolkit is needed on a Mac.

The app's **Device** setting defaults to `auto`, which selects CUDA if
available, otherwise Apple's Metal GPU (`mps`) if available, otherwise CPU.
Choose `mps` explicitly to require the Mac GPU, or `cpu` for CPU execution.
An unavailable explicitly selected GPU produces an error instead of silently
switching devices. Programmatic configurations still default to CPU; set
`hyperparameters.device` to `auto` or `mps` to enable Metal.

Check whether your PyTorch installation can access Metal:

```bash
python -c 'import torch; print(torch.backends.mps.is_available())'
```

Training uses float32 on Metal. Double-precision metrics, stacking, and OOD
analysis run on CPU when necessary; OOD scores return to the training device.
Seeded permutations and observation noise are generated on CPU for MPS.
Optional PyKAN discovery and correction evaluation also run on CPU, returning
correction predictions to the training device. The Mamba implementation uses
ordinary PyTorch and does not require CUDA extensions.

Performance depends on model size, sequence length, and batch size; compare
short CPU and MPS runs before launching a large experiment. The Mac GPU does
not provide CUDA compatibility. If your PyTorch version reports an unsupported
Metal operator, CPU remains available; `PYTORCH_ENABLE_MPS_FALLBACK=1` can also
enable CPU fallback for supported fallback cases when set before launching.
It does not enable double precision on Metal.

## Provide the data

The CAMELS-CH dataset is **not included** in this repository. Point the app at
your locally extracted dataset root. It must contain at least:

```text
camels_ch/
└── timeseries/
    ├── observation_based/
    │   └── CAMELS_CH_obs_based_<basin_id>.csv
    └── simulation_based/
        └── CAMELS_CH_sim_based_<basin_id>.csv
```

The app matches basin IDs across those files and needs continuous daily data
for the dates and input-window length you choose. Optional EStreams and
CAMELS-CH-Chem datasets are read from their own extracted directories; their
paths can be supplied at launch or in **Data and features**. The datasets are
not copied into the repository.

## Run the app

```bash
complexity-ensemble-app --data-root /path/to/camels_ch
```

Open the local URL printed by Gradio (default `http://127.0.0.1:7860`). Select
target basins and dates, choose the data and model setup, then run an experiment
in **Results**. After it finishes, **Rainfall–runoff replay** can select a
rainfall episode, inspect each forecast day, and generate a GIF. The replay
requires the original CAMELS-CH files as well as the run predictions.

Catchments without complete daily input windows for their required date splits
are skipped automatically. Training uses the remaining catchments, and the
completion message lists those skipped. If no usable training catchments or
prediction targets remain, the app explains why the experiment cannot run.

To use optional datasets and keep completed runs in the leaderboard:

```bash
complexity-ensemble-app \
  --data-root /path/to/camels_ch \
  --estreams-root /path/to/estreams_dataset \
  --camels-chem-root /path/to/camels_ch_chem \
  --save-results true \
  --results-dir artifacts/hydrology/run_registry
```

Saving is off by default. The basin map loads Leaflet assets from a CDN, so
the map view needs browser internet access. Training and the replay read local
data.

## Re-run a saved experiment

Saved runs include `config.json`, predictions, metrics, and a bundle. From the
repository root:

```bash
complexity-ensemble-run \
  --config artifacts/hydrology/run_registry/RUN_ID/config.json \
  --data-root /path/to/camels_ch
```

To rerun a configuration saved on another machine, override its device with
`--device auto`, `--device mps`, or `--device cpu`.

See [results registry](ensemble_review/RESULTS_REGISTRY.md) for saving and
comparison details, and [events and forecasts](ensemble_review/EVENTS_AND_FORECASTS.md)
for forecast, event, scenario, and replay definitions.

## Run tests

```bash
python -m pytest -q ensemble_review/tests
```

## HTTP API and shared worker

### Custom research workspace

The custom browser frontend uses a white-to-silver gradient with sky-blue accents (`#00BFFF`).
It shares one Python server with the HTTP API and Gradio, while a separate worker
runs training. Catchment selection stays in the browser; map geometry is fetched
once per dataset, charts load per catchment, and progress polling does not redraw
the rest of the page. No Node build or frontend dependency installation is needed.

```bash
python -m pip install -e './ensemble_review[api,animations]'
cp api-server.example.json api-server.json
```

Set the dataset paths in `api-server.json`, then run these in separate terminals:

```bash
complexity-ensemble-studio --settings api-server.json
complexity-ensemble-worker --settings api-server.json
```

Open **http://127.0.0.1:7860**. The workspace includes experiment configuration,
catchment selection, progress/cancellation, a run library with charts and downloads,
comparisons, rainfall replay/GIF export, and paved-site scenarios.

Completed BAFU high-flow runs include **Flood event performance** in the Run library.
The Swiss catchment map can display event precision, event recall, event false-alarm
ratio (FAR), peak magnitude MAE (mm/day), or peak timing MAE (days). These are
historical verification scores, not flood probabilities. Catchment scores average
available seeds; the run average gives each catchment equal weight and reports
contributing counts. Peak errors exclude boundary-censored events. Scores cover
the full test period and all forecast leads, independently of the discharge lead
selector. Existing saved runs with the one-to-one flood verifier can be viewed
without retraining; unavailable scores appear as a dash and grey map areas.

BAFU high-flow results also include **Documented floods · HANZE** when the server
registration has `"hanze_root": "../datasets/hanze"`. Restart both studio and worker
after changing server settings. Existing saved runs can be evaluated when opened;
new runs save the report in `hanze_evaluation.json` and the archive. The map metric
selector includes HANZE catalogue-linked precision and recall, with per-catchment
scores, equal-catchment run means, and event matching/coverage details. For rolling runs, these scores default to **Rolling blocks · all leads**, joining
the saved forecast blocks into one daily series per seed. This mixes forecast
leads according to the saved schedule; overlapping blocks use the shortest available
lead per target date. The HANZE forecast-series selector also exposes individual
leads, which may be sparse and fail complete-window coverage. `GET /api/v1/runs/{run_id}/hanze` returns the report.

HANZE is an impact catalogue, not a complete gauge-level flood-label dataset. The
initial integration uses Swiss `River` records only and assigns gauges to reported
cantons using gauge coordinates. Each complete predicted high-flow spell can match
at most one documented event by exact date overlap. Recall requires the entire
catalogue event window to have finite predictions and observations. Boundary/gap
forecasts and forecasts overlapping excluded catalogue windows are excluded from
precision. Unlinked forecasts are **not confirmed false alarms**. Missing catalogue
events, unmapped gauges, or insufficient coverage produce unavailable scores, not
assumed negatives. For a documented case, HANZE event `1466` covers 19–26 August
2005 in multiple Swiss cantons; the test period must cover the complete event and
surrounding days. Select stations in the affected cantons and verify BAFU coverage.

Use **Data source** above the catchment map to choose **CAMELS-CH**, **BAFU only**,
or **BAFU + CAMELS predictors**. The map and list follow station compatibility;
unavailable catchments show a reason and cannot be selected. BAFU availability
here checks local station metadata and catchment area, plus CAMELS predictor
availability in combined mode. **Check benchmark coverage** audits daily discharge
and date windows. BAFU setup validation and submission also run this audit before
queueing; strict cohorts with incomplete coverage are rejected.

For BAFU drought experiments, choose **BAFU benchmark → Event task → Drought
events**. Predictions remain daily discharge; seven-day low flows are evaluated
as drought days and events using thresholds fitted only to training observations.
Use **Forecasting → Rolling** for multi-day trajectories. Results include a
drought threshold chart and drought day, event and score exports. CAMELS-CH
rainfall–runoff defaults remain unchanged. See
[BAFU drought settings](ensemble_review/BAFU_BENCHMARKS.md#drought).

Configuration
drafts are saved in browser local storage; an active job reconnects after refreshing
the same tab. Result charts let you select a catchment and forecast lead day, with
time series averaged across seeds. Complete predictions remain in the exports.

To also use the original experiment interface at `/gradio`, add
`--native-dataset camels-ch` to the studio command. Its training uses the same API
and worker. The existing `complexity-ensemble-app` command remains available.

For an authenticated server, set `RUNOFF_API_TOKEN` before launching the studio,
then enter it through the workspace connection button. The token stays in tab
session storage and is sent in request headers, including downloads. The native
Gradio interface uses username `runoff` and that token as its password. The studio
does not open a public share tunnel. CPU/RAM/GPU limits still apply when the worker
shares the same machine; this UI change does not reserve training resources.

Workspace checks:

```bash
node --test ensemble_review/tests/studio_core.test.mjs
python -m unittest discover -s ensemble_review/tests -p 'test_studio.py'
```

The API exposes experiments, progress/cancellation, run comparisons, predictions,
rainfall replay, GIF export, and paved-site scenarios. Gradio and the CLI remain
available and can optionally submit training through the same worker.

```bash
python -m pip install -e './ensemble_review[api,animations]'
cp api-server.example.json api-server.json
```

Edit the dataset paths in `api-server.json`, then start these in separate terminals:

```bash
complexity-ensemble-api --settings api-server.json
complexity-ensemble-worker --settings api-server.json
```

Open `http://127.0.0.1:8000/docs`. Set `RUNOFF_API_TOKEN` for authenticated
integration. Health and OpenAPI work before datasets are available.
See [the API guide](ensemble_review/API.md) for requests, deployment settings,
job lifecycle, storage, and remote Gradio/CLI execution.

## BAFU extreme-flow benchmarking

The API workspace supports BAFU-only and combined predictor experiments,
read-only coverage audits, fixed evaluation fingerprints and shared flood
verification. See [BAFU benchmarks](ensemble_review/BAFU_BENCHMARKS.md).

## Hydropower monitoring map

The Gradio application includes a **Hydropower monitoring** tab. On first launch it attempts to download and cache the official Swiss Federal Office of Energy WASTA hydropower-station inventory as `data/hydropower_ch.gpkg` (override the location with `HYDROPOWER_GPKG`). Source: [WASTA statistics on hydropower plants](https://opendata.swiss/en/dataset/statistik-der-wasserkraftanlagen-wasta). The source inventory covers Swiss hydropower plants with output of at least 300 kW.

In the Hydropower monitoring tab, select a saved model run (loaded from the run registry and its `predictions.csv`) or leave the selector on the current session, choose a target date, and update the map. This lets you inspect previously completed experiments without retraining. Plant points are colored red-to-green by the change in the modelled basin discharge relative to the previous calendar day. The tooltip/table show the predicted and previous-day flow (m³/s where a basin area attribute is available), change percentage, and matched CAMELS-CH basin. Plants that cannot be spatially matched to a catchment with a prediction remain gray. The current model's predictions are experiment/back-test outputs; they are not automatically a live forecast for tomorrow. A genuine operational day-ahead workflow requires future-target forecasts and the corresponding meteorological inputs.

The inventory is fetched automatically if no local file is present, so the first launch needs internet access. For offline deployments, place the official WASTA GeoPackage at `data/hydropower_ch.gpkg` or set `HYDROPOWER_GPKG` to its path before launching the app.
