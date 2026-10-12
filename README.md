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

The model uses CAMELS-CH simulation-based precipitation, temperature, and
potential evapotranspiration (PET), previous observed discharge, seasonal day
encoding, and available basin descriptors. Optional EStreams vegetation and
snow-cover data and CAMELS-CH-Chem catchment predictors can be added. The
ensemble can route between a complex Mamba or MLP expert and a simpler RBF or
Fourier expert; a linear-reservoir relation provides a physics constraint.

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
