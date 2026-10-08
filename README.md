# Regional rainfall–runoff ensemble

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
mode, the app dependencies, GIF export (`Pillow`), optional TDA and KAN methods,
and `pytest`. The package dependency definitions live in
[ensemble_review/pyproject.toml](ensemble_review/pyproject.toml).

For a smaller installation with the app and GIF export, omit the optional
modeling and test dependencies:

```bash
python -m pip install -e './ensemble_review[animations]'
```

CPU execution is supported; a compatible PyTorch/CUDA installation is needed
only if you select a CUDA device. GIF export does not require FFmpeg.

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

See [results registry](ensemble_review/RESULTS_REGISTRY.md) for saving and
comparison details, and [events and forecasts](ensemble_review/EVENTS_AND_FORECASTS.md)
for forecast, event, scenario, and replay definitions.

## Run tests

```bash
python -m pytest -q ensemble_review/tests
```
