# Hydrology API and shared application services

Gradio, the CLI, and the HTTP API use the same application services. Existing
local commands continue to work. The HTTP service submits long computations to
a dedicated worker; it does not train inside an HTTP request.

## Installation and startup

From the repository root, activate your Python environment and install:

```bash
python -m pip install -e './ensemble_review[api,animations]'
cp api-server.example.json api-server.json
```

Edit `api-server.json` to register the extracted datasets on the server. Paths
are resolved relative to the settings file, except absolute paths. Optional
`estreams_root`, `camels_chem_root`, and `hyd_responses_root` may be registered under each dataset.
Clients refer to the dataset ID (`camels-ch` in this example), not these paths.
Do not commit credentials or a machine-specific settings file.

Run these commands in separate terminals using the same environment and
settings file:

```bash
export RUNOFF_API_TOKEN='replace-with-your-shared-secret'
complexity-ensemble-api --settings api-server.json
```

```bash
complexity-ensemble-worker --settings api-server.json
```

The server defaults to `http://127.0.0.1:8000`. Interactive OpenAPI documentation
is at `/docs`; the machine-readable schema is at `/openapi.json`. Use the
**Authorize** button to enter the bearer token. Documentation is public; all
`/api/v1` routes require the token when configured. Without a token, the command
only permits binding to a loopback address. For another platform's access, bind
to the appropriate interface with `--host`, set the token, and use HTTPS at your
existing reverse proxy. The token identifies one trusted integration; this
version does not implement per-user tenancy or authorization.

Health and capabilities do not load datasets. You can start the server with an
empty `datasets` object before downloading data. `/health` reports whether the
worker is ready separately from HTTP service availability. Dataset validation
reports missing data when requested; experiment submission requires a valid
registered dataset and eligible basin IDs.

## Submit, monitor, and retrieve an experiment

Create `experiment.json`:

```json
{
  "dataset_id": "camels-ch",
  "config": {
    "basins": {
      "target_basins": ["4009"],
      "training_scope": "all_basins"
    },
    "dates": {
      "train": {"start": "2010-01-01", "end": "2016-08-06"},
      "validation": {"start": "2016-08-07", "end": "2018-10-18"},
      "test": {"start": "2018-10-19", "end": "2020-12-31"}
    },
    "model": {"complex_expert": "mlp", "simple_expert": "fourier"},
    "strategy": {"approach": "no_routing", "epochs": 2},
    "hyperparameters": {"device": "auto", "seeds": [0]}
  },
  "use_estreams": false,
  "use_camels_chem": false,
  "save_to_leaderboard": false
}
```

All configuration sections have the same defaults as the existing domain
configuration. Basins and date splits are required. Model, training, physics,
extreme-event and forecast options are documented in OpenAPI. Unknown fields
are rejected, including nested misspellings. Numbers must be JSON numbers;
NaN and infinity are rejected. Optional feature selections use
`config.estreams_features` and `config.camels_chem_features`.

HYD-RESPONSES is another optional registered dataset. Add `hyd_responses_root`
to the server dataset settings, then set `use_hyd_responses: true` in requests.
Choose `config.hyd_responses_features` from `hyd_soil_moisture_layer_1` through
`hyd_soil_moisture_layer_4`, `hyd_snowmelt_oshd`, `hyd_swe_oshd`, and
`hyd_swe_spass`. Set `config.hyd_responses_static: true` to add physical
catchment descriptors. Soil moisture uses m³/m³; missing data are masked.
The `/datasets/{dataset_id}/features` endpoint lists available selections.

Complexity routing is selected through `config.hyperparameters.complexity_method`:
`morse` (default), `ph`, `takens`, or `lyapunov`. The older experimental name
`tda` is accepted as an alias for PH. These settings apply to soft/hard routing,
distillation, and routed physics-distillation teachers. Static blends, stacking,
OOD fallback, and no-routing approaches otherwise ignore them. The CLI and
saved-run configurations use these same fields. For example:

```json
{
  "complexity_method": "ph",
  "routing_context_length": 365,
  "embedding_dim": 3,
  "embedding_delay": 7,
  "ph_min_persistence": 0.1,
  "ph_score": "total_persistence"
}
```

Place these fields inside `config.hyperparameters` and select a routed training
approach. `ph_score` also accepts `persistent_holes` and `persistence_entropy`.
Takens uses effective covariance rank of the delay embedding, rather than PH.
Lyapunov additionally accepts `lyapunov_theiler_window` (30 days),
`lyapunov_fit_horizon` (20 subsampled steps), and `lyapunov_min_r2` (0.8).
Routing history must cover the model sequence and delay embedding; Lyapunov
also requires enough points for its fitting horizon and neighbor exclusion.
Only training estimates fit the routing threshold. Rolling forecasts recompute
scores after replacing future observed discharge with predictions.

`GET /api/v1/capabilities` lists `complexity_methods` and reports Ripser availability
as `optional_methods.ph`. Install `./ensemble_review[tda]` in the worker's
environment to run PH. Results persist the selected settings and include
`complexity_method`, `complexity_routing_active`, and, for history-based routes,
`valid_complexity_fraction` and `mean_complexity_score` in routing diagnostics.

```bash
curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  http://127.0.0.1:8000/api/v1/datasets

curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @experiment.json \
  http://127.0.0.1:8000/api/v1/experiments/validate

curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: example-experiment-001' \
  --data-binary @experiment.json \
  http://127.0.0.1:8000/api/v1/experiments
```

Submission returns HTTP 202 and a `job_id` with status and links. Validation
checks configuration, dataset layout, basin eligibility and optional sources;
continuous date coverage is checked while preparing the experiment. The HTTP
process and worker need the same environment and registered files. The worker
revalidates before execution.

```bash
curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  http://127.0.0.1:8000/api/v1/jobs/JOB_ID

curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  http://127.0.0.1:8000/api/v1/jobs/JOB_ID/result

curl -H "Authorization: Bearer $RUNOFF_API_TOKEN" \
  'http://127.0.0.1:8000/api/v1/runs/RUN_ID/predictions?basin_id=4009&offset=0&limit=1000'
```

`ensemble_review/examples/api_client.py` implements submission, polling and
paginated prediction downloads with only the Python standard library:

```bash
python ensemble_review/examples/api_client.py \
  --request experiment.json --output predictions.json
```

Reuse an idempotency key when retrying the same submission. Reusing that key
with different contents or a different operation returns HTTP 409. Keys remain
associated with their jobs across restarts; use a new key to explicitly rerun.

## Route reference

All routes below are relative to `/api/v1`.

| Method and path | Purpose |
| --- | --- |
| `GET /health`, `/capabilities` | HTTP status, worker readiness, available devices/methods |
| `GET /datasets` | Registered sources without filesystem paths |
| `GET /datasets/{id}/validation` | Validate main and registered optional dataset layouts |
| `GET /datasets/{id}/basins` | Basin metadata and geometry; `offset`, `limit` |
| `GET /datasets/{id}/features` | Available core, static and optional feature groups |
| `POST /experiments/validate` | Configuration and source preflight |
| `POST /experiments` | Queue an experiment |
| `GET /jobs/{id}` | Status, progress, timestamps, warnings/result links |
| `POST /jobs/{id}/cancel` | Request cooperative cancellation |
| `GET /jobs/{id}/result` | Run summary, scenario JSON, or GIF |
| `GET /runs` | Filtered runs; `filters` JSON object, `offset`, `limit` |
| `GET /runs/facets` | Available values for the existing leaderboard filters |
| `GET /runs/{id}` | Configuration, metrics, traces, events and diagnostics |
| `GET /runs/{id}/predictions` | Rows filtered by basin, seed, start/end date and lead day |
| `GET /runs/{id}/metrics`, `/events`, `/losses` | Structured analysis results |
| `GET /runs/{id}/artifacts/{name}` | `predictions`, `metrics`, `config`, or `archive` |
| `POST /runs/{id}/save` | Promote a complete run to the configured leaderboard |
| `POST /comparisons` | Run/config comparisons and metric deltas; `run_a`, `run_b` |
| `GET /runs/{id}/replay/episodes?basin_id=...` | Candidate rainfall episodes |
| `POST /runs/{id}/replay` | Replay series and daily forecast snapshots |
| `POST /runs/{id}/replay/gif` | Queue GIF export |
| `POST /runs/{id}/scenarios` | Queue a paved-site comparison |

Replay requests contain `basin_id`, `episode` (ISO date), and optional
`event_days` (default 14). Scenario requests contain `basin_id` and a `site`
object matching the documented `PavedSiteRequest`. Site area and paved fractions
are **fractions from 0 to 1**, rather than the UI's percentages. Storage and
release are millimetres over the site and millimetres per day respectively.
All discharge values are mm/day. Predictions preserve individual seeds, issue
dates, target dates and lead days. For plotting the UI's mean forecast, average
prediction rows by basin and target date across seeds. Replay uses the same
existing aggregation behavior as the UI.

Unavailable or non-finite result metrics are JSON `null`. Replay's illustrative
bucket remains uncalibrated and distinct from trained model storage. Rolling
forecasts retain the existing perfect-weather hindcast semantics.

Errors use `{code, message, details}`. Invalid input returns 422, missing
resources 404, authentication failures 401, and state/idempotency conflicts 409.
Worker exceptions are logged server-side; job errors contain a stable code and
summary rather than a traceback. Inspect `failures` on runs even when the job
succeeds: some seeds can fail while others yield usable predictions. A run
with no usable predictions is a failed job with inspectable saved output.

## Jobs, storage, cancellation and recovery

Jobs follow `queued -> running -> succeeded | failed | cancelled`.
Cancellation of a queued job is immediate. Running jobs expose
`cancel_requested` until computation reaches a cancellation checkpoint. Training
preserves partial output when available. Scenario/GIF jobs check cancellation
at operation boundaries; a rendering or discovery step is not forcibly killed.
A result link can remain available on a cancelled job.

SQLite stores job state and idempotency keys. The worker owns an OS file lock,
so only one worker can execute against a storage directory at a time. Starting
a second worker fails. The worker heartbeat drives readiness. After a crash,
the next worker marks abandoned running jobs `failed` with
`worker_interrupted`; queued jobs remain queued. Interrupted work is not
silently retried. `--once` processes at most one queued job and exits.

API results always persist under `storage_dir/runs`, independently of
leaderboard saving. Scenarios/GIFs live under `storage_dir/job-results`.
`save_to_leaderboard: true` or the save endpoint copies a complete run into
`results_dir` under the same run ID. Cancelled runs cannot be promoted. Existing
Gradio/CLI manifests remain readable. Replay/scenarios for older runs require a
registered dataset whose root matches their original `data_root`.

There is **no automatic retention cleanup**. Keep job state, results and source
data for as long as clients need them. Administrative removal should happen
with the worker stopped and should account for stored job result references.
This first version is for one machine with local storage; multiple API
processes can share the SQLite queue, but it is not a distributed worker system.

## Existing UI and CLI

These still execute locally by default:

```bash
complexity-ensemble-app --data-root /path/to/camels_ch
complexity-ensemble-run --config /path/to/config.json --device auto
```

To send their training through the shared API worker, add:

```bash
--api-url http://127.0.0.1:8000 --dataset-id camels-ch
```

Both read `RUNOFF_API_TOKEN` from the environment. The CLI keeps its waiting and
terminal progress behavior, and Ctrl-C requests remote cancellation. Gradio
keeps its existing progress and cancel controls; remote run replay preparation
and paved-site comparisons also use the API. Gradio still needs a local source
catalog for its basin-selection/map UI, so this mode is intended for a shared
server or a frontend host with that catalog. The external platform can use the
HTTP API without mounting datasets locally.

The existing `--save-results` flag retains its local leaderboard-save behavior;
remote execution additionally retains the server's private job results. Avoid
running standalone GPU training concurrently with the shared worker if you
want the queue's single-job resource policy to cover all training.

## Architecture and verification

- `configuration.py`: reusable saved/local configuration parsing.
- `services.py`: experiment, comparison, replay, scenario and run-storage operations.
- `serialization.py`: domain-to-JSON conversion.
- `api_models.py`: strict request contracts and typed primary responses.
- `application.py`, `server_settings.py`: registered-source application context.
- `jobs.py`, `worker.py`: persistent queue, exclusive execution and lifecycle.
- `api.py`: HTTP/authentication/OpenAPI adapter.
- `remote.py`: HTTP execution adapter for existing entry points.

The CLI re-exports `config_from_mapping` for backward compatibility. Local
configuration parsing remains tolerant of saved-run metadata; HTTP contracts
reject unrecognized request fields. Scientific implementation stays in the
existing model and regional modules.

Run the synthetic integration and queue tests without downloading CAMELS-CH:

```bash
python -m unittest discover -s ensemble_review/tests -p 'test_jobs.py' -v
python -m unittest discover -s ensemble_review/tests -p 'test_api.py' -v
```

Run all existing regression tests with the dev extra installed:

```bash
python -m pytest -q ensemble_review/tests
```

This API reproduces the current UI's experiments and analyses. It does not
persist trained weights or provide inference from saved checkpoints; adding
that remains a separate model-serving feature.
