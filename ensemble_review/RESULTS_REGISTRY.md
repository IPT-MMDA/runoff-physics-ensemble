# Shared experiment registry

Regional CLI and Gradio runs use the same versioned registry. Persistence is
opt-in: a completed run enters the registry only when the process is launched
with `--save-results true`.

## Gradio

```bash
complexity-ensemble-app \
  --data-root /path/to/camels_ch \
  --save-results true \
  --results-dir artifacts/hydrology/run_registry
```

Without `--save-results true`, the current result remains available in the
Gradio session but no run directory or download bundle is created.

The **Saved-run leaderboard** tab loads valid registry manifests at startup.
Its two setup panels expose only configuration values present in saved runs.
After a run finishes in the same app process, use **Refresh saved runs** to
refresh every selector. Exactly two concrete runs feed the comparison table,
aggregate chart, per-basin chart, and loss-trace chart.

## CLI

The canonical regional runner consumes the same JSON shape written as
`config.json` by a saved run:

```bash
complexity-ensemble-run \
  --config artifacts/hydrology/run_registry/RUN_ID/config.json \
  --data-root /path/to/camels_ch \
  --save-results true \
  --results-dir artifacts/hydrology/run_registry
```

`--data-root` is optional when the path in the configuration remains valid.
The default for `--save-results` is `false` in both entry points.

## Run layout

Each immutable run directory contains:

- `run.json`: schema version, timestamp, resolved setup, metrics, diagnostics,
  loss traces, extremes, failures and artifact names;
- `predictions.csv`: seed/basin/date predictions and routing diagnostics;
- `metrics.csv`: aggregate and per-basin metrics;
- `config.json`: rerunnable resolved configuration;
- `regional_experiment.zip`: portable CSV/config bundle.

Writes use a staging directory and are renamed into place only after every
artifact and the manifest has been produced. A malformed or incomplete run is
reported and skipped without preventing the remaining leaderboard from loading.
