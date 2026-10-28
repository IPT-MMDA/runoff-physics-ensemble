# Launching the Hydropower-enabled app

The `Hydropower monitoring` tab is included in `ensemble_review/complexity_ensemble/app.py`.
If you do not see it, you may be launching an older installed checkout or the original archive.

From this project directory, launch the app using the local-source launcher:

```bash
python run_hydropower_app.py --data-root /path/to/CAMELS-CH
```

This launcher puts this project's `ensemble_review` source directory first on Python's import path,
so an older globally installed `complexity_ensemble` package does not shadow this version.

The hydropower station GeoPackage can be placed at `data/hydropower_ch.gpkg`, or set an explicit path
with `HYDROPOWER_GPKG`. The app attempts to fetch the Swiss WASTA GeoPackage only when that file is
missing; if network access is unavailable, the rest of the app should still start.

After launching, open the local URL printed in the terminal and select **Hydropower monitoring**.
For previously saved experiments, use the **Saved model run** selector in that tab. Saved runs are
read from the configured results registry; they are not retrained.
