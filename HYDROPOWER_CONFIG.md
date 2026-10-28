# Configuring the hydropower GeoPackage

The Studio server accepts an optional `hydropower` object in `api-server.json`:

```json
"hydropower": {
  "enabled": true,
  "geopackage": "data/hydropower_ch.gpkg",
  "layer": "hydropower_ch1__hydropowerplant"
}
```

- `enabled` (default `true`) enables or disables hydropower monitoring in Studio and station loading in the optional Gradio interface.
- `geopackage` is a path to a local GeoPackage. Relative paths are resolved relative to the `api-server.json` file. If omitted, the app uses its built-in default path and may download the official Swiss WASTA inventory.
- `layer` optionally names the exact layer to read. If omitted, the loader automatically chooses a suitable point layer and prefers names containing `hydropowerplant`.

The configured GeoPackage must exist and the selected layer must contain Point or MultiPoint geometries with a declared coordinate reference system. Geometries are transformed to EPSG:4326 for browser display. If the configured file or layer cannot be loaded, the rest of the application remains available and the Hydropower view displays the error.

These settings apply to the main Studio **Hydropower** view at `/#hydropower`.
Launch Studio normally with `complexity-ensemble-studio --settings api-server.json`;
no `--native-dataset` flag is needed. Station data are loaded on demand when the
view opens. Choose a dataset and completed run in the view; the API verifies the
run's source dataset before matching plants to its catchments. The worker command
does not change, and viewing completed predictions does not require a running worker.

The same settings are also passed to the optional native Gradio interface at
`/gradio` when Studio is launched with `--native-dataset DATASET_ID`.
