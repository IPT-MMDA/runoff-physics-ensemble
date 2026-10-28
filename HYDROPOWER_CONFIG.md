# Configuring the hydropower GeoPackage

The Studio server accepts an optional `hydropower` object in `api-server.json`:

```json
"hydropower": {
  "enabled": true,
  "geopackage": "data/hydropower_ch.gpkg",
  "layer": "hydropower_ch1__hydropowerplant"
}
```

- `enabled` (default `true`) enables or disables the hydropower monitoring tab's data loading.
- `geopackage` is a path to a local GeoPackage. Relative paths are resolved relative to the `api-server.json` file. If omitted, the app uses its built-in default path and may download the official Swiss WASTA inventory.
- `layer` optionally names the exact layer to read. If omitted, the loader automatically chooses a suitable point layer and prefers names containing `hydropowerplant`.

The configured GeoPackage must exist and the selected layer must contain Point or MultiPoint geometries with a declared coordinate reference system. Geometries are transformed to EPSG:4326 for browser display. If the configured file or layer cannot be loaded, the rest of the application remains available and the hydropower tab displays the error.

This setting is passed to the native Gradio interface when launching Studio with `--native-dataset`.
