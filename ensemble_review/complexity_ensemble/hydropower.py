"""Swiss hydropower site data and forecast-map helpers.

The station inventory is sourced from the Swiss Federal Office of Energy's
WASTA GeoPackage. It is downloaded automatically on first use and cached as a
local GeoPackage so the user does not need to upload a layer in the GUI.
"""
from __future__ import annotations

import math
import os
import tempfile
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

WASTA_GPKG_URL = (
    "https://data.geo.admin.ch/ch.bfe.statistik-wasserkraftanlagen/"
    "statistik-wasserkraftanlagen/statistik-wasserkraftanlagen_2056.gpkg"
)


def default_hydropower_path() -> Path:
    """Return a writable project data path, overridable for deployments."""
    configured = os.environ.get("HYDROPOWER_GPKG")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "hydropower_ch.gpkg").resolve()


def ensure_hydropower_gpkg(path: str | Path, *, timeout: int = 8) -> Path:
    """Return a local WASTA GeoPackage, downloading it if it is not cached.

    Download is atomic. Existing non-empty local files are respected, so the
    dataset can be replaced manually with a newer official release.
    """
    target = Path(path)
    if target.is_file() and target.stat().st_size > 10_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".hydropower-", suffix=".gpkg", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        request = urllib.request.Request(
            WASTA_GPKG_URL,
            headers={"User-Agent": "complexity-ensemble/0.1 hydropower-data-loader"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if temp_path.stat().st_size <= 10_000:
            raise OSError("Downloaded hydropower GeoPackage is unexpectedly small.")
        with temp_path.open("rb") as source:
            if source.read(16) != b"SQLite format 3\x00":
                raise OSError("Official hydropower download is not a valid SQLite/GeoPackage file.")
        temp_path.replace(target)
        return target
    finally:
        temp_path.unlink(missing_ok=True)


def load_hydropower_sites(path: str | Path, *, layer: str | None = None):
    """Read the largest point layer from an official or user-supplied GeoPackage.

    The source schema is retained. Geometries are normalized to EPSG:4326 for
    browser display; choosing the largest point layer avoids duplicating station
    locations when a source also contains auxiliary label layers.
    """
    import geopandas as gpd
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        layers = gpd.list_layers(source)
        # GeoPandas/Fiona report geometry_type without reading feature data.
        # Use it to avoid loading potentially large attribute-only and polygon
        # layers just to discover that they cannot render as station points.
        if "geometry_type" in layers.columns:
            point_layers = layers[layers["geometry_type"].astype(str).str.lower().isin(
                {"point", "multipoint"}
            )]
            layer_names = list(point_layers["name"].astype(str))
        else:
            layer_names = list(layers["name"].astype(str))
    except (AttributeError, ValueError, OSError):
        import fiona
        layer_names = list(fiona.listlayers(source))
    if layer is not None:
        if layer not in layer_names:
            # Layer metadata may omit geometry types; verify against all GeoPackage layers.
            try:
                import fiona
                all_layers = list(fiona.listlayers(source))
            except Exception:
                all_layers = layer_names
            if layer not in all_layers:
                raise ValueError(f"Configured hydropower layer {layer!r} was not found in {source}")
        layer_names = [layer]
    frames = []
    for layer_name in layer_names:
        try:
            frame = gpd.read_file(source, layer=layer_name)
        except (ValueError, OSError, RuntimeError):
            continue
        # GeoPackages can contain ordinary attribute tables alongside spatial
        # layers. geopandas.read_file returns a pandas.DataFrame for those tables,
        # which has no `.geometry` accessor. Skip them and continue searching.
        if frame.empty or not isinstance(frame, gpd.GeoDataFrame):
            continue
        try:
            geometry_name = frame.geometry.name
        except (AttributeError, ValueError):
            continue
        if geometry_name not in frame.columns:
            continue
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        if frame.empty:
            continue
        geom_types = set(frame.geometry.geom_type.dropna().unique())
        if not geom_types or not geom_types.issubset({"Point", "MultiPoint"}):
            continue
        if frame.crs is None:
            continue
        frame = frame.to_crs(epsg=4326)
        frame["_source_layer"] = layer_name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No point layer with a declared CRS was found in hydropower GeoPackage {source!s}" + (f" for configured layer {layer!r}." if layer else "."))
    # Prefer the explicitly named hydropower plant layer when present. Some
    # GeoPackages contain auxiliary point layers with more label/centroid
    # features than the actual inventory.
    preferred = [candidate for candidate in frames
                 if "hydropowerplant" in str(candidate["_source_layer"].iloc[0]).lower()]
    # Avoid relying on pandas in the expression above; source-layer names are
    # constant within each candidate frame.
    frame = (max(preferred, key=len) if preferred else max(frames, key=len)).copy()
    # Leaflet circle markers require individual Point geometries. Some source
    # GeoPackages store stations as MultiPoint features, which were previously
    # silently skipped by the browser renderer.
    if frame.geometry.geom_type.eq("MultiPoint").any():
        frame = frame.explode(index_parts=False, ignore_index=True)
    frame = frame[frame.geometry.geom_type.eq("Point")].copy()
    frame = frame.reset_index(drop=True)
    frame["_site_id"] = [f"hydro_{i + 1}" for i in range(len(frame))]
    frame["_site_name"] = frame.apply(_site_name, axis=1)
    frame["_capacity_kw"] = frame.apply(_capacity_kw, axis=1)
    return frame


def _site_name(row) -> str:
    candidates = ("name", "anlagenname", "kraftwerk", "werk_name", "name_de", "name_fr",
                  "name_it", "plant_name", "zentralenname", "anlage", "bezeichnung", "objektname")
    columns = {str(key).lower(): key for key in row.index}
    for candidate in candidates:
        key = columns.get(candidate)
        if key is not None:
            value = row.get(key)
            if value is not None and str(value).strip() and str(value).lower() != "nan":
                return str(value).strip()
    for key, value in row.items():
        lower = str(key).lower()
        if any(word in lower for word in ("name", "kraftwerk", "zentral")) and value is not None:
            if str(value).strip() and str(value).lower() != "nan":
                return str(value).strip()
    return "Hydropower plant"


def _capacity_kw(row) -> float | None:
    for key, value in row.items():
        lower = str(key).lower()
        if not any(word in lower for word in ("leistung", "capacity", "power", "kw", "mw")):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number) or number <= 0:
            continue
        if "mw" in lower and "kw" not in lower:
            number *= 1000
        return number
    return None


def _basin_area_km2(record) -> float | None:
    attrs = getattr(record, "static_attributes", {}) or {}
    # CAMELS-CH area attributes are normally in km². Prefer explicit names.
    preferred = ("area_km2", "area_km²", "area", "area_gages2", "catchment_area", "catchment_area_km2")
    lowered = {str(key).lower(): value for key, value in attrs.items()}
    for key in preferred:
        if key in lowered:
            try:
                value = float(lowered[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
    for key, raw in lowered.items():
        if "area" in key and ("km" in key or key == "area"):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
    return None


def _mean_predictions(result) -> dict[tuple[str, date], float]:
    grouped: dict[tuple[str, date], list[float]] = defaultdict(list)
    for item in getattr(result, "predictions", []) or []:
        value = getattr(item, "predicted_mm_day", None)
        basin_id = str(getattr(item, "basin_id", ""))
        target_date = getattr(item, "target_date", None)
        if isinstance(target_date, str):
            try:
                target_date = date.fromisoformat(target_date[:10])
            except ValueError:
                continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if basin_id and isinstance(target_date, date) and math.isfinite(value):
            grouped[(basin_id, target_date)].append(value)
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}



def load_saved_run_predictions(run_path: str | Path):
    """Load persisted predictions.csv into the lightweight shape used by the map.

    The full training result is not reconstructed: only columns needed by the
    hydropower map are read, so historical runs can be inspected without
    rerunning model training.
    """
    import csv
    from types import SimpleNamespace

    path = Path(run_path)
    if path.is_dir():
        path = path / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Saved run has no predictions.csv: {path}")
    predictions = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"seed", "basin_id", "target_date", "predicted_mm_day"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Saved predictions.csv is missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            try:
                day = date.fromisoformat(str(row["target_date"])[:10])
                seed = int(float(row["seed"]))
                value = float(row["predicted_mm_day"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            predictions.append(SimpleNamespace(
                seed=seed, basin_id=str(row["basin_id"]), target_date=day,
                predicted_mm_day=value,
            ))
    if not predictions:
        raise ValueError(f"No valid predictions found in {path}")
    return SimpleNamespace(predictions=predictions, source_path=path)

def prepare_hydropower_forecast(result, catalog, sites, target_date: str | date | None = None):
    """Attach basin forecast and day-over-day change to each plant.

    Strict point-in-polygon matching is used against CAMELS-CH basin geometry.
    Stations outside the available catchment coverage remain unmatched and are
    shown gray rather than being assigned an invented forecast.
    """
    import geopandas as gpd
    import pandas as pd
    means = _mean_predictions(result)
    available_dates = sorted({day for _, day in means})
    if not available_dates:
        raise ValueError("The current run contains no usable discharge predictions.")
    if target_date is None or str(target_date).strip() == "":
        selected_date = available_dates[-1]
    elif isinstance(target_date, date):
        selected_date = target_date
    else:
        try:
            selected_date = date.fromisoformat(str(target_date).strip()[:10])
        except ValueError as error:
            raise ValueError("Date must use YYYY-MM-DD format.") from error
    if selected_date not in available_dates:
        available = ", ".join(day.isoformat() for day in available_dates[-5:])
        raise ValueError(f"No predictions for {selected_date.isoformat()}. Latest available dates: {available}")
    yesterday = selected_date - timedelta(days=1)
    records = [record for record in getattr(catalog, "records", ()) if getattr(record, "geometry", None) is not None]
    # Cache geometry bounds once. The former nested scan called covers() and
    # buffer() for every plant x basin pair, making large inventories slow.
    records_with_bounds = []
    for record in records:
        geometry = record.geometry
        try:
            minx, miny, maxx, maxy = geometry.bounds
        except Exception:
            continue
        records_with_bounds.append((record, minx, miny, maxx, maxy))
    out = sites.copy()
    basin_ids, predicted, previous, deltas, pct_changes, flows, previous_flows = [], [], [], [], [], [], []
    statuses = []
    for _, site in out.iterrows():
        point = site.geometry
        candidates = []
        if point is None or point.is_empty:
            candidate_records = ()
        else:
            px, py = point.x, point.y
            # Cheap bounding-box rejection avoids expensive geometry operations
            # for almost all unrelated catchments.
            candidate_records = (record for record, minx, miny, maxx, maxy in records_with_bounds
                                 if minx - 1e-9 <= px <= maxx + 1e-9 and miny - 1e-9 <= py <= maxy + 1e-9)
        for record in candidate_records:
            geometry = record.geometry
            try:
                hit = geometry.covers(point)
                if not hit and geometry.distance(point) <= 1e-9:
                    hit = True
            except Exception:
                hit = False
            if hit:
                candidates.append(record)
        # Where boundaries overlap, prefer the smallest eligible catchment with
        # a forecast for the selected day; otherwise leave the station unmatched.
        candidates.sort(key=lambda rec: (0 if (str(rec.basin_id), selected_date) in means else 1,
                                         getattr(rec.geometry, "area", float("inf"))))
        match = next((rec for rec in candidates if (str(rec.basin_id), selected_date) in means), None)
        if match is None:
            basin_ids.append(None); predicted.append(float("nan")); previous.append(float("nan"))
            deltas.append(float("nan")); pct_changes.append(float("nan")); flows.append(float("nan")); previous_flows.append(float("nan"))
            statuses.append("No matched basin forecast")
            continue
        basin = str(match.basin_id)
        now = means.get((basin, selected_date), float("nan"))
        before = means.get((basin, yesterday), float("nan"))
        delta = now - before if math.isfinite(before) else float("nan")
        pct = (delta / abs(before) * 100.0) if math.isfinite(delta) and abs(before) > 1e-9 else float("nan")
        area = _basin_area_km2(match)
        q_now = now * area / 86.4 if area is not None else float("nan")
        q_before = before * area / 86.4 if area is not None and math.isfinite(before) else float("nan")
        basin_ids.append(basin); predicted.append(now); previous.append(before); deltas.append(delta)
        pct_changes.append(pct); flows.append(q_now); previous_flows.append(q_before)
        statuses.append("Increase" if math.isfinite(pct) and pct > 0 else "Decrease" if math.isfinite(pct) and pct < 0 else "Stable" if math.isfinite(pct) else "Yesterday unavailable")
    out["basin_id"] = basin_ids
    out["predicted_mm_day"] = predicted
    out["previous_mm_day"] = previous
    out["delta_mm_day"] = deltas
    out["change_pct"] = pct_changes
    out["predicted_m3_s"] = flows
    out["previous_m3_s"] = previous_flows
    out["forecast_date"] = selected_date.isoformat()
    out["status"] = statuses
    return out, selected_date


def hydropower_map_payload(sites, *, target_date: str | None = None) -> dict:
    """Create the serializable Leaflet payload; red means falling flow, green rising."""
    features = []
    for _, row in sites.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPoint":
            geom = max(geom.geoms, key=lambda item: item.length)
        properties = {
            "site_id": str(row.get("_site_id", "")),
            "name": str(row.get("_site_name", "Hydropower plant")),
            "basin_id": str(row.get("basin_id", "") or ""),
            "change_pct": _json_number(row.get("change_pct")),
            "predicted_mm_day": _json_number(row.get("predicted_mm_day")),
            "previous_mm_day": _json_number(row.get("previous_mm_day")),
            "predicted_m3_s": _json_number(row.get("predicted_m3_s")),
            "previous_m3_s": _json_number(row.get("previous_m3_s")),
            "capacity_kw": _json_number(row.get("_capacity_kw")),
            "status": str(row.get("status", "No forecast")),
        }
        features.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": properties})
    return {"type": "FeatureCollection", "features": features, "target_date": target_date}


def _json_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def hydropower_table_rows(sites) -> list[list[object]]:
    """Compact table for Gradio, sorted by largest absolute flow change."""
    rows = []
    for _, row in sites.iterrows():
        pct = _json_number(row.get("change_pct"))
        rows.append([
            str(row.get("_site_name", "Hydropower plant")),
            str(row.get("basin_id", "") or "—"),
            _json_number(row.get("predicted_m3_s")),
            _json_number(row.get("previous_m3_s")),
            _json_number(row.get("change_pct")),
            str(row.get("status", "No forecast")),
        ])
    rows.sort(key=lambda item: abs(item[4]) if isinstance(item[4], (int, float)) else -1, reverse=True)
    return rows
