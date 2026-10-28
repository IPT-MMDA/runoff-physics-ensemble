from datetime import date
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from complexity_ensemble.hydropower import (
    ensure_hydropower_gpkg,
    hydropower_map_payload,
    hydropower_table_rows,
    load_saved_run_predictions,
    prepare_hydropower_forecast,
)
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog, PredictionRecord, ExperimentResult


def _result():
    rows = []
    for seed in (1, 2):
        rows.extend([
            PredictionRecord(seed, "basin-1", date(2024, 1, 1), 4.0, 4.0, 0.5),
            PredictionRecord(seed, "basin-1", date(2024, 1, 2), 7.0, 8.0, 0.5),
        ])
    return ExperimentResult(rows, {}, {}, {}, {}, [], [], {})


def test_hydropower_map_uses_ensemble_mean_and_day_over_day_percent():
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(
        "basin-1", "Test basin", {"area": 86.4}, eligible=True, geometry=box(0, 0, 2, 2)
    )])
    sites = gpd.GeoDataFrame({"_site_id": ["p1"], "_site_name": ["Test plant"], "_capacity_kw": [1200]},
                              geometry=[Point(1, 1)], crs="EPSG:4326")

    forecast, target_date = prepare_hydropower_forecast(_result(), catalog, sites, "2024-01-02")

    assert target_date == date(2024, 1, 2)
    assert forecast.loc[0, "basin_id"] == "basin-1"
    assert forecast.loc[0, "predicted_mm_day"] == pytest.approx(8.0)
    assert forecast.loc[0, "previous_mm_day"] == pytest.approx(4.0)
    assert forecast.loc[0, "change_pct"] == pytest.approx(100.0)
    assert forecast.loc[0, "predicted_m3_s"] == pytest.approx(8.0)
    payload = hydropower_map_payload(forecast, target_date=target_date.isoformat())
    assert payload["target_date"] == "2024-01-02"
    assert payload["features"][0]["properties"]["change_pct"] == pytest.approx(100.0)
    assert hydropower_table_rows(forecast)[0][0] == "Test plant"


def test_hydropower_stations_without_spatial_match_are_not_given_fake_forecast():
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(
        "basin-1", "Test basin", {"area": 10}, eligible=True, geometry=box(0, 0, 1, 1)
    )])
    sites = gpd.GeoDataFrame({"_site_id": ["p1"], "_site_name": ["Outside plant"]},
                              geometry=[Point(10, 10)], crs="EPSG:4326")
    forecast, _ = prepare_hydropower_forecast(_result(), catalog, sites, "2024-01-02")
    assert forecast.loc[0, "basin_id"] is None
    assert forecast.loc[0, "status"] == "No matched basin forecast"
    assert hydropower_map_payload(forecast)["features"][0]["properties"]["change_pct"] is None


def test_hydropower_date_must_exist_in_current_run():
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(
        "basin-1", "Test basin", {"area": 10}, eligible=True, geometry=box(0, 0, 1, 1)
    )])
    sites = gpd.GeoDataFrame({"_site_id": ["p1"], "_site_name": ["Test plant"]},
                              geometry=[Point(.5, .5)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="No predictions"):
        prepare_hydropower_forecast(_result(), catalog, sites, "2024-02-01")


def test_existing_local_gpkg_is_reused_without_network(tmp_path):
    path = tmp_path / "hydropower.gpkg"
    path.write_bytes(b"x" * 10_001)
    assert ensure_hydropower_gpkg(path) == path


def test_hydropower_geopackage_loader_normalizes_points(tmp_path):
    from complexity_ensemble.hydropower import load_hydropower_sites

    source = tmp_path / "stations.gpkg"
    frame = gpd.GeoDataFrame({"ANLAGENNAME": ["Testwerk"], "LEISTUNG_MW": [2.5]},
                             geometry=[Point(2600000, 1200000)], crs="EPSG:2056")
    frame.to_file(source, layer="stations", driver="GPKG")
    loaded = load_hydropower_sites(source)
    assert len(loaded) == 1
    assert loaded.crs.to_epsg() == 4326
    assert loaded.loc[0, "_site_name"] == "Testwerk"
    assert loaded.loc[0, "_capacity_kw"] == pytest.approx(2500)


def test_saved_run_predictions_load_without_training(tmp_path):
    run_dir = tmp_path / "saved-run"
    run_dir.mkdir()
    (run_dir / "predictions.csv").write_text(
        "seed,basin_id,target_date,predicted_mm_day\n"
        "1,basin-1,2024-01-01,4.5\n"
        "2,basin-1,2024-01-01,5.5\n"
        "1,basin-1,2024-01-02,8.0\n",
        encoding="utf-8",
    )
    result = load_saved_run_predictions(run_dir)
    assert len(result.predictions) == 3
    assert result.predictions[0].basin_id == "basin-1"
    assert result.predictions[0].target_date == date(2024, 1, 1)
    assert result.predictions[0].predicted_mm_day == pytest.approx(4.5)


def test_hydropower_loader_skips_nonspatial_attribute_tables(tmp_path, monkeypatch):
    import pandas as pd
    from complexity_ensemble.hydropower import load_hydropower_sites

    source = tmp_path / "mixed.gpkg"
    source.write_bytes(b"test placeholder")
    spatial = gpd.GeoDataFrame(
        {"name": ["Valid plant"]}, geometry=[Point(7.4, 46.9)], crs="EPSG:4326"
    )
    monkeypatch.setattr(gpd, "list_layers", lambda _: pd.DataFrame({"name": ["metadata", "stations"]}))
    original_read_file = gpd.read_file

    def fake_read_file(path, layer=None, **kwargs):
        if layer == "metadata":
            return pd.DataFrame({"key": ["source"], "value": ["WASTA"]})
        if layer == "stations":
            return spatial.copy()
        return original_read_file(path, layer=layer, **kwargs)

    monkeypatch.setattr(gpd, "read_file", fake_read_file)
    loaded = load_hydropower_sites(source)
    assert len(loaded) == 1
    assert loaded.loc[0, "_site_name"] == "Valid plant"


def test_hydropower_loader_explodes_multipoint_to_renderable_points(tmp_path):
    from shapely.geometry import MultiPoint
    from complexity_ensemble.hydropower import load_hydropower_sites

    source = tmp_path / "multipoint.gpkg"
    frame = gpd.GeoDataFrame({"name": ["Two plants"]},
                             geometry=[MultiPoint([(7.4, 46.9), (7.5, 47.0)])],
                             crs="EPSG:4326")
    frame.to_file(source, layer="stations", driver="GPKG")
    loaded = load_hydropower_sites(source)
    assert len(loaded) == 2
    assert set(loaded.geometry.geom_type) == {"Point"}
    payload = hydropower_map_payload(loaded)
    assert len(payload["features"]) == 2
    assert all(feature["geometry"]["type"] == "Point" for feature in payload["features"])


def test_hydropower_loader_can_select_explicit_layer(tmp_path):
    from complexity_ensemble.hydropower import load_hydropower_sites

    source = tmp_path / "multiple_layers.gpkg"
    gpd.GeoDataFrame({"name": ["Auxiliary"]}, geometry=[Point(7.4, 46.9)], crs="EPSG:4326").to_file(
        source, layer="auxiliary_points", driver="GPKG"
    )
    gpd.GeoDataFrame({"name": ["Configured plant"]}, geometry=[Point(7.5, 47.0)], crs="EPSG:4326").to_file(
        source, layer="custom_station_layer", driver="GPKG", mode="a"
    )
    loaded = load_hydropower_sites(source, layer="custom_station_layer")
    assert len(loaded) == 1
    assert loaded.loc[0, "_site_name"] == "Configured plant"
    assert loaded.loc[0, "_source_layer"] == "custom_station_layer"


def test_hydropower_loader_rejects_missing_configured_layer(tmp_path):
    from complexity_ensemble.hydropower import load_hydropower_sites

    source = tmp_path / "stations.gpkg"
    gpd.GeoDataFrame({"name": ["Plant"]}, geometry=[Point(7.4, 46.9)], crs="EPSG:4326").to_file(
        source, layer="stations", driver="GPKG"
    )
    with pytest.raises(ValueError, match="Configured hydropower layer"):
        load_hydropower_sites(source, layer="wrong_layer")
