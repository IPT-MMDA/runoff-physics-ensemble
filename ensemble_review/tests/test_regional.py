from datetime import date, timedelta
import csv

import pytest
import torch

from complexity_ensemble.app import BasinMapComponent, DEFAULT_BASINS, _config_from_ui_values
from complexity_ensemble.camels_ch_chem import CAMELSCHChemPressures
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.estreams import EStreamsVegetationSnow
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, CancellationToken,
    DateRange, DateSplitConfig, ExtremeEventConfig, HyperparameterConfig,
    ModelArchitectureConfig, PhysicsModelConfig, RegionalExperimentConfig, TrainingStrategyConfig,
    make_regional_hydrology_data, run_regional_experiment,
)


def _series(basin: str, offset: float = 0.0) -> HydrologySeries:
    first = date(2001, 1, 1)
    dates = tuple(first + timedelta(days=index) for index in range(65))
    forcing = torch.stack((
        torch.tensor([2.0 if index % 5 == 0 else 0.2 for index in range(65)]),
        torch.linspace(2, 10, 65), torch.full((65,), 0.5),
    ), dim=1)
    return HydrologySeries(dates, torch.linspace(1 + offset, 2 + offset, 65), forcing,
                           ("precipitation", "temperature", "pet"), basin, "Switzerland")


def _dates() -> DateSplitConfig:
    return DateSplitConfig(DateRange("2001-01-08", "2001-02-02"),
                           DateRange("2001-02-03", "2001-02-12"),
                           DateRange("2001-02-13", "2001-03-01"))


def _write_estreams(root, gauge_ids=("a", "target")):
    attributes = root / "attributes" / "temporal_attributes"
    gauges = root / "streamflow_gauges"
    attributes.mkdir(parents=True)
    gauges.mkdir()
    with (gauges / "estreams_gauging_stations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(
            target, fieldnames=["basin_id", "gauge_id", "gauge_provider"]
        )
        writer.writeheader()
        for index, gauge_id in enumerate(gauge_ids):
            writer.writerow({
                "basin_id": f"CH{index:06d}", "gauge_id": gauge_id,
                "gauge_provider": "CH_CAMELS",
            })
    filenames = {
        "estreams_LAI_monhtly.csv": (1.0, 2.0),
        "estreams_NDVI_monhtly.csv": (0.2, 0.4),
        "estreams_snowcover_monhtly.csv": (25.0, 10.0),
    }
    fields = ["date", *(f"CH{index:06d}" for index in range(len(gauge_ids)))]
    for filename, monthly_values in filenames.items():
        with (attributes / filename).open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"date": "2001-01-31", **{
                field: monthly_values[0] + index
                for index, field in enumerate(fields[1:])
            }})
            writer.writerow({"date": "2001-02-28", **{
                field: monthly_values[1] + index
                for index, field in enumerate(fields[1:])
            }})


def _write_camels_chem(root, gauge_ids=("a", "target")):
    groups = {
        "agricultural_data": ("swisscrops", ["total_arable"], [100.0, 110.0]),
        "livestock_data": ("livestock", ["gve_ha"], [0.5, 0.6]),
        "atmospheric_deposition": ("atmdepo", ["dntotal"], [20.0, 18.0]),
    }
    metadata = root / "gauges_metadata"
    metadata.mkdir(parents=True)
    with (metadata / "camels_ch_chem_gauges_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=["gauge_id"])
        writer.writeheader()
        for gauge_id in gauge_ids:
            writer.writerow({"gauge_id": gauge_id})
    aggregated = root / "catchment_aggregated_data"
    for directory, (prefix, columns, annual_values) in groups.items():
        path = aggregated / directory
        path.mkdir(parents=True)
        for basin_index, gauge_id in enumerate(gauge_ids):
            with (path / f"camels_ch_chem_{prefix}_{gauge_id}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.DictWriter(target, fieldnames=["date", *columns])
                writer.writeheader()
                writer.writerow({
                    "date": "2000",
                    **{column: value + basin_index for column, value in zip(columns, annual_values)},
                })
                writer.writerow({
                    "date": "2001",
                    **{column: value + 10 + basin_index for column, value in zip(columns, annual_values)},
                })
    isotopes = aggregated / "rain_water_isotopes"
    isotopes.mkdir(parents=True)
    for basin_index, gauge_id in enumerate(gauge_ids):
        with (isotopes / f"camels_ch_chem_rainisotopes_{gauge_id}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(target, fieldnames=["date", "delta_18o"])
            writer.writeheader()
            writer.writerow({"date": "2001-01-15", "delta_18o": -12.0 + basin_index})
            writer.writerow({"date": "2001-02-15", "delta_18o": -10.0 + basin_index})


def test_estreams_monthly_features_are_causally_aligned(tmp_path):
    _write_estreams(tmp_path)
    source = {"a": _series("a"), "missing": _series("missing")}

    augmented = EStreamsVegetationSnow(tmp_path).augment(source)

    assert augmented["a"].forcing_names[-6:] == (
        "estreams_lai", "estreams_lai_available",
        "estreams_ndvi", "estreams_ndvi_available",
        "estreams_snow_cover_fraction", "estreams_snow_cover_fraction_available",
    )
    jan_30 = augmented["a"].dates.index(date(2001, 1, 30))
    jan_31 = augmented["a"].dates.index(date(2001, 1, 31))
    feb_27 = augmented["a"].dates.index(date(2001, 2, 27))
    feb_28 = augmented["a"].dates.index(date(2001, 2, 28))
    external = augmented["a"].forcings[:, -6:]
    assert external[jan_30].tolist() == [0.0] * 6
    assert external[jan_31].tolist() == pytest.approx([1.0, 1.0, 0.2, 1.0, 0.25, 1.0])
    assert external[feb_27].tolist() == pytest.approx([1.0, 1.0, 0.2, 1.0, 0.25, 1.0])
    assert external[feb_28].tolist() == pytest.approx([2.0, 1.0, 0.4, 1.0, 0.10, 1.0])
    assert torch.count_nonzero(augmented["missing"].forcings[:, -6:]) == 0

    ndvi_only = EStreamsVegetationSnow(tmp_path).augment(
        {"a": source["a"]}, ("estreams_ndvi",)
    )["a"]
    assert ndvi_only.forcing_names[-2:] == (
        "estreams_ndvi", "estreams_ndvi_available",
    )
    assert ndvi_only.forcings.shape[1] == source["a"].forcings.shape[1] + 2

    with pytest.raises(ValueError, match="at least one"):
        EStreamsVegetationSnow(tmp_path).augment({"a": source["a"]}, ())


def test_regional_run_can_add_estreams_features(tmp_path):
    _write_estreams(tmp_path)
    series = {"a": _series("a"), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(key) for key in series])
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "exclude_targets"), _dates(),
        ModelArchitectureConfig(
            complex_expert="mlp", simple_expert="fourier",
            mlp_widths=(8,), fourier_frequencies=4,
        ),
        TrainingStrategyConfig("no_routing", 1, 1, 1, 1),
        HyperparameterConfig(sequence_length=7, batch_size=128),
        estreams_root=tmp_path,
    )

    result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)

    assert result.predictions and not result.failures
    assert result.resolved_config["estreams_root"] == str(tmp_path)
    assert result.resolved_config["analysis_task"] == "daily_discharge_forecasting"
    assert result.resolved_config["feature_names"][3:9] == [
        "estreams_lai", "estreams_lai_available",
        "estreams_ndvi", "estreams_ndvi_available",
        "estreams_snow_cover_fraction", "estreams_snow_cover_fraction_available",
    ]


def test_camels_chem_pressures_are_selectable_and_causally_aligned(tmp_path):
    _write_camels_chem(tmp_path)
    source = {"a": _series("a"), "missing": _series("missing")}
    selected = (
        "chem_agriculture_total_arable",
        "chem_livestock_gve_per_ha",
        "chem_deposition_n_total",
        "chem_rain_delta_18o",
    )

    augmented = CAMELSCHChemPressures(tmp_path).augment(source, selected)

    assert augmented["a"].forcing_names[-8:] == tuple(
        item for feature in selected for item in (feature, f"{feature}_available")
    )
    jan_14 = augmented["a"].dates.index(date(2001, 1, 14))
    jan_15 = augmented["a"].dates.index(date(2001, 1, 15))
    values = augmented["a"].forcings[:, -8:]
    assert values[jan_14].tolist() == pytest.approx([
        100.0, 1.0, 0.5, 1.0, 20.0, 1.0, 0.0, 0.0,
    ])
    assert values[jan_15].tolist() == pytest.approx([
        100.0, 1.0, 0.5, 1.0, 20.0, 1.0, -12.0, 1.0,
    ])
    assert torch.count_nonzero(augmented["missing"].forcings[:, -8:]) == 0
    with pytest.raises(ValueError, match="at least one"):
        CAMELSCHChemPressures(tmp_path).augment({"a": source["a"]}, ())


def test_regional_run_can_add_camels_chem_features(tmp_path):
    _write_camels_chem(tmp_path)
    series = {"a": _series("a"), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(key) for key in series])
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "exclude_targets"), _dates(),
        ModelArchitectureConfig(
            complex_expert="mlp", simple_expert="fourier",
            mlp_widths=(8,), fourier_frequencies=4,
        ),
        TrainingStrategyConfig("no_routing", 1, 1, 1, 1),
        HyperparameterConfig(sequence_length=7, batch_size=128),
        camels_chem_root=tmp_path,
        camels_chem_features=("chem_livestock_gve_per_ha",),
    )

    result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)

    assert result.predictions and not result.failures
    assert result.resolved_config["camels_chem_root"] == str(tmp_path)
    assert "chem_livestock_gve_per_ha" in result.resolved_config["feature_names"]


def test_scope_and_target_date_splits_are_leakage_free():
    series = {"a": _series("a"), "b": _series("b", .2), "target": _series("target", .4)}
    data = make_regional_hydrology_data(series,
        {key: {"height": value, "missing": None if key == "target" else value}
         for value, key in enumerate(series)},
        BasinScopeConfig(("target",), "all_except_targets"), _dates(), sequence_length=7)
    assert set(data.train.basin_ids) == {"a", "b"}
    assert set(data.validation.basin_ids) == {"target"}
    assert set(data.test.basin_ids) == {"target"}
    assert data.validation.target_dates[0] == date(2001, 2, 3)
    assert torch.allclose(data.train.inputs.reshape(-1, 6).mean(0), torch.zeros(6), atol=1e-5)


def test_all_strategies_run_on_synthetic_regional_data():
    series = {"a": _series("a"), "b": _series("b", .2), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [
        BasinCatalogRecord(key, key, {"height": index})
        for index, key in enumerate(series)
    ])
    for approach in (
        "soft_routing", "hard_routing", "distillation", "no_routing",
        "static_50_50", "ood_fallback",
    ):
        config = RegionalExperimentConfig(".", BasinScopeConfig(("target",), "exclude_targets"),
            _dates(), ModelArchitectureConfig(complex_expert="mlp", simple_expert="fourier",
                mlp_widths=(8,), fourier_frequencies=4),
            TrainingStrategyConfig(approach, 1, 1, 1, 1),
            HyperparameterConfig(sequence_length=7, batch_size=128),
            extremes=ExtremeEventConfig("statistical"))
        result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)
        assert result.predictions and not result.failures
        assert set(result.per_basin_metrics) == {"target"}
        assert result.extreme_events
        if approach == "no_routing":
            assert result.routing_diagnostics["mean_complex_weight"] == 1.0
        elif approach == "static_50_50":
            assert result.routing_diagnostics["mean_complex_weight"] == 0.5
        elif approach == "ood_fallback":
            assert "ood_fraction" in result.routing_diagnostics


def test_pre_cancelled_run_returns_partial_result():
    series = {"a": _series("a"), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(key) for key in series])
    token = CancellationToken(); token.cancel()
    config = RegionalExperimentConfig(".", BasinScopeConfig(("target",), "exclude_targets"),
        _dates(), strategy=TrainingStrategyConfig(epochs=1),
        hyperparameters=HyperparameterConfig(sequence_length=7))
    result = run_regional_experiment(config, cancellation_token=token,
                                     catalog=catalog, series_by_basin=series)
    assert result.cancelled and not result.predictions


def test_stacking_uses_target_validation_after_target_training():
    series = {"a": _series("a"), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(key) for key in series])
    config = RegionalExperimentConfig(
        ".",
        BasinScopeConfig(("target",), "all_basins"),
        _dates(),
        ModelArchitectureConfig(
            complex_expert="mlp", simple_expert="fourier",
            mlp_widths=(8,), fourier_frequencies=4,
        ),
        TrainingStrategyConfig("stacking", 1, 1, 1, 1),
        HyperparameterConfig(sequence_length=7, batch_size=128),
    )

    result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)

    weight = result.routing_diagnostics["fitted_complex_weight"]
    assert 0.0 <= weight <= 1.0
    assert result.routing_diagnostics["stacking_calibration_scope"] == "pooled_target_validation"
    assert result.resolved_config["fitted_diagnostics"] == result.routing_diagnostics


def test_stacking_rejects_excluded_target_training_scope():
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "exclude_targets"), _dates(),
        strategy=TrainingStrategyConfig(approach="stacking"),
    )

    with pytest.raises(ValueError, match="requires target basins in training"):
        config.validate()


def test_kan_physics_rejects_excluded_target_training_scope():
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "exclude_targets"), _dates(),
        physics=PhysicsModelConfig(optimization_mode="balanced"),
    )

    with pytest.raises(ValueError, match="requires target basins in training"):
        config.validate()


def test_numeric_kan_physics_distills_selected_teacher():
    pytest.importorskip("kan")
    series = {"target": _series("target")}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord("target")])
    config = RegionalExperimentConfig(
        ".", BasinScopeConfig(("target",), "targets_only"), _dates(),
        ModelArchitectureConfig(
            complex_expert="mlp", simple_expert="fourier",
            mlp_widths=(8,), fourier_frequencies=4,
        ),
        TrainingStrategyConfig("static_50_50", 1, 1, 1, 1),
        HyperparameterConfig(sequence_length=7, batch_size=128),
        PhysicsModelConfig(
            optimization_mode="distillation", distillation_teacher="hard_routing",
            kan_candidates=1, kan_steps=1, symbolic=False,
        ),
    )

    result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)

    assert result.predictions and not result.failures
    assert result.routing_diagnostics["physics_distillation_teacher"] == "hard_routing"
    assert result.routing_diagnostics["kan_selected_candidate_indices"] == [0]
    assert "delta_q =" in result.routing_diagnostics["kan_selected_formulas"][0]
    for split in ("train", "validation", "test"):
        trace = result.loss_traces[f"kan_standalone_{split}_observation_mse"]
        assert len(trace) == 1
        assert torch.isfinite(torch.tensor(trace)).all()


def test_gradio_values_map_to_the_correct_configuration_fields():
    values = [
        "2001-01-08", "2001-02-02", "2001-02-03", "2001-02-12",
        "2001-02-13", "2001-03-01", "distillation", "mlp", "fourier",
        3, 7, "4,5", 16, 0.002, 0.1, 75, 0.2, 0.3, 0.4,
        6, 7, 8, "12,10", 14, 2, 48, 9, 11, "statistical",
        0.5, 0.6, "cpu", "quantile", 0.9,
    ]
    config = _config_from_ui_values("/data", ["target"], "exclude_targets", values)
    assert config.model.mlp_widths == (12, 10)
    assert config.model.mamba_feed_forward_size == 48
    assert config.strategy.complex_teacher_epochs == 6
    assert config.hyperparameters.routing_weight == 0.5
    assert config.hyperparameters.compute_weight == 0.6
    assert config.extremes.mode == "statistical"
    assert config.extremes.value == 0.9


def test_gradio_values_include_kan_physics_controls():
    values = [
        "2001-01-08", "2001-02-02", "2001-02-03", "2001-02-12",
        "2001-02-13", "2001-03-01", "hard_routing", "mlp", "fourier",
        3, 7, "4", 16, 0.002, 0.1, 75, 0.2, 0.3, 0.4,
        6, 7, 8, "12,10", 14, 2, 48, 9, 11, "none",
        0.5, 0.6, "cpu", "quantile", 0.9, 0.99, 0.1,
        "distillation", "static_50_50", 4, 60, 5, 3, 3, 2, 0.002,
        "0.03,0.1", 0.07, False,
    ]

    config = _config_from_ui_values("/data", ["target"], "targets_only", values)

    assert config.physics.optimization_mode == "distillation"
    assert config.physics.distillation_teacher == "static_50_50"
    assert config.physics.kan_candidates == 4
    assert config.physics.robustness_noise_levels == (0.03, 0.1)
    assert config.physics.symbolic is False

    configured = _config_from_ui_values(
        "/data", ["target"], "targets_only",
        [
            *values,
            True, "/mnt/c/Downloads/estreams_dataset", ["estreams_ndvi"],
            True, "/mnt/c/Downloads/camels-ch-chem", ["chem_deposition_n_total"],
        ],
    )
    assert configured.estreams_root == "/mnt/c/Downloads/estreams_dataset"
    assert configured.estreams_features == ("estreams_ndvi",)
    assert configured.camels_chem_root == "/mnt/c/Downloads/camels-ch-chem"
    assert configured.camels_chem_features == ("chem_deposition_n_total",)

    event_setup = _config_from_ui_values(
        "/data", ["target"], "targets_only",
        [
            *values,
            False, "", ["estreams_ndvi"], False, "", ["chem_deposition_n_total"],
            "rolling", 14, ["low_flow_spell", "rapid_rise"],
            0.85, 0.15, 0.9, 5, 4, 2,
        ],
    )
    assert event_setup.forecast.mode == "rolling"
    assert event_setup.forecast.horizon_days == 14
    assert event_setup.extremes.event_types == ("low_flow_spell", "rapid_rise")
    assert event_setup.extremes.minimum_days == 5

    disabled = _config_from_ui_values(
        "/data", ["target"], "targets_only",
        [
            *values,
            False, "/invalid/path", ["estreams_ndvi"],
            False, "/invalid/chem", ["chem_deposition_n_total"],
        ],
    )
    assert disabled.estreams_root is None
    assert disabled.camels_chem_root is None


def test_established_basin_set_is_the_application_default():
    assert DEFAULT_BASINS == (
        "4009", "4008", "2488", "2109", "2011", "2126", "5001", "2247",
        "4022", "4023", "5009", "5010", "5016", "3014", "3015",
    )


def test_basin_map_payload_contains_geojson_and_selection_state():
    class Geometry:
        area = 2.0
        __geo_interface__ = {
            "type": "Polygon",
            "coordinates": (((7.0, 46.0), (8.0, 46.0), (8.0, 47.0), (7.0, 46.0)),),
        }

    catalog = CAMELSCHCatalog(".", [
        BasinCatalogRecord("a", "Alpha", eligible=True, geometry=Geometry()),
        BasinCatalogRecord("b", "Beta", eligible=False),
    ])
    payload = BasinMapComponent(catalog).payload(
        ["a"], active="a", training=["a"], interaction="toggle"
    )
    assert payload["selected"] == ["a"]
    assert payload["eligible"] == ["a"]
    assert payload["active"] == "a"
    assert payload["interaction"] == "toggle"
    assert payload["geojson"]["features"][0]["properties"] == {
        "basin_id": "a", "name": "Alpha", "eligible": True,
    }
    assert payload["geojson"]["features"][0]["geometry"]["type"] == "Polygon"


def test_catalog_joins_numeric_shapefile_gauge_ids(tmp_path, monkeypatch):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    observed = tmp_path / "timeseries" / "observation_based"
    simulated = tmp_path / "timeseries" / "simulation_based"
    shapes = tmp_path / "catchment_delineations"
    observed.mkdir(parents=True); simulated.mkdir(parents=True); shapes.mkdir()
    (observed / "CAMELS_CH_obs_based_4009.csv").touch()
    (simulated / "CAMELS_CH_sim_based_4009.csv").touch()
    shape_path = shapes / "CAMELS_CH_catchments.shp"
    shape_path.touch()
    frame = geopandas.GeoDataFrame(
        {"gauge_id": [4009.0]},
        geometry=[Polygon([
            (2_600_000, 1_200_000), (2_601_000, 1_200_000),
            (2_601_000, 1_201_000), (2_600_000, 1_200_000),
        ])],
        crs="EPSG:2056",
    )
    monkeypatch.setattr(geopandas, "read_file", lambda path: frame)

    record = CAMELSCHCatalog(tmp_path).get("4009")
    assert record.geometry is not None
    minx, miny, maxx, maxy = record.geometry.bounds
    assert 5 < minx < maxx < 11
    assert 45 < miny < maxy < 48


def test_regional_catalog_does_not_require_optional_annual_landcover(tmp_path):
    observed = tmp_path / "timeseries" / "observation_based"
    simulated = tmp_path / "timeseries" / "simulation_based"
    observed.mkdir(parents=True); simulated.mkdir(parents=True)
    with (observed / "CAMELS_CH_obs_based_4009.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["date", "discharge_spec(mm/d)"])
        writer.writeheader()
        for index in range(5):
            writer.writerow({"date": f"2001/01/{index + 1:02d}", "discharge_spec(mm/d)": 1 + index})
    with (simulated / "CAMELS_CH_sim_based_4009.csv").open("w", newline="") as target:
        fields = ["date", "precipitation_sim(mm/d)", "temperature_sim(degC)", "pet_sim(mm/d)"]
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader()
        for index in range(5):
            writer.writerow({"date": f"2001-01-{index + 1:02d}",
                "precipitation_sim(mm/d)": 2, "temperature_sim(degC)": 5,
                "pet_sim(mm/d)": .5})
    catalog = CAMELSCHCatalog(tmp_path)
    series = catalog.load_series("4009", date(2001, 1, 1), date(2001, 1, 5))
    assert len(series.dates) == 5
    assert series.forcing_names == ("precipitation", "temperature", "pet")
