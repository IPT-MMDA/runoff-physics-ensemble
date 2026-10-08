from datetime import date
import json

import pytest

from complexity_ensemble.regional import ExperimentResult, PredictionRecord
from complexity_ensemble.regional_cli import build_parser, config_from_mapping
from complexity_ensemble.result_registry import (
    available_values, discover_runs, filter_runs, leaderboard_rows, parse_bool, save_run,
)


def _config(approach="ood_fallback", epochs=100, physics="none"):
    return {
        "data_root": "/data/camels",
        "basins": {
            "target_basins": ["4009"], "training_scope": "all_basins",
            "eligible_basins": None,
        },
        "dates": {
            "train": {"start": "2010-01-01", "end": "2016-01-01"},
            "validation": {"start": "2016-01-02", "end": "2018-01-01"},
            "test": {"start": "2018-01-02", "end": "2020-01-01"},
        },
        "model": {"complex_expert": "mlp", "simple_expert": "rbf"},
        "strategy": {"approach": approach, "epochs": epochs},
        "hyperparameters": {
            "sequence_length": 30, "seeds": [0], "batch_size": 64,
            "learning_rate": 0.001, "training_noise": 0.0,
        },
        "physics": {"optimization_mode": physics, "distillation_teacher": "hard_routing"},
        "extremes": {"mode": "none"},
    }


def _result(approach="ood_fallback", epochs=100, physics="none"):
    return ExperimentResult(
        predictions=[PredictionRecord(0, "4009", date(2020, 1, 1), 1.0, 1.1, 0.0)],
        aggregate_metrics={"nse": 0.8, "kge": 0.7, "rmse_mm_day": 0.1,
                           "physics_error": 0.01},
        per_basin_metrics={"4009": {"nse": 0.8, "kge": 0.7,
                                      "rmse_mm_day": 0.1, "physics_error": 0.01}},
        routing_diagnostics={"ood_fraction": 0.2},
        loss_traces={"joint": [1.0, 0.5]},
        extreme_events=[], failures=[],
        resolved_config=_config(approach, epochs, physics),
        parameter_count=123, runtime_seconds=4.5,
        forecast_metrics={"1": {"samples": 4, "nse": 0.8, "kge": 0.7,
                                "rmse_mm_day": 0.1}},
    )


def test_registry_round_trip_and_leaderboard(tmp_path):
    saved = save_run(_result(), tmp_path)

    snapshot = discover_runs(tmp_path)

    assert not snapshot.warnings
    assert snapshot.runs == (saved,)
    assert (saved.path / "run.json").is_file()
    assert (saved.path / "predictions.csv").is_file()
    assert (saved.path / "regional_experiment.zip").is_file()
    assert saved.forecast_metrics["1"]["samples"] == 4
    assert leaderboard_rows(snapshot.runs)[0][2:7] == [
        "ood_fallback", "mlp", "rbf", "100", "none",
    ]


def test_registry_filters_only_existing_combinations(tmp_path):
    first = save_run(_result(), tmp_path)
    second = save_run(_result("soft_routing", 20, "balanced"), tmp_path)
    runs = discover_runs(tmp_path).runs

    assert filter_runs(runs, {
        "strategy.approach": "ood_fallback", "model.complex_expert": "mlp",
    }) == [first]
    assert available_values(
        runs, {"strategy.approach": "soft_routing"}, "strategy.epochs"
    ) == ["20"]
    assert filter_runs(runs, {"strategy.approach": "missing"}) == []
    assert {item.run_id for item in runs} == {first.run_id, second.run_id}


def test_discovery_skips_corrupt_entries(tmp_path):
    save_run(_result(), tmp_path)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "run.json").write_text("{not json", encoding="utf-8")

    snapshot = discover_runs(tmp_path)

    assert len(snapshot.runs) == 1
    assert len(snapshot.warnings) == 1


@pytest.mark.parametrize("raw, expected", [
    ("true", True), ("YES", True), ("1", True),
    ("false", False), ("off", False), ("0", False),
])
def test_explicit_boolean_parser(raw, expected):
    assert parse_bool(raw) is expected


def test_cli_defaults_to_not_saving_and_accepts_explicit_true(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")
    parser = build_parser()

    assert parser.parse_args(["--config", str(config)]).save_results is False
    assert parser.parse_args([
        "--config", str(config), "--save-results", "true",
    ]).save_results is True


def test_saved_config_can_be_reconstructed_for_cli():
    config = config_from_mapping(_config(), data_root="/override")

    assert str(config.data_root) == "/override"
    assert config.strategy.approach == "ood_fallback"
    assert config.strategy.epochs == 100
    assert config.model.complex_expert == "mlp"
    assert config.hyperparameters.seeds == (0,)
