import json
from pathlib import Path

import pytest

from complexity_ensemble.server_settings import ServerSettings


def test_server_settings_load_hydropower_path_relative_to_config(tmp_path):
    config = tmp_path / "api-server.json"
    config.write_text(json.dumps({
        "storage_dir": "artifacts/api",
        "datasets": {},
        "hydropower": {"enabled": True, "geopackage": "data/stations.gpkg", "layer": "custom_layer"},
    }), encoding="utf-8")
    settings = ServerSettings.load(config)
    assert settings.hydropower.enabled is True
    assert settings.hydropower.geopackage == (tmp_path / "data/stations.gpkg").resolve()
    assert settings.hydropower.layer == "custom_layer"


def test_server_settings_reject_unknown_hydropower_option(tmp_path):
    config = tmp_path / "api-server.json"
    config.write_text(json.dumps({"datasets": {}, "hydropower": {"bad_key": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown hydropower settings"):
        ServerSettings.load(config)
