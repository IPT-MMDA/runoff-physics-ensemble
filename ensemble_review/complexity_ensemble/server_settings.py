"""Server-owned dataset registration and storage configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class DatasetRegistration:
    data_root: Path
    estreams_root: Path | None = None
    camels_chem_root: Path | None = None
    hyd_responses_root: Path | None = None
    bafu_root: Path | None = None
    bafu_statistics_zip: Path | None = None
    bafu_return_levels: Path | None = None
    hanze_root: Path | None = None


@dataclass(frozen=True)
class ServerSettings:
    storage_dir: Path
    datasets: dict[str, DatasetRegistration] = field(default_factory=dict)
    results_dir: Path | None = None
    token: str | None = None
    cors_origins: tuple[str, ...] = ()

    @property
    def runs_dir(self):
        return self.storage_dir / "runs"

    @property
    def jobs_path(self):
        return self.storage_dir / "jobs.sqlite3"

    @classmethod
    def load(cls, path: str | Path, *, token: str | None = None):
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        unknown = set(raw) - {"storage_dir", "datasets", "results_dir", "cors_origins"}
        if unknown:
            raise ValueError(f"unknown server settings: {sorted(unknown)}")
        def local(value):
            p = Path(value).expanduser()
            return p.resolve() if p.is_absolute() else (path.parent / p).resolve()
        datasets = {}
        for key, value in raw.get("datasets", {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
                raise ValueError("dataset IDs must contain only letters, numbers, underscores, or hyphens")
            if set(value) - {"data_root", "estreams_root", "camels_chem_root", "hyd_responses_root", "bafu_root", "bafu_statistics_zip", "bafu_return_levels", "hanze_root"}:
                raise ValueError(f"unknown dataset settings for {key}")
            datasets[key] = DatasetRegistration(
                local(value["data_root"]),
                local(value["estreams_root"]) if value.get("estreams_root") else None,
                local(value["camels_chem_root"]) if value.get("camels_chem_root") else None,
                local(value["hyd_responses_root"]) if value.get("hyd_responses_root") else None,
                local(value["bafu_root"]) if value.get("bafu_root") else None,
                local(value["bafu_statistics_zip"]) if value.get("bafu_statistics_zip") else None,
                local(value["bafu_return_levels"]) if value.get("bafu_return_levels") else None,
                local(value["hanze_root"]) if value.get("hanze_root") else None,
            )
        return cls(local(raw.get("storage_dir", "artifacts/api")), datasets,
                   local(raw["results_dir"]) if raw.get("results_dir") else None,
                   token, tuple(raw.get("cors_origins", ())))
