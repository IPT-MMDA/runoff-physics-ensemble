"""Server application context; registered datasets are resolved only here."""
from __future__ import annotations

from pathlib import Path

from .configuration import config_from_mapping
from .devices import resolve_device
from .regional import CAMELSCHCatalog
from .services import RunRepository, validate_optional_dataset


class Application:
    def __init__(self, settings):
        self.settings = settings
        self.runs = RunRepository(settings.runs_dir, settings.results_dir)

    def registration(self, dataset_id):
        try:
            return self.settings.datasets[dataset_id]
        except KeyError:
            raise KeyError("dataset not found") from None

    def catalog(self, dataset_id):
        return CAMELSCHCatalog(self.registration(dataset_id).data_root)

    def experiment_config(self, payload, *, check_data=True):
        registration = self.registration(payload["dataset_id"])
        cfg = payload["config"]
        for kind in ("estreams", "camels_chem", "hyd_responses"):
            if payload.get("use_" + kind) and getattr(registration, kind + "_root") is None:
                raise ValueError(f"optional {kind} dataset is not configured")
        config = config_from_mapping(
            cfg, data_root=registration.data_root,
            bafu_root=registration.bafu_root,
            bafu_statistics_zip=registration.bafu_statistics_zip,
            bafu_return_levels=registration.bafu_return_levels,
            estreams_root=registration.estreams_root if payload.get("use_estreams") else None,
            camels_chem_root=registration.camels_chem_root if payload.get("use_camels_chem") else None,
            hyd_responses_root=registration.hyd_responses_root if payload.get("use_hyd_responses") else None,
        )
        config.validate()
        resolve_device(config.hyperparameters.device)
        if payload.get("save_to_leaderboard") and self.settings.results_dir is None:
            raise ValueError("leaderboard saving is not configured")
        if check_data:
            catalog = self.catalog(payload["dataset_id"])
            eligible = set(catalog.eligible_ids)
            if config.benchmark.mode != "none":
                from dataclasses import replace
                from .bafu import basin_compatibility
                compatibility = basin_compatibility(catalog, registration.bafu_root)
                eligible = {basin for basin, modes in compatibility.items()
                            if modes[config.benchmark.mode]['eligible']}
                invalid = set(config.basins.target_basins) - eligible
                if invalid:
                    raise ValueError('Incompatible BAFU target stations: ' + '; '.join(
                        basin + ': ' + ', '.join(compatibility.get(basin, {}).get(
                            config.benchmark.mode, {}).get('reasons', ['Unknown station']))
                        for basin in sorted(invalid)))
                if config.basins.eligible_basins is None:
                    config = replace(config, basins=replace(config.basins, eligible_basins=tuple(sorted(eligible))))
            if set(config.basins.target_basins) - eligible:
                raise ValueError("target basins must be eligible basins in this dataset")
            if config.basins.eligible_basins is not None and set(config.basins.eligible_basins) - eligible:
                raise ValueError("eligible_basins contains unavailable basins")
            for kind in ("estreams", "camels_chem", "hyd_responses"):
                if payload.get("use_" + kind):
                    validate_optional_dataset(kind, getattr(registration, kind + "_root"))
        return config

    def dataset_for_run(self, run_id):
        run = self.runs.get(run_id)
        dataset_id = run.config.get("dataset_id")
        if dataset_id:
            self.registration(dataset_id)
            return dataset_id
        # Older Gradio/CLI runs have paths but no registered dataset identifier.
        root = Path(run.config.get("data_root", "")).resolve()
        for key, registration in self.settings.datasets.items():
            if registration.data_root.resolve() == root:
                return key
        raise ValueError("the run's source dataset is not registered on this server")

    def catalog_for_run(self, run_id):
        return self.catalog(self.dataset_for_run(run_id))
