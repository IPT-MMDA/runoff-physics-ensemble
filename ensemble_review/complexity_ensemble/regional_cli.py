"""Run one regional hydrology configuration from the command line."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from .configuration import config_from_mapping
from .services import execute_experiment
from .devices import DEVICE_CHOICES
from .result_registry import DEFAULT_RESULTS_DIRECTORY, parse_bool, save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Regional experiment JSON")
    parser.add_argument("--data-root", type=Path, help="Override data_root from the JSON file")
    parser.add_argument("--device", choices=DEVICE_CHOICES, help="Override the configured training device")
    parser.add_argument(
        "--estreams-root", type=Path,
        help="Read EStreams vegetation and snow from this extracted dataset directory",
    )
    parser.add_argument(
        "--camels-chem-root", type=Path,
        help="Read CAMELS-CH-Chem pressures from this extracted dataset directory",
    )
    parser.add_argument(
        "--save-results", type=parse_bool, default=False, metavar="true|false",
        help="Persist the completed run in the shared leaderboard registry (default: false)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY,
        help="Shared run-registry directory",
    )
    parser.add_argument("--api-url", help="Submit execution to an API worker")
    parser.add_argument("--dataset-id", help="Registered server dataset ID")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    executor = None
    if args.api_url:
        if not args.dataset_id:
            parser.error("--dataset-id is required with --api-url")
        from .remote import RemoteExecutor
        executor = RemoteExecutor(args.api_url, args.dataset_id, os.environ.get("RUNOFF_API_TOKEN"))
    try:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("the top-level configuration must be a JSON object")
        config = config_from_mapping(
            raw, data_root=args.data_root, estreams_root=args.estreams_root,
            camels_chem_root=args.camels_chem_root,
        )
        if args.device is not None:
            config = replace(config, hyperparameters=replace(config.hyperparameters, device=args.device))
        result = execute_experiment(
            config,
            lambda fraction, message: print(f"[{fraction:6.1%}] {message}", flush=True),
            executor=executor,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    summary = {
        "aggregate_metrics": result.aggregate_metrics,
        "per_basin_metrics": result.per_basin_metrics,
        "parameters": result.parameter_count,
        "runtime_seconds": result.runtime_seconds,
        "failures": result.failures,
        "cancelled": result.cancelled,
    }
    from .serialization import json_safe
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    if args.save_results and not result.cancelled:
        saved = save_run(result, args.results_dir)
        print(f"saved run {saved.run_id} to {saved.path}")
    else:
        print("result was not persisted (--save-results false)")


if __name__ == "__main__":
    main()
