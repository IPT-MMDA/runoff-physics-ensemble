"""JSON-safe domain serialization, independent of any presentation framework."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from collections.abc import Mapping
import math


def json_safe(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def public_config(config):
    return {k: v for k, v in json_safe(config).items()
            if k not in {"data_root", "estreams_root", "camels_chem_root"}}
