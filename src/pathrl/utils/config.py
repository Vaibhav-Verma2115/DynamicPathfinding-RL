"""YAML configuration loading with layered overrides.

Every experiment is described by `configs/default.yaml` plus an experiment file
that overrides only the keys it changes. Command-line style overrides such as
"env.grid_size=50" are applied last, so a single run is fully described by
(default, experiment file, overrides) and can be saved next to its results.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where nested keys of `override` replace those of `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like 'a.b=value', got {item!r}")
    dotted, raw = item.split("=", 1)
    # yaml.safe_load turns "50" into 50, "0.1" into 0.1, "true" into True.
    return dotted.split("."), yaml.safe_load(raw)


def load_config(
    experiment: str | Path | None = None,
    overrides: list[str] | None = None,
    default: str | Path = DEFAULT_CONFIG,
) -> dict:
    """Load default config, then an optional experiment file, then dotted overrides."""
    with open(default) as f:
        cfg = yaml.safe_load(f) or {}

    if experiment is not None:
        with open(experiment) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    for item in overrides or []:
        keys, value = _parse_override(item)
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section {'.'.join(keys)!r}")
            node = node[key]
        if keys[-1] not in node:
            # Typos like "env.grid_szie" should fail loudly, not silently add a key.
            raise KeyError(f"Unknown config key {'.'.join(keys)!r}")
        node[keys[-1]] = value

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the fully resolved config next to results so every run is reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
