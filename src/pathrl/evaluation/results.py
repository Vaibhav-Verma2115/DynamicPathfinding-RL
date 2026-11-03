"""Saving raw results with full provenance.

Each CSV gets a sibling `<name>.meta.json` holding the resolved configuration, the hardware
snapshot, the git commit and a timestamp, so every number can be traced to the code and
machine that produced it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from pathrl.utils.config import PROJECT_ROOT
from pathrl.utils.hardware import hardware_info


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def results_dir(cfg: dict, experiment: str) -> Path:
    path = PROJECT_ROOT / cfg["paths"]["results_dir"] / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(df: pd.DataFrame, cfg: dict, experiment: str, name: str, extra: dict | None = None) -> Path:
    out = results_dir(cfg, experiment) / f"{name}.csv"
    df.to_csv(out, index=False)
    meta = {"experiment": experiment, "file": out.name, "rows": len(df), "git_commit": git_commit(),
            "hardware": hardware_info(), "config": cfg, **(extra or {})}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def load_results(cfg: dict, experiment: str, name: str) -> pd.DataFrame | None:
    path = PROJECT_ROOT / cfg["paths"]["results_dir"] / experiment / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None
