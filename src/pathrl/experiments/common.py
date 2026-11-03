"""Helpers shared by the experiment scripts."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from typing import Callable, Iterable, Sequence

from tqdm import tqdm

from pathrl.environment import EnvConfig
from pathrl.utils.config import PROJECT_ROOT
from pathrl.utils.seeding import derive_seeds


def env_config(cfg: dict, **overrides) -> EnvConfig:
    """EnvConfig from the experiment config, with per-condition overrides (grid size, slip, ...)."""
    return replace(EnvConfig.from_config(cfg), **overrides)


def run_seeds(cfg: dict, n: int | None = None, offset: int = 0) -> list[int]:
    """Independent per-run seeds derived from the master seed. `offset` gives disjoint families
    (e.g. test maps vs. training maps) from the same master seed."""
    return derive_seeds(cfg["seed"] + offset, n if n is not None else cfg["num_seeds"])


def models_dir(cfg: dict):
    path = PROJECT_ROOT / cfg["paths"]["models_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _init_worker() -> None:
    # One BLAS/torch thread per worker so parallel workers do not oversubscribe the CPU.
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def parallel_map(fn: Callable, items: Sequence, workers: int = 1, desc: str = "") -> list:
    """Run fn over items, in order. workers <= 1 runs serially in this process (clean timings)."""
    if workers <= 1:
        _init_worker()
        return [fn(item) for item in tqdm(items, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))


def flatten(rows: Iterable[list[dict]]) -> list[dict]:
    return [r for chunk in rows for r in chunk]
