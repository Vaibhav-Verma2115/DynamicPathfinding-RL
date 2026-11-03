"""Deterministic seeding.

Two rules keep experiments reproducible:

1. Never use the global `np.random.*` functions inside library code. Every
   component receives its own `np.random.Generator`, so adding a new random
   call in one module cannot shift the random numbers seen by another.
2. Independent runs get seeds derived from one master seed with
   `np.random.SeedSequence.spawn`, which guarantees statistically
   independent streams (unlike seed, seed+1, seed+2, which can correlate).
"""

from __future__ import annotations

import os
import random

import numpy as np


def derive_seeds(master_seed: int, n: int) -> list[int]:
    """Derive `n` independent 32-bit seeds from one master seed."""
    children = np.random.SeedSequence(master_seed).spawn(n)
    return [int(c.generate_state(1)[0]) for c in children]


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy generator to pass explicitly to maps, dynamics and agents."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and PyTorch global state (needed for network init and dropout).

    `deterministic_torch` asks PyTorch for deterministic kernels. This can be
    slower and only covers operations PyTorch supports; GPU results can still
    differ across hardware, which we disclose in the methodology.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch  # local import so classical-only runs do not pay torch import time

    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
