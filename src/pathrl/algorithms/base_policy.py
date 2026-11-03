"""Shared interface for policies that choose one action per step (VI, Q-Learning, DQN).

Training and acting are separate methods. `act` never learns, so any time measured
around `act` is pure inference cost, and training cost is recorded separately in a
`TrainingLog`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingLog:
    episodes: list[dict] = field(default_factory=list)     # one row per training episode
    evaluations: list[dict] = field(default_factory=list)  # one row per greedy evaluation window
    train_time_s: float = 0.0   # wall time spent acting + learning (excludes evaluation)
    eval_time_s: float = 0.0    # wall time spent in greedy evaluation during training
    env_steps: int = 0
    gradient_steps: int = 0

    def episodes_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.episodes)

    def evaluations_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.evaluations)


class LearnedPolicy(ABC):
    name = "policy"

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        """Greedy action for the current observation. Must not change any learned parameters."""

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "LearnedPolicy":
        raise NotImplementedError


def linear_schedule(start: float, end: float, duration: float, t: float) -> float:
    """Linearly interpolate from `start` to `end` over `duration`, then hold `end`."""
    if duration <= 0:
        return end
    return end + (start - end) * max(0.0, 1.0 - t / duration)
