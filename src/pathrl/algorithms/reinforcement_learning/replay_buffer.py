"""Experience replay: a fixed-size circular buffer of transitions (s, a, r, s', terminated).

Why replay? Consecutive transitions are strongly correlated (the agent walks a path).
Training on random mini-batches from a large buffer breaks that correlation and reuses
each experience many times.

Observations are stored as uint8 (they are 0/1 grids), using 4x less memory than float32.
At 100,000 transitions of 3x10x10 that is 2 x 30 MB.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity  # overwrite the oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx].astype(np.float32), self.actions[idx], self.rewards[idx],
                self.next_obs[idx].astype(np.float32), self.terminated[idx])

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.obs, self.next_obs, self.actions, self.rewards, self.terminated))
