"""Scenario sampling for training on many maps / conditions.

A sampler holds one or more environments (e.g. different change intervals or slip
probabilities) and an optional list of maps. Each training episode draws one environment
and one map. This is how DQN is trained on 1,000 maps (E5) and across dynamics (E3/E4).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, generate_map


class ScenarioSampler:
    def __init__(self, envs: Sequence[GridWorldEnv], maps: Sequence[GridMap] | None = None):
        if not envs:
            raise ValueError("need at least one environment")
        self.envs, self.maps = list(envs), list(maps) if maps else None

    def sample(self, rng: np.random.Generator) -> tuple[GridWorldEnv, dict | None]:
        env = self.envs[int(rng.integers(len(self.envs)))]
        if self.maps is None:
            return env, None
        m = self.maps[int(rng.integers(len(self.maps)))]
        return env, {"map": m.with_dynamics(env.config.dynamics)}


def make_map_set(cfg: EnvConfig, count: int, seed: int) -> list[GridMap]:
    """`count` maps, each from its own child seed. Train and test sets use different master seeds,
    so no map is shared between them (checked in tests)."""
    seeds = np.random.SeedSequence(seed).generate_state(count)
    return [generate_map(cfg, np.random.default_rng(int(s))) for s in seeds]
