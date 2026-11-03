"""Typed, validated environment configuration.

The YAML loader returns plain dicts. Converting them into frozen dataclasses
here means an invalid setting (e.g. slip_prob=1.5, an even local view) fails
when the environment is built, not halfway through a 30-seed experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

DYNAMICS = ("static", "random_walk", "oscillating_corridor")
PLACEMENTS = ("corners", "random")
OBSERVATIONS = ("global", "local")


@dataclass(frozen=True)
class RewardConfig:
    step: float = -1.0
    collision: float = -50.0
    goal: float = 100.0


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 10
    obstacle_density: float = 0.15
    placement: str = "corners"
    min_start_goal_distance: float = 0.5
    require_reachable: bool = True
    max_generation_attempts: int = 1000

    dynamics: str = "static"
    dynamic_obstacle_density: float = 0.02
    change_interval: int = 0
    oscillation_amplitude: int = 2

    slip_prob: float = 0.0
    connectivity: int = 4
    diagonal_cost: float = math.sqrt(2)
    allow_corner_cutting: bool = False

    observation: str = "global"
    local_view: int = 7
    max_steps_factor: float = 1.0
    terminate_on_collision: bool = False

    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        def check(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"Invalid env config: {message}")

        check(self.grid_size >= 2, "grid_size must be >= 2")
        check(0.0 <= self.obstacle_density < 1.0, "obstacle_density must be in [0, 1)")
        check(self.placement in PLACEMENTS, f"placement must be one of {PLACEMENTS}")
        check(0.0 <= self.min_start_goal_distance <= 2.0, "min_start_goal_distance must be in [0, 2]")
        check(self.max_generation_attempts >= 1, "max_generation_attempts must be >= 1")
        check(self.dynamics in DYNAMICS, f"dynamics must be one of {DYNAMICS}")
        check(0.0 <= self.dynamic_obstacle_density < 1.0, "dynamic_obstacle_density must be in [0, 1)")
        check(self.change_interval >= 0, "change_interval must be >= 0")
        check(
            self.dynamics == "static" or self.change_interval >= 1,
            "moving obstacles need change_interval >= 1",
        )
        check(self.oscillation_amplitude >= 1, "oscillation_amplitude must be >= 1")
        check(0.0 <= self.slip_prob < 1.0, "slip_prob must be in [0, 1)")
        check(self.connectivity in (4, 8), "connectivity must be 4 or 8")
        check(self.diagonal_cost > 0, "diagonal_cost must be positive")
        check(self.observation in OBSERVATIONS, f"observation must be one of {OBSERVATIONS}")
        check(self.local_view >= 3 and self.local_view % 2 == 1, "local_view must be odd and >= 3")
        check(self.max_steps_factor > 0, "max_steps_factor must be positive")

    @property
    def max_steps(self) -> int:
        return max(1, round(self.max_steps_factor * self.grid_size**2))

    @classmethod
    def from_config(cls, cfg: dict) -> "EnvConfig":
        """Build from the full config dict returned by `load_config` (uses 'env' and 'rewards')."""
        env = dict(cfg.get("env", {}))
        known = {f.name for f in fields(cls)} - {"rewards"}
        unknown = set(env) - known
        if unknown:
            raise KeyError(f"Unknown env config keys: {sorted(unknown)}")
        return cls(**env, rewards=RewardConfig(**cfg.get("rewards", {})))
