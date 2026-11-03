"""GridWorld: the single environment shared by every planner and learner.

One step works like this:
  1. Slip: with probability `slip_prob` the chosen action is replaced by a perpendicular one.
  2. Move: the agent moves, or collides and stays put (rules in `dynamics.attempt_move`).
  3. Reward: one outcome per step (goal / collision / move), see `rewards.py`.
  4. Obstacles: if this step is a multiple of `change_interval`, moving obstacles update.
  5. Observe: global (3 x N x N) or local (3 x V x V) tensor.

Randomness has two independent streams, one for slips and one for obstacle moves.
Both are derived from the `seed` passed to `reset`. The map itself is fixed when
the environment is built (or replaced via `reset(options={"map": ...})`), so
seeding an episode never silently changes the map.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pathrl.environment.config import EnvConfig
from pathrl.environment.dynamics import Cell, action_deltas, attempt_move, move_cost, sample_delta
from pathrl.environment.map_generator import GridMap, generate_map
from pathrl.environment.obstacles import make_obstacles
from pathrl.environment.rewards import Outcome, compute_reward

# RGB colours (Okabe-Ito colour-blind-safe palette) used by render("rgb_array")
_PALETTE = np.array(
    [
        [255, 255, 255],  # 0 free
        [43, 43, 43],     # 1 wall
        [230, 159, 0],    # 2 moving obstacle
        [0, 158, 115],    # 3 start
        [213, 94, 0],     # 4 goal
        [0, 114, 178],    # 5 agent
    ],
    dtype=np.uint8,
)


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        config: EnvConfig | None = None,
        grid_map: GridMap | None = None,
        map_seed: int = 0,
        render_mode: str | None = None,
    ):
        self.config = config or EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode

        self._deltas = action_deltas(self.config.connectivity)
        self.action_space = spaces.Discrete(len(self._deltas))
        side = self.config.grid_size if self.config.observation == "global" else self.config.local_view
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3, side, side), dtype=np.float32)

        self._slip_rng: np.random.Generator | None = None
        self._obstacle_rng: np.random.Generator | None = None
        self._set_map(grid_map if grid_map is not None else generate_map(self.config, np.random.default_rng(map_seed)))

    # ------------------------------------------------------------------ setup

    def _set_map(self, grid_map: GridMap) -> None:
        if grid_map.size != self.config.grid_size:
            raise ValueError(f"map is {grid_map.size}x{grid_map.size} but config.grid_size={self.config.grid_size}")
        if grid_map.dynamics != self.config.dynamics:
            raise ValueError(f"map dynamics {grid_map.dynamics!r} != config dynamics {self.config.dynamics!r}")
        self.grid_map = grid_map
        self.obstacles = make_obstacles(grid_map)
        self._reset_state()

    def _reset_state(self) -> None:
        self.obstacles.reset()
        self._occupied = self.grid_map.walls.copy()
        self._occupied[tuple(self.obstacles.positions.T)] = True
        self.agent_pos: Cell = self.grid_map.start
        self.steps = 0

    @property
    def occupancy(self) -> np.ndarray:
        """Read-only view of walls + moving obstacles (True = blocked) at the current step."""
        view = self._occupied.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None or self._slip_rng is None:
            entropy = seed if seed is not None else int(self.np_random.integers(2**63))
            slip_ss, obstacle_ss = np.random.SeedSequence(entropy).spawn(2)
            self._slip_rng = np.random.default_rng(slip_ss)
            self._obstacle_rng = np.random.default_rng(obstacle_ss)
        if options and "map" in options:
            self._set_map(options["map"])
        else:
            self._reset_state()
        return self._observe(), self._info(Outcome.MOVED, False, [])

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")
        cfg = self.config

        delta, slipped = sample_delta(self._deltas, int(action), cfg.slip_prob, self._slip_rng)
        self.agent_pos, collided = attempt_move(self.agent_pos, delta, self._occupied, cfg.allow_corner_cutting)
        self.steps += 1

        if self.agent_pos == self.grid_map.goal:
            outcome = Outcome.REACHED_GOAL
        elif collided:
            outcome = Outcome.COLLIDED
        else:
            outcome = Outcome.MOVED
        reward = compute_reward(outcome, cfg.rewards, move_cost(delta, cfg.diagonal_cost))

        terminated = outcome is Outcome.REACHED_GOAL or (collided and cfg.terminate_on_collision)
        truncated = not terminated and self.steps >= cfg.max_steps

        changed: list[Cell] = []
        if not terminated and self.obstacles.count and cfg.change_interval and self.steps % cfg.change_interval == 0:
            changed = self.obstacles.update(self._occupied, self.agent_pos, self._obstacle_rng)

        return self._observe(), float(reward), terminated, truncated, self._info(outcome, slipped, changed)

    # --------------------------------------------------------- observations

    def _observe(self) -> np.ndarray:
        if self.config.observation == "global":
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[0] = self._occupied
            obs[1][self.agent_pos] = 1.0
            obs[2][self.grid_map.goal] = 1.0
            return obs
        return self._observe_local()

    def _observe_local(self) -> np.ndarray:
        """Egocentric V x V window centred on the agent.

        Cells outside the grid appear as obstacles. If the goal is outside the window,
        it is marked on the window border in its direction (clipped offset), like a
        compass. So the local agent knows *where the goal is* but not the map beyond
        its view. This information assumption is documented in docs/methodology.md.
        """
        view = self.config.local_view
        k = view // 2
        r, c = self.agent_pos
        padded = np.pad(self._occupied, k, constant_values=True)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = padded[r : r + view, c : c + view]
        obs[1][k, k] = 1.0
        gr = int(np.clip(self.grid_map.goal[0] - r, -k, k))
        gc = int(np.clip(self.grid_map.goal[1] - c, -k, k))
        obs[2][k + gr, k + gc] = 1.0
        return obs

    def _info(self, outcome: Outcome, slipped: bool, changed: list[Cell]) -> dict[str, Any]:
        return {
            "agent_pos": self.agent_pos,
            "step": self.steps,
            "outcome": outcome.value,
            "collided": outcome is Outcome.COLLIDED,
            "slipped": slipped,
            "changed_cells": changed,
            "obstacle_positions": [tuple(map(int, p)) for p in self.obstacles.positions],
        }

    # ------------------------------------------------------------- rendering

    def _codes(self) -> np.ndarray:
        codes = self.grid_map.walls.astype(np.uint8)
        for r, c in self.obstacles.positions:
            codes[r, c] = 2
        codes[self.grid_map.start] = 3
        codes[self.grid_map.goal] = 4
        codes[self.agent_pos] = 5
        return codes

    def render(self):
        if self.render_mode == "ansi":
            chars = np.array(list(".#oSGA"))[self._codes()]
            return "\n".join(" ".join(row) for row in chars)
        if self.render_mode == "rgb_array":
            scale = max(1, 256 // self.config.grid_size)
            image = _PALETTE[self._codes()]
            return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        gym.logger.warn("render() called without a render_mode; pass render_mode='ansi' or 'rgb_array'.")
        return None
