"""Moving-obstacle controllers.

Rules shared by all controllers (also listed in docs/methodology.md):
  * Obstacles move only on update steps, which happen every `change_interval` agent steps,
    after the agent has moved.
  * An obstacle never enters a wall, the goal, the agent's current cell, or another
    obstacle's cell. If its chosen move is invalid it waits for that update.
  * `update` returns the cells whose occupancy changed. D* Lite (Stage 5) uses this
    list to know exactly which edges to repair.
"""

from __future__ import annotations

import numpy as np

from pathrl.environment.dynamics import ACTIONS_4, Cell
from pathrl.environment.map_generator import GridMap


class StaticObstacles:
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.positions = np.zeros((0, 2), dtype=np.int64)

    @property
    def count(self) -> int:
        return len(self.positions)

    def reset(self) -> None:
        self.positions = np.array(self.grid_map.dynamic_init, dtype=np.int64).reshape(-1, 2)

    def update(self, occupied: np.ndarray, agent_pos: Cell, rng: np.random.Generator) -> list[Cell]:
        return []

    def _can_enter(self, cell: Cell, occupied: np.ndarray, agent_pos: Cell) -> bool:
        n = occupied.shape[0]
        r, c = cell
        return (0 <= r < n and 0 <= c < n and not occupied[r, c]
                and cell != agent_pos and cell != self.grid_map.goal)

    def _move(self, i: int, target: Cell, occupied: np.ndarray, before: dict) -> None:
        source = (int(self.positions[i, 0]), int(self.positions[i, 1]))
        for cell in (source, target):
            before.setdefault(cell, bool(occupied[cell]))
        occupied[source] = False
        occupied[target] = True
        self.positions[i] = target

    @staticmethod
    def _net_changes(occupied: np.ndarray, before: dict) -> list[Cell]:
        # A cell vacated by one obstacle and filled by another in the same update is not a change.
        return sorted(cell for cell, was in before.items() if was != bool(occupied[cell]))


class RandomWalkObstacles(StaticObstacles):
    """Each update, every obstacle picks uniformly from {stay, up, down, left, right}."""

    def update(self, occupied, agent_pos, rng):
        # One draw per obstacle on every update, so the same seed always gives the same
        # proposals regardless of where the agent is (common random numbers).
        choices = rng.integers(5, size=self.count)
        before: dict[Cell, bool] = {}
        for i, choice in enumerate(choices):
            if choice == 0:
                continue
            dr, dc = ACTIONS_4[choice - 1]
            target = (int(self.positions[i, 0]) + dr, int(self.positions[i, 1]) + dc)
            if self._can_enter(target, occupied, agent_pos):
                self._move(i, target, occupied, before)
        return self._net_changes(occupied, before)


class OscillatingObstacles(StaticObstacles):
    """Each obstacle slides one cell along its track per update and reverses at the ends.

    This is deterministic: a corridor cell on the shortest path is blocked and freed
    on a fixed schedule. It waits only if the agent stands on its next cell.
    """

    def reset(self):
        super().reset()
        self.index = np.array(self.grid_map.track_start, dtype=np.int64)
        self.direction = np.ones(self.count, dtype=np.int64)

    def update(self, occupied, agent_pos, rng):
        before: dict[Cell, bool] = {}
        for i, track in enumerate(self.grid_map.tracks):
            nxt = self.index[i] + self.direction[i]
            if not 0 <= nxt < len(track):
                self.direction[i] *= -1
                nxt = self.index[i] + self.direction[i]
            target = (int(track[nxt, 0]), int(track[nxt, 1]))
            if self._can_enter(target, occupied, agent_pos):
                self._move(i, target, occupied, before)
                self.index[i] = nxt
        return self._net_changes(occupied, before)


def make_obstacles(grid_map: GridMap) -> StaticObstacles:
    controllers = {
        "static": StaticObstacles,
        "random_walk": RandomWalkObstacles,
        "oscillating_corridor": OscillatingObstacles,
    }
    return controllers[grid_map.dynamics](grid_map)
