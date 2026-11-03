"""The grid as a graph: which moves are legal and what they cost.

Every planner (BFS, Dijkstra, A*, D* Lite) and the evaluation code use this one
neighbour function. It is built on the environment's own `attempt_move`, so a
planner can never find a "path" the environment would refuse to execute.

On a grid, edges are symmetric: if a -> b is legal then b -> a is legal with the
same cost. The diagonal corner-cutting rule checks the same two corner cells in
both directions. D* Lite relies on this when it treats predecessors as neighbours.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np

from pathrl.environment.dynamics import Cell, action_deltas, attempt_move, move_cost


class GridGraph:
    def __init__(self, connectivity: int = 4, diagonal_cost: float = math.sqrt(2),
                 allow_corner_cutting: bool = False):
        self.connectivity = connectivity
        self.diagonal_cost = diagonal_cost
        self.allow_corner_cutting = allow_corner_cutting
        self.deltas = action_deltas(connectivity)
        self._costs = tuple(move_cost(d, diagonal_cost) for d in self.deltas)

    @classmethod
    def from_env_config(cls, cfg) -> "GridGraph":
        return cls(cfg.connectivity, cfg.diagonal_cost, cfg.allow_corner_cutting)

    def neighbors(self, cell: Cell, occupancy: np.ndarray) -> Iterator[tuple[Cell, float]]:
        """Yield (neighbour, edge_cost) for every legal move out of `cell`."""
        for delta, cost in zip(self.deltas, self._costs):
            target, collided = attempt_move(cell, delta, occupancy, self.allow_corner_cutting)
            if not collided:
                yield target, cost

    def edge_cost(self, a: Cell, b: Cell, occupancy: np.ndarray) -> float:
        """Cost of moving a -> b, or infinity if that move is not legal right now."""
        delta = (b[0] - a[0], b[1] - a[1])
        if delta not in self.deltas:
            return math.inf
        target, collided = attempt_move(a, delta, occupancy, self.allow_corner_cutting)
        return math.inf if collided else move_cost(delta, self.diagonal_cost)

    def path_cost(self, path: Sequence[Cell], occupancy: np.ndarray) -> float:
        """Total cost of a path, or infinity if any step is illegal (so it doubles as a validity check)."""
        return sum(self.edge_cost(a, b, occupancy) for a, b in zip(path, path[1:]))

    def is_valid_path(self, path: Sequence[Cell] | None, start: Cell, goal: Cell, occupancy: np.ndarray) -> bool:
        return (bool(path) and path[0] == start and path[-1] == goal
                and math.isfinite(self.path_cost(path, occupancy)))
