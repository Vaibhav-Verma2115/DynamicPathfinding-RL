"""Shared planner interface and the counters every search algorithm reports.

The counters are machine-independent: they measure the work an algorithm does,
whatever the speed of the computer. Report them next to wall-clock time, which
depends on hardware, Python version and background load.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pathrl.algorithms.grid_graph import GridGraph
from pathrl.environment.dynamics import Cell


@dataclass
class SearchStats:
    expansions: int = 0       # nodes removed from the frontier and expanded (neighbours examined)
    generated: int = 0        # neighbour edges examined during expansions
    pushes: int = 0           # insertions into the queue / heap
    pops: int = 0             # removals, including stale heap entries that are skipped
    max_frontier: int = 0     # largest queue / heap size (proxy for memory)
    states_touched: int = 0   # distinct states that received a cost/parent entry
    vertex_updates: int = 0   # D* Lite UpdateVertex calls (0 for non-incremental planners)

    def add(self, other: "SearchStats") -> None:
        """Accumulate another call's counters (max_frontier and states_touched take the maximum)."""
        for key, value in other.__dict__.items():
            if key in ("max_frontier", "states_touched"):
                setattr(self, key, max(getattr(self, key), value))
            else:
                setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PlanResult:
    path: list[Cell] | None   # start ... goal, or None if the goal is unreachable
    cost: float               # sum of edge costs (math.inf if unreachable)
    stats: SearchStats
    expanded: list[Cell] = field(default_factory=list)  # expansion order, for visualisation

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int:
        """Number of moves (path cells minus one)."""
        return len(self.path) - 1 if self.path else 0


class Planner(ABC):
    """Common interface: plan on the current occupancy grid, optionally react to changes.

    Stateless planners (BFS, Dijkstra, A*) ignore `update` and simply plan again from
    scratch. Incremental planners (D* Lite, Stage 5) override `update` to repair their
    previous search using the list of changed cells.
    """

    name = "planner"

    def __init__(self, graph: GridGraph | None = None, record_expanded: bool = False):
        self.graph = graph or GridGraph()
        self.record_expanded = record_expanded

    @abstractmethod
    def plan(self, occupancy: np.ndarray, start: Cell, goal: Cell) -> PlanResult:
        ...

    def update(self, changed_cells: list[Cell], occupancy: np.ndarray) -> None:
        """Called when the environment changes. Default: nothing to repair."""

    def reset(self) -> None:
        """Forget any state kept between calls."""

    @staticmethod
    def reconstruct(parent: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
        """Follow parent pointers from the goal back to the start, then reverse.

        The start is the only cell whose parent is None. Each cell's parent is the
        cell it was reached from on its best-known route.
        """
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    @staticmethod
    def unreachable(stats: SearchStats, expanded: list[Cell]) -> PlanResult:
        return PlanResult(None, math.inf, stats, expanded)
