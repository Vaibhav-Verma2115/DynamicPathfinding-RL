"""A* search (Hart, Nilsson & Raphael, 1968).

Like Dijkstra, but the heap is ordered by f(n) = g(n) + w * h(n), where h(n)
estimates the remaining cost to the goal.

Key properties:
  * Admissible h (never overestimates): the first time the goal is popped, its path is optimal.
  * Consistent h (h(a) <= cost(a,b) + h(b)): additionally every node's g is final when
    popped, so a closed set never needs reopening. Manhattan (4-conn) and octile
    (8-conn) distances are both consistent on our grids.
  * h = 0 turns A* into Dijkstra.
  * w > 1 (weighted A*) is inadmissible: usually far fewer expansions, and the cost is
    guaranteed to be at most w times the optimum. E2 measures this trade-off.

Tie-breaking: among equal f, we prefer the smaller h (the node closer to the goal).
On open grids, many nodes share the same f, and this choice can change expansions
by an order of magnitude. It is part of the algorithm specification, so report it.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable

from pathrl.algorithms.base_planner import PlanResult, Planner, SearchStats
from pathrl.algorithms.grid_graph import GridGraph
from pathrl.environment.dynamics import Cell

Heuristic = Callable[[Cell, Cell], float]


def zero(a: Cell, b: Cell) -> float:
    return 0.0


def manhattan(a: Cell, b: Cell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def octile(a: Cell, b: Cell, diagonal_cost: float = math.sqrt(2)) -> float:
    """Exact distance on an empty 8-connected grid: diagonal moves for the shorter axis, straight for the rest."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (diagonal_cost - 1.0) * min(dr, dc)


HEURISTICS = {"zero": zero, "manhattan": manhattan, "octile": octile}


class AStar(Planner):
    name = "astar"

    def __init__(self, graph: GridGraph | None = None, heuristic: str = "auto", weight: float = 1.0,
                 record_expanded: bool = False):
        super().__init__(graph, record_expanded)
        if heuristic == "auto":
            heuristic = "manhattan" if self.graph.connectivity == 4 else "octile"
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic must be one of {sorted(HEURISTICS)} or 'auto'")
        if weight < 1.0:
            raise ValueError("weight must be >= 1")
        self.heuristic_name = heuristic
        self.weight = weight
        if heuristic == "octile":
            d = self.graph.diagonal_cost
            self._h: Heuristic = lambda a, b: octile(a, b, d)
        else:
            self._h = HEURISTICS[heuristic]

    @property
    def admissible(self) -> bool:
        if self.weight > 1.0:
            return False
        # Manhattan overestimates when cheap diagonals exist (2 straight moves vs one sqrt 2 move).
        return not (self.heuristic_name == "manhattan" and self.graph.connectivity == 8)

    def plan(self, occupancy, start, goal) -> PlanResult:
        stats, expanded = SearchStats(), []
        if occupancy[goal]:
            return self.unreachable(stats, expanded)

        h = lambda cell: self.weight * self._h(cell, goal)
        g = {start: 0.0}
        parent = {start: None}
        closed = set()
        tie = itertools.count()
        h0 = h(start)
        heap = [(h0, h0, next(tie), start)]  # (f, h, insertion order, cell)
        stats.pushes = 1

        while heap:
            stats.max_frontier = max(stats.max_frontier, len(heap))
            _, _, _, cell = heapq.heappop(heap)
            stats.pops += 1
            if cell in closed:
                continue
            closed.add(cell)
            stats.expansions += 1
            if self.record_expanded:
                expanded.append(cell)
            if cell == goal:
                break
            for nbr, step in self.graph.neighbors(cell, occupancy):
                stats.generated += 1
                if nbr in closed:  # consistent h: a closed node's g is already final
                    continue
                new_g = g[cell] + step
                if new_g < g.get(nbr, math.inf):
                    g[nbr] = new_g
                    parent[nbr] = cell
                    h_n = h(nbr)
                    heapq.heappush(heap, (new_g + h_n, h_n, next(tie), nbr))
                    stats.pushes += 1

        stats.states_touched = len(g)
        if goal not in closed:
            return self.unreachable(stats, expanded)
        return PlanResult(self.reconstruct(parent, goal), g[goal], stats, expanded)
