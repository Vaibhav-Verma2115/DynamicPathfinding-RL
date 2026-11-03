"""D* Lite (Koenig & Likhachev, 2002), optimized version with the k_m key modifier.

D* Lite searches BACKWARDS, from the goal towards the agent. So g(s) is the cost
from s to the goal, and the agent's moves only change which cell is "start".

Two values per cell:
    g(s)    current cost-to-goal estimate
    rhs(s)  one-step lookahead: min over neighbours s' of  c(s, s') + g(s'),  with rhs(goal) = 0

A cell is consistent when g(s) == rhs(s). It is
    over-consistent  (g > rhs) when a cheaper route has appeared, e.g. an obstacle vanished;
    under-consistent (g < rhs) when its route got more expensive, e.g. an obstacle appeared.
Only inconsistent cells sit in the priority queue. When the map changes, only cells whose
edges changed are re-examined (UpdateVertex). The repair then spreads only as far as the
change actually matters. This is what makes D* Lite incremental: g/rhs values far from
the change are kept from the previous search, not recomputed.

Priority key of s (lexicographic):  [min(g, rhs) + h(start, s) + k_m ,  min(g, rhs)]
When the agent moves, h(start, .) changes for every queued cell. Instead of re-keying the
whole queue, D* Lite adds the distance moved to k_m. Old keys then remain lower bounds, and
stale ones are fixed lazily when popped (the "k_old < k_new" branch).

Interface: call plan(occupancy, start, goal) whenever a path is needed, and update(changed_cells,
occupancy) when the environment reports changes. The first plan (or a new goal) initializes
the search. Every later plan only repairs it.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from pathrl.algorithms.base_planner import PlanResult, Planner, SearchStats
from pathrl.algorithms.classical.astar import make_heuristic
from pathrl.algorithms.grid_graph import GridGraph
from pathrl.environment.dynamics import Cell

INF = math.inf
KEY_DECIMALS = 9


class DStarLite(Planner):
    name = "dstar_lite"

    def __init__(self, graph: GridGraph | None = None, heuristic: str = "auto", record_expanded: bool = False):
        super().__init__(graph, record_expanded)
        self.heuristic_name, self._h = make_heuristic(self.graph, heuristic)
        self.reset()

    # ------------------------------------------------------------ public API

    def reset(self) -> None:
        self._occ: np.ndarray | None = None
        self._goal: Cell | None = None
        self._pending: set[Cell] = set()
        self.initializations = 0           # how many times the search started from scratch
        self.total_stats = SearchStats()   # counters accumulated over all plan() calls

    def update(self, changed_cells, occupancy) -> None:
        """Record cells whose occupancy changed. They are repaired on the next plan()."""
        if self._occ is not None:
            self._pending.update((int(r), int(c)) for r, c in changed_cells)

    def plan(self, occupancy: np.ndarray, start: Cell, goal: Cell) -> PlanResult:
        self._stats, self._expanded = SearchStats(), []
        if self._occ is None or goal != self._goal or occupancy.shape != self._occ.shape:
            self._initialize(occupancy, start, goal)
        else:
            self._start = start
            # Cells reported via update(), plus any difference from our copy of the map,
            # so plan() stays correct even if a caller forgets to call update().
            diff = {(int(r), int(c)) for r, c in np.argwhere(occupancy != self._occ)}
            changed = self._pending | diff
            self._pending = set()
            if changed:
                self._km += self._h(self._last, start)
                self._last = start
                for cell in changed:
                    self._occ[cell] = occupancy[cell]
                # A changed cell alters its own edges and, for diagonal moves, the corner rule of
                # edges between its neighbours, so every cell in the 3x3 block may need repair.
                affected = {(r + dr, c + dc) for r, c in changed for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                for u in sorted(affected):
                    if self._in_bounds(u):
                        self._update_vertex(u)

        self._compute_shortest_path()
        path = self._extract_path()
        stats = self._stats
        stats.states_touched = len(self._g.keys() | self._rhs.keys())
        self.total_stats.add(stats)
        if path is None:
            return self.unreachable(stats, self._expanded)
        return PlanResult(path, self.graph.path_cost(path, occupancy), stats, self._expanded)

    # ---------------------------------------------------------- algorithm

    def _initialize(self, occupancy, start, goal) -> None:
        self.initializations += 1
        self._occ = occupancy.copy()
        self._pending = set()
        self._start = self._last = start
        self._goal = goal
        self._km = 0.0
        self._g: dict[Cell, float] = {}
        self._rhs: dict[Cell, float] = {goal: 0.0}
        self._heap: list = []
        self._open: dict[Cell, tuple[float, float]] = {}
        self._tie = itertools.count()
        self._push(goal, self._key(goal))

    def _key(self, s: Cell) -> tuple[float, float]:
        # Keys are rounded because sums of sqrt(2) taken in different orders differ in the
        # last bits (e.g. 8.242640687119286 vs ...284). Without rounding, two mathematically
        # equal keys compare unequal and the termination test can stop the search too early.
        m = min(self._g.get(s, INF), self._rhs.get(s, INF))
        return (round(m + self._h(self._start, s) + self._km, KEY_DECIMALS), round(m, KEY_DECIMALS))

    def _push(self, s: Cell, key) -> None:
        self._open[s] = key
        heapq.heappush(self._heap, (key, next(self._tie), s))
        self._stats.pushes += 1
        self._stats.max_frontier = max(self._stats.max_frontier, len(self._heap))

    def _top(self):
        """Smallest valid (key, cell), discarding stale heap entries (lazy deletion)."""
        while self._heap:
            key, _, s = self._heap[0]
            if self._open.get(s) == key:
                return key, s
            heapq.heappop(self._heap)
            self._stats.pops += 1
        return None

    def _in_bounds(self, s: Cell) -> bool:
        n = self._occ.shape[0]
        return 0 <= s[0] < n and 0 <= s[1] < n

    def _neighbors(self, s: Cell):
        for dr, dc in self.graph.deltas:
            v = (s[0] + dr, s[1] + dc)
            if self._in_bounds(v):
                yield v

    def _cost(self, a: Cell, b: Cell) -> float:
        # Blocked cells have no usable edges in either direction, which keeps costs symmetric.
        if self._occ[a] or self._occ[b]:
            return INF
        return self.graph.edge_cost(a, b, self._occ)

    def _update_vertex(self, u: Cell) -> None:
        self._stats.vertex_updates += 1
        if u != self._goal:
            best = INF
            if not self._occ[u]:
                for v in self._neighbors(u):
                    self._stats.generated += 1
                    best = min(best, self._cost(u, v) + self._g.get(v, INF))
            if best == INF:
                self._rhs.pop(u, None)
            else:
                self._rhs[u] = best
        if self._g.get(u, INF) != self._rhs.get(u, INF):
            self._push(u, self._key(u))
        else:
            self._open.pop(u, None)

    def _compute_shortest_path(self) -> None:
        start = self._start
        while True:
            top = self._top()
            start_inconsistent = self._g.get(start, INF) != self._rhs.get(start, INF)
            if top is None or not (top[0] < self._key(start) or start_inconsistent):
                return
            k_old, u = top
            heapq.heappop(self._heap)
            self._stats.pops += 1
            del self._open[u]
            k_new = self._key(u)
            if k_old < k_new:  # key was stale because the agent moved: re-queue with the right key
                self._push(u, k_new)
                continue
            self._stats.expansions += 1
            if self.record_expanded:
                self._expanded.append(u)
            g_u, rhs_u = self._g.get(u, INF), self._rhs.get(u, INF)
            if g_u > rhs_u:  # over-consistent: lower g and propagate the improvement
                self._g[u] = rhs_u
                for s in self._neighbors(u):
                    self._update_vertex(s)
            else:  # under-consistent: invalidate g and let neighbours find new routes
                self._g.pop(u, None)
                self._update_vertex(u)
                for s in self._neighbors(u):
                    self._update_vertex(s)

    def _extract_path(self) -> list[Cell] | None:
        """Walk greedily from start, always to the neighbour minimising c(s, s') + g(s')."""
        if self._g.get(self._start, INF) == INF:
            return None
        path, cell = [self._start], self._start
        for _ in range(self._occ.size):
            if cell == self._goal:
                return path
            best, best_cost = None, INF
            for v in self._neighbors(cell):
                cost = self._cost(cell, v) + self._g.get(v, INF)
                if cost < best_cost:
                    best, best_cost = v, cost
            if best is None:
                return None
            path.append(best)
            cell = best
        return None  # defensive: a consistent search never loops
