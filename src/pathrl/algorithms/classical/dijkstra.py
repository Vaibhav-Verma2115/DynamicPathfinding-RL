"""Dijkstra's algorithm with a binary heap (heapq).

g(n) is the cheapest known cost from the start to n. Each iteration removes the
frontier cell with the smallest g. Because edge costs are non-negative, nothing
found later can make that cell cheaper, so its g is final when it is popped
("settled"). We stop when the goal is popped.

heapq has no decrease-key operation. When a cheaper route to n is found, we push a
new entry and leave the old one in the heap. When an old ("stale") entry is popped
later, its g is larger than the recorded best, so we skip it. This is called lazy
deletion. The heap can hold up to E entries, so the time is O(E log E) = O(E log V),
and on a grid (E <= 8V) that is O(V log V).
"""

from __future__ import annotations

import heapq
import itertools
import math

from pathrl.algorithms.base_planner import PlanResult, Planner, SearchStats


class Dijkstra(Planner):
    name = "dijkstra"

    def plan(self, occupancy, start, goal) -> PlanResult:
        stats, expanded = SearchStats(), []
        if occupancy[goal]:
            return self.unreachable(stats, expanded)

        g = {start: 0.0}
        parent = {start: None}
        closed = set()
        tie = itertools.count()  # FIFO among equal costs, so results are deterministic
        heap = [(0.0, next(tie), start)]
        stats.pushes = 1

        while heap:
            stats.max_frontier = max(stats.max_frontier, len(heap))
            cost, _, cell = heapq.heappop(heap)
            stats.pops += 1
            if cell in closed:  # stale entry: a cheaper copy was already settled
                continue
            closed.add(cell)
            stats.expansions += 1
            if self.record_expanded:
                expanded.append(cell)
            if cell == goal:
                break
            for nbr, step in self.graph.neighbors(cell, occupancy):
                stats.generated += 1
                new_cost = cost + step
                if new_cost < g.get(nbr, math.inf):  # "relax" the edge
                    g[nbr] = new_cost
                    parent[nbr] = cell
                    heapq.heappush(heap, (new_cost, next(tie), nbr))
                    stats.pushes += 1

        stats.states_touched = len(g)
        if goal not in closed:
            return self.unreachable(stats, expanded)
        return PlanResult(self.reconstruct(parent, goal), g[goal], stats, expanded)
