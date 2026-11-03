"""Breadth-first search.

BFS expands cells in order of the number of moves from the start (level by level),
using a FIFO queue. It finds the path with the FEWEST MOVES. That equals the
cheapest path only when every move costs the same, which is true for the
4-connected grid. With the 8-connected ablation (diagonals cost sqrt 2) BFS can
return a costlier path, and we report that honestly instead of calling BFS optimal.

Time O(V + E), space O(V): every cell enters the queue at most once.
"""

from __future__ import annotations

from collections import deque

from pathrl.algorithms.base_planner import PlanResult, Planner, SearchStats


class BFS(Planner):
    name = "bfs"

    def plan(self, occupancy, start, goal) -> PlanResult:
        stats, expanded = SearchStats(), []
        if occupancy[goal]:
            return self.unreachable(stats, expanded)

        parent = {start: None}  # doubles as the "visited" set
        queue = deque([start])
        stats.pushes = 1

        while queue:
            stats.max_frontier = max(stats.max_frontier, len(queue))
            cell = queue.popleft()
            stats.pops += 1
            stats.expansions += 1
            if self.record_expanded:
                expanded.append(cell)
            if cell == goal:
                break
            for nbr, _ in self.graph.neighbors(cell, occupancy):
                stats.generated += 1
                if nbr not in parent:  # first visit is via a shortest (fewest-moves) route
                    parent[nbr] = cell
                    queue.append(nbr)
                    stats.pushes += 1

        stats.states_touched = len(parent)
        if goal not in parent:
            return self.unreachable(stats, expanded)
        path = self.reconstruct(parent, goal)
        return PlanResult(path, self.graph.path_cost(path, occupancy), stats, expanded)
