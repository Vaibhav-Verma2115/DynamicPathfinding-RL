# Complexity analysis

Notation: V = number of free cells (≤ N²). E = number of legal moves (≤ 4V for 4-connected,
≤ 8V for 8-connected). On our grids E = Θ(V), which collapses several textbook bounds.
Measured counters are defined in `algorithms/base_planner.py::SearchStats`.

These are **theoretical** bounds. Measured behaviour is reported only from experiments
(see RESULTS.md), and the two must not be confused.

## BFS (`algorithms/classical/bfs.py`)
- Each cell enters the FIFO queue at most once, and each edge is examined at most once per endpoint.
- Time **O(V + E)** = O(V) on a grid. Space **O(V)** for the parent map plus the queue.
- Optimal for path *length in moves*. That equals cost-optimal only when all moves cost the same
  (true for 4-connected, false for the 8-connected ablation with √2 diagonals).

## Dijkstra (`algorithms/classical/dijkstra.py`), binary heap, lazy deletion
- A cell can be pushed once per improving relaxation, so the heap holds at most O(E) entries.
- Each push/pop costs O(log(heap size)) = O(log E) = O(log V), because E ≤ V².
- Time **O((V + E) log V)** = O(V log V) on a grid. Space **O(V + E)** = O(V).
- On a 4-connected grid with unit costs, a cell's first relaxation is already optimal, so
  pushes = expansions (observed in the Stage 4 demo). With √2 diagonals, re-pushes can occur.

## A* (`algorithms/classical/astar.py`)
- Worst case equals Dijkstra: **O((V + E) log V)**. If h gives no guidance (e.g. h = 0, or walls
  force the search away from the goal), A* expands as much as Dijkstra.
- With a consistent heuristic, A* expands every node with f(n) < C* and some nodes with f(n) = C*,
  where C* is the optimal cost. A better heuristic (larger without overestimating) shrinks that set.
  So the practical speed-up depends on the heuristic, the map and the tie-breaking rule, and
  A* is **not** guaranteed to be faster than Dijkstra on every instance.
- Weighted A* (f = g + w·h, w > 1) finds a path of cost ≤ w · C*.

| Heuristic | Graph | Admissible | Consistent |
|---|---|---|---|
| h = 0 | any | yes | yes (A* = Dijkstra) |
| Manhattan | 4-connected | yes | yes |
| Manhattan | 8-connected | **no** (overestimates diagonals) | no |
| Octile | 8-connected | yes | yes |
| w · Manhattan, w = 1.5 | 4-connected | no | no; cost ≤ 1.5 C* |

Admissibility and consistency are checked on an open grid by
`tests/test_classical.py::test_heuristics_are_admissible_and_consistent`.

## Pending
D* Lite (Stage 5), Value Iteration and Q-Learning (Stage 6), DQN/Double DQN (Stage 7).
