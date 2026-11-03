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

## D* Lite (`algorithms/classical/dstar_lite.py`)
- Initial search: same worst case as A* (backward from the goal), O((V + E) log V) time, O(V) space.
  Unlike A*, its g/rhs tables and priority queue are *kept* between calls.
- Repair after k changed cells: each change triggers O(1) UpdateVertex calls (the 3×3 block).
  The work then depends on how many cells' cost-to-goal actually changes, not on k or V.
  - Best case (change far from the optimal route, or behind the agent): near-zero expansions.
  - Worst case (e.g. a wall closing near the goal, invalidating most g-values): can exceed a fresh
    search, because under-consistent cells are expanded twice (raised to ∞, then lowered).
  So D* Lite is not guaranteed to beat a fresh A* search on every change. E3 measures this.
- The k_m term avoids re-keying the queue (O(|U| log |U|)) every time the agent moves.
- Implementation note: keys are rounded to 9 decimals. With √2 costs, floating-point rounding
  made mathematically equal keys compare unequal and stopped the search too early (found by
  `tests/test_dstar_lite.py`).

## Value Iteration (`algorithms/model_based/value_iteration.py`)
- One synchronous sweep costs O(S · A · K): S states, A actions, K ≤ 3 slip outcomes. Vectorised with NumPy.
- The Bellman operator is a γ-contraction: ‖V_{k+1} − V_k‖∞ ≤ γ ‖V_k − V_{k−1}‖∞ (tested).
  Values must first propagate along the longest shortest path (roughly the path length in
  sweeps), then the residual shrinks geometrically. Sweeps to reach θ grow with path length
  and with log(1/θ)/log(1/γ).
- Only goal-connected cells are updated. Walled-off pockets would otherwise dominate the sweep count.

## Q-Learning (`algorithms/reinforcement_learning/q_learning.py`)
- O(A) time per update, O(S · A) memory (N² × 4 floats; 40,000 entries at 100 × 100).
- Sample complexity (episodes needed) has no simple closed form here. It depends on
  exploration, α, γ and map difficulty, so E6 measures it empirically.

## Pending
DQN/Double DQN (Stage 7).
