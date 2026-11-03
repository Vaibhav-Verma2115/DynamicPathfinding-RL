# Methodology

This file records the experimental rules as they are implemented. Each section
names the code that enforces it.

## 1. Environment specification

### 1.1 Grid and coordinates
- An N × N grid with cells indexed `(row, col)`. Row 0 is the top, so UP decreases the row index.
- A `GridMap` (`environment/map_generator.py`) fixes the static walls, start, goal and
  initial moving-obstacle layout. It is immutable, and matched comparisons share the same object.

### 1.2 Map generation (`generate_map`)
1. Start and goal are placed first:
   - `corners`: start (0,0), goal (N−1,N−1).
   - `random`: uniform start, then a goal at Manhattan distance ≥ ⌈`min_start_goal_distance`·N⌉.
2. Exactly round(`obstacle_density`·N²) walls are sampled uniformly, never on start or goal.
3. If `require_reachable`, the map is rejected unless BFS finds a **4-connected** start→goal path
   through static walls. Every 4-path is also an 8-path, so one map set serves both connectivities.
4. Rejected maps are resampled from the same RNG stream, up to `max_generation_attempts` times.
   After that, `MapGenerationError` is raised. Generation never loops forever.
5. The same seed always produces the same map. This is tested.

Because of the rejection step, maps at density 0.30 are a *conditional* sample: random maps,
**given** that they are solvable. Report this when discussing difficulty.

### 1.3 Actions and movement (`environment/dynamics.py`)
| Connectivity | Actions | Step cost |
|---|---|---|
| 4 (primary) | UP, DOWN, LEFT, RIGHT | 1 |
| 8 (ablation) | + UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT | √2 (`diagonal_cost`) |

### 1.4 Collision rules (`attempt_move`)
A move **collides**, and the agent stays in its current cell, if any of these holds:
1. The target cell is outside the grid.
2. The target cell holds a static wall.
3. The target cell holds a moving obstacle (as of the start of the step).
4. (8-connected, `allow_corner_cutting: false`) The move is diagonal and either orthogonal cell it
   squeezes past is occupied.

By default a collision does not end the episode (`terminate_on_collision: false`).

### 1.5 Stochastic transitions (slip)
With probability `slip_prob` the executed move is one of the two moves perpendicular to the
intended one, each chosen with probability `slip_prob`/2. A slip never reverses the agent. The
slipped move then follows the same collision rules.

    P(intended) = 1 − p,   P(left of intended) = P(right of intended) = p / 2

Value Iteration (Stage 6) will read this distribution from the same function the environment
samples from (`action_outcomes` / `sample_delta`), so the planner's model and the real dynamics
cannot drift apart.

### 1.6 Rewards (`environment/rewards.py`)
Each step has exactly one outcome:

| Outcome | Reward |
|---|---|
| Reached goal | +100 (episode terminates) |
| Collision | −50 |
| Normal move | −1 × step cost |

The return of an L-step collision-free optimal path is therefore −(L−1) + 100.
Episodes are truncated after round(`max_steps_factor`·N²) steps.

### 1.7 Moving obstacles (`environment/obstacles.py`)
Order within one step: **agent moves → reward → obstacles update (if due) → observation.**

- Obstacles update when `step % change_interval == 0`.
- An obstacle never enters a wall, the goal, the agent's cell or another obstacle's cell. If its
  chosen move is invalid, it waits.
- `random_walk`: round(`dynamic_obstacle_density`·N²) obstacles (at least 1), placed uniformly on free
  cells. Each picks uniformly from {stay, up, down, left, right}.
- `oscillating_corridor`: obstacles are placed on cells of the static shortest path. Each slides
  along a straight track perpendicular to the path, up to `oscillation_amplitude` cells each side,
  and reverses at the ends. Tracks never overlap and have ≥ 2 cells, so every blocked path cell
  reopens periodically.
- **Reachability with moving obstacles:** guaranteed only for the static layer. Moving obstacles can
  block a route temporarily but never permanently.
- Each step reports `info["changed_cells"]`, the cells whose occupancy actually changed. This is
  the information incremental planners use for repair.

### 1.8 Randomness and seeds
- Map randomness comes from `map_seed`. Episode randomness comes from `reset(seed=...)`, which
  derives two independent streams: one for slips, one for obstacle moves.
- **Common random numbers:** every step draws a fixed amount of randomness, whether or not a slip
  happens and wherever the agent is. Two methods run with the same seed therefore face the same slip
  draws and the same obstacle proposals. Their environments diverge only where an algorithm's own
  choices change the outcome, e.g. the agent standing where an obstacle wanted to move. This makes
  paired comparisons less noisy.

### 1.9 Observations and information access
| Mode | Tensor | What the agent knows |
|---|---|---|
| `global` | 3 × N × N | Full current occupancy (walls + moving obstacles), own position, goal |
| `local` | 3 × V × V (V=7) | Occupancy within its window (outside the grid = blocked), goal location if visible, otherwise goal **direction** marked on the window border |

Channels: 0 = obstacles, 1 = agent, 2 = goal.

**Information-assumption disclosure:** classical planners (BFS, Dijkstra, A*, D* Lite) are given the
global occupancy map. Any comparison between a global planner and a `local` RL agent must be labelled
as using different information. Neither mode exposes *future* obstacle positions.

## 2. Convergence definitions (`analysis/convergence.py`)

Each algorithm family has its own definition. They are deliberately not interchangeable.

| Algorithm | Converged when | Reported as |
|---|---|---|
| Value Iteration | Bellman residual max\|V_{k+1} − V_k\| < θ (θ = 1e-6), over goal-connected cells | sweeps, seconds |
| Q-Learning, DQN, Double DQN | W = 3 consecutive greedy evaluations each with success ≥ 0.9, and mean-return spread ≤ 5% of its magnitude | training episodes (and env steps, train seconds) at the *confirming* evaluation |
| BFS / Dijkstra / A* | terminate exactly | not applicable |
| D* Lite | queue empty or start consistent with top key ≥ key(start) | repair work (expansions), not convergence |

- Greedy evaluations use no exploration and run every `eval_interval` training episodes on
  `eval_episodes` episodes. Their time is recorded separately from training time.
- Runs that never meet the criterion are reported as **non-convergent**, with their final success
  rate. They are never dropped, and never assigned the training budget as a fake convergence point.

## 3. Hardware and measurement protocol

- Classical planners, Value Iteration, Q-Learning and all **inference/action-selection benchmarks**
  run on the CPU (Apple Silicon, see the hardware JSON next to every result file).
- **DQN/Double DQN training** runs on the Apple GPU (PyTorch MPS backend). Profiling showed that
  95% of CPU training time is convolution forward/backward, and this CPU build lacks MKLDNN.
  MPS training was 4–5× faster in the Stage 7 benchmark.
  Consequences, disclosed with every training-cost result:
  - Training wall time is GPU time, and is also reported in machine-independent units (env steps,
    gradient steps).
  - Some MPS kernels are non-deterministic. Two training runs with the same seed can differ slightly.
- Trained checkpoints are loaded on the CPU for inference, so per-decision costs of DQN and the
  planners are measured on the same processor.
- Wall-clock times use `time.perf_counter`. tracemalloc measures Python-heap allocations only and
  misses most PyTorch tensor memory. DQN memory is reported as parameter + replay-buffer bytes.
