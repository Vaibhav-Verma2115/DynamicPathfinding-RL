# Architecture

This document grows with each stage. Only the design decisions that have been implemented are recorded here.

## Layers

| Layer | Package | Status |
|---|---|---|
| 1 Configuration manager | `pathrl.utils.config`, `configs/*.yaml`, `pathrl.environment.config` | done |
| 2 GridWorld environment | `pathrl.environment` | done |
| 3A Classical planning | `pathrl.algorithms.classical` | BFS, Dijkstra, A*, D* Lite done |
| 3B Learning / model-based | `pathrl.algorithms.model_based`, `pathrl.algorithms.reinforcement_learning` | pending |
| 4 Evaluation framework | `pathrl.evaluation` | done: profiling, rollout, navigation, scenarios, results |
| 5 Analysis & visualization | `pathrl.analysis`, `pathrl.visualization` | grid drawing done |
| 6 Dashboard & report | `pathrl.dashboard`, `docs/` | pending |

## Configuration
`default.yaml` → experiment file → dotted overrides (`env.grid_size=50`). Unknown keys raise
`KeyError`. The `env` and `rewards` sections are converted into frozen, validated dataclasses
(`EnvConfig`, `RewardConfig`), so invalid values fail when the environment is built.

## Environment module map

```
environment/
  config.py         EnvConfig / RewardConfig: typed, validated settings
  map_generator.py  GridMap (immutable scenario) + generate_map + BFS reachability
  dynamics.py       action tables, slip model, collision rules (shared with Value Iteration)
  rewards.py        Outcome enum + reward function
  obstacles.py      Static / RandomWalk / Oscillating obstacle controllers
  grid_world.py     GridWorldEnv (gymnasium.Env), registered as "pathrl/GridWorld-v0"
```

Dependencies only point downward: `grid_world` uses all the others, and `dynamics` and
`rewards` depend on nothing else in the environment package. Later, planners will import
`dynamics` (for neighbours and costs) without importing the Gym environment.

Behavioural rules (collisions, slip, obstacle movement, observations) are specified in
[methodology.md](methodology.md).

## Seeding
- `derive_seeds(master, n)` produces independent per-run seeds via `SeedSequence.spawn`.
- Map randomness (`map_seed`) is separate from episode randomness (`reset(seed)`), which is split
  into slip and obstacle streams.
- `set_global_seed(seed)` seeds Python, NumPy and PyTorch for network initialisation.

## Hardware disclosure
`pathrl.utils.hardware` records CPU, OS, library versions and CUDA/MPS availability alongside results.

## Planner interface (`algorithms/base_planner.py`)
```
Planner.plan(occupancy, start, goal) -> PlanResult(path, cost, stats, expanded)
Planner.update(changed_cells, occupancy)   # no-op for BFS/Dijkstra/A*; D* Lite repairs here
Planner.reset()
```
- All planners search `GridGraph` (`algorithms/grid_graph.py`), which is built on the environment's
  `attempt_move`. A test checks that planner neighbours equal the moves the environment allows.
- `SearchStats` holds machine-independent counters: expansions, generated, pushes, pops,
  max_frontier, states_touched.
- `make_planner(name, env_cfg, **kwargs)` builds a planner with the experiment's movement rules.

## Measurement (`evaluation/profiling.py`)
`measure(fn)` takes the median wall time of several runs (GC disabled), then runs once more
under tracemalloc for peak Python memory. These are separate runs because tracemalloc slows code down.

## Planned
`LearnedPolicy` (train / act) for RL in Stage 7, with `train()` kept separate from `act()`
so training is never timed as inference.
