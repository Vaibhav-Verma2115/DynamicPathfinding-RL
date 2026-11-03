# Architecture

This document grows with each stage. Only the design decisions that have been implemented are recorded here.

## Layers

| Layer | Package | Status |
|---|---|---|
| 1 Configuration manager | `pathrl.utils.config`, `configs/*.yaml`, `pathrl.environment.config` | done |
| 2 GridWorld environment | `pathrl.environment` | done |
| 3A Classical planning | `pathrl.algorithms.classical` | pending |
| 3B Learning / model-based | `pathrl.algorithms.model_based`, `pathrl.algorithms.reinforcement_learning` | pending |
| 4 Evaluation framework | `pathrl.evaluation` | pending |
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

## Planned interfaces
`Planner` / `LearnedPolicy` will give the evaluation harness one way to drive every method.
For example, `update(changed_cells)` lets D* Lite repair its search, and `train()` is kept
separate from `act()` so DQN training is never timed as inference.
