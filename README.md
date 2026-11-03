# DynamicPathfinding-RL

**Comparative Performance Analysis of Classical Pathfinding vs. Model-Free Deep Reinforcement Learning in Dynamic Environments**

A reproducible experimental framework that asks a narrow, testable question:

> **When does the cumulative cost of repeated classical path planning exceed the
> amortized cost of training once and then executing a learned RL policy?**

We do **not** assume RL is better. Each method is measured on identical maps,
seeds and dynamics, and the conditions under which each one wins are reported.
Over a horizon of H environment changes, the two costs are:

```
C_RL(H)       = C_training + H · C_inference
C_planning(H) = C_initial  + Σ_{i=1..H} C_repair(i)
```

The crossover point H* is where these two curves intersect, if they intersect at all.

## Status

| Stage | Module | Status |
|---|---|---|
| 1 | Project setup, config, seeding, hardware disclosure | ✅ done |
| 2 | GridWorld and visualization | ✅ done |
| 3 | Environment tests and Gymnasium integration | ✅ done (merged with Stage 2) |
| 4 | BFS, Dijkstra, A* | ✅ done |
| 5 | D* Lite | ✅ done |
| 6 | Value Iteration, Q-Learning | ✅ done |
| 7 | DQN, Double DQN | ✅ done |
| 8 | Evaluation and profiling framework | ⏳ pending |
| 9 | Experiments E1–E6 | ⏳ pending |
| 10 | Statistics and crossover analysis | ⏳ pending |
| 11 | Streamlit dashboard | ⏳ pending |
| 12 | Final documentation and report | ⏳ pending |

**No experimental results exist yet.** See [docs/RESULTS.md](docs/RESULTS.md).

## Setup

Requires Python 3.11. Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -e .
```

Or with plain pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

`requirements.txt` lists compatible version ranges. `requirements-lock.txt` pins the
exact versions used to produce the reported results. Install from it to reproduce them exactly:
`uv pip install -r requirements-lock.txt -e .`

## Verify the installation

```bash
pytest                              # run the test suite
python main.py hardware             # print the hardware/software snapshot
```

## Try the environment

```bash
# 10x10 random-walking obstacles, random policy, figure saved to outputs/figures/
python main.py demo --config configs/dynamic.yaml --steps 80 --seed 4

# 20x20 oscillating corridors, obstacles move every step
python main.py demo --config configs/dynamic.yaml \
    --set env.dynamics=oscillating_corridor env.change_interval=1 env.grid_size=20 --steps 150 --seed 2
```

In Python, use the Gymnasium API:

```python
import gymnasium as gym
from pathrl.environment import EnvConfig

env = gym.make("pathrl/GridWorld-v0", config=EnvConfig(grid_size=20, slip_prob=0.1), map_seed=0)
obs, info = env.reset(seed=0)            # obs: float32 (3, 20, 20) = obstacles, agent, goal
obs, reward, terminated, truncated, info = env.step(3)   # 3 = RIGHT
```

The environment rules (collisions, slip, rewards, moving obstacles, observations) are specified in
[docs/methodology.md](docs/methodology.md).

## Try the planners

```bash
# BFS, Dijkstra, A* (h=0, Manhattan, 1.5x Manhattan) on one map: counters, time, memory, figure
python main.py plan --set env.grid_size=30 --seed 1
```

```python
from pathrl.algorithms import make_planner
from pathrl.environment import EnvConfig, generate_map
from pathrl.utils.seeding import make_rng

cfg = EnvConfig(grid_size=30)
m = generate_map(cfg, make_rng(1))
result = make_planner("astar", cfg, heuristic="manhattan").plan(m.walls, m.start, m.goal)
print(result.cost, result.stats.expansions)
```

Complexity bounds are in [docs/complexity_analysis.md](docs/complexity_analysis.md).

## Repository layout

```
main.py                     command-line entry point (hardware, demo, ...)
configs/
  default.yaml              base parameters (grid, dynamics, rewards, paths)
  static.yaml, dynamic.yaml, stochastic.yaml   scenario presets
  experiments/*.yaml        per-experiment overrides (E1–E6)
src/pathrl/
  environment/              GridWorld, map generation, dynamics, rewards, obstacles
  algorithms/classical/     BFS, Dijkstra, A*, D* Lite
  algorithms/model_based/   Value Iteration
  algorithms/reinforcement_learning/  Q-Learning, DQN, Double DQN
  evaluation/               shared harness, metrics, profiling
  analysis/                 statistics (means, 95% CI), crossover analysis
  visualization/            grid rendering and publication figures
  experiments/              E1–E6 entry points
  dashboard/                Streamlit app
  utils/                    config loading, seeding, hardware disclosure
tests/                      pytest suite
outputs/results/            raw CSV results (committed)
outputs/figures/            generated figures (committed)
outputs/models/             model checkpoints (ignored by git)
docs/                       architecture.md, methodology.md, complexity_analysis.md, RESULTS.md
```

## Reproducibility rules

1. Every experimental parameter lives in `configs/`. Library code takes values from the config instead of hardcoding them.
2. Every random component receives an explicit `np.random.Generator`. Per-run seeds are
   derived from a master seed with `np.random.SeedSequence.spawn`.
3. Every result file is saved alongside its resolved config and a hardware snapshot.
4. Training time, inference time and adaptation cost are always reported separately.
5. Unrun experiments are marked *pending*. Results that were not measured are never shown.
