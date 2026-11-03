# DynamicPathfinding-RL

**Comparative Performance Analysis of Classical Pathfinding vs. Model-Free Deep Reinforcement Learning in Dynamic Environments**

A reproducible experimental framework that asks a narrow, testable question:

> **When does the cumulative cost of repeated classical path planning exceed the
> amortized cost of training once and then executing a learned RL policy?**

We do **not** assume RL is better. Each method is measured on identical maps,
seeds and dynamics, and the conditions under which each one wins are reported.

```
C_RL(H)       = C_training + H · C_inference
C_planning(H) = C_initial  + Σ_{i=1..H} C_repair(i)        crossover H*: C_RL(H*) = C_planning(H*)
```

H* is a *computational* break-even point. Success rate and path quality are always reported beside it.

- **Results (numbers):** [docs/RESULTS.md](docs/RESULTS.md), generated from the CSVs
- **Report (interpretation, limitations):** [docs/report.md](docs/report.md)
- **Methodology / rules / information assumptions:** [docs/methodology.md](docs/methodology.md)
- **Architecture:** [docs/architecture.md](docs/architecture.md) · **Complexity:** [docs/complexity_analysis.md](docs/complexity_analysis.md)

## Status

| Stage | Module | Status |
|---|---|---|
| 1 | Project setup, config, seeding, hardware disclosure | ✅ |
| 2–3 | GridWorld (static / random-walk / oscillating obstacles, slip, 4/8-conn, global/local obs), Gymnasium | ✅ |
| 4 | BFS, Dijkstra, A* | ✅ |
| 5 | D* Lite (incremental repair) | ✅ |
| 6 | Value Iteration, Q-Learning, convergence definitions | ✅ |
| 7 | DQN, Double DQN (replay, target net, Huber, Adam, checkpoints) | ✅ |
| 8 | Evaluation harness, provenance, statistics | ✅ |
| 9 | Experiments E1–E6 | ✅ run (see RESULTS.md for what is pending) |
| 10 | Statistics, crossover analysis, figures | ✅ |
| 11 | Streamlit dashboard | ✅ |
| 12 | Documentation and report | ✅ |

## Setup

Requires Python 3.11. Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt -e .
```

Or with plain pip: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -e .`

`requirements-lock.txt` pins the exact versions used for the reported results.

## Usage

```bash
pytest                                   # full test suite (~160 tests)
pytest -m "not slow"                     # skip the end-to-end DQN learning tests

python main.py hardware                  # hardware/software snapshot
python main.py demo --config configs/dynamic.yaml --steps 80 --seed 4   # random policy in the env
python main.py plan --set env.grid_size=30 --seed 1                     # BFS/Dijkstra/A* on one map

# Experiments (raw CSV + .meta.json written to outputs/results/<experiment>/)
python main.py experiment e1             # static scaling            (serial: clean timings)
python main.py experiment e2             # heuristic quality
python main.py experiment train_dqn      # generalist DQN/Double DQN (GPU if available), needed by E3-E5
python main.py experiment e3             # dynamic obstacles + crossover data (serial)
python main.py experiment e4 --workers 8 # stochastic transitions
python main.py experiment e5             # zero-shot generalization
python main.py experiment e6 --workers 12  # convergence + learning-rate sensitivity

python main.py figures                   # summary tables (outputs/results/summary) + figures (outputs/figures)
python main.py report                    # regenerate docs/RESULTS.md from the summary tables

streamlit run src/pathrl/dashboard/app.py   # interactive dashboard
```

Any config value can be overridden, e.g. `python main.py experiment e1 --set num_seeds=5 'e1.grid_sizes=[10,20]'`.
Quote list values in zsh.

## Repository layout

```
main.py                     command-line entry point
configs/
  default.yaml              all base parameters (env, rewards, RL hyperparameters, convergence, paths)
  static.yaml, dynamic.yaml, stochastic.yaml   scenario presets
  experiments/*.yaml        per-experiment overrides (train_dqn, E1-E6)
src/pathrl/
  environment/              GridWorld env, map generation, dynamics (slip/collisions), rewards, obstacles
  algorithms/
    grid_graph.py           neighbour/cost function shared by all planners (built on env rules)
    base_planner.py         Planner interface + machine-independent counters
    base_policy.py          LearnedPolicy interface (act never learns) + TrainingLog
    classical/              BFS, Dijkstra, A*, D* Lite
    model_based/            Value Iteration
    reinforcement_learning/ Q-Learning, CNN Q-network, replay buffer, DQN, Double DQN
  evaluation/               profiling, rollouts, planner navigation, scenario sampling, results I/O
  analysis/                 statistics (t-CI, Wilson), convergence criteria, crossover
  visualization/            grid drawing, plotting helpers, demos
  experiments/              train_dqn, E1-E6, make_figures, make_report
  dashboard/app.py          Streamlit app
  utils/                    config loading, seeding, hardware disclosure
tests/                      pytest suite
outputs/results/            raw CSVs + provenance JSON (committed); summary/ tables
outputs/figures/            generated figures (committed)
outputs/models/             checkpoints (ignored by git; regenerate with train_dqn)
docs/                       methodology, architecture, complexity, RESULTS (generated), report
```

## Reproducibility rules

1. Every experimental parameter lives in `configs/`. Library code takes values from the config.
2. Every random component receives an explicit `np.random.Generator`. Seeds are derived with
   `SeedSequence.spawn`, and train/validation/test maps come from disjoint seed families.
3. Every result CSV is saved with its resolved config, a hardware snapshot and the git commit.
4. Training time, evaluation time, inference time and repair cost are reported separately.
5. Figures and RESULTS.md are generated from CSVs by code. Unrun experiments are marked *pending*.

## Information assumptions (read before comparing numbers)

- Classical planners and Value Iteration receive the **global** occupancy map. VI also knows the slip model.
- The trained DQN / Double DQN agents use a **local 7×7 egocentric view plus goal direction**
  (chosen by validation-set model selection, see methodology §4). They see *less* than the planners.
- DQN training ran on the Apple GPU (MPS). All inference and planning costs were measured on the CPU.
