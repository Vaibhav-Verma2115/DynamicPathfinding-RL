"""E4 - Stochastic transitions.

Methods and their information assumptions (stated with every result):
  A* / D* Lite  know the map, assume deterministic moves, replan after every slip
  Value Iteration  knows the map AND the slip model, so it is the model-based optimum
  Q-Learning    trained model-free on the test map itself (per map, per slip level)
  DQN / DDQN    trained once on other maps with mixed slip levels (zero-shot here)

Per (grid size, slip, map) each method runs the same `episodes_per_map` episode seeds.
Path deviation = executed path cost - static optimal cost (successful episodes only).
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from pathrl.algorithms import make_planner
from pathrl.algorithms.model_based import ValueIteration
from pathrl.algorithms.reinforcement_learning import QLearning
from pathrl.environment import GridWorldEnv, generate_map
from pathrl.evaluation.navigation import run_planner_episode
from pathrl.evaluation.results import save_results
from pathrl.evaluation.rollout import run_policy_episode
from pathrl.experiments.common import env_config, flatten, parallel_map, run_seeds
from pathrl.experiments.e3_dynamic_obstacles import load_generalists
from pathrl.utils.seeding import derive_seeds, make_rng


def _summarise(rows: list[dict], optimal: float) -> dict:
    ok = [r for r in rows if r["success"]]
    return {
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_collisions": float(np.mean([r["collisions"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_path_deviation": float(np.mean([r["path_cost"] - optimal for r in ok])) if ok else np.nan,
        "episodes": len(rows),
    }


def _task(args) -> list[dict]:
    cfg, n, slip, map_index, map_seed = args
    e = cfg["e4"]
    ec = env_config(cfg, grid_size=n, slip_prob=slip)
    m = generate_map(ec, make_rng(map_seed))
    env = GridWorldEnv(ec, grid_map=m)
    optimal = make_planner("dijkstra", ec).plan(m.walls, m.start, m.goal).cost
    episode_seeds = derive_seeds(map_seed, e["episodes_per_map"])
    base = {"grid_size": n, "slip_prob": slip, "map_index": map_index, "map_seed": map_seed, "static_optimal_cost": optimal}
    out = []

    for p in e["planners"]:
        rows = [run_planner_episode(env, make_planner(p["planner"], ec), seed=s) for s in episode_seeds]
        out.append({**base, "method": p["label"], "information": "map, deterministic model",
                    "prep_time_s": 0.0, "replans_per_episode": float(np.mean([r["replans"] for r in rows])),
                    **_summarise(rows, optimal)})

    vi = ValueIteration(ec, gamma=cfg["rl"]["gamma"], theta=cfg["rl"]["value_iteration"]["theta"]).solve(m.walls, m.goal)
    rows = [run_policy_episode(env, vi, seed=s) for s in episode_seeds]
    out.append({**base, "method": "Value Iteration", "information": "map + slip model",
                "prep_time_s": vi.solve_time_s, "sweeps": vi.sweeps, **_summarise(rows, optimal)})

    q = cfg["rl"]["q_learning"]
    agent = QLearning(n, alpha=q["alpha"], gamma=cfg["rl"]["gamma"], epsilon_start=q["epsilon_start"],
                      epsilon_end=q["epsilon_end"], epsilon_decay_fraction=q["epsilon_decay_fraction"], seed=map_seed)
    t0 = time.perf_counter()
    log = agent.train(GridWorldEnv(ec, grid_map=m), e["q_learning_episodes"], seed=map_seed,
                      eval_interval=e["q_learning_episodes"], eval_episodes=1)
    rows = [run_policy_episode(env, agent, seed=s) for s in episode_seeds]
    out.append({**base, "method": "Q-Learning", "information": "experience on this map",
                "prep_time_s": log.train_time_s, "train_env_steps": log.env_steps, **_summarise(rows, optimal)})

    if n == e["rl_grid_size"]:
        rl_env = GridWorldEnv(replace(ec, observation=e["rl_observation"]), grid_map=m)
        for (algo, t_seed), policy in load_generalists(cfg, n, e["rl_algorithms"], e["rl_training_seeds"],
                                                        e["rl_observation"]).items():
            rows = [run_policy_episode(rl_env, policy, seed=s) for s in episode_seeds]
            out.append({**base, "method": "DQN" if algo == "dqn" else "Double DQN", "train_seed": t_seed,
                        "information": f"experience on other maps (zero-shot), {e['rl_observation']} view",
                        "prep_time_s": 0.0,
                        **_summarise(rows, optimal)})
    return out


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    e = cfg["e4"]
    seeds = run_seeds(cfg, offset=e["test_map_offset"])
    tasks = [(cfg, n, slip, i, s) for n in e["grid_sizes"] for slip in e["slip_probs"] for i, s in enumerate(seeds)]
    df = pd.DataFrame(flatten(parallel_map(_task, tasks, workers, desc="E4")))
    save_results(df, cfg, "e4_stochastic", "e4_raw", extra={"workers": workers})
    return df
