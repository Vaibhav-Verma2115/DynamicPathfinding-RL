"""E3 - Dynamic obstacles: the cost of keeping a plan up to date vs. a trained policy.

For every (grid size, change interval, test seed): one unseen map with random-walking
obstacles. Every method runs one episode on it with the SAME episode seed, so all methods
face the same obstacle proposals (common random numbers). Information assumption: planners
observe the full current occupancy grid; RL policies observe `rl_observation` (local 7 x 7 view).

Per episode we record success, path cost, collisions, and computation:
  planners: initial plan time + repair/replan time, expansions, replans
  RL:       inference time (forward passes on CPU). Training cost comes from train_dqn_summary.csv.
The crossover analysis (analysis/crossover.py) combines both.
"""

from __future__ import annotations

import pandas as pd
import torch

from dataclasses import replace

from pathrl.algorithms import make_planner
from pathrl.algorithms.reinforcement_learning import DQN
from pathrl.environment import GridWorldEnv, generate_map
from pathrl.evaluation.navigation import run_planner_episode
from pathrl.evaluation.results import save_results
from pathrl.evaluation.rollout import run_policy_episode
from pathrl.experiments.common import env_config, models_dir, run_seeds
from pathrl.utils.seeding import make_rng
from tqdm import tqdm


def load_generalists(cfg: dict, grid_size: int, algorithms, n_seeds: int, observation: str = "global") -> dict:
    agents = {}
    for algo in algorithms:
        for i in range(n_seeds):
            path = models_dir(cfg) / f"generalist_N{grid_size}_{observation}_{algo}_seed{i}.pt"
            if path.exists():
                agents[(algo, i)] = DQN.load(path, device="cpu")
    return agents


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    e = cfg["e3"]
    torch.set_num_threads(1)  # single-threaded inference, like the single-threaded planners
    seeds = run_seeds(cfg, offset=e["test_map_offset"])
    agents = load_generalists(cfg, e["rl_grid_size"], e["rl_algorithms"], e["rl_training_seeds"], e["rl_observation"])
    if not agents:
        print("WARNING: no DQN checkpoints found; E3 will contain classical planners only.")

    rows = []
    conditions = [(n, k) for n in e["classical_grid_sizes"] for k in e["change_intervals"]]
    for n, interval in tqdm(conditions, desc="E3"):
        ec = env_config(cfg, grid_size=n, change_interval=interval)
        for i, seed in enumerate(seeds):
            m = generate_map(ec, make_rng(seed))
            optimal = make_planner("dijkstra", ec).plan(m.walls, m.start, m.goal).cost
            env = GridWorldEnv(ec, grid_map=m)
            base = {"grid_size": n, "change_interval": interval, "seed_index": i, "map_seed": seed,
                    "n_moving_obstacles": len(m.dynamic_init), "static_optimal_cost": optimal}
            for p in e["planners"]:
                out = run_planner_episode(env, make_planner(p["planner"], ec), seed=seed, replan=p["replan"])
                rows.append({**base, "method": p["label"], "family": "classical", "train_seed": None,
                             "observation": "global",
                             "computation_time_s": out["total_planning_time_s"], **out})
            if n == e["rl_grid_size"]:
                # Same map, dynamics and episode seed; only the observation differs if the policy is local.
                rl_env = GridWorldEnv(replace(ec, observation=e["rl_observation"]), grid_map=m)
                for (algo, t_seed), agent in agents.items():
                    out = run_policy_episode(rl_env, agent, seed=seed)
                    label = "DQN" if algo == "dqn" else "Double DQN"
                    rows.append({**base, "method": label, "family": "rl", "train_seed": t_seed,
                                 "observation": e["rl_observation"],
                                 "computation_time_s": out["inference_time_s"], **out})

    df = pd.DataFrame(rows)
    df["path_cost_ratio"] = df["path_cost"] / df["static_optimal_cost"]
    save_results(df, cfg, "e3_dynamic_obstacles", "e3_raw",
                 extra={"workers": 1, "torch_threads": 1, "checkpoints": [f"{a}_seed{s}" for a, s in agents]})
    return df
