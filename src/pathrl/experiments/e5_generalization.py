"""E5 - Generalization to unseen maps (zero-shot, no further training).

Each generalist checkpoint is evaluated greedily on
  * `n_test_maps` maps from a seed family never used for training (zero-shot), and
  * `n_train_maps_eval` of its own training maps.
The difference between the two is the generalization gap. A* runs on the same test maps
as a reference: these maps are solvable by construction, so a correct planner succeeds on all of them.

Rows are per (checkpoint, map). Success rates are summarised per training seed (the unit of
replication for RL) and also pooled with a Wilson interval.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import torch
from tqdm import tqdm

from pathrl.algorithms import make_planner
from pathrl.environment import GridWorldEnv
from pathrl.evaluation.navigation import run_planner_episode
from pathrl.evaluation.results import save_results
from pathrl.evaluation.rollout import run_policy_episode
from pathrl.evaluation.scenarios import make_map_set
from pathrl.experiments.common import env_config
from pathrl.experiments.e3_dynamic_obstacles import load_generalists
from pathrl.experiments.train_dqn import train_maps
from pathrl.utils.config import PROJECT_ROOT, load_config


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    e = cfg["e5"]
    torch.set_num_threads(1)
    ec = env_config(cfg)
    test_maps = make_map_set(ec, e["n_test_maps"], seed=cfg["seed"] + e["test_map_offset"])
    train_cfg = load_config(PROJECT_ROOT / e["train_config"])
    all_train = train_maps(train_cfg)
    seen = [m.with_dynamics("static") for m in all_train[: e["n_train_maps_eval"]]]
    key = lambda m: (m.walls.tobytes(), m.start, m.goal)
    overlap = len({key(m) for m in test_maps} & {key(m) for m in all_train})

    n = ec.grid_size
    agents = load_generalists(cfg, n, e["rl_algorithms"], e["rl_training_seeds"], e["rl_observation"])
    rl_cfg = replace(ec, observation=e["rl_observation"])
    rows = []
    for split, maps in (("test (unseen)", test_maps), ("train (seen)", seen)):
        for i, m in enumerate(tqdm(maps, desc=f"E5 {split}")):
            optimal = make_planner("dijkstra", ec).plan(m.walls, m.start, m.goal).cost
            base = {"split": split, "map_index": i, "static_optimal_cost": optimal}
            if split.startswith("test"):
                out = run_planner_episode(GridWorldEnv(ec, grid_map=m), make_planner("astar", ec), seed=i)
                rows.append({**base, "method": "A*", "train_seed": None, **out})
            env = GridWorldEnv(rl_cfg, grid_map=m)
            for (algo, t_seed), agent in agents.items():
                out = run_policy_episode(env, agent, seed=i)
                rows.append({**base, "method": "DQN" if algo == "dqn" else "Double DQN", "train_seed": t_seed, **out})
    df = pd.DataFrame(rows)
    df["optimality_ratio"] = df["path_cost"].where(df["success"]) / df["static_optimal_cost"]
    save_results(df, cfg, "e5_generalization", "e5_raw",
                 extra={"train_test_map_overlap": overlap, "observation": e["rl_observation"]})
    print(f"train/test map overlap: {overlap} (must be 0)")
    return df
