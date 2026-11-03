"""Crossover analysis: when does training once beat planning repeatedly?

Cost model over a deployment horizon of H episodes (or H decisions):

    C_planning(H) = H * c_plan                  c_plan = mean planning + repair time per episode
    C_RL(H)       = C_train + H * c_infer       c_infer = mean inference time per episode

C_planning includes the initial plan of each episode, because every episode starts on a new map.
The crossover horizon is where the two lines meet:

    H* = C_train / (c_plan - c_infer)     if c_plan > c_infer, otherwise never (H* = inf)

H* is a COMPUTATIONAL break-even point only. It says nothing about which method reaches the goal
more often, so success rate and path quality are always reported beside it. A cheaper but
less successful policy has not "won".
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch

from pathrl.algorithms.reinforcement_learning.network import QNetwork
from pathrl.analysis.statistics import mean_ci


def crossover_horizon(train_cost: float, plan_cost: float, infer_cost: float) -> float:
    gap = plan_cost - infer_cost
    return train_cost / gap if gap > 0 else math.inf


def crossover_table(e3: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """One row per (change interval, planner, RL algorithm) at the RL grid size."""
    rl_sizes = e3.loc[e3["family"] == "rl", "grid_size"].unique()
    rows = []
    for n in rl_sizes:
        sub = e3[e3["grid_size"] == n]
        for k, g in sub.groupby("change_interval"):
            for planner, gp in g[g["family"] == "classical"].groupby("method"):
                plan_ep = mean_ci(gp["computation_time_s"])
                plan_step = gp["computation_time_s"].sum() / gp["steps"].sum()
                for algo, gr in g[g["family"] == "rl"].groupby("method"):
                    algo_key = "dqn" if algo == "DQN" else "double_dqn"
                    train = mean_ci(training.loc[training["algorithm"] == algo_key, "train_time_s"])
                    infer_ep = mean_ci(gr["computation_time_s"])
                    infer_step = gr["computation_time_s"].sum() / gr["steps"].sum()
                    rows.append({
                        "grid_size": n, "change_interval": k, "planner": planner, "rl_method": algo,
                        "train_time_s": train["mean"], "train_time_ci_low": train["ci_low"],
                        "train_time_ci_high": train["ci_high"],
                        "plan_time_per_episode_s": plan_ep["mean"], "infer_time_per_episode_s": infer_ep["mean"],
                        "plan_time_per_step_s": plan_step, "infer_time_per_step_s": infer_step,
                        "crossover_episodes": crossover_horizon(train["mean"], plan_ep["mean"], infer_ep["mean"]),
                        "crossover_decisions": crossover_horizon(train["mean"], plan_step, infer_step),
                        "planner_success_rate": gp["success"].mean(), "rl_success_rate": gr["success"].mean(),
                        "planner_path_ratio": gp.loc[gp["success"], "path_cost_ratio"].mean(),
                        "rl_path_ratio": gr.loc[gr["success"], "path_cost_ratio"].mean(),
                        "rl_n_episodes": len(gr), "planner_n_episodes": len(gp),
                    })
    return pd.DataFrame(rows)


def measure_inference_scaling(grid_sizes, repeats: int = 200, seed: int = 0) -> pd.DataFrame:
    """CPU time of one greedy action (single forward pass) for the DQN architecture at each grid size.

    The forward-pass cost does not depend on the weights, so an untrained network gives the true
    inference cost of that architecture. It does NOT tell us whether a trained network would succeed.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rows = []
    for n in grid_sizes:
        net = QNetwork(n).eval()
        x = torch.zeros(1, 3, n, n)
        with torch.no_grad():
            for _ in range(20):
                net(x)  # warm-up
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                net(x).argmax(dim=1)
                times.append(time.perf_counter() - t0)
        rows.append({"grid_size": n, "infer_time_per_step_s": float(np.median(times)),
                     "parameters": sum(p.numel() for p in net.parameters())})
    return pd.DataFrame(rows)
