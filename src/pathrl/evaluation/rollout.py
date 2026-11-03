"""Running a policy in the environment and recording what happened.

Used both for greedy evaluation during training (convergence curves) and for the
final benchmarks. Inference time is measured around `policy.act` only.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from pathrl.algorithms.base_policy import LearnedPolicy
from pathrl.environment import GridMap, GridWorldEnv
from pathrl.environment.dynamics import move_cost


def run_policy_episode(env: GridWorldEnv, policy: LearnedPolicy, seed: int | None = None,
                       grid_map: GridMap | None = None, record_path: bool = False) -> dict:
    options = {"map": grid_map} if grid_map is not None else None
    obs, info = env.reset(seed=seed, options=options)
    total_reward, collisions, path_cost, inference_s = 0.0, 0, 0.0, 0.0
    path = [env.agent_pos]
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = policy.act(obs, info)
        inference_s += time.perf_counter() - t0
        before = env.agent_pos
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        collisions += info["collided"]
        moved = (env.agent_pos[0] - before[0], env.agent_pos[1] - before[1])
        if moved != (0, 0):
            path_cost += move_cost(moved, env.config.diagonal_cost)
        if record_path:
            path.append(env.agent_pos)
    result = {
        "success": info["outcome"] == "reached_goal",
        "return": total_reward,
        "steps": env.steps,
        "collisions": collisions,
        "path_cost": path_cost,
        "inference_time_s": inference_s,
        "inference_time_per_step_s": inference_s / max(env.steps, 1),
    }
    if record_path:
        result["path"] = path
    return result


def evaluate_policy(policy: LearnedPolicy, env: GridWorldEnv, episodes: int, seed: int,
                    maps: Sequence[GridMap] | None = None) -> dict:
    """Greedy evaluation over `episodes` episodes (cycling through `maps` if given)."""
    seeds = np.random.SeedSequence(seed).generate_state(episodes)
    rows = [run_policy_episode(env, policy, seed=int(s), grid_map=maps[i % len(maps)] if maps else None)
            for i, s in enumerate(seeds)]
    return {
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_collisions": float(np.mean([r["collisions"] for r in rows])),
        "inference_time_per_step_s": float(np.mean([r["inference_time_per_step_s"] for r in rows])),
    }
