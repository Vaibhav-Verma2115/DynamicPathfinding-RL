"""Environment demo: run a random policy and draw the map before and after.

This is a sanity check of the environment, not an experiment: a random policy
is expected to perform badly, and its numbers are not research results.
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathrl.environment import EnvConfig, GridWorldEnv
from pathrl.utils.config import PROJECT_ROOT
from pathrl.visualization.grid_visualizer import draw_grid, save_figure


def run_demo(cfg: dict, steps: int = 60, seed: int | None = None) -> dict:
    seed = cfg["seed"] if seed is None else seed
    env_cfg = EnvConfig.from_config(cfg)
    env = GridWorldEnv(env_cfg, map_seed=seed, render_mode="ansi")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    print(f"Map (seed={seed}, {env_cfg.grid_size}x{env_cfg.grid_size}, dynamics={env_cfg.dynamics}, "
          f"slip={env_cfg.slip_prob}):\n{env.grid_map.to_ascii()}\n")

    trajectory, outcomes, total_reward, changes = [env.agent_pos], Counter(), 0.0, 0
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        trajectory.append(info["agent_pos"])
        outcomes[info["outcome"]] += 1
        outcomes["slipped"] += info["slipped"]
        changes += bool(info["changed_cells"])
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Final state after {env.steps} random steps:\n{env.render()}\n")
    summary = {"steps": env.steps, "total_reward": total_reward, "obstacle_updates_with_change": changes,
               **dict(outcomes)}
    print("Summary:", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_grid(axes[0], env.grid_map, agent=env.grid_map.start, title="Initial map", legend=False)
    draw_grid(axes[1], env.grid_map, obstacle_positions=info["obstacle_positions"], agent=env.agent_pos,
              path=trajectory, title=f"Random policy, {env.steps} steps")
    name = f"demo_env_{env_cfg.dynamics}_N{env_cfg.grid_size}_seed{seed}"
    out = save_figure(fig, name, PROJECT_ROOT / cfg["paths"]["figures_dir"])
    print(f"Saved figure: {out}")
    summary["figure"] = str(out)
    return summary
