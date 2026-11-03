"""Side-by-side picture of what BFS, Dijkstra and A* explore on the same map.

This is a teaching aid for a single map, not an experiment. E1/E2 (Stage 9) run
many seeds and report confidence intervals.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathrl.algorithms import make_planner
from pathrl.environment import EnvConfig, generate_map
from pathrl.evaluation.profiling import measure
from pathrl.utils.config import PROJECT_ROOT
from pathrl.utils.seeding import make_rng
from pathrl.visualization.grid_visualizer import draw_grid, save_figure

# "auto" = Manhattan on 4-connected grids, octile on 8-connected grids
DEMO_PLANNERS = [
    ("BFS", "bfs", {}),
    ("Dijkstra", "dijkstra", {}),
    ("A* (h=0)", "astar", {"heuristic": "zero"}),
    ("A* (h={h})", "astar", {"heuristic": "auto"}),
    ("A* (1.5 x {h})", "astar", {"heuristic": "auto", "weight": 1.5}),
]


def run_planner_demo(cfg: dict, seed: int | None = None) -> list[dict]:
    seed = cfg["seed"] if seed is None else seed
    env_cfg = EnvConfig.from_config(cfg)
    grid_map = generate_map(env_cfg, make_rng(seed))
    occupancy = grid_map.walls
    h_name = "Manhattan" if env_cfg.connectivity == 4 else "octile"
    planners = [(label.format(h=h_name), name, kw) for label, name, kw in DEMO_PLANNERS]

    rows = []
    fig, axes = plt.subplots(1, len(planners), figsize=(4.2 * len(planners), 4.6))
    optimal = None
    for ax, (label, name, kwargs) in zip(axes, planners):
        planner = make_planner(name, env_cfg, record_expanded=True, **kwargs)
        m = measure(lambda: planner.plan(occupancy, grid_map.start, grid_map.goal), repeats=5)
        res = m.result
        if name == "dijkstra":
            optimal = res.cost
        rows.append({"planner": label, "found": res.found, "cost": res.cost, "moves": res.length,
                     **res.stats.as_dict(), "time_ms": m.wall_time_s * 1e3,
                     "peak_kib": m.peak_memory_bytes / 1024})
        draw_grid(ax, grid_map, path=res.path, explored=res.expanded, legend=False,
                  title=f"{label}\nexpanded {res.stats.expansions}, cost {res.cost:.2f}")

    for row in rows:
        row["optimality_ratio"] = row["cost"] / optimal if optimal else float("nan")

    name = f"demo_planners_N{env_cfg.grid_size}_d{env_cfg.obstacle_density}_c{env_cfg.connectivity}_seed{seed}"
    out = save_figure(fig, name, PROJECT_ROOT / cfg["paths"]["figures_dir"])

    header = f"{'planner':<22}{'cost':>8}{'ratio':>7}{'expand':>8}{'pushes':>8}{'frontier':>9}{'time ms':>9}{'peak KiB':>10}"
    print(f"{env_cfg.grid_size}x{env_cfg.grid_size}, density {env_cfg.obstacle_density}, "
          f"{env_cfg.connectivity}-connected, map seed {seed} (single map, single run)\n{header}")
    for r in rows:
        print(f"{r['planner']:<22}{r['cost']:>8.2f}{r['optimality_ratio']:>7.3f}{r['expansions']:>8}"
              f"{r['pushes']:>8}{r['max_frontier']:>9}{r['time_ms']:>9.2f}{r['peak_kib']:>10.1f}")
    print(f"Saved figure: {out}")
    return rows
