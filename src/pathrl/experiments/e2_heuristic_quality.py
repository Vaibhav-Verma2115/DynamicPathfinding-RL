"""E2 - Heuristic quality: speed vs. optimality for A* with h=0, Manhattan, 1.5 x Manhattan.

The optimality ratio rho = cost / optimal cost uses Dijkstra as the exact baseline. h=0 and
Manhattan are admissible, so rho must be exactly 1. 1.5 x Manhattan is inadmissible, so
rho is bounded by 1.5 but measured.
"""

from __future__ import annotations

import pandas as pd

from pathrl.algorithms import make_planner
from pathrl.environment import generate_map
from pathrl.evaluation.profiling import measure
from pathrl.evaluation.results import save_results
from pathrl.experiments.common import env_config, flatten, parallel_map, run_seeds
from pathrl.utils.seeding import make_rng


def _task(args) -> list[dict]:
    ec, seed, variants, repeats = args
    m = generate_map(ec, make_rng(seed))
    optimal = make_planner("dijkstra", ec).plan(m.walls, m.start, m.goal).cost
    rows = []
    for v in variants:
        meas = measure(lambda: make_planner("astar", ec, heuristic=v["heuristic"], weight=v["weight"])
                       .plan(m.walls, m.start, m.goal), repeats=repeats)
        r = meas.result
        rows.append({"variant": v["label"], "heuristic": v["heuristic"], "weight": v["weight"],
                     "grid_size": ec.grid_size, "density": ec.obstacle_density, "seed": seed,
                     "path_cost": r.cost, "optimal_cost": optimal, "optimality_ratio": r.cost / optimal,
                     "optimal": abs(r.cost - optimal) < 1e-9, "wall_time_s": meas.wall_time_s,
                     "peak_memory_bytes": meas.peak_memory_bytes, **r.stats.as_dict()})
    return rows


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    e = cfg["e2"]
    seeds = run_seeds(cfg, offset=e["map_offset"])
    tasks = [(env_config(cfg, grid_size=n, obstacle_density=d), s, e["variants"], e["timing_repeats"])
             for n in e["grid_sizes"] for d in e["densities"] for s in seeds]
    df = pd.DataFrame(flatten(parallel_map(_task, tasks, workers, desc="E2")))
    save_results(df, cfg, "e2_heuristic_quality", "e2_raw", extra={"workers": workers})
    return df
