"""E1 - Static scaling of BFS, Dijkstra, A* (Manhattan) and D* Lite (initial search).

For every (grid size, density, seed) one map is generated and each planner solves the same
start -> goal query. Recorded per planner: median wall time, peak Python memory (separate
run), machine-independent counters, path cost/length and optimality ratio vs. Dijkstra.

Runs serially by default so that wall-clock times are not distorted by parallel workers.
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
    ec, seed, planners, repeats = args
    m = generate_map(ec, make_rng(seed))
    optimal = make_planner("dijkstra", ec).plan(m.walls, m.start, m.goal).cost
    rows = []
    for name in planners:
        meas = measure(lambda: make_planner(name, ec).plan(m.walls, m.start, m.goal), repeats=repeats)
        r = meas.result
        rows.append({"planner": name, "grid_size": ec.grid_size, "density": ec.obstacle_density, "seed": seed,
                     "found": r.found, "path_cost": r.cost, "path_length": r.length,
                     "optimality_ratio": r.cost / optimal, "wall_time_s": meas.wall_time_s,
                     "peak_memory_bytes": meas.peak_memory_bytes, **r.stats.as_dict()})
    return rows


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    e = cfg["e1"]
    seeds = run_seeds(cfg, offset=e["map_offset"])
    tasks = [(env_config(cfg, grid_size=n, obstacle_density=d), s, e["planners"], e["timing_repeats"])
             for n in e["grid_sizes"] for d in e["densities"] for s in seeds]
    df = pd.DataFrame(flatten(parallel_map(_task, tasks, workers, desc="E1")))
    save_results(df, cfg, "e1_static_scaling", "e1_raw", extra={"workers": workers})
    return df
