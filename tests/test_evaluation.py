"""Stage 8 tests: planner navigation harness, result provenance and statistics."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pathrl.algorithms import make_planner
from pathrl.analysis.statistics import mean_ci, summarize, wilson_interval
from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, generate_map
from pathrl.environment.map_generator import bfs_distances
from pathrl.evaluation.navigation import run_planner_episode
from pathrl.evaluation.results import save_results
from pathrl.utils.config import load_config


@pytest.mark.parametrize("name", ["bfs", "dijkstra", "astar", "dstar_lite"])
def test_static_navigation_follows_optimal_path_without_replanning(name):
    cfg = EnvConfig(grid_size=15, obstacle_density=0.25)
    env = GridWorldEnv(cfg, map_seed=4)
    out = run_planner_episode(env, make_planner(name), seed=0)
    m = env.grid_map
    assert out["success"] and out["collisions"] == 0 and out["replans"] == 0
    assert out["path_cost"] == bfs_distances(m.walls, m.start)[m.goal]
    assert out["initial_expansions"] > 0 and out["repair_expansions"] == 0


def test_dynamic_navigation_replans_and_dstar_never_restarts():
    cfg = EnvConfig(grid_size=15, dynamics="random_walk", change_interval=2, dynamic_obstacle_density=0.04)
    for seed in range(5):
        env = GridWorldEnv(cfg, map_seed=seed)
        dstar = make_planner("dstar_lite")
        out = run_planner_episode(env, dstar, seed=seed)
        assert dstar.initializations == 1
        if out["change_steps"]:
            assert out["replans"] >= 1
        astar = run_planner_episode(env, make_planner("astar"), seed=seed)
        # Same map + seed: identical initial plan cost, and A* also replans whenever the map changes
        assert astar["initial_expansions"] > 0
        if astar["change_steps"]:
            assert astar["replans"] >= 1 and astar["repair_expansions"] > 0


def test_on_block_replans_no_more_than_on_change():
    cfg = EnvConfig(grid_size=20, dynamics="random_walk", change_interval=1, dynamic_obstacle_density=0.03)
    env = GridWorldEnv(cfg, map_seed=1)
    eager = run_planner_episode(env, make_planner("astar"), seed=1, replan="on_change")
    lazy = run_planner_episode(env, make_planner("astar"), seed=1, replan="on_block")
    assert lazy["replans"] <= eager["replans"]


def test_slip_forces_replans_but_planner_still_reaches_goal():
    cfg = EnvConfig(grid_size=12, obstacle_density=0.15, slip_prob=0.2, max_steps_factor=2)
    env = GridWorldEnv(cfg, map_seed=0)
    out = run_planner_episode(env, make_planner("astar"), seed=3)
    assert out["success"] and out["replans"] > 0


def test_unreachable_goal_is_reported_as_failure():
    m = GridMap.from_ascii("""
    S . # . .
    . . # . .
    # # # . .
    . . . . .
    . . . . G
    """)
    env = GridWorldEnv(EnvConfig(grid_size=5, max_steps_factor=0.4), grid_map=m)
    out = run_planner_episode(env, make_planner("astar"), seed=0)
    assert not out["success"] and out["blocked_steps"] == out["steps"] == 10


def test_mean_ci_matches_t_distribution():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = mean_ci(x)
    half = stats.t.ppf(0.975, 4) * np.std(x, ddof=1) / math.sqrt(5)
    assert s["mean"] == 3.0 and s["ci_high"] - 3.0 == pytest.approx(half) and s["n"] == 5
    assert math.isnan(mean_ci([7.0])["ci_low"]) and mean_ci([float("nan"), 2.0])["n"] == 1


def test_wilson_interval_stays_in_bounds():
    lo, hi = wilson_interval(10, 10)
    assert 0.6 < lo < 1.0 and hi == pytest.approx(1.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.4


def test_summarize_groups():
    df = pd.DataFrame({"algo": ["a"] * 3 + ["b"] * 3, "t": [1, 2, 3, 10, 10, 10]})
    out = summarize(df, ["algo"], ["t"])
    assert list(out["t_mean"]) == [2.0, 10.0] and out.loc[1, "t_ci_low"] == 10.0


def test_results_saved_with_provenance(tmp_path):
    cfg = load_config(overrides=[f"paths.results_dir={tmp_path}"])
    path = save_results(pd.DataFrame({"x": [1, 2]}), cfg, "unit", "demo", extra={"note": "test"})
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert path.exists() and meta["rows"] == 2 and meta["hardware"]["python"].startswith("3.11")
    assert meta["config"]["env"]["grid_size"] == 10 and meta["note"] == "test"
