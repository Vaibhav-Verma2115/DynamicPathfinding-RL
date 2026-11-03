"""Stage 10 tests: crossover arithmetic and inference-cost measurement."""

import math

import pandas as pd
import pytest

from pathrl.analysis.crossover import crossover_horizon, crossover_table, measure_inference_scaling


def test_crossover_horizon_formula():
    # C_train = 100 s, planning 2 s/episode, inference 0.5 s/episode -> H* = 100 / 1.5
    assert crossover_horizon(100.0, 2.0, 0.5) == pytest.approx(100 / 1.5)
    assert math.isinf(crossover_horizon(100.0, 0.5, 0.5))   # equal per-episode cost: never
    assert math.isinf(crossover_horizon(100.0, 0.1, 0.5))   # inference dearer than planning: never


def test_crossover_table_uses_matching_rows():
    e3 = pd.DataFrame([
        {"grid_size": 10, "change_interval": 1, "family": "classical", "method": "A*", "computation_time_s": 0.02,
         "steps": 20, "success": True, "path_cost_ratio": 1.1},
        {"grid_size": 10, "change_interval": 1, "family": "classical", "method": "A*", "computation_time_s": 0.04,
         "steps": 20, "success": True, "path_cost_ratio": 1.0},
        {"grid_size": 10, "change_interval": 1, "family": "rl", "method": "DQN", "computation_time_s": 0.01,
         "steps": 25, "success": False, "path_cost_ratio": 1.4},
    ])
    training = pd.DataFrame({"algorithm": ["dqn", "dqn"], "train_time_s": [10.0, 14.0]})
    t = crossover_table(e3, training)
    row = t.iloc[0]
    assert row["plan_time_per_episode_s"] == pytest.approx(0.03)
    assert row["train_time_s"] == pytest.approx(12.0)
    assert row["crossover_episodes"] == pytest.approx(12.0 / (0.03 - 0.01))
    assert row["rl_success_rate"] == 0.0 and row["planner_success_rate"] == 1.0


def test_inference_scaling_measures_each_size():
    df = measure_inference_scaling([10, 50], repeats=20)
    assert list(df["grid_size"]) == [10, 50]
    assert (df["infer_time_per_step_s"] > 0).all() and (df["parameters"] > 0).all()
