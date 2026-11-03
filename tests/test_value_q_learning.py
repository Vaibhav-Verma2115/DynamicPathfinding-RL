"""Stage 6 tests: Value Iteration, Q-Learning and convergence definitions."""

import numpy as np
import pandas as pd
import pytest

from pathrl.algorithms.model_based import ValueIteration
from pathrl.algorithms.reinforcement_learning import QLearning
from pathrl.analysis.convergence import ConvergenceCriterion, bellman_convergence, rl_convergence
from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, generate_map
from pathrl.environment.map_generator import bfs_distances
from pathrl.evaluation.rollout import evaluate_policy, run_policy_episode

GAMMA = 0.99

CORRIDOR = """
S . # . .
. . # . .
. . . . .
. . # . .
. . # . G
"""


def discounted_shortest_return(length: int, gamma=GAMMA, step=-1.0, goal=100.0) -> float:
    """Return of an L-step collision-free path: L-1 steps of -1, then +100 on arrival."""
    return sum(step * gamma**t for t in range(length - 1)) + goal * gamma ** (length - 1)


# ---------------------------------------------------------------- Value Iteration

def test_vi_values_match_shortest_paths_without_slip():
    for seed in range(5):
        cfg = EnvConfig(grid_size=12, obstacle_density=0.25)
        m = generate_map(cfg, np.random.default_rng(seed))
        vi = ValueIteration(cfg, gamma=GAMMA).solve(m.walls, m.goal)
        dist = bfs_distances(m.walls, m.goal)
        for cell in [m.start, (m.goal[0] - 1, m.goal[1])]:
            if not m.walls[cell] and dist[cell] > 0:
                assert vi.value(cell) == pytest.approx(discounted_shortest_return(dist[cell]), abs=1e-4)


def test_vi_greedy_policy_follows_a_shortest_path():
    cfg = EnvConfig(grid_size=5)
    m = GridMap.from_ascii(CORRIDOR)
    vi = ValueIteration(cfg, gamma=GAMMA).solve(m.walls, m.goal)
    env = GridWorldEnv(cfg, grid_map=m)
    result = run_policy_episode(env, vi, seed=0)
    assert result["success"] and result["steps"] == 8 and result["collisions"] == 0


def test_bellman_residual_contracts_and_converges():
    cfg = EnvConfig(grid_size=15, slip_prob=0.2)
    m = generate_map(cfg, np.random.default_rng(1))
    vi = ValueIteration(cfg, gamma=GAMMA, theta=1e-6).solve(m.walls, m.goal)
    res = np.array(vi.residuals)
    assert vi.converged and res[-1] < 1e-6
    # Contraction: residual_{k+1} <= gamma * residual_k (small float slack)
    assert np.all(res[1:] <= GAMMA * res[:-1] + 1e-9)
    info = bellman_convergence(vi.residuals, 1e-6)
    assert info["converged"] and info["sweeps_to_convergence"] == vi.sweeps


def test_vi_model_matches_environment_dynamics_under_slip():
    """VI's value of the start state must equal the Monte Carlo return of its own policy
    in the real environment. This only holds if VI's model = the environment's dynamics."""
    cfg = EnvConfig(grid_size=6, obstacle_density=0.15, slip_prob=0.2, max_steps_factor=10)
    m = generate_map(cfg, np.random.default_rng(3))
    vi = ValueIteration(cfg, gamma=GAMMA, theta=1e-10).solve(m.walls, m.goal)
    env = GridWorldEnv(cfg, grid_map=m)
    env.reset(seed=0)
    returns = []
    for _ in range(4000):
        obs, info = env.reset()
        g, discount, done = 0.0, 1.0, False
        while not done:
            obs, r, term, trunc, info = env.step(vi.act(obs, info))
            g += discount * r
            discount *= GAMMA
            done = term or trunc
        returns.append(g)
    se = np.std(returns) / np.sqrt(len(returns))
    assert abs(np.mean(returns) - vi.value(m.start)) < 4 * se + 0.05


def test_vi_with_slip_avoids_walls_more_than_deterministic_plan():
    """With slip, the stochastic-aware policy should value safety: its expected return under slip
    is at least that of the deterministic-optimal policy evaluated under slip."""
    m = GridMap.from_ascii(CORRIDOR)
    slip_cfg = EnvConfig(grid_size=5, slip_prob=0.2)
    aware = ValueIteration(slip_cfg, gamma=GAMMA).solve(m.walls, m.goal)
    naive = ValueIteration(EnvConfig(grid_size=5), gamma=GAMMA).solve(m.walls, m.goal)
    env = GridWorldEnv(slip_cfg, grid_map=m)
    r_aware = np.mean([run_policy_episode(env, aware, seed=s)["return"] for s in range(400)])
    r_naive = np.mean([run_policy_episode(env, naive, seed=s)["return"] for s in range(400)])
    assert r_aware >= r_naive - 2.0


# ------------------------------------------------------------------ Q-Learning

def test_q_learning_update_rule():
    q = QLearning(grid_size=2, alpha=0.5, gamma=0.9)
    q.Q[1] = [0.0, 10.0, 0.0, 0.0]
    td = q.update(s=0, a=3, r=-1.0, s_next=1, terminated=False)
    assert td == pytest.approx(-1 + 0.9 * 10)
    assert q.Q[0, 3] == pytest.approx(0.5 * 8.0)
    q.update(s=0, a=2, r=100.0, s_next=1, terminated=True)  # terminal: no bootstrap
    assert q.Q[0, 2] == pytest.approx(50.0)


def test_q_learning_learns_optimal_path_on_small_map():
    cfg = EnvConfig(grid_size=5, max_steps_factor=4)
    m = GridMap.from_ascii(CORRIDOR)
    env = GridWorldEnv(cfg, grid_map=m)
    agent = QLearning(5, alpha=0.5, gamma=GAMMA, seed=0)
    log = agent.train(env, episodes=400, seed=0, eval_interval=50, eval_episodes=5)
    final = run_policy_episode(env, agent, seed=123)
    assert final["success"] and final["steps"] == 8
    evals = log.evaluations_df()
    assert evals["success_rate"].iloc[-1] == 1.0
    assert log.train_time_s > 0 and log.eval_time_s > 0 and log.env_steps > 0


def test_q_learning_training_is_reproducible(tmp_path):
    cfg = EnvConfig(grid_size=6, slip_prob=0.1)
    env_a, env_b = GridWorldEnv(cfg, map_seed=2), GridWorldEnv(cfg, map_seed=2)
    a = QLearning(6, seed=4); a.train(env_a, 60, seed=9, eval_interval=30, eval_episodes=2)
    b = QLearning(6, seed=4); b.train(env_b, 60, seed=9, eval_interval=30, eval_episodes=2)
    assert np.array_equal(a.Q, b.Q)
    a.save(tmp_path / "q.npz")
    assert np.array_equal(QLearning.load(tmp_path / "q.npz").Q, a.Q)


def test_act_does_not_learn():
    cfg = EnvConfig(grid_size=5)
    env = GridWorldEnv(cfg, map_seed=0)
    agent = QLearning(5)
    before = agent.Q.copy()
    evaluate_policy(agent, env, episodes=3, seed=0)
    assert np.array_equal(before, agent.Q)


# ------------------------------------------------------------------ convergence

def _evals(success, returns):
    return pd.DataFrame({"episode": np.arange(1, len(success) + 1) * 100, "success_rate": success,
                         "mean_return": returns, "env_steps": np.arange(1, len(success) + 1) * 1000,
                         "train_time_s": np.arange(1, len(success) + 1) * 1.0})


def test_rl_convergence_requires_consecutive_stable_windows():
    crit = ConvergenceCriterion(success_threshold=0.9, windows=3, return_tolerance=0.05)
    evals = _evals([0.2, 0.95, 0.5, 0.95, 1.0, 1.0, 1.0], [10, 80, 40, 82, 83, 82, 83])
    out = rl_convergence(evals, crit)
    assert out["converged"] and out["episodes_to_convergence"] == 600  # windows 4,5,6 -> confirmed at 6
    assert out["steps_to_convergence"] == 6000 and out["time_to_convergence_s"] == 6.0


def test_rl_convergence_rejects_unstable_returns_and_reports_nonconvergence():
    crit = ConvergenceCriterion(success_threshold=0.9, windows=3, return_tolerance=0.05)
    out = rl_convergence(_evals([1.0] * 5, [50, 90, 60, 95, 55]), crit)
    assert not out["converged"] and np.isnan(out["episodes_to_convergence"])
    assert out["final_success_rate"] == 1.0
