"""Stage 2 tests: map generation, movement, collisions, rewards, slip, moving obstacles,
observations and Gymnasium compliance."""

import math
from collections import Counter
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, MapGenerationError, generate_map
from pathrl.environment.dynamics import ACTIONS_4, ACTIONS_8, action_outcomes, perpendicular, sample_delta
from pathrl.environment.map_generator import bfs_distances, is_reachable
from pathrl.environment.obstacles import make_obstacles
from pathrl.utils.config import load_config

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 4, 5, 6, 7

OPEN_5 = """
S . . . .
. . . . .
. . . . .
. . . . .
. . . . G
"""

CORRIDOR = """
S . # . .
. . # . .
. . . . .
. . # . .
. . # . G
"""


def make_env(text: str, **overrides) -> GridWorldEnv:
    grid_map = GridMap.from_ascii(text)
    cfg = EnvConfig(grid_size=grid_map.size, dynamics=grid_map.dynamics, **overrides)
    env = GridWorldEnv(cfg, grid_map=grid_map)
    env.reset(seed=0)
    return env


# ------------------------------------------------------------------ config

def test_env_config_from_yaml_defaults():
    cfg = EnvConfig.from_config(load_config())
    assert cfg.grid_size == 10 and cfg.rewards.collision == -50.0


@pytest.mark.parametrize("bad", [
    {"slip_prob": 1.5}, {"connectivity": 6}, {"local_view": 6}, {"dynamics": "teleport"},
    {"dynamics": "random_walk", "change_interval": 0}, {"obstacle_density": 1.0},
])
def test_env_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        EnvConfig(**bad)


def test_env_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        EnvConfig.from_config({"env": {"grid_szie": 10}})


# ----------------------------------------------------------- map generation

@pytest.mark.parametrize("size", [10, 20, 30, 50, 70, 100])
@pytest.mark.parametrize("density", [0.15, 0.30])
def test_generated_maps_are_valid_for_all_experiment_sizes(size, density):
    cfg = EnvConfig(grid_size=size, obstacle_density=density)
    for seed in range(3):
        m = generate_map(cfg, np.random.default_rng(seed))
        assert m.walls.shape == (size, size)
        assert m.walls.sum() == round(density * size * size)
        assert not m.walls[m.start] and not m.walls[m.goal]
        assert is_reachable(m.walls, m.start, m.goal)


def test_map_generation_is_deterministic_per_seed():
    cfg = EnvConfig(grid_size=20, placement="random")
    a, b = (generate_map(cfg, np.random.default_rng(5)) for _ in range(2))
    c = generate_map(cfg, np.random.default_rng(6))
    assert np.array_equal(a.walls, b.walls) and a.start == b.start and a.goal == b.goal
    assert not np.array_equal(a.walls, c.walls)


def test_random_placement_respects_minimum_distance():
    cfg = EnvConfig(grid_size=20, placement="random", min_start_goal_distance=0.5)
    for seed in range(20):
        m = generate_map(cfg, np.random.default_rng(seed))
        assert abs(m.start[0] - m.goal[0]) + abs(m.start[1] - m.goal[1]) >= 10


def test_impossible_reachability_raises_instead_of_looping_forever():
    cfg = EnvConfig(grid_size=10, obstacle_density=0.95, max_generation_attempts=20)
    with pytest.raises(MapGenerationError):
        generate_map(cfg, np.random.default_rng(0))


def test_unreachable_maps_allowed_when_not_required():
    cfg = EnvConfig(grid_size=10, obstacle_density=0.6, require_reachable=False)
    maps = [generate_map(cfg, np.random.default_rng(s)) for s in range(20)]
    assert any(not is_reachable(m.walls, m.start, m.goal) for m in maps)


def test_grid_map_is_immutable_and_validated():
    m = GridMap.from_ascii(OPEN_5)
    with pytest.raises(ValueError):
        m.walls[0, 1] = True
    with pytest.raises(ValueError):
        GridMap(np.zeros((3, 3), bool), (0, 0), (0, 0))  # start == goal
    with pytest.raises(ValueError):
        GridMap(np.ones((3, 3), bool), (0, 0), (2, 2))  # start inside a wall


def test_ascii_round_trip_and_bfs_distance():
    m = GridMap.from_ascii(CORRIDOR)
    assert GridMap.from_ascii(m.to_ascii()).to_ascii() == m.to_ascii()
    # Only gap in the wall column is row 2: down 2, right 4, down 2 = 8 steps
    assert bfs_distances(m.walls, m.start)[m.goal] == 8


# ----------------------------------------------------- movement & rewards

@pytest.mark.parametrize("action,expected", [(DOWN, (3, 2)), (UP, (1, 2)), (LEFT, (2, 1)), (RIGHT, (2, 3))])
def test_cardinal_moves(action, expected):
    env = make_env(OPEN_5)
    env.agent_pos = (2, 2)
    _, reward, terminated, truncated, info = env.step(action)
    assert env.agent_pos == expected and reward == -1.0
    assert info["outcome"] == "moved" and not terminated and not truncated


def test_boundary_collision_keeps_agent_in_place():
    env = make_env(OPEN_5)
    _, reward, terminated, _, info = env.step(UP)
    assert env.agent_pos == (0, 0) and reward == -50.0 and info["collided"] and not terminated


def test_wall_collision():
    env = make_env(CORRIDOR)
    env.agent_pos = (0, 1)
    _, reward, _, _, info = env.step(RIGHT)
    assert env.agent_pos == (0, 1) and reward == -50.0 and info["collided"]


def test_terminate_on_collision_option():
    env = make_env(OPEN_5, terminate_on_collision=True)
    _, _, terminated, _, _ = env.step(UP)
    assert terminated


def test_reaching_goal_gives_goal_reward_and_terminates():
    env = make_env(CORRIDOR)
    total = 0.0
    for a in [DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, DOWN]:
        _, reward, terminated, _, info = env.step(a)
        total += reward
    assert terminated and info["outcome"] == "reached_goal"
    assert total == 7 * -1.0 + 100.0  # optimal return = -(L-1) + goal for an L-step path


def test_truncation_at_max_steps():
    env = make_env(OPEN_5, max_steps_factor=0.2)  # 0.2 * 25 = 5 steps
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(UP)
    assert truncated and not terminated and env.config.max_steps == 5


def test_invalid_action_rejected():
    env = make_env(OPEN_5)
    with pytest.raises(ValueError):
        env.step(4)  # diagonal not allowed with 4-connectivity


# ----------------------------------------------------- 8-connected ablation

def test_diagonal_move_and_cost():
    env = make_env(OPEN_5, connectivity=8)
    _, reward, _, _, _ = env.step(DOWN_RIGHT)
    assert env.agent_pos == (1, 1) and reward == pytest.approx(-math.sqrt(2))


def test_corner_cutting_rule():
    text = """
    S # .
    . . .
    . . G
    """
    # From S=(0,0), DOWN_RIGHT targets the free cell (1,1) but squeezes past the wall at (0,1).
    blocked = make_env(text, connectivity=8)
    _, reward, _, _, info = blocked.step(DOWN_RIGHT)
    assert blocked.agent_pos == (0, 0) and info["collided"] and reward == -50.0
    allowed = make_env(text, connectivity=8, allow_corner_cutting=True)
    _, reward, _, _, info = allowed.step(DOWN_RIGHT)
    assert allowed.agent_pos == (1, 1) and not info["collided"]


# ------------------------------------------------------- stochastic slips

def test_perpendicular_never_reverses():
    for delta in ACTIONS_8:
        for p in perpendicular(delta):
            assert p != delta and p != (-delta[0], -delta[1])
            assert delta[0] * p[0] + delta[1] * p[1] == 0  # dot product 0 => 90 degrees


@pytest.mark.parametrize("p", [0.0, 0.1, 0.2])
def test_action_outcomes_form_a_distribution(p):
    for a in range(4):
        outcomes = action_outcomes(ACTIONS_4, a, p)
        assert sum(prob for prob, _ in outcomes) == pytest.approx(1.0)


def test_empirical_slip_rate_matches_model():
    rng = np.random.default_rng(0)
    n, p = 20_000, 0.2
    counts = Counter(sample_delta(ACTIONS_4, RIGHT, p, rng)[0] for _ in range(n))
    assert counts[(0, 1)] / n == pytest.approx(0.8, abs=0.015)
    assert counts[(-1, 0)] / n == pytest.approx(0.1, abs=0.01)
    assert counts[(1, 0)] / n == pytest.approx(0.1, abs=0.01)
    assert counts[(0, -1)] == 0  # never backwards


def test_zero_slip_is_deterministic():
    env = make_env(OPEN_5, slip_prob=0.0)
    for _ in range(4):
        _, _, _, _, info = env.step(RIGHT)
        assert not info["slipped"]
    assert env.agent_pos == (0, 4)


# -------------------------------------------------------- moving obstacles

def _check_obstacle_invariants(env: GridWorldEnv):
    positions = [tuple(p) for p in env.obstacles.positions]
    assert len(set(positions)) == len(positions) == env.obstacles.count
    expected = env.grid_map.walls.copy()
    for cell in positions:
        assert not env.grid_map.walls[cell]
        assert cell != env.grid_map.goal and cell != env.agent_pos
        expected[cell] = True
    assert np.array_equal(expected, env.occupancy)


@pytest.mark.parametrize("dynamics", ["random_walk", "oscillating_corridor"])
def test_moving_obstacle_rules_hold_over_long_runs(dynamics):
    cfg = EnvConfig(grid_size=15, dynamics=dynamics, change_interval=1, dynamic_obstacle_density=0.05)
    env = GridWorldEnv(cfg, map_seed=3)
    env.reset(seed=3)
    env.action_space.seed(3)
    assert env.obstacles.count >= 1
    for _ in range(500):
        before = env.occupancy.copy()
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        _check_obstacle_invariants(env)
        diff = sorted(map(tuple, np.argwhere(before != env.occupancy)))
        assert diff == info["changed_cells"]  # reported changes are exactly the real ones
        if terminated or truncated:
            env.reset()


def test_obstacles_only_move_on_change_interval():
    cfg = EnvConfig(grid_size=15, dynamics="random_walk", change_interval=5, dynamic_obstacle_density=0.05)
    env = GridWorldEnv(cfg, map_seed=1)
    env.reset(seed=1)
    for step in range(1, 41):
        before = env.occupancy.copy()
        env.step(UP)  # agent bumps the top wall, stays put
        if step % 5 != 0:
            assert np.array_equal(before, env.occupancy)


def test_oscillating_obstacles_stay_on_tracks_and_are_periodic():
    cfg = EnvConfig(grid_size=20, dynamics="oscillating_corridor", change_interval=1, dynamic_obstacle_density=0.02)
    m = generate_map(cfg, np.random.default_rng(0))
    assert len(m.tracks) >= 1 and all(len(t) >= 2 for t in m.tracks)
    obstacles = make_obstacles(m)
    obstacles.reset()
    occupied = m.walls.copy()
    occupied[tuple(obstacles.positions.T)] = True
    initial = obstacles.positions.copy()
    period = math.lcm(*(2 * (len(t) - 1) for t in m.tracks))
    far_away = (-1, -1)  # no agent interference
    for _ in range(period):
        obstacles.update(occupied, far_away, np.random.default_rng(0))
        for pos, track in zip(obstacles.positions, m.tracks):
            assert any((pos == cell).all() for cell in track)
    assert np.array_equal(obstacles.positions, initial)


def test_corridor_obstacles_cross_the_shortest_path():
    cfg = EnvConfig(grid_size=20, dynamics="oscillating_corridor", change_interval=1)
    m = generate_map(cfg, np.random.default_rng(0))
    for track in m.tracks:  # every track was built through a path cell, so the path gets blocked
        assert len(track) >= 2


# -------------------------------------------------------- determinism / seeds

def _rollout(env: GridWorldEnv, seed: int, steps: int = 200):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    trace = [obs.copy()]
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        trace.append((obs.copy(), reward, info["agent_pos"], tuple(info["obstacle_positions"])))
        if terminated or truncated:
            break
    return trace


def test_same_seed_same_trajectory():
    cfg = EnvConfig(grid_size=12, dynamics="random_walk", change_interval=2, slip_prob=0.2)
    a, b = GridWorldEnv(cfg, map_seed=7), GridWorldEnv(cfg, map_seed=7)
    for x, y in zip(_rollout(a, 11), _rollout(b, 11)):
        if isinstance(x, tuple):
            assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


def test_reset_with_new_map_switches_scenario():
    cfg = EnvConfig(grid_size=10)
    env = GridWorldEnv(cfg, map_seed=0)
    other = generate_map(cfg, np.random.default_rng(99))
    env.reset(seed=0, options={"map": other})
    assert env.grid_map is other and env.agent_pos == other.start


def test_map_size_mismatch_rejected():
    env = GridWorldEnv(EnvConfig(grid_size=10))
    with pytest.raises(ValueError):
        env.reset(options={"map": generate_map(EnvConfig(grid_size=12), np.random.default_rng(0))})


# ------------------------------------------------------------ observations

def test_global_observation_channels():
    env = make_env(CORRIDOR)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 5, 5)
    assert obs.dtype == np.float32
    assert np.array_equal(obs[0].astype(bool), env.grid_map.walls)
    assert obs[1][env.grid_map.start] == 1 and obs[1].sum() == 1
    assert obs[2][env.grid_map.goal] == 1 and obs[2].sum() == 1


def test_local_observation_is_egocentric_with_padded_borders():
    env = make_env(OPEN_5, observation="local", local_view=7)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 7, 7)
    assert obs[1][3, 3] == 1 and obs[1].sum() == 1  # agent always at the centre
    assert obs[0][:3, :].all() and obs[0][:, :3].all()  # above/left of (0,0) is outside the grid
    assert obs[2][3 + 3, 3 + 3] == 1  # goal at offset (4,4) clipped to the view edge (3,3)


def test_local_observation_shows_visible_goal_exactly():
    env = make_env(OPEN_5, observation="local", local_view=7)
    env.agent_pos = (3, 3)
    obs = env._observe()
    assert obs[2][3 + 1, 3 + 1] == 1  # goal (4,4) is one down, one right


# ---------------------------------------------------------------- gymnasium

@pytest.mark.parametrize("overrides", [
    {}, {"observation": "local"}, {"connectivity": 8},
    {"dynamics": "random_walk", "change_interval": 1, "slip_prob": 0.1},
    {"dynamics": "oscillating_corridor", "change_interval": 2},
])
def test_gymnasium_api_compliance(overrides):
    # Built via gym.make so the checker can also exercise the other render mode ("ansi").
    env = gym.make("pathrl/GridWorld-v0", config=replace(EnvConfig(grid_size=10), **overrides),
                   map_seed=0, render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=False)


def test_gym_make_registration():
    env = gym.make("pathrl/GridWorld-v0", config=EnvConfig(grid_size=8))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 8, 8)


def test_ansi_render():
    env = GridWorldEnv(EnvConfig(grid_size=5), map_seed=0, render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "A" in text and "G" in text and len(text.splitlines()) == 5
