"""Stage 4 tests: BFS, Dijkstra and A* correctness, optimality, counters and edge cases."""

import math

import numpy as np
import pytest

from pathrl.algorithms import GridGraph, make_planner
from pathrl.algorithms.classical import BFS, AStar, Dijkstra
from pathrl.algorithms.classical.astar import manhattan, octile
from pathrl.environment import EnvConfig, GridMap, generate_map
from pathrl.environment.dynamics import ACTIONS_8, attempt_move
from pathrl.environment.map_generator import bfs_distances
from pathrl.evaluation.profiling import measure

ALL = ["bfs", "dijkstra", "astar"]

CORRIDOR = """
S . # . .
. . # . .
. . . . .
. . # . .
. . # . G
"""

WALLED_OFF = """
S . # . .
. . # . .
# # # . .
. . . . .
. . . . G
"""

SERPENTINE = """
S . . . . . .
# # # # # # .
. . . . . . .
. # # # # # #
. . . . . . .
# # # # # # .
G . . . . . .
"""


def plan(name, text, **kwargs):
    m = GridMap.from_ascii(text)
    planner = make_planner(name, **kwargs) if name != "astar" else AStar(**kwargs)
    return m, planner.plan(m.walls, m.start, m.goal)


# --------------------------------------------------------------- grid graph

def test_graph_neighbours_match_environment_moves():
    """Planners must search exactly the moves the environment executes."""
    rng = np.random.default_rng(0)
    m = generate_map(EnvConfig(grid_size=12, obstacle_density=0.3), rng)
    for corner_cutting in (False, True):
        graph = GridGraph(connectivity=8, allow_corner_cutting=corner_cutting)
        for r in range(12):
            for c in range(12):
                if m.walls[r, c]:
                    continue
                expected = {attempt_move((r, c), d, m.walls, corner_cutting)[0] for d in ACTIONS_8} - {(r, c)}
                assert {n for n, _ in graph.neighbors((r, c), m.walls)} == expected


def test_graph_edges_are_symmetric():
    m = generate_map(EnvConfig(grid_size=12, obstacle_density=0.3), np.random.default_rng(1))
    graph = GridGraph(connectivity=8)
    for r in range(12):
        for c in range(12):
            if not m.walls[r, c]:
                for nbr, cost in graph.neighbors((r, c), m.walls):
                    assert graph.edge_cost(nbr, (r, c), m.walls) == cost


# --------------------------------------------------- hand-made known answers

@pytest.mark.parametrize("name", ALL)
def test_known_shortest_path_through_gap(name):
    m, res = plan(name, CORRIDOR)
    assert res.found and res.cost == 8 and res.length == 8
    assert (2, 2) in res.path  # the only gap in the wall column
    assert GridGraph().is_valid_path(res.path, m.start, m.goal, m.walls)


@pytest.mark.parametrize("name", ALL)
def test_serpentine_forces_long_detour(name):
    m, res = plan(name, SERPENTINE)
    # Manhattan distance is 6, but the walls force a zig-zag: 6 right, 2 down, 6 left, 2 down, ...
    assert res.cost == 6 + 2 + 6 + 2 + 6 + 2 + 6
    assert GridGraph().is_valid_path(res.path, m.start, m.goal, m.walls)


@pytest.mark.parametrize("name", ALL)
def test_unreachable_goal(name):
    m, res = plan(name, WALLED_OFF)
    assert not res.found and res.path is None and res.cost == math.inf and res.length == 0
    # It must have exhausted the start's component (4 free cells) before giving up
    assert res.stats.expansions == 4


@pytest.mark.parametrize("name", ALL)
def test_goal_blocked_by_obstacle(name):
    m = GridMap.from_ascii(CORRIDOR)
    occupancy = m.walls.copy()
    occupancy[m.goal] = True  # e.g. a moving obstacle sitting on the goal
    planner = make_planner(name)
    assert not planner.plan(occupancy, m.start, m.goal).found


@pytest.mark.parametrize("name", ALL)
def test_start_equals_goal(name):
    walls = np.zeros((3, 3), dtype=bool)
    res = make_planner(name).plan(walls, (1, 1), (1, 1))
    assert res.path == [(1, 1)] and res.cost == 0 and res.length == 0


# ------------------------------------------- optimality on many random maps

def _random_maps(n_maps=40):
    for i in range(n_maps):
        size = [10, 20, 30][i % 3]
        density = [0.15, 0.30][i % 2]
        cfg = EnvConfig(grid_size=size, obstacle_density=density, placement="random")
        yield generate_map(cfg, np.random.default_rng(1000 + i))


def test_four_connected_all_planners_are_optimal():
    """Compare against an independent oracle: the map generator's BFS distance table."""
    for m in _random_maps():
        oracle = bfs_distances(m.walls, m.start)[m.goal]
        for name in ALL:
            res = make_planner(name).plan(m.walls, m.start, m.goal)
            assert res.cost == oracle, (name, m.to_ascii())
            assert GridGraph().is_valid_path(res.path, m.start, m.goal, m.walls)


def test_eight_connected_astar_octile_matches_dijkstra():
    cfg8 = EnvConfig(connectivity=8)
    for m in _random_maps():
        d = make_planner("dijkstra", cfg8).plan(m.walls, m.start, m.goal)
        a = make_planner("astar", cfg8).plan(m.walls, m.start, m.goal)
        b = make_planner("bfs", cfg8).plan(m.walls, m.start, m.goal)
        assert a.cost == pytest.approx(d.cost)
        assert b.cost >= d.cost - 1e-9  # BFS minimises moves, not cost, with sqrt 2 diagonals
        assert GridGraph(8).is_valid_path(a.path, m.start, m.goal, m.walls)


def test_bfs_is_not_cost_optimal_with_diagonals():
    """A documented limitation: with sqrt 2 diagonals, fewest moves != cheapest path.

    On these 40 seeded maps BFS finds a strictly costlier path on some of them (3 at the
    time of writing), always with the same number of moves as Dijkstra or fewer.
    """
    cfg8 = EnvConfig(connectivity=8)
    worse = 0
    for m in _random_maps():
        b = make_planner("bfs", cfg8).plan(m.walls, m.start, m.goal)
        d = make_planner("dijkstra", cfg8).plan(m.walls, m.start, m.goal)
        assert b.length <= d.length
        worse += b.cost > d.cost + 1e-9
    assert worse >= 1


def test_weighted_astar_respects_suboptimality_bound():
    for m in _random_maps():
        optimal = make_planner("dijkstra").plan(m.walls, m.start, m.goal).cost
        res = AStar(heuristic="manhattan", weight=1.5).plan(m.walls, m.start, m.goal)
        assert optimal <= res.cost <= 1.5 * optimal
        assert GridGraph().is_valid_path(res.path, m.start, m.goal, m.walls)


# ------------------------------------------------------------- heuristics

def test_heuristics_are_admissible_and_consistent():
    walls = np.zeros((15, 15), dtype=bool)
    goal = (11, 4)
    for conn, h in ((4, manhattan), (8, octile)):
        graph = GridGraph(conn)
        true_dist = {}
        res = Dijkstra(graph)
        for r in range(15):
            for c in range(15):
                true_dist[(r, c)] = res.plan(walls, (r, c), goal).cost
                assert h((r, c), goal) <= true_dist[(r, c)] + 1e-9  # admissible
                for nbr, cost in graph.neighbors((r, c), walls):
                    assert h((r, c), goal) <= cost + h(nbr, goal) + 1e-9  # consistent


def test_manhattan_is_flagged_inadmissible_on_eight_connected():
    assert AStar(GridGraph(4), heuristic="manhattan").admissible
    assert not AStar(GridGraph(8), heuristic="manhattan").admissible
    assert not AStar(GridGraph(4), heuristic="manhattan", weight=1.5).admissible
    with pytest.raises(ValueError):
        AStar(heuristic="euclidean-ish")
    with pytest.raises(ValueError):
        AStar(weight=0.5)


# ------------------------------------------------------------- counters

def test_astar_with_zero_heuristic_reproduces_dijkstra_exactly():
    """h = 0 makes A* identical to Dijkstra (same keys, same tie-breaking), so the counters match exactly."""
    for m in _random_maps(10):
        d = Dijkstra(record_expanded=True).plan(m.walls, m.start, m.goal)
        a = AStar(heuristic="zero", record_expanded=True).plan(m.walls, m.start, m.goal)
        assert a.stats.as_dict() == d.stats.as_dict() and a.expanded == d.expanded


def test_informed_heuristic_reduces_expansions_on_open_grid():
    walls = np.zeros((20, 20), dtype=bool)
    d = Dijkstra().plan(walls, (0, 0), (19, 19))
    a = AStar().plan(walls, (0, 0), (19, 19))
    assert d.stats.expansions == 400  # Dijkstra settles every cell with g <= 38
    assert a.stats.expansions <= 2 * a.length  # A* walks almost straight to the goal


def test_counter_invariants():
    for m in _random_maps(10):
        for name in ALL:
            s = make_planner(name).plan(m.walls, m.start, m.goal).stats
            free = int((~m.walls).sum())
            assert 1 <= s.expansions <= free
            assert s.expansions <= s.pops <= s.pushes
            assert s.states_touched <= free and s.max_frontier >= 1
            assert s.generated <= 4 * s.expansions


def test_planning_is_deterministic_and_stateless():
    m = next(_random_maps(1))
    for name in ALL:
        planner = make_planner(name, record_expanded=True)
        a = planner.plan(m.walls, m.start, m.goal)
        planner.reset()
        b = planner.plan(m.walls, m.start, m.goal)
        assert a.path == b.path and a.stats.as_dict() == b.stats.as_dict() and a.expanded == b.expanded


def test_make_planner_uses_env_movement_rules():
    p = make_planner("astar", EnvConfig(connectivity=8, allow_corner_cutting=True), heuristic="octile")
    assert p.graph.connectivity == 8 and p.graph.allow_corner_cutting
    with pytest.raises(ValueError):
        make_planner("teleport")


# ------------------------------------------------------------- profiling

def test_measure_reports_time_and_memory():
    m = next(_random_maps(1))
    out = measure(lambda: Dijkstra().plan(m.walls, m.start, m.goal), repeats=3)
    assert out.result.found and out.wall_time_s > 0 and out.peak_memory_bytes > 0
    assert measure(lambda: 1, track_memory=False).peak_memory_bytes is None
