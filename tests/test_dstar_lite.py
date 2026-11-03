"""Stage 5 tests: D* Lite optimality and genuine incremental repair."""

import math

import numpy as np
import pytest

from pathrl.algorithms import GridGraph
from pathrl.algorithms.classical import AStar, Dijkstra, DStarLite
from pathrl.environment import EnvConfig, GridMap, GridWorldEnv, generate_map

CORRIDOR = """
S . # . .
. . # . .
. . . . .
. . # . .
. . # . G
"""


def random_map(i, size=20, density=0.2, placement="random"):
    return generate_map(EnvConfig(grid_size=size, obstacle_density=density, placement=placement),
                        np.random.default_rng(500 + i))


@pytest.mark.parametrize("conn", [4, 8])
def test_initial_plan_is_optimal(conn):
    graph = GridGraph(conn)
    for i in range(30):
        m = random_map(i, size=[10, 20, 30][i % 3], density=[0.15, 0.3][i % 2])
        d = Dijkstra(graph).plan(m.walls, m.start, m.goal)
        res = DStarLite(graph).plan(m.walls, m.start, m.goal)
        assert res.cost == pytest.approx(d.cost)
        assert graph.is_valid_path(res.path, m.start, m.goal, m.walls)


@pytest.mark.parametrize("conn", [4, 8])
def test_repaired_plans_stay_optimal_under_random_changes(conn):
    """The agent walks along its plan while cells randomly toggle. After every repair, the
    cost must equal a fresh A* search on the current map (both infinite if unreachable)."""
    graph = GridGraph(conn)
    checks = 0
    for i in range(12):
        m = random_map(i, size=15, density=0.2)
        rng = np.random.default_rng(i)
        occ = m.walls.copy()
        planner = DStarLite(graph)
        start = m.start
        res = planner.plan(occ, start, m.goal)
        for _ in range(25):
            if res.found and len(res.path) > 1:
                start = res.path[1]  # agent takes one step
            if start == m.goal:
                break
            changed = []
            for _ in range(rng.integers(1, 6)):
                cell = (int(rng.integers(15)), int(rng.integers(15)))
                if cell not in (start, m.goal):
                    occ[cell] = not occ[cell]
                    changed.append(cell)
            planner.update(changed, occ)
            res = planner.plan(occ, start, m.goal)
            fresh = AStar(graph).plan(occ, start, m.goal)
            assert res.cost == pytest.approx(fresh.cost) or (math.isinf(res.cost) and math.isinf(fresh.cost))
            if res.found:
                assert graph.is_valid_path(res.path, start, m.goal, occ)
            checks += 1
    assert planner.initializations == 1 and checks > 100


def test_repair_preserves_search_state_instead_of_restarting():
    m = random_map(0, size=30, density=0.15, placement="corners")
    planner = DStarLite()
    first = planner.plan(m.walls, m.start, m.goal)
    g_table = planner._g
    occ = m.walls.copy()
    blocked = first.path[len(first.path) // 2]
    occ[blocked] = True
    planner.update([blocked], occ)
    second = planner.plan(occ, m.start, m.goal)
    assert planner.initializations == 1
    assert planner._g is g_table and len(g_table) > 0  # same table, repaired in place
    assert blocked not in second.path
    assert second.cost == pytest.approx(AStar().plan(occ, m.start, m.goal).cost)


def test_small_changes_cost_less_than_replanning_from_scratch():
    """Repair work vs. the work of a fresh D* Lite search on the changed map, averaged over maps.

    A single change near the agent, on its path, is D* Lite's favourable case. (Changes
    near the goal can be expensive to repair. E3 measures realistic mixtures.)
    """
    repair, scratch = [], []
    for i in range(20):
        m = random_map(i, size=30, density=0.15, placement="corners")
        planner = DStarLite()
        res = planner.plan(m.walls, m.start, m.goal)
        occ = m.walls.copy()
        cell = res.path[3]
        occ[cell] = True
        planner.update([cell], occ)
        repair.append(planner.plan(occ, res.path[1], m.goal).stats.expansions)
        scratch.append(DStarLite().plan(occ, res.path[1], m.goal).stats.expansions)
    assert np.mean(repair) < 0.5 * np.mean(scratch)


def test_no_change_means_almost_no_work():
    m = random_map(1, size=30, density=0.15)
    planner = DStarLite()
    res = planner.plan(m.walls, m.start, m.goal)
    again = planner.plan(m.walls, res.path[1], m.goal)  # agent moved, map unchanged
    assert again.stats.expansions <= 2
    assert again.cost == pytest.approx(res.cost - GridGraph().edge_cost(res.path[0], res.path[1], m.walls))


def test_blocked_then_reopened():
    m = GridMap.from_ascii(CORRIDOR)
    planner = DStarLite()
    occ = m.walls.copy()
    assert planner.plan(occ, m.start, m.goal).cost == 8
    occ[2, 2] = True  # close the only gap
    planner.update([(2, 2)], occ)
    assert not planner.plan(occ, m.start, m.goal).found
    occ[2, 2] = False
    planner.update([(2, 2)], occ)
    res = planner.plan(occ, m.start, m.goal)
    assert res.cost == 8 and planner.initializations == 1


def test_changes_detected_without_update_call():
    m = GridMap.from_ascii(CORRIDOR)
    planner = DStarLite()
    occ = m.walls.copy()
    planner.plan(occ, m.start, m.goal)
    occ[2, 2] = True
    assert not planner.plan(occ, m.start, m.goal).found  # diff against its own map copy


def test_new_goal_reinitializes_and_reset_clears():
    m = GridMap.from_ascii(CORRIDOR)
    planner = DStarLite()
    planner.plan(m.walls, m.start, m.goal)
    planner.plan(m.walls, m.start, (0, 4))
    assert planner.initializations == 2
    planner.reset()
    assert planner.initializations == 0 and planner._occ is None


def test_works_with_environment_change_reports():
    """Drive D* Lite with the env's own changed_cells on a random-walk map."""
    cfg = EnvConfig(grid_size=15, dynamics="random_walk", change_interval=1, dynamic_obstacle_density=0.05)
    env = GridWorldEnv(cfg, map_seed=2)
    env.reset(seed=2)
    planner = DStarLite()
    res = planner.plan(env.occupancy, env.agent_pos, env.grid_map.goal)
    for _ in range(60):
        if not res.found or len(res.path) < 2:
            break
        nxt = res.path[1]
        action = GridGraph().deltas.index((nxt[0] - env.agent_pos[0], nxt[1] - env.agent_pos[1]))
        _, _, terminated, _, info = env.step(action)
        if terminated:
            break
        planner.update(info["changed_cells"], env.occupancy)
        res = planner.plan(env.occupancy, env.agent_pos, env.grid_map.goal)
        fresh = AStar().plan(env.occupancy, env.agent_pos, env.grid_map.goal)
        assert res.cost == pytest.approx(fresh.cost) or (math.isinf(res.cost) and math.isinf(fresh.cost))
