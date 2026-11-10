"""Worked examples computed by running the project's own code when the guide is built.

Every number printed in an "example" box of the guide comes from these functions, so the
examples always agree with the implementation.
"""

from __future__ import annotations

import math

import numpy as np

from pathrl.algorithms import make_planner
from pathrl.algorithms.model_based.value_iteration import ValueIteration
from pathrl.algorithms.reinforcement_learning.network import QNetwork
from pathrl.algorithms.reinforcement_learning.q_learning import QLearning
from pathrl.analysis.crossover import crossover_horizon
from pathrl.analysis.statistics import mean_ci, wilson_interval
from pathrl.environment import EnvConfig, GridMap
from pathrl.utils.seeding import derive_seeds

DEMO_MAP = """
S . . . # . . . . .
. # # . # . # # # .
. . # . . . . . # .
# . # # # # # . # .
. . . . . . # . . .
. # # # # . # # # .
. . . # . . . . # .
# # . # . # # . # .
. . . . . # . . . .
. # # # . . . # # G
"""


def demo_map() -> GridMap:
    return GridMap.from_ascii(DEMO_MAP)


def _render(m: GridMap, path=None, expanded=None) -> str:
    grid = np.where(m.walls, "#", ".").astype("<U1")
    for cell in expanded or []:
        grid[cell] = "x"
    for cell in path or []:
        grid[cell] = "*"
    grid[m.start], grid[m.goal] = "S", "G"
    return "\n".join(" ".join(row) for row in grid)


def planner_comparison() -> dict:
    """Run every planner on DEMO_MAP and report cost and work counters."""
    m, cfg = demo_map(), EnvConfig(grid_size=10)
    rows, renders = [], {}
    for label, name, kw in [("BFS", "bfs", {}), ("Dijkstra", "dijkstra", {}), ("A* (h = 0)", "astar", {"heuristic": "zero"}),
                            ("A* (Manhattan)", "astar", {}), ("A* (1.5 x Manhattan)", "astar", {"weight": 1.5}),
                            ("D* Lite (first plan)", "dstar_lite", {})]:
        p = make_planner(name, cfg, record_expanded=True, **kw)
        r = p.plan(m.walls, m.start, m.goal)
        rows.append({"planner": label, "path cost": r.cost, "expansions": r.stats.expansions,
                     "pushes": r.stats.pushes, "max frontier": r.stats.max_frontier})
        renders[label] = _render(m, r.path, r.expanded)
    return {"rows": rows, "renders": renders, "ascii": m.to_ascii()}


def dstar_repair() -> dict:
    """Plan with D* Lite, block a cell on the path, then repair. Compare with A* from scratch."""
    m, cfg = demo_map(), EnvConfig(grid_size=10)
    occ = m.walls.copy()
    d = make_planner("dstar_lite", cfg)
    first = d.plan(occ, m.start, m.goal)
    # Agent walks 4 steps along the plan, then a cell further along the plan becomes blocked.
    pos = first.path[4]
    blocked = first.path[8]
    occ[blocked] = True
    d.update([blocked], occ)
    repaired = d.plan(occ, pos, m.goal)
    fresh = make_planner("astar", cfg).plan(occ, pos, m.goal)
    return {"first_cost": first.cost, "first_exp": first.stats.expansions, "pos": pos, "blocked": blocked,
            "rep_cost": repaired.cost, "rep_exp": repaired.stats.expansions,
            "rep_updates": repaired.stats.vertex_updates, "astar_cost": fresh.cost,
            "astar_exp": fresh.stats.expansions, "inits": d.initializations,
            "render": _render(GridMap(occ, pos, m.goal), repaired.path)}


ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}

TINY = """
S . . . .
. # # # .
. . . # .
# # . . .
. . . # G
"""


def value_iteration_example() -> dict:
    m = GridMap.from_ascii(TINY)
    out = {}
    for slip in (0.0, 0.2):
        vi = ValueIteration(EnvConfig(grid_size=5, slip_prob=slip), gamma=0.99, theta=1e-6).solve(m.walls, m.goal)
        n = 5
        values, policy = [], []
        for r in range(n):
            vrow, prow = [], []
            for c in range(n):
                if m.walls[r, c]:
                    vrow.append("  ###  "); prow.append("#")
                elif (r, c) == m.goal:
                    vrow.append("   G   "); prow.append("G")
                else:
                    vrow.append(f"{vi.value((r, c)):7.2f}"); prow.append(ARROWS[int(np.argmax(vi.Q[r * n + c]))])
            values.append(" ".join(vrow)); policy.append(" ".join(prow))
        out[slip] = {"values": "\n".join(values), "policy": "\n".join(policy), "sweeps": vi.sweeps,
                     "residuals": vi.residuals[:4], "start_value": vi.value(m.start)}
    return out


def td_update_example() -> dict:
    """One Q-Learning update with concrete numbers, executed through QLearning.update."""
    agent = QLearning(grid_size=5, alpha=0.1, gamma=0.99)
    s, a, s_next = 7, 3, 8
    agent.Q[s, a] = 2.0
    agent.Q[s_next] = [1.0, 5.0, 3.0, 4.0]
    td = agent.update(s, a, r=-1.0, s_next=s_next, terminated=False)
    return {"q_before": 2.0, "max_next": 5.0, "r": -1.0, "target": -1.0 + 0.99 * 5.0, "td": td, "q_after": agent.Q[s, a]}


def network_layers(side: int) -> list[dict]:
    net = QNetwork(side)
    rows = []
    for name, module in list(net.features.named_children()) + [("pool", net.pool)] + list(net.head.named_children()):
        params = sum(p.numel() for p in module.parameters())
        rows.append({"layer": f"{type(module).__name__} ({name})", "parameters": params})
    rows.append({"layer": "TOTAL", "parameters": sum(p.numel() for p in net.parameters())})
    return rows


def network_totals() -> list[dict]:
    return [{"input side": s, "used for": u, "parameters": sum(p.numel() for p in QNetwork(s).parameters())}
            for s, u in [(7, "local 7x7 view (trained generalists)"), (10, "global 10x10"), (20, "global 20x20"),
                         (50, "global 50x50"), (100, "global 100x100")]]


def stats_examples() -> dict:
    sample = [12.0, 15.0, 11.0, 14.0, 13.0]
    ci = mean_ci(sample)
    return {"sample": sample, "ci": ci, "wilson_0_of_20": wilson_interval(0, 20), "wilson_18_of_20": wilson_interval(18, 20),
            "seeds": derive_seeds(0, 3), "h_example": crossover_horizon(500.0, 0.003, 0.001)}


def heuristic_numbers() -> dict:
    a, b = (0, 0), (3, 7)
    from pathrl.algorithms.classical.astar import manhattan, octile
    return {"manhattan": manhattan(a, b), "octile": octile(a, b), "euclid": math.dist(a, b)}
