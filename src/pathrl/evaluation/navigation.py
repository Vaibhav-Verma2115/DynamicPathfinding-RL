"""Running a classical planner as an agent in the (possibly changing, possibly slippery) environment.

The planner sees the global occupancy map (its information assumption). The agent follows the
current plan one cell per step. It asks the planner for a new plan when:

  replan="on_change"  the map changed this step, OR the agent is not where the plan said
                      (slip or collision). This keeps an optimal plan for the current map at
                      all times. It is the setting where incremental repair (D* Lite) should
                      help most.
  replan="on_block"   only when the next cells of the current plan are now blocked, or the
                      agent is off the plan. A "lazy" baseline that saves work but may follow
                      a no-longer-optimal plan.

Every environment change is passed to planner.update(), so D* Lite always knows what changed,
even when no new plan is requested.

If no path currently exists (moving obstacles can temporarily cut the goal off), the agent
takes the legal move that most reduces the heuristic distance to the goal, or bumps a wall
if every move is blocked. These steps are counted as `blocked_steps`.

Costs are split into the initial plan and the later repairs/replans. Wall time is measured
around plan() and update() only (pure computation, no environment stepping).
"""

from __future__ import annotations

import time

from pathrl.algorithms.base_planner import Planner, SearchStats
from pathrl.algorithms.classical.astar import make_heuristic
from pathrl.environment import GridMap, GridWorldEnv
from pathrl.environment.dynamics import move_cost

LOOKAHEAD = 3  # on_block checks this many upcoming plan cells


def run_planner_episode(env: GridWorldEnv, planner: Planner, seed: int | None = None,
                        grid_map: GridMap | None = None, replan: str = "on_change") -> dict:
    if replan not in ("on_change", "on_block"):
        raise ValueError("replan must be 'on_change' or 'on_block'")
    options = {"map": grid_map} if grid_map is not None else None
    env.reset(seed=seed, options=options)
    planner.reset()
    goal, graph = env.grid_map.goal, planner.graph
    _, h = make_heuristic(graph, "auto")

    t0 = time.perf_counter()
    result = planner.plan(env.occupancy, env.agent_pos, goal)
    initial_time = time.perf_counter() - t0
    initial = result.stats
    repair = SearchStats()
    path, idx = result.path, 0
    replans, repair_time, blocked_steps, change_steps = 0, 0.0, 0, 0
    total_reward, collisions, path_cost = 0.0, 0, 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        pos = env.agent_pos
        if path is not None and idx + 1 < len(path):
            nxt = path[idx + 1]
            action = graph.deltas.index((nxt[0] - pos[0], nxt[1] - pos[1]))
        else:
            blocked_steps += 1
            options_ = [(h(n, goal), graph.deltas.index((n[0] - pos[0], n[1] - pos[1])))
                        for n, _ in graph.neighbors(pos, env.occupancy)]
            action = min(options_)[1] if options_ else 0
            nxt = None

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        collisions += info["collided"]
        moved = (env.agent_pos[0] - pos[0], env.agent_pos[1] - pos[1])
        if moved != (0, 0):
            path_cost += move_cost(moved, env.config.diagonal_cost)
        if terminated or truncated:
            break

        changed = info["changed_cells"]
        change_steps += bool(changed)
        on_plan = nxt is not None and env.agent_pos == nxt
        if on_plan:
            idx += 1
        t1 = time.perf_counter()
        planner.update(changed, env.occupancy)
        repair_time += time.perf_counter() - t1

        if replan == "on_change":
            need = bool(changed) or not on_plan
        else:
            upcoming = path[idx + 1 : idx + 1 + LOOKAHEAD] if path else []
            need = not on_plan or path is None or any(env.occupancy[c] for c in upcoming)
        if need:
            t1 = time.perf_counter()
            result = planner.plan(env.occupancy, env.agent_pos, goal)
            repair_time += time.perf_counter() - t1
            repair.add(result.stats)
            replans += 1
            path, idx = result.path, 0

    return {
        "success": info["outcome"] == "reached_goal",
        "return": total_reward,
        "steps": env.steps,
        "collisions": collisions,
        "path_cost": path_cost,
        "initial_plan_time_s": initial_time,
        "initial_expansions": initial.expansions,
        "initial_pushes": initial.pushes,
        "replans": replans,
        "repair_time_s": repair_time,
        "repair_expansions": repair.expansions,
        "repair_pushes": repair.pushes,
        "repair_vertex_updates": repair.vertex_updates,
        "total_planning_time_s": initial_time + repair_time,
        "total_expansions": initial.expansions + repair.expansions,
        "planning_time_per_step_s": (initial_time + repair_time) / max(env.steps, 1),
        "change_steps": change_steps,
        "blocked_steps": blocked_steps,
    }
