"""Planners and learners behind one interface. `make_planner` builds one by name."""

from pathrl.algorithms.base_planner import PlanResult, Planner, SearchStats
from pathrl.algorithms.grid_graph import GridGraph


def make_planner(name: str, env_cfg=None, **kwargs) -> Planner:
    """Build a planner by name, using the environment config's movement rules.

    Examples: make_planner("astar", cfg, heuristic="manhattan", weight=1.5)
    """
    from pathrl.algorithms.classical import BFS, AStar, Dijkstra

    registry = {"bfs": BFS, "dijkstra": Dijkstra, "astar": AStar}
    if name not in registry:
        raise ValueError(f"unknown planner {name!r}; available: {sorted(registry)}")
    graph = GridGraph.from_env_config(env_cfg) if env_cfg is not None else GridGraph()
    return registry[name](graph=graph, **kwargs)


__all__ = ["GridGraph", "PlanResult", "Planner", "SearchStats", "make_planner"]
