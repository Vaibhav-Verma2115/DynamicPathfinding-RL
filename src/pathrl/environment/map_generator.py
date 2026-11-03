"""Seeded, validated map generation.

A `GridMap` is the complete, immutable description of one scenario: static
walls, start, goal, and the initial layout of moving obstacles. Two algorithms
given the same GridMap face exactly the same problem, so it is the unit that
matched experiments share.

Reachability is checked with 4-connected moves on the static walls only. A 4-path
is also a valid 8-path, so the same maps can be used for the 8-connected
ablation. Moving obstacles may block a route temporarily, but never permanently
(see docs/methodology.md).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pathrl.environment.config import DYNAMICS, EnvConfig
from pathrl.environment.dynamics import ACTIONS_4, Cell


class MapGenerationError(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class GridMap:
    walls: np.ndarray                     # bool (N, N), True = static wall
    start: Cell
    goal: Cell
    dynamics: str = "static"
    dynamic_init: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    tracks: tuple[np.ndarray, ...] = ()   # oscillating corridors: ordered cells each obstacle slides along
    track_start: tuple[int, ...] = ()     # initial index of each obstacle on its track

    def __post_init__(self) -> None:
        walls = np.array(self.walls, dtype=bool)
        walls.setflags(write=False)  # a shared map must never be modified in place
        dyn = np.array(self.dynamic_init, dtype=np.int64).reshape(-1, 2)
        dyn.setflags(write=False)
        object.__setattr__(self, "walls", walls)
        object.__setattr__(self, "dynamic_init", dyn)
        object.__setattr__(self, "start", (int(self.start[0]), int(self.start[1])))
        object.__setattr__(self, "goal", (int(self.goal[0]), int(self.goal[1])))

        n_rows, n_cols = walls.shape
        if n_rows != n_cols:
            raise ValueError("GridMap must be square")
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not (0 <= cell[0] < n_rows and 0 <= cell[1] < n_cols):
                raise ValueError(f"{name} {cell} is outside the grid")
            if walls[cell]:
                raise ValueError(f"{name} {cell} is inside a wall")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")
        if self.dynamics not in DYNAMICS:
            raise ValueError(f"dynamics must be one of {DYNAMICS}")
        cells = [tuple(map(int, p)) for p in dyn]
        if len(set(cells)) != len(cells):
            raise ValueError("moving obstacles must start in distinct cells")
        for cell in cells:
            if walls[cell] or cell in (self.start, self.goal):
                raise ValueError(f"moving obstacle at {cell} overlaps a wall, start or goal")
        if self.dynamics == "oscillating_corridor" and not (len(self.tracks) == len(self.track_start) == len(cells)):
            raise ValueError("oscillating maps need one track and start index per obstacle")

    @property
    def size(self) -> int:
        return self.walls.shape[0]

    @classmethod
    def from_ascii(cls, text: str) -> "GridMap":
        """Build a map from a drawing: '#' wall, '.' free, 'S' start, 'G' goal, 'o' random walker.

        Handy for hand-made test cases with known shortest paths.
        """
        rows = [line.split() if " " in line.strip() else list(line.strip())
                for line in text.strip().splitlines()]
        walls = np.array([[ch == "#" for ch in row] for row in rows])
        find = lambda target: [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == target]
        (start,), (goal,) = find("S"), find("G")
        movers = find("o")
        return cls(walls, start, goal, "random_walk" if movers else "static",
                   np.array(movers, dtype=np.int64).reshape(-1, 2))

    def to_ascii(self) -> str:
        grid = np.where(self.walls, "#", ".").astype("<U1")
        for r, c in self.dynamic_init:
            grid[r, c] = "o"
        grid[self.start] = "S"
        grid[self.goal] = "G"
        return "\n".join(" ".join(row) for row in grid)


def bfs_distances(walls: np.ndarray, source: Cell) -> np.ndarray:
    """4-connected step distance from `source` to every cell (-1 if unreachable)."""
    n = walls.shape[0]
    dist = np.full(walls.shape, -1, dtype=np.int64)
    dist[source] = 0
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTIONS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not walls[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def is_reachable(walls: np.ndarray, start: Cell, goal: Cell) -> bool:
    return bfs_distances(walls, start)[goal] >= 0


def static_shortest_path(walls: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """One 4-connected shortest path, used to place corridor obstacles where they matter.

    (The instrumented BFS planner with metrics is built in Stage 4; this is only a map utility.)
    """
    dist = bfs_distances(walls, goal)
    if dist[start] < 0:
        return None
    path, cell, n = [start], start, walls.shape[0]
    while cell != goal:  # walk downhill in distance-to-goal
        r, c = cell
        for dr, dc in ACTIONS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and dist[nr, nc] == dist[r, c] - 1:
                cell = (nr, nc)
                break
        path.append(cell)
    return path


def generate_map(cfg: EnvConfig, rng: np.random.Generator) -> GridMap:
    """Sample walls, start and goal until the constraints hold, then place moving obstacles."""
    for _ in range(cfg.max_generation_attempts):
        start, goal = _place_start_goal(cfg, rng)
        walls = _sample_walls(cfg.grid_size, cfg.obstacle_density, start, goal, rng)
        if cfg.require_reachable and not is_reachable(walls, start, goal):
            continue
        if cfg.dynamics == "random_walk":
            return GridMap(walls, start, goal, "random_walk", _place_walkers(cfg, walls, start, goal, rng))
        if cfg.dynamics == "oscillating_corridor":
            init, tracks, track_start = _place_corridors(cfg, walls, start, goal, rng)
            return GridMap(walls, start, goal, "oscillating_corridor", init, tracks, track_start)
        return GridMap(walls, start, goal)
    raise MapGenerationError(
        f"No valid {cfg.grid_size}x{cfg.grid_size} map at density {cfg.obstacle_density} "
        f"after {cfg.max_generation_attempts} attempts"
    )


def _place_start_goal(cfg: EnvConfig, rng: np.random.Generator) -> tuple[Cell, Cell]:
    n = cfg.grid_size
    if cfg.placement == "corners":
        return (0, 0), (n - 1, n - 1)
    start = (int(rng.integers(n)), int(rng.integers(n)))
    rows, cols = np.indices((n, n))
    dist = np.abs(rows - start[0]) + np.abs(cols - start[1])
    candidates = np.flatnonzero(dist >= math.ceil(cfg.min_start_goal_distance * n))
    if candidates.size == 0:  # tiny grid: fall back to the farthest cells
        candidates = np.flatnonzero(dist == dist.max())
    goal = np.unravel_index(int(rng.choice(candidates)), (n, n))
    return start, (int(goal[0]), int(goal[1]))


def _sample_walls(n: int, density: float, start: Cell, goal: Cell, rng: np.random.Generator) -> np.ndarray:
    """Exactly round(density * N^2) walls, never on start or goal."""
    reserved = {start[0] * n + start[1], goal[0] * n + goal[1]}
    candidates = np.array([i for i in range(n * n) if i not in reserved])
    count = min(round(density * n * n), candidates.size)
    walls = np.zeros(n * n, dtype=bool)
    walls[rng.choice(candidates, size=count, replace=False)] = True
    return walls.reshape(n, n)


def _num_dynamic(cfg: EnvConfig) -> int:
    return max(1, round(cfg.dynamic_obstacle_density * cfg.grid_size**2))


def _place_walkers(cfg: EnvConfig, walls: np.ndarray, start: Cell, goal: Cell, rng) -> np.ndarray:
    n = cfg.grid_size
    free = [i for i in np.flatnonzero(~walls.ravel()) if i not in (start[0] * n + start[1], goal[0] * n + goal[1])]
    count = min(_num_dynamic(cfg), len(free))
    chosen = rng.choice(np.array(free), size=count, replace=False)
    return np.stack(np.unravel_index(chosen, (n, n)), axis=1)


def _place_corridors(cfg: EnvConfig, walls: np.ndarray, start: Cell, goal: Cell, rng):
    """Put sliding obstacles across the static shortest path so they actually open and close it.

    Each obstacle gets a straight track perpendicular to the path direction, at most
    `oscillation_amplitude` cells to each side. Tracks never overlap, and every track has
    at least 2 cells, so no corridor obstacle can block the path forever.
    """
    path = static_shortest_path(walls, start, goal)
    if path is None or len(path) < 3:
        return np.zeros((0, 2), dtype=np.int64), (), ()
    n, used = cfg.grid_size, set()

    def usable(cell: Cell) -> bool:
        r, c = cell
        return (0 <= r < n and 0 <= c < n and not walls[r, c]
                and cell not in (start, goal) and cell not in used)

    tracks, track_start = [], []
    for i in rng.permutation(len(path) - 2):
        if len(tracks) == _num_dynamic(cfg):
            break
        prev, here = path[i], path[i + 1]
        if not usable(here):
            continue
        d_in = (here[0] - prev[0], here[1] - prev[1])
        axis = (d_in[1], d_in[0])  # perpendicular to the incoming move
        sides = []
        for sign in (-1, 1):
            side, cell = [], here
            for _ in range(cfg.oscillation_amplitude):
                cell = (cell[0] + sign * axis[0], cell[1] + sign * axis[1])
                if not usable(cell):
                    break
                side.append(cell)
            sides.append(side)
        track = sides[0][::-1] + [here] + sides[1]
        if len(track) < 2:
            continue
        used.update(track)
        tracks.append(np.array(track, dtype=np.int64))
        track_start.append(int(rng.integers(len(track))))

    init = np.array([t[s] for t, s in zip(tracks, track_start)], dtype=np.int64).reshape(-1, 2)
    return init, tuple(tracks), tuple(track_start)
