"""Matplotlib drawing of grids, paths and explored cells.

Used by the demo now, and later by the planners (explored nodes), the RL
evaluation (trajectories) and the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pathrl.environment.dynamics import Cell
from pathrl.environment.map_generator import GridMap

# Okabe-Ito colour-blind-safe palette
COLORS = {
    "free": "#FFFFFF",
    "wall": "#2B2B2B",
    "moving": "#E69F00",
    "start": "#009E73",
    "goal": "#D55E00",
    "agent": "#0072B2",
    "path": "#56B4E9",
    "explored": "#F0E442",
}


def draw_grid(
    ax: plt.Axes,
    grid_map: GridMap,
    obstacle_positions: Iterable[Cell] | None = None,
    agent: Cell | None = None,
    path: Sequence[Cell] | None = None,
    explored: Iterable[Cell] | None = None,
    title: str | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Draw one grid state. `obstacle_positions` defaults to the map's initial moving obstacles."""
    n = grid_map.size
    codes = grid_map.walls.astype(int)  # 0 free, 1 wall, 2 explored, 3 moving
    if explored is not None:
        for cell in explored:
            if not grid_map.walls[cell]:
                codes[cell] = 2
    movers = list(obstacle_positions) if obstacle_positions is not None else [tuple(p) for p in grid_map.dynamic_init]
    for cell in movers:
        codes[tuple(cell)] = 3
    cmap = ListedColormap([COLORS["free"], COLORS["wall"], COLORS["explored"], COLORS["moving"]])
    ax.imshow(codes, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

    # Faint grid lines only when cells are big enough to see them
    if n <= 50:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if path:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color=COLORS["path"], linewidth=2, zorder=3)
    marker = max(3, 120 / n)
    for cell, key, symbol in ((grid_map.start, "start", "s"), (grid_map.goal, "goal", "*")):
        ax.plot(cell[1], cell[0], symbol, color=COLORS[key], markersize=marker * (1.4 if symbol == "*" else 1), zorder=4)
    if agent is not None:
        ax.plot(agent[1], agent[0], "o", color=COLORS["agent"], markersize=marker, zorder=5)

    if legend:
        handles = [Patch(color=COLORS["wall"], label="wall"), Patch(color=COLORS["start"], label="start"),
                   Patch(color=COLORS["goal"], label="goal")]
        if movers:
            handles.append(Patch(color=COLORS["moving"], label="moving obstacle"))
        if agent is not None:
            handles.append(Patch(color=COLORS["agent"], label="agent"))
        if path:
            handles.append(Patch(color=COLORS["path"], label="path"))
        if explored is not None:
            handles.append(Patch(color=COLORS["explored"], label="explored"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def save_figure(fig: plt.Figure, name: str, figures_dir: str | Path = "outputs/figures") -> Path:
    """Save as PNG (300 dpi) and close the figure. Returns the file path."""
    out = Path(figures_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
