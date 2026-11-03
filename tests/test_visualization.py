import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathrl.environment import EnvConfig, generate_map
from pathrl.visualization.grid_visualizer import draw_grid, save_figure


def test_draw_and_save_grid(tmp_path):
    m = generate_map(EnvConfig(grid_size=10, dynamics="random_walk", change_interval=1), np.random.default_rng(0))
    fig, ax = plt.subplots()
    draw_grid(ax, m, agent=m.start, path=[m.start, (0, 1)], explored=[(1, 1)], title="test")
    out = save_figure(fig, "grid", tmp_path)
    assert out.exists() and out.stat().st_size > 0
