"""Publication-style plotting helpers. Every function takes measured data frames, never constants.

Error bars and bands are 95% confidence intervals of the mean across independent seeds
(Student t, see analysis/statistics.py).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathrl.analysis.statistics import summarize

# Okabe-Ito colour-blind-safe palette, assigned in a fixed order
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })


def line_ci(ax, df: pd.DataFrame, x: str, y: str, hue: str, order: Sequence | None = None,
            logy: bool = False, band: bool = False, label_fmt: str = "{}") -> None:
    """Mean of `y` per (hue, x) with 95% CI error bars (or a shaded band)."""
    s = summarize(df, [hue, x], [y])
    levels = order if order is not None else list(dict.fromkeys(s[hue]))
    for i, level in enumerate(levels):
        g = s[s[hue] == level].sort_values(x)
        if g.empty:
            continue
        mean = g[f"{y}_mean"].to_numpy()
        low = np.nan_to_num(mean - g[f"{y}_ci_low"].to_numpy())
        high = np.nan_to_num(g[f"{y}_ci_high"].to_numpy() - mean)
        color, marker = PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]
        if band:
            ax.plot(g[x], mean, color=color, label=label_fmt.format(level), linewidth=1.4)
            ax.fill_between(g[x], mean - low, mean + high, color=color, alpha=0.18, linewidth=0)
        else:
            ax.errorbar(g[x], mean, yerr=[low, high], color=color, marker=marker, markersize=4,
                        capsize=2.5, linewidth=1.2, label=label_fmt.format(level))
    if logy:
        ax.set_yscale("log")


def bar_ci(ax, df: pd.DataFrame, x: str, y: str, hue: str | None = None, order=None, hue_order=None) -> None:
    by = [x] if hue is None else [hue, x]
    s = summarize(df, by, [y])
    xs = order if order is not None else sorted(s[x].unique())
    hues = [None] if hue is None else (hue_order if hue_order is not None else list(dict.fromkeys(s[hue])))
    width = 0.8 / len(hues)
    for j, h in enumerate(hues):
        g = s if h is None else s[s[hue] == h]
        g = g.set_index(x).reindex(xs)
        mean = g[f"{y}_mean"].to_numpy()
        err = [np.nan_to_num(mean - g[f"{y}_ci_low"].to_numpy()), np.nan_to_num(g[f"{y}_ci_high"].to_numpy() - mean)]
        pos = np.arange(len(xs)) + (j - (len(hues) - 1) / 2) * width
        ax.bar(pos, mean, width, yerr=err, capsize=2, color=PALETTE[j % len(PALETTE)],
               label=None if h is None else str(h), edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(xs)), [str(v) for v in xs])


def heatmap(ax, table: pd.DataFrame, fmt: str = "{:.0f}", cmap: str = "viridis", missing: str = "n/c") -> None:
    """Annotated heatmap. NaN cells (e.g. no converged runs) are shown hatched with `missing`."""
    data = table.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(r) for r in table.index])
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", color="#999999", lw=0))
                ax.text(j, i, missing, ha="center", va="center", fontsize=7, color="#555555")
            else:
                norm = (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-12)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if norm < 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
