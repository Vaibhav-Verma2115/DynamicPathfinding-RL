"""Stage 10: build summary tables and figures from the raw CSVs in outputs/results/.

Nothing here invents data. If an experiment's CSV is missing, its figures are skipped and
listed as pending. Run:  python main.py figures
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from pathrl.analysis.crossover import crossover_table, measure_inference_scaling
from pathrl.analysis.statistics import summarize
from pathrl.evaluation.results import load_results, save_results
from pathrl.utils.config import PROJECT_ROOT
from pathrl.visualization.grid_visualizer import save_figure
from pathrl.visualization.plots import PALETTE, bar_ci, heatmap, line_ci, style

PLANNER_NAMES = {"bfs": "BFS", "dijkstra": "Dijkstra", "astar": "A* (Manhattan)", "dstar_lite": "D* Lite (initial)"}


def _fig_dir(cfg):
    return PROJECT_ROOT / cfg["paths"]["figures_dir"]


def _save(fig, name, cfg, made):
    made.append(str(save_figure(fig, name, _fig_dir(cfg)).relative_to(PROJECT_ROOT)))


def e1_figures(cfg, made, pending):
    df = load_results(cfg, "e1_static_scaling", "e1_raw")
    if df is None:
        pending.append("E1")
        return
    df["planner"] = df["planner"].map(PLANNER_NAMES)
    df["wall_time_ms"] = df["wall_time_s"] * 1e3
    df["peak_memory_kib"] = df["peak_memory_bytes"] / 1024
    order = list(PLANNER_NAMES.values())
    summary = summarize(df, ["density", "grid_size", "planner"],
                        ["wall_time_ms", "expansions", "peak_memory_kib", "pushes", "path_cost", "optimality_ratio"])
    save_results(summary, cfg, "summary", "e1_summary")
    for metric, label, name in [("wall_time_ms", "Wall time per query (ms)", "e1_runtime_vs_grid"),
                                ("expansions", "Node expansions", "e1_expansions_vs_grid"),
                                ("peak_memory_kib", "Peak Python memory (KiB, tracemalloc)", "e1_memory_vs_grid")]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
        for ax, d in zip(axes, sorted(df["density"].unique())):
            line_ci(ax, df[df["density"] == d], "grid_size", metric, "planner", order=order, logy=True)
            ax.set_title(f"Obstacle density {d}")
            ax.set_xlabel("Grid size N (N x N)")
        axes[0].set_ylabel(label)
        axes[1].legend(frameon=False)
        fig.suptitle(f"E1 static scaling, mean and 95% CI over {df['seed'].nunique()} maps", fontsize=10)
        fig.tight_layout()
        _save(fig, name, cfg, made)


def e2_figures(cfg, made, pending):
    df = load_results(cfg, "e2_heuristic_quality", "e2_raw")
    if df is None:
        pending.append("E2")
        return
    df["wall_time_ms"] = df["wall_time_s"] * 1e3
    order = ["h=0", "Manhattan", "1.5 x Manhattan"]
    summary = summarize(df, ["density", "grid_size", "variant"], ["expansions", "wall_time_ms", "optimality_ratio"])
    summary["fraction_optimal"] = df.groupby(["density", "grid_size", "variant"])["optimal"].mean().to_numpy()
    save_results(summary, cfg, "summary", "e2_summary")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for row, d in enumerate(sorted(df["density"].unique())):
        sub = df[df["density"] == d]
        for ax, (metric, label, logy) in zip(axes[row], [("expansions", "Node expansions", True),
                                                          ("wall_time_ms", "Wall time (ms)", True),
                                                          ("optimality_ratio", "Optimality ratio cost/C*", False)]):
            line_ci(ax, sub, "grid_size", metric, "variant", order=order, logy=logy)
            ax.set_ylabel(label)
            ax.set_title(f"density {d}")
            ax.set_xlabel("Grid size N")
    axes[0, 2].legend(frameon=False)
    fig.suptitle(f"E2 heuristic quality for A*, mean and 95% CI over {df['seed'].nunique()} maps", fontsize=10)
    fig.tight_layout()
    _save(fig, "e2_heuristic_quality", cfg, made)


def training_figures(cfg, made, pending):
    ev = load_results(cfg, "train_dqn", "train_dqn_evaluations")
    if ev is None:
        pending.append("DQN training curves")
        return
    ev["method"] = ev["algorithm"].map({"dqn": "DQN", "double_dqn": "Double DQN"})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for ax, metric, label in [(axes[0], "success_rate", "Validation success rate (greedy)"),
                              (axes[1], "mean_return", "Validation mean return")]:
        ev["step_bin"] = (ev["env_steps"] // 10000) * 10000
        line_ci(ax, ev, "step_bin", metric, "method", band=True)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Generalist training on 1,000 maps, 50 unseen validation maps (band: 95% CI over "
                 f"{ev['train_seed'].nunique()} training seeds)", fontsize=9)
    fig.tight_layout()
    _save(fig, "train_dqn_validation_curves", cfg, made)


def _interval_axis(ax, intervals) -> None:
    """Log-scaled change-interval axis with the actual intervals as tick labels, frequent change on the right."""
    ax.set_xscale("log")
    ax.set_xticks(sorted(intervals))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.invert_xaxis()


def e3_figures(cfg, made, pending):
    df = load_results(cfg, "e3_dynamic_obstacles", "e3_raw")
    if df is None:
        pending.append("E3")
        return
    df["computation_ms"] = df["computation_time_s"] * 1e3
    df["computation_per_step_ms"] = df["computation_ms"] / df["steps"].clip(lower=1)
    df["success_f"] = df["success"].astype(float)
    summary = summarize(df, ["grid_size", "change_interval", "method"],
                        ["computation_ms", "computation_per_step_ms", "success_f", "path_cost_ratio",
                         "replans", "total_expansions", "collisions", "steps"])
    save_results(summary, cfg, "summary", "e3_summary")

    # Classical scaling with change rate and grid size
    cl = df[df["family"] == "classical"]
    sizes = sorted(cl["grid_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.0 * len(sizes), 3.0), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), sizes):
        line_ci(ax, cl[cl["grid_size"] == n], "change_interval", "computation_ms", "method", logy=True)
        _interval_axis(ax, cl["change_interval"].unique())
        ax.set_title(f"{n} x {n}")
        ax.set_xlabel("Change interval (steps, more frequent to the right)")
    np.atleast_1d(axes)[0].set_ylabel("Planning + repair time per episode (ms)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("E3 classical planning cost vs. environmental change rate (95% CI over 30 maps)", fontsize=10)
    fig.tight_layout()
    _save(fig, "e3_planning_cost_vs_change_rate", cfg, made)

    rl_sizes = df.loc[df["family"] == "rl", "grid_size"].unique()
    if len(rl_sizes) == 0:
        pending.append("E3 RL comparison (no checkpoints)")
        return
    n = int(rl_sizes[0])
    sub = df[df["grid_size"] == n]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (metric, label, logy) in zip(axes, [("computation_ms", "Computation per episode (ms)", True),
                                                ("success_f", "Success rate", False),
                                                ("path_cost_ratio", "Path cost / static optimum", False)]):
        line_ci(ax, sub, "change_interval", metric, "method", logy=logy)
        _interval_axis(ax, cl["change_interval"].unique())
        ax.set_xlabel("Change interval (steps)")
        ax.set_ylabel(label)
    axes[1].legend(frameon=False, fontsize=7)
    obs = sub.loc[sub["family"] == "rl", "observation"].iloc[0] if "observation" in sub else "global"
    fig.suptitle(f"E3 at {n} x {n}: planners (global map) vs. trained policies ({obs} view), 95% CI", fontsize=10)
    fig.tight_layout()
    _save(fig, "e3_methods_vs_change_rate", cfg, made)

    training = load_results(cfg, "train_dqn", "train_dqn_summary")
    if training is None:
        pending.append("E3 crossover (no training summary)")
        return
    table = crossover_table(df, training)
    save_results(table, cfg, "summary", "e3_crossover")
    infer = measure_inference_scaling(sorted(df["grid_size"].unique()))
    save_results(infer, cfg, "summary", "e3_inference_scaling",
                 extra={"note": "forward-pass time of untrained networks: cost only, success not measured"})

    # Cumulative cost vs horizon for the fastest-changing and slowest-changing conditions
    intervals = sorted(table["change_interval"].unique())
    show = [intervals[-1], intervals[0]] if len(intervals) > 1 else intervals
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 3.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), show):
        t = table[table["change_interval"] == k]
        finite = t["crossover_episodes"].replace(np.inf, np.nan).dropna()
        h_max = float(finite.max()) * 10.0 if len(finite) else 1e7   # all "never": show a long horizon
        h = np.geomspace(1.0, max(h_max, 10.0), 200)
        for i, (planner, g) in enumerate(t.groupby("planner")):
            ax.plot(h, h * g["plan_time_per_episode_s"].iloc[0], color=PALETTE[i], label=f"{planner}")
        for j, (rl, g) in enumerate(t.groupby("rl_method")):
            r = g.iloc[0]
            ax.plot(h, r["train_time_s"] + h * r["infer_time_per_episode_s"], color=PALETTE[4 + j],
                    linestyle="--", label=f"{rl}: train + inference")
            ax.fill_between(h, r["train_time_ci_low"] + h * r["infer_time_per_episode_s"],
                            r["train_time_ci_high"] + h * r["infer_time_per_episode_s"], color=PALETTE[4 + j], alpha=0.12)
        ax.set_xlabel("Deployment horizon H (episodes)")
        ax.set_title(f"Obstacles move every {k} step(s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
    np.atleast_1d(axes)[0].set_ylabel("Cumulative computation (s)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"E3 crossover at {n} x {n}: C_plan(H) = H c_plan vs C_RL(H) = C_train + H c_infer "
                 "(computation only; see success rates)", fontsize=9)
    fig.tight_layout()
    _save(fig, "e3_cumulative_cost_crossover", cfg, made)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, ((planner, rl), g) in enumerate(table.groupby(["planner", "rl_method"])):
        g = g.sort_values("change_interval")
        y = g["crossover_episodes"].replace(np.inf, np.nan)
        ax.plot(g["change_interval"], y, marker=MARKERS_SAFE[i % 8], color=PALETTE[i % 8], label=f"{planner} vs {rl}")
    ax.set_xscale("log")
    if np.isfinite(table["crossover_episodes"]).any():
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=6)
    else:
        # Measured inference per episode >= planning per episode everywhere: H* does not exist
        ax.text(0.5, 0.5, "H* = never in every condition:\nper-episode inference cost >= per-episode planning cost",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_yticks([])
    ax.invert_xaxis()
    ax.set_xlabel("Change interval (steps, more frequent to the right)")
    ax.set_ylabel("Crossover horizon H* (episodes)")
    ax.set_title(f"E3 crossover horizon vs change rate ({n} x {n}); missing = never", fontsize=9)
    fig.tight_layout()
    _save(fig, "e3_crossover_horizon", cfg, made)

    # Per-decision cost of planners (measured, all sizes) vs DQN-architecture inference (measured, untrained)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    fastest = cl[cl["change_interval"] == cl["change_interval"].min()].copy()
    line_ci(ax, fastest, "grid_size", "computation_per_step_ms", "method", logy=True)
    ax.plot(infer["grid_size"], infer["infer_time_per_step_s"] * 1e3, "k--", marker="x",
            label="CNN forward pass (architecture cost, untrained)")
    ax.set_xlabel("Grid size N")
    ax.set_ylabel("Computation per decision (ms)")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title(f"Per-decision cost, obstacles move every {int(fastest['change_interval'].min())} step", fontsize=9)
    fig.tight_layout()
    _save(fig, "e3_per_decision_cost_vs_grid", cfg, made)


MARKERS_SAFE = ["o", "s", "^", "D", "v", "P", "X", "*"]


def e4_figures(cfg, made, pending):
    df = load_results(cfg, "e4_stochastic", "e4_raw")
    if df is None:
        pending.append("E4")
        return
    summary = summarize(df, ["grid_size", "slip_prob", "method"],
                        ["success_rate", "mean_return", "mean_collisions", "mean_path_deviation", "mean_steps", "prep_time_s"])
    save_results(summary, cfg, "summary", "e4_summary")
    sizes = sorted(df["grid_size"].unique())
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(len(sizes), 3, figsize=(11, 3.0 * len(sizes)), squeeze=False)
    for row, n in enumerate(sizes):
        sub = df[df["grid_size"] == n]
        for ax, (metric, label) in zip(axes[row], [("success_rate", "Success rate"), ("mean_return", "Mean return"),
                                                    ("mean_collisions", "Collisions per episode")]):
            line_ci(ax, sub, "slip_prob", metric, "method", order=methods)
            ax.set_xlabel("Slip probability")
            ax.set_ylabel(label)
            ax.set_title(f"{n} x {n}")
        axes[row, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("E4 stochastic transitions, mean and 95% CI over 30 maps (20 episodes each)", fontsize=10)
    fig.tight_layout()
    _save(fig, "e4_success_vs_slip", cfg, made)


def e5_figures(cfg, made, pending):
    df = load_results(cfg, "e5_generalization", "e5_raw")
    if df is None:
        pending.append("E5")
        return
    rl = df[df["method"] != "A*"]
    per_seed = rl.groupby(["method", "split", "train_seed"]).agg(success_rate=("success", "mean"),
                                                                   optimality_ratio=("optimality_ratio", "mean")).reset_index()
    summary = summarize(per_seed, ["method", "split"], ["success_rate", "optimality_ratio"])
    astar = df[df["method"] == "A*"]
    summary = pd.concat([summary, pd.DataFrame([{"method": "A*", "split": "test (unseen)",
                                                 "success_rate_mean": astar["success"].mean(), "n": len(astar)}])])
    save_results(summary, cfg, "summary", "e5_summary")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bar_ci(ax, per_seed, "method", "success_rate", hue="split", hue_order=["train (seen)", "test (unseen)"])
    ax.set_ylabel("Success rate (greedy)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title(f"E5 zero-shot generalization (95% CI over {per_seed['train_seed'].nunique()} training seeds)", fontsize=9)
    fig.tight_layout()
    _save(fig, "e5_generalization", cfg, made)


def e6_figures(cfg, made, pending):
    vi = load_results(cfg, "e6_convergence", "e6_vi_runs")
    res = load_results(cfg, "e6_convergence", "e6_vi_residuals")
    if vi is not None:
        save_results(summarize(vi, ["grid_size", "density", "slip_prob"], ["sweeps", "solve_time_s"]), cfg, "summary", "e6_vi_summary")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        r = res[(res["density"] == res["density"].min()) & (res["slip_prob"] == 0.1)]
        for i, (n, g) in enumerate(r.groupby("grid_size")):
            axes[0].plot(g["sweep"], g["residual"], color=PALETTE[i % 8], label=f"{n} x {n}")
        axes[0].axhline(cfg["rl"]["value_iteration"]["theta"], color="k", linestyle=":", linewidth=0.8)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Sweep")
        axes[0].set_ylabel("Bellman residual max|V_k+1 - V_k|")
        axes[0].set_title("Residual (slip 0.1, one map per size)")
        axes[0].legend(frameon=False, fontsize=7)
        line_ci(axes[1], vi, "grid_size", "sweeps", "slip_prob", label_fmt="slip {}")
        axes[1].set_xlabel("Grid size N")
        axes[1].set_ylabel("Sweeps to residual < theta")
        axes[1].legend(frameon=False)
        line_ci(axes[2], vi, "grid_size", "solve_time_s", "slip_prob", logy=True, label_fmt="slip {}")
        axes[2].set_xlabel("Grid size N")
        axes[2].set_ylabel("Solve time (s)")
        fig.suptitle("E6 Value Iteration convergence (sweeps, not episodes), 95% CI over seeds", fontsize=10)
        fig.tight_layout()
        _save(fig, "e6_value_iteration", cfg, made)
    else:
        pending.append("E6 value iteration")

    runs = load_results(cfg, "e6_convergence", "e6_rl_runs")
    evals = load_results(cfg, "e6_convergence", "e6_rl_evaluations")
    if runs is None:
        pending.append("E6 RL")
        return
    runs["converged_f"] = runs["converged"].astype(float)
    save_results(summarize(runs, ["algorithm", "grid_size", "learning_rate"],
                           ["converged_f", "episodes_to_convergence", "steps_to_convergence",
                            "time_to_convergence_s", "final_success_rate", "train_time_s"]), cfg, "summary", "e6_rl_summary")
    names = {"q_learning": "Q-Learning", "dqn": "DQN", "double_dqn": "Double DQN"}
    for algo, g_runs in runs.groupby("algorithm"):
        g_eval = evals[evals["algorithm"] == algo]
        sizes = sorted(g_runs["grid_size"].unique())
        # Convergence curves: greedy return and success vs episodes, one panel per grid size
        for metric, label, fname in [("mean_return", "Greedy evaluation return", "reward"),
                                     ("success_rate", "Greedy success rate", "success")]:
            fig, axes = plt.subplots(1, len(sizes), figsize=(3.1 * len(sizes), 3.0), squeeze=False)
            for ax, n in zip(axes[0], sizes):
                line_ci(ax, g_eval[g_eval["grid_size"] == n], "episode", metric, "learning_rate", band=True, label_fmt="lr {}")
                ax.set_title(f"{n} x {n}")
                ax.set_xlabel("Training episodes")
            axes[0, 0].set_ylabel(label)
            axes[0, -1].legend(frameon=False, fontsize=6)
            fig.suptitle(f"E6 {names[algo]}: {label.lower()} vs training episodes (95% CI over seeds)", fontsize=10)
            fig.tight_layout()
            _save(fig, f"e6_{algo}_{fname}_curves", cfg, made)
        # Episodes-to-convergence bars (converged runs only) + fraction converged
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        conv = g_runs[g_runs["converged"]]
        lrs = sorted(g_runs["learning_rate"].unique())
        if not conv.empty:
            bar_ci(axes[0], conv, "learning_rate", "episodes_to_convergence", hue="grid_size", order=lrs)
        axes[0].set_xlabel("Learning rate")
        axes[0].set_ylabel("Episodes to convergence (converged runs)")
        axes[0].legend(frameon=False, fontsize=6, title="N")
        bar_ci(axes[1], g_runs, "learning_rate", "converged_f", hue="grid_size", order=lrs)
        axes[1].set_xlabel("Learning rate")
        axes[1].set_ylabel("Fraction of runs converged")
        axes[1].set_ylim(0, 1.1)
        fig.suptitle(f"E6 {names[algo]}: learning-rate sensitivity", fontsize=10)
        fig.tight_layout()
        _save(fig, f"e6_{algo}_lr_bars", cfg, made)
        # Heatmaps: lr x grid size
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        ep = conv.groupby(["learning_rate", "grid_size"])["episodes_to_convergence"].mean().unstack().reindex(index=lrs, columns=sizes)
        heatmap(axes[0], ep, fmt="{:.0f}")
        axes[0].set_title("Mean episodes to convergence (n/c = no run converged)")
        sr = g_runs.groupby(["learning_rate", "grid_size"])["final_success_rate"].mean().unstack().reindex(index=lrs, columns=sizes)
        heatmap(axes[1], sr, fmt="{:.2f}", cmap="magma")
        axes[1].set_title("Final greedy success rate")
        for ax in axes:
            ax.set_xlabel("Grid size N")
            ax.set_ylabel("Learning rate")
        fig.suptitle(f"E6 {names[algo]}: learning rate x grid size", fontsize=10)
        fig.tight_layout()
        _save(fig, f"e6_{algo}_heatmap", cfg, made)

    # Grid size vs convergence time / episodes, best learning rate per algorithm and size
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    best = []
    for algo, g in runs[runs["converged"]].groupby("algorithm"):
        best_lr = g.groupby("learning_rate")["converged_f"].count().idxmax()
        best.append(g[g["learning_rate"] == best_lr].assign(label=f"{names[algo]} (lr {best_lr})"))
    if best:
        best = pd.concat(best)   # one call so each method gets its own colour and marker
        line_ci(axes[0], best, "grid_size", "episodes_to_convergence", "label")
        line_ci(axes[1], best, "grid_size", "time_to_convergence_s", "label", logy=True)
        axes[0].set_ylim(bottom=0)   # t-intervals with n = 2 can extend below zero
    for ax, label in zip(axes, ["Episodes to convergence", "Training time to convergence (s)"]):
        ax.set_xlabel("Grid size N")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E6 convergence vs grid size (converged runs, most frequently converging lr)", fontsize=10)
    fig.tight_layout()
    _save(fig, "e6_convergence_vs_grid", cfg, made)


def run(cfg: dict, workers: int = 1) -> dict:
    style()
    made, pending = [], []
    for fn in (e1_figures, e2_figures, training_figures, e3_figures, e4_figures, e5_figures, e6_figures):
        fn(cfg, made, pending)
    print("Figures:\n  " + "\n  ".join(made))
    print("Pending (no data):", pending or "none")
    return {"figures": made, "pending": pending}
