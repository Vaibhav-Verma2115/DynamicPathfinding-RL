"""DynamicPathfinding-RL research dashboard.

    streamlit run src/pathrl/dashboard/app.py

Every number shown is either computed live when you press a button (on this machine) or loaded
from the experiment CSVs in outputs/results. Nothing is simulated or hard-coded.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pathrl.algorithms import GridGraph, make_planner
from pathrl.algorithms.model_based import ValueIteration
from pathrl.algorithms.reinforcement_learning import DQN, DQNConfig, DoubleDQN, QLearning
from pathrl.analysis.convergence import ConvergenceCriterion, rl_convergence
from pathrl.environment import EnvConfig, GridWorldEnv, generate_map
from pathrl.evaluation.navigation import run_planner_episode
from pathrl.evaluation.profiling import measure
from pathrl.evaluation.rollout import evaluate_policy
from pathrl.evaluation.scenarios import ScenarioSampler
from pathrl.utils.config import PROJECT_ROOT, load_config
from pathrl.utils.seeding import make_rng
from pathrl.visualization.grid_visualizer import draw_grid

GRID_SIZES = [10, 20, 30, 50, 70, 100]
LEARNING_RATES = {"Q-Learning": [0.01, 0.05, 0.1, 0.3, 0.5],
                  "DQN": [0.0001, 0.0005, 0.001, 0.005],
                  "Double DQN": [0.0001, 0.0005, 0.001, 0.005]}
CFG = load_config()
RESULTS = PROJECT_ROOT / CFG["paths"]["results_dir"]
FIGURES = PROJECT_ROOT / CFG["paths"]["figures_dir"]

st.set_page_config(page_title="DynamicPathfinding-RL", layout="wide")
st.title("DynamicPathfinding-RL")
st.caption("Classical pathfinding vs. model-free deep RL in dynamic grid worlds. "
           "All values are computed live or loaded from experiment CSVs.")


def figure(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Environment")
    grid_size = st.selectbox("Grid size", GRID_SIZES, index=0)
    density = st.select_slider("Obstacle density", [0.15, 0.30], value=0.15)
    dynamics = st.selectbox("Obstacle dynamics", ["static", "random_walk", "oscillating_corridor"])
    change_interval = st.select_slider("Obstacles move every k steps", [1, 2, 5, 10, 20], value=5,
                                       disabled=dynamics == "static")
    slip = st.select_slider("Slip probability", [0.0, 0.1, 0.2], value=0.0)
    connectivity = st.radio("Movement", [4, 8], horizontal=True)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)

env_cfg = EnvConfig(grid_size=grid_size, obstacle_density=density, dynamics=dynamics,
                    change_interval=change_interval if dynamics != "static" else 0, slip_prob=slip,
                    connectivity=connectivity, max_steps_factor=2.0)
grid_map = generate_map(env_cfg, make_rng(int(seed)))

tab_sim, tab_cmp, tab_rl, tab_res = st.tabs(
    ["Environment simulator", "Algorithm comparison", "RL training (E6)", "Research results"])

# --------------------------------------------------------------- simulator
with tab_sim:
    st.subheader("Run one planner as an agent in this environment")
    c1, c2 = st.columns([1, 2])
    with c1:
        planner_name = st.selectbox("Planner", ["astar", "dstar_lite", "dijkstra", "bfs"],
                                    format_func={"astar": "A*", "dstar_lite": "D* Lite",
                                                 "dijkstra": "Dijkstra", "bfs": "BFS"}.get)
        replan = st.radio("Replanning", ["on_change", "on_block"], horizontal=True)
        if st.button("Run episode", type="primary"):
            env = GridWorldEnv(env_cfg, grid_map=grid_map)
            frames = []

            class Recorder:  # wraps env.step to capture each frame for the replay slider
                def __init__(self, env):
                    self.env, self._step = env, env.step

                def __call__(self, action):
                    out = self._step(action)
                    frames.append((self.env.agent_pos, out[4]["obstacle_positions"]))
                    return out

            env.step = Recorder(env)
            result = run_planner_episode(env, make_planner(planner_name, env_cfg), seed=int(seed), replan=replan)
            st.session_state["sim"] = {"frames": [(grid_map.start, [tuple(p) for p in grid_map.dynamic_init])] + frames,
                                       "result": result}
        if "sim" in st.session_state:
            r = st.session_state["sim"]["result"]
            st.metric("Outcome", "reached goal" if r["success"] else "failed")
            st.write(pd.Series({k: r[k] for k in ["steps", "path_cost", "collisions", "replans",
                                                  "total_expansions", "total_planning_time_s", "blocked_steps"]}))
    with c2:
        if "sim" in st.session_state:
            frames = st.session_state["sim"]["frames"]
            t = st.slider("Step", 0, len(frames) - 1, len(frames) - 1)
            fig, ax = plt.subplots(figsize=(5, 5))
            trail = [f[0] for f in frames[: t + 1]]
            draw_grid(ax, grid_map, obstacle_positions=frames[t][1], agent=frames[t][0], path=trail, title=f"step {t}")
            figure(fig)
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            draw_grid(ax, grid_map, agent=grid_map.start, title="Generated map")
            figure(fig)

# -------------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("All planners on the same map (static occupancy)")
    st.caption("Single map, single run: an illustration, not a statistical result. See E1/E2 for 30-seed results.")
    if st.button("Compare planners", type="primary"):
        variants = [("BFS", "bfs", {}), ("Dijkstra", "dijkstra", {}), ("A* (h=0)", "astar", {"heuristic": "zero"}),
                    ("A*", "astar", {}), ("A* (1.5 x h)", "astar", {"weight": 1.5}), ("D* Lite", "dstar_lite", {})]
        rows, results = [], []
        for label, name, kw in variants:
            m = measure(lambda: make_planner(name, env_cfg, record_expanded=True, **kw).plan(
                grid_map.walls, grid_map.start, grid_map.goal), repeats=3)
            res = m.result
            results.append((label, res))
            rows.append({"planner": label, "found": res.found, "cost": res.cost, "expansions": res.stats.expansions,
                         "pushes": res.stats.pushes, "time_ms": m.wall_time_s * 1e3,
                         "peak_memory_KiB": m.peak_memory_bytes / 1024})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, (label, res) in zip(axes.ravel(), results):
            draw_grid(ax, grid_map, path=res.path, explored=res.expanded, legend=False,
                      title=f"{label}: {res.stats.expansions} expanded")
        figure(fig)

# -------------------------------------------------------------- RL training
with tab_rl:
    st.subheader("Live convergence experiment on this map")
    st.caption("Trains on the map selected in the sidebar and evaluates the greedy policy periodically. "
               "Convergence = 3 consecutive evaluations with success >= 0.9 and stable return (analysis/convergence.py).")
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["Q-Learning", "DQN", "Double DQN", "Value Iteration"])
    if algo == "Value Iteration":
        lr = None
        c2.write("Value Iteration has no learning rate; convergence = Bellman residual < 1e-6.")
    else:
        lr = c2.select_slider("Learning rate", LEARNING_RATES[algo], value=LEARNING_RATES[algo][2 if algo == "Q-Learning" else 1])
    budget = c3.number_input("Episodes (Q-Learning) / env steps (DQN)", min_value=100, max_value=500_000,
                             value=2000 if algo == "Q-Learning" else 30_000, step=100)
    if grid_size > 20 and algo in ("DQN", "Double DQN"):
        st.warning("DQN on large grids is slow on CPU; expect minutes per 10k steps.")
    if st.button("Run experiment", type="primary"):
        train_cfg = replace(env_cfg, max_steps_factor=1.0)
        env, eval_env = GridWorldEnv(train_cfg, grid_map=grid_map), GridWorldEnv(train_cfg, grid_map=grid_map)
        bar = st.progress(0.0, text="training...")
        last = [0.0]

        def progress(frac):
            if frac - last[0] >= 0.01 or frac >= 1.0:
                bar.progress(min(frac, 1.0), text=f"training... {frac:.0%}")
                last[0] = frac

        t0 = time.perf_counter()
        if algo == "Value Iteration":
            vi = ValueIteration(train_cfg).solve(grid_map.walls, grid_map.goal)
            bar.progress(1.0, text="done")
            st.metric("Sweeps to convergence", vi.sweeps)
            st.metric("Solve time (s)", f"{vi.solve_time_s:.3f}")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.semilogy(vi.residuals)
            ax.axhline(1e-6, color="k", ls=":")
            ax.set_xlabel("sweep")
            ax.set_ylabel("Bellman residual")
            figure(fig)
        else:
            episodes_hint = int(budget) if algo == "Q-Learning" else max(1, int(budget) // (grid_size * 4))
            interval = max(1, episodes_hint // 60)
            if algo == "Q-Learning":
                agent = QLearning(grid_size, train_cfg.connectivity, alpha=lr, seed=int(seed))
                log = agent.train(env, int(budget), seed=int(seed), eval_interval=interval, eval_episodes=1, progress=progress)
            else:
                cls = DQN if algo == "DQN" else DoubleDQN
                agent = cls((3, grid_size, grid_size), env.action_space.n,
                            DQNConfig(learning_rate=lr, total_steps=int(budget), learning_starts=min(1000, int(budget) // 5),
                                      target_update_interval=1000, device="cpu"), seed=int(seed))
                log = agent.train(ScenarioSampler([env]), seed=int(seed), eval_interval=interval,
                                  evaluator=lambda p: evaluate_policy(p, eval_env, 1, seed=1), progress=progress)
            bar.progress(1.0, text=f"done in {time.perf_counter() - t0:.1f}s")
            ev, ep = log.evaluations_df(), log.episodes_df()
            conv = rl_convergence(ev, ConvergenceCriterion.from_config(CFG))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Converged", "yes" if conv["converged"] else "no")
            m2.metric("Episodes to convergence", "-" if not conv["converged"] else int(conv["episodes_to_convergence"]))
            m3.metric("Training time (s)", f"{log.train_time_s:.2f}")
            m4.metric("Evaluation time (s)", f"{log.eval_time_s:.2f}")
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
            axes[0].plot(ep["episode"], ep["return"].rolling(20, min_periods=1).mean(), label="training (20-ep moving avg)")
            axes[0].plot(ev["episode"], ev["mean_return"], "o-", ms=3, label="greedy evaluation")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Return")
            axes[0].legend(frameon=False)
            axes[1].plot(ep["episode"], ep["success"].astype(float).rolling(20, min_periods=1).mean(), label="training")
            axes[1].plot(ev["episode"], ev["success_rate"], "o-", ms=3, label="greedy evaluation")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Success rate")
            axes[1].legend(frameon=False)
            figure(fig)
            row = {"timestamp": pd.Timestamp.now().isoformat(), "algorithm": algo, "grid_size": grid_size,
                   "density": density, "learning_rate": lr, "seed": int(seed), "budget": int(budget),
                   "train_time_s": log.train_time_s, **conv}
            log_path = Path(os.environ.get("PATHRL_DASHBOARD_LOG", RESULTS / "dashboard" / "dashboard_runs.csv"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            st.caption(f"Run appended to {log_path} (real measurements from this machine).")

    st.divider()
    st.subheader("Experiment E6 results (from CSV)")
    runs_csv = RESULTS / "e6_convergence" / "e6_rl_runs.csv"
    if runs_csv.exists():
        runs = pd.read_csv(runs_csv)
        algo_pick = st.selectbox("Algorithm (E6 results)", sorted(runs["algorithm"].unique()))
        sub = runs[runs["algorithm"] == algo_pick]
        table = sub.groupby(["learning_rate", "grid_size"]).agg(
            converged=("converged", "mean"), episodes_to_convergence=("episodes_to_convergence", "mean"),
            final_success=("final_success_rate", "mean"), runs=("seed_index", "count")).reset_index()
        st.dataframe(table, hide_index=True)
        for name in [f"e6_{algo_pick}_reward_curves", f"e6_{algo_pick}_success_curves",
                     f"e6_{algo_pick}_lr_bars", f"e6_{algo_pick}_heatmap"]:
            if (FIGURES / f"{name}.png").exists():
                st.image(str(FIGURES / f"{name}.png"))
    else:
        st.info("E6 has not been run yet (pending). Run: python main.py experiment e6")

# ----------------------------------------------------------------- results
with tab_res:
    st.subheader("Research comparison graphs (generated from outputs/results by `python main.py figures`)")
    groups = {"E1 static scaling": "e1_", "E2 heuristic quality": "e2_", "DQN training": "train_dqn",
              "E3 dynamic obstacles / crossover": "e3_", "E4 stochastic transitions": "e4_",
              "E5 generalization": "e5_", "E6 convergence": "e6_"}
    for title, prefix in groups.items():
        files = sorted(FIGURES.glob(f"{prefix}*.png"))
        with st.expander(f"{title} ({len(files)} figures)" if files else f"{title} (pending)"):
            if not files:
                st.write("No results yet for this experiment.")
            for f in files:
                st.image(str(f), caption=f.name)
    summary_dir = RESULTS / "summary"
    if summary_dir.exists():
        st.subheader("Summary tables (mean, 95% CI)")
        pick = st.selectbox("Table", sorted(p.name for p in summary_dir.glob("*.csv")))
        df = pd.read_csv(summary_dir / pick)
        st.dataframe(df, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=pick)
