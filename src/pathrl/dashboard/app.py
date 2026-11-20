"""DynamicPathfinding-RL dashboard.

    streamlit run src/pathrl/dashboard/app.py

Four steps, top to bottom: build a map, watch one planner, compare all planners, train an AI agent,
then see the saved experiment results. Every number shown is either computed live when you press a
button (on this machine) or loaded from the experiment CSVs in outputs/results. Nothing is simulated
or hard-coded.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/: run from a fresh clone (e.g. Streamlit Cloud)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pathrl.algorithms import make_planner
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
DYNAMICS = {"static": "Still", "random_walk": "Wander", "oscillating_corridor": "Swing across"}
PLANNERS = {"astar": "A*", "dstar_lite": "D* Lite", "dijkstra": "Dijkstra", "bfs": "BFS"}
PLANNER_NOTES = {
    "astar": "Searches toward the goal using a hint (distance if there were no walls). "
             "Plans a fresh route when something changes.",
    "dstar_lite": "Plans once, then repairs only the part of the route that changed.",
    "dijkstra": "Spreads out from the start, cheapest cell first. No hint, so it checks more cells.",
    "bfs": "Spreads out layer by layer. Finds the shortest route because every step costs 1.",
}
EXPERIMENTS = {  # figure prefix -> what the experiment asks, in one line
    "e1_": ("E1 · Speed vs. map size", "How does search time grow as the map gets bigger?"),
    "e2_": ("E2 · A* hints", "How much does a better hint cut the number of cells A* checks?"),
    "train_dqn": ("DQN training", "How the neural-network agents learned over time."),
    "e3_": ("E3 · Moving obstacles", "Planning vs. learning when obstacles move. Does training ever pay off?"),
    "e4_": ("E4 · Slippery moves", "What happens when moves sometimes slip sideways?"),
    "e5_": ("E5 · Brand-new maps", "Can the AI handle maps it never saw in training?"),
    "e6_": ("E6 · Learning speed", "How fast does each learner reach a good policy, for each learning rate?"),
}
CFG = load_config()
RESULTS = PROJECT_ROOT / CFG["paths"]["results_dir"]
FIGURES = PROJECT_ROOT / CFG["paths"]["figures_dir"]

st.set_page_config(page_title="DynamicPathfinding-RL", layout="wide", initial_sidebar_state="collapsed")

# Monochrome look: monospace labels and headings, plain sans body, square black buttons.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');
html, body, p, label, li, input, textarea { font-family: 'Inter', sans-serif; }
h1, h2, h3, .mono, [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
.stTabs [data-baseweb="tab"] p, .stButton button p { font-family: 'JetBrains Mono', monospace !important; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebarCollapsedControl"] { display: none; }
.eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: .08em;
           text-transform: uppercase; color: #111; margin: 0 0 18px; }
.eyebrow:before { content: ''; display: inline-block; width: 10px; height: 10px; background: #111;
                  margin-right: 10px; vertical-align: 0; }
.stMarkdown h1.hero { font-family: 'JetBrains Mono', monospace !important; font-weight: 500; font-size: 64px;
        line-height: 1.04; letter-spacing: -.04em; margin: 0 0 22px; padding: 0; color: #111 !important; }
.stMarkdown h1.hero span.dim { color: #8A8A86 !important; font-family: inherit !important; }
.lede { font-size: 18px; line-height: 1.6; color: #3A3A38; max-width: 560px; margin-bottom: 24px; }
.figlabel { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .1em;
            text-transform: uppercase; color: #6B6B67; display: flex; justify-content: space-between; }
.strip { display: flex; gap: 0; border-top: 1px dashed #D6D6D2; margin-top: 18px; padding-top: 14px; }
.strip div { padding-right: 28px; margin-right: 28px; border-right: 1px solid #E4E4E0; }
.strip div:last-child { border-right: 0; }
.strip small { display: block; font-family: 'JetBrains Mono', monospace; font-size: 11px;
               letter-spacing: .1em; color: #8A8A86; text-transform: uppercase; }
.strip b { font-family: 'JetBrains Mono', monospace; font-weight: 500; font-size: 15px; }
.step { font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: .08em;
        text-transform: uppercase; color: #8A8A86; margin: 6px 0 2px; }
.vs { border: 1px solid #E4E4E0; padding: 22px 24px; height: 100%; }
.vs h3 { font-size: 20px; margin: 0 0 4px; }
.vs .tagline { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .08em;
               text-transform: uppercase; color: #8A8A86; margin-bottom: 14px; }
.vs.dark { background: #111; color: #fff; border-color: #111; }
.vs.dark h3, .vs.dark p, .vs.dark li { color: #fff; }
.vs ul { padding-left: 18px; margin: 0; } .vs li { margin-bottom: 6px; font-size: 15px; }
.big { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 500; line-height: 1.1; }
.big span { color: #8A8A86; font-size: 18px; }
.takeaway { border-left: 3px solid #111; padding: 10px 16px; background: #F4F4F2; margin: 12px 0; }
.stButton button[kind="primary"] { background: #111; color: #fff; border: 1px solid #111; padding: .6rem 1.4rem; }
.stButton button[kind="primary"]:hover { background: #333; border-color: #333; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid #E4E4E0; }
.stTabs [aria-selected="true"] p { font-weight: 700; }
[data-testid="stMetric"] { border: 1px solid #E4E4E0; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-size: 26px; }
</style>
""", unsafe_allow_html=True)

# Charts match the page: monospace text, no box, black and grey lines.
plt.rcParams.update({"font.family": "monospace", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#111111", "axes.prop_cycle": plt.cycler(color=["#111111", "#8A8A86"]),
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})


def figure(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def step_label(text):
    st.markdown(f'<p class="step">{text}</p>', unsafe_allow_html=True)


# --------------------------------------------------------------- hero + map
left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown('<p class="eyebrow">ADA project / pathfinding in changing grids</p>'
                '<h1 class="hero">Plan the path.<br><span class="dim">Or learn it?</span></h1>'
                '<p class="lede">A robot must cross a grid with walls and moving obstacles. Classic search '
                '<b>plans</b> a route; an AI agent <b>learns</b> one by trial and error. '
                'Build a map below, then try both.</p>', unsafe_allow_html=True)
    step_label("01 — Build a map")
    c1, c2 = st.columns(2)
    grid_size = c1.selectbox("Grid size", GRID_SIZES, index=0, format_func=lambda n: f"{n} × {n}")
    density = c2.segmented_control("Walls", [0.15, 0.30], default=0.15,
                                   format_func=lambda d: f"{d:.0%}") or 0.15
    c3, c4 = st.columns(2)
    dynamics = c3.selectbox("Obstacles", list(DYNAMICS), format_func=DYNAMICS.get)
    seed = c4.number_input("Map number (seed)", min_value=0, max_value=10_000, value=0, step=1)
    with st.expander("More options"):
        change_interval = st.select_slider("Obstacles move every k steps", [1, 2, 5, 10, 20], value=5,
                                           disabled=dynamics == "static")
        slip = st.select_slider("Chance a move slips sideways", [0.0, 0.1, 0.2], value=0.0,
                                format_func=lambda p: f"{p:.0%}")
        connectivity = st.radio("Moves allowed", [4, 8], horizontal=True,
                                format_func={4: "4 (no diagonals)", 8: "8 (with diagonals)"}.get)

env_cfg = EnvConfig(grid_size=grid_size, obstacle_density=density, dynamics=dynamics,
                    change_interval=change_interval if dynamics != "static" else 0, slip_prob=slip,
                    connectivity=connectivity, max_steps_factor=2.0)
grid_map = generate_map(env_cfg, make_rng(int(seed)))

with right:
    st.markdown(f'<div class="figlabel"><span>Fig. 01 — your map / {grid_size} × {grid_size}</span>'
                f'<span>Seed {int(seed)}</span></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_grid(ax, grid_map, agent=grid_map.start)
    figure(fig)

st.markdown(f'<div class="strip"><div><small>Grid</small><b>{grid_size} × {grid_size}</b></div>'
            f'<div><small>Walls</small><b>{density:.0%}</b></div>'
            f'<div><small>Obstacles</small><b>{DYNAMICS[dynamics]}</b></div>'
            f'<div><small>Slip</small><b>{slip:.0%}</b></div>'
            f'<div><small>Moves</small><b>{connectivity}-way</b></div></div>', unsafe_allow_html=True)
st.write("")

tab_sim, tab_cmp, tab_rl, tab_vs, tab_res = st.tabs(
    ["02  Watch one planner", "03  Compare planners", "04  Train an AI", "05  Planning vs. learning", "06  Results"])

# --------------------------------------------------------------- simulator
with tab_sim:
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        step_label("02 — Pick a planner and press run")
        planner_name = st.radio("Planner", list(PLANNERS), horizontal=True, format_func=PLANNERS.get)
        st.caption(PLANNER_NOTES[planner_name])
        replan = st.radio("Plan a new route", ["on_change", "on_block"], horizontal=True,
                          format_func={"on_change": "Whenever the map changes",
                                       "on_block": "Only when blocked"}.get)
        if st.button("Run ▸", type="primary", key="run_episode"):
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Outcome", "Reached" if r["success"] else "Failed")
            m2.metric("Steps", r["steps"])
            m3.metric("New routes", r["replans"])
            m4, m5, m6 = st.columns(3)
            m4.metric("Cells checked", r["total_expansions"])
            m5.metric("Bumps", r["collisions"])
            m6.metric("Planning time", f"{r['total_planning_time_s'] * 1e3:.2f} ms")
    with c2:
        if "sim" in st.session_state:
            frames = st.session_state["sim"]["frames"]
            t = st.slider("Replay: drag to move through the steps", 0, len(frames) - 1, len(frames) - 1)
            fig, ax = plt.subplots(figsize=(5, 5))
            trail = [f[0] for f in frames[: t + 1]]
            draw_grid(ax, grid_map, obstacle_positions=frames[t][1], agent=frames[t][0], path=trail, title=f"step {t}")
            figure(fig)
        else:
            st.markdown('<div class="takeaway">Press <b>Run ▸</b> to watch the robot cross your map. '
                        'You can then replay it step by step.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------- comparison
with tab_cmp:
    step_label("03 — Every planner on the same map")
    st.caption("One map, one run: good for explaining, not a statistical result. "
               "The experiments in Results use 30 maps each.")
    if st.button("Compare all planners ▸", type="primary", key="compare"):
        variants = [("BFS", "bfs", {}), ("Dijkstra", "dijkstra", {}), ("A* (no hint)", "astar", {"heuristic": "zero"}),
                    ("A*", "astar", {}), ("A* (strong hint 1.5×)", "astar", {"weight": 1.5}), ("D* Lite", "dstar_lite", {})]
        rows, results = [], []
        for label, name, kw in variants:
            m = measure(lambda: make_planner(name, env_cfg, record_expanded=True, **kw).plan(
                grid_map.walls, grid_map.start, grid_map.goal), repeats=3)
            res = m.result
            results.append((label, res))
            rows.append({"Planner": label, "Found a route": "yes" if res.found else "no", "Route length": res.cost,
                         "Cells checked": res.stats.expansions, "Time (ms)": round(m.wall_time_s * 1e3, 3),
                         "Memory (KB)": round(m.peak_memory_bytes / 1024, 1)})
        table = pd.DataFrame(rows)
        checked = table.set_index("Planner")["Cells checked"]
        if checked["A*"] > 0:
            st.markdown(f'<div class="takeaway">A* checked <b>{checked["A*"]}</b> cells; Dijkstra checked '
                        f'<b>{checked["Dijkstra"]}</b>, so A* did <b>{checked["Dijkstra"] / checked["A*"]:.1f}× less work</b> '
                        f'for a route of the same length.</div>', unsafe_allow_html=True)
        st.dataframe(table, hide_index=True, width="stretch")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, (label, res) in zip(axes.ravel(), results):
            draw_grid(ax, grid_map, path=res.path, explored=res.expanded, legend=False,
                      title=f"{label}: {res.stats.expansions} cells checked")
        figure(fig)
        st.caption("Yellow = cells the planner checked. Blue line = the route it found.")

# -------------------------------------------------------------- RL training
with tab_rl:
    step_label("04 — Train an AI agent on this map")
    st.caption("The agent practises on your map and is tested every few episodes. "
               "It counts as trained once it reaches the goal at least 90% of the time, three tests in a row.")
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Learner", ["Q-Learning", "DQN", "Double DQN", "Value Iteration"])
    if algo == "Value Iteration":
        lr = None
        c2.caption("Value Iteration knows the map, so it needs no learning rate. It stops when the scores stop changing.")
    else:
        lr = c2.select_slider("Learning rate", LEARNING_RATES[algo], value=LEARNING_RATES[algo][2 if algo == "Q-Learning" else 1])
    budget = c3.number_input("Practice length (episodes for Q-Learning, steps for DQN)", min_value=100,
                             max_value=500_000, value=2000 if algo == "Q-Learning" else 30_000, step=100, key="budget")
    if grid_size > 20 and algo in ("DQN", "Double DQN"):
        st.warning("DQN on big maps is slow on a CPU: expect minutes per 10k steps.")
    if st.button("Train ▸", type="primary", key="train"):
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
            m1, m2 = st.columns(2)
            m1.metric("Sweeps to finish", vi.sweeps)
            m2.metric("Solve time", f"{vi.solve_time_s:.3f} s")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.semilogy(vi.residuals)
            ax.axhline(1e-6, color="#8A8A86", ls=":")
            ax.set_xlabel("sweep")
            ax.set_ylabel("change in scores")
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Converged", "yes" if conv["converged"] else "no")
            m2.metric("Episodes needed", "-" if not conv["converged"] else int(conv["episodes_to_convergence"]))
            m3.metric("Training time", f"{log.train_time_s:.2f} s")
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
            axes[0].plot(ep["episode"], ep["return"].rolling(20, min_periods=1).mean(), label="practice (20-ep average)")
            axes[0].plot(ev["episode"], ev["mean_return"], "o-", ms=3, label="test")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Reward")
            axes[0].legend(frameon=False)
            axes[1].plot(ep["episode"], ep["success"].astype(float).rolling(20, min_periods=1).mean(), label="practice")
            axes[1].plot(ev["episode"], ev["success_rate"], "o-", ms=3, label="test")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Reached the goal")
            axes[1].legend(frameon=False)
            figure(fig)
            row = {"timestamp": pd.Timestamp.now().isoformat(), "algorithm": algo, "grid_size": grid_size,
                   "density": density, "learning_rate": lr, "seed": int(seed), "budget": int(budget),
                   "train_time_s": log.train_time_s, **conv}
            log_path = Path(os.environ.get("PATHRL_DASHBOARD_LOG", RESULTS / "dashboard" / "dashboard_runs.csv"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            st.caption(f"Run saved to {log_path} (real measurements from this machine).")

    runs_csv = RESULTS / "e6_convergence" / "e6_rl_runs.csv"
    with st.expander("Saved results from experiment E6"):
        if runs_csv.exists():
            runs = pd.read_csv(runs_csv)
            algo_pick = st.selectbox("Learner (E6 results)", sorted(runs["algorithm"].unique()))
            sub = runs[runs["algorithm"] == algo_pick]
            table = sub.groupby(["learning_rate", "grid_size"]).agg(
                converged=("converged", "mean"), episodes_to_convergence=("episodes_to_convergence", "mean"),
                final_success=("final_success_rate", "mean"), runs=("seed_index", "count")).reset_index()
            st.dataframe(table, hide_index=True, width="stretch")
            for name in [f"e6_{algo_pick}_reward_curves", f"e6_{algo_pick}_success_curves"]:
                if (FIGURES / f"{name}.png").exists():
                    st.image(str(FIGURES / f"{name}.png"))
        else:
            st.info("E6 has not been run yet. Run: python main.py experiment e6")


# ------------------------------------------------------- planning vs. learning
def span(lo, hi, fmt):
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


with tab_vs:
    step_label("05 — Planning vs. learning: the verdict")
    a, b = st.columns(2, gap="medium")
    a.markdown('<div class="vs dark"><p class="tagline">Classical planning · A*, D* Lite</p>'
               '<h3>Look at the map, work out the route.</h3><ul>'
               '<li>Needs the map and the movement rules</li><li>No training: plans straight away</li>'
               '<li>Map changed? Plans again or repairs the route</li>'
               '<li>Always finds the shortest route if one exists</li>'
               '<li>Every decision can be traced and explained</li></ul></div>', unsafe_allow_html=True)
    b.markdown('<div class="vs"><p class="tagline">Reinforcement learning · DQN, Double DQN</p>'
               '<h3>Practise a lot, learn which move is good.</h3><ul>'
               '<li>Needs only a reward after each move</li><li>Must train first (minutes to hours)</li>'
               '<li>Map changed? Reacts with what it learned</li>'
               '<li>No guarantee: may wander, bump or fail</li>'
               '<li>Decisions hidden inside a neural network</li></ul></div>', unsafe_allow_html=True)

    summary_dir = RESULTS / "summary"
    e3_csv, cross_csv, e5_csv = (summary_dir / f for f in ("e3_summary.csv", "e3_crossover.csv", "e5_summary.csv"))
    if not (e3_csv.exists() and cross_csv.exists()):
        st.info("Run experiment E3 to fill this section: python main.py experiment e3, then python main.py analyze")
    else:
        e3 = pd.read_csv(e3_csv)
        cross = pd.read_csv(cross_csv)
        size = int(cross["grid_size"].min())
        rows = e3[e3["grid_size"] == size].groupby("method").agg(
            s_lo=("success_f_mean", "min"), s_hi=("success_f_mean", "max"),
            p_lo=("path_cost_ratio_mean", "min"), p_hi=("path_cost_ratio_mean", "max"),
            t_lo=("computation_per_step_ms_mean", "min"), t_hi=("computation_per_step_ms_mean", "max"))
        train = cross.groupby("rl_method")["train_time_s"].mean()
        planners = [m for m in rows.index if m not in train.index]
        learners = [m for m in rows.index if m in train.index]
        plan_rows, rl_rows = rows.loc[planners], rows.loc[learners]
        never = int(np.isinf(cross["crossover_episodes"]).sum())

        st.write("")
        st.markdown(f'<p class="step">Measured on {size} × {size} maps with moving obstacles '
                    f'(experiment E3, 30 maps per setting, 5 obstacle speeds)</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        for col, label, plan_v, rl_v in [
            (k1, "Reaches the goal", span(plan_rows.s_lo.min(), plan_rows.s_hi.max(), "{:.0%}"),
             span(rl_rows.s_lo.min(), rl_rows.s_hi.max(), "{:.0%}")),
            (k2, "Route length vs. best", span(plan_rows.p_lo.min(), plan_rows.p_hi.max(), "{:.2f}×"),
             span(rl_rows.p_lo.min(), rl_rows.p_hi.max(), "{:.1f}×")),
            (k3, "Fastest time per move", f"{plan_rows.t_lo.min():.3f} ms", f"{rl_rows.t_lo.min():.3f} ms"),
            (k4, "Training before use", "none", span(train.min() / 60, train.max() / 60, "{:.0f}") + " min"),
        ]:
            col.markdown(f'<div class="vs"><p class="tagline">{label}</p>'
                         f'<p class="big">{plan_v} <span>planning</span></p>'
                         f'<p class="big" style="color:#8A8A86">{rl_v} <span>learning</span></p></div>',
                         unsafe_allow_html=True)

        st.markdown(f'<div class="takeaway"><b>Does training ever pay off?</b> In {never} of {len(cross)} '
                    f'planner-vs-AI pairings, never: one AI move is slower than one planner move, so the '
                    f'training time is never won back.</div>', unsafe_allow_html=True)

        table = pd.DataFrame({
            "Method": rows.index,
            "Type": ["Learning" if m in train.index else "Planning" for m in rows.index],
            "Reaches goal": [span(r.s_lo, r.s_hi, "{:.0%}") for r in rows.itertuples()],
            "Route length vs. best": [span(r.p_lo, r.p_hi, "{:.2f}×") for r in rows.itertuples()],
            "Time per move (ms)": [span(r.t_lo, r.t_hi, "{:.3f}") for r in rows.itertuples()],
            "Training (min)": [f"{train[m] / 60:.0f}" if m in train.index else "none" for m in rows.index],
        }).sort_values("Type", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            sub = e3[e3["grid_size"] == size]
            for method, g in sub.groupby("method"):
                learner = method in train.index
                ax.plot(g["change_interval"], g["success_f_mean"] * 100, "o-" if learner else "s--",
                        color="#8A8A86" if learner else "#111111", label=method, ms=4)
            ax.set_xscale("log")
            ax.set_xticks(sorted(sub["change_interval"].unique()), labels=sorted(sub["change_interval"].unique()))
            ax.set_xlabel("obstacles move every k steps (left = faster)")
            ax.set_ylabel("reached the goal (%)")
            ax.set_ylim(0, 105)
            ax.set_title("Reaching the goal as obstacles move")
            ax.legend(frameon=False, fontsize=7)
            figure(fig)
        with c2:
            if e5_csv.exists():
                e5 = pd.read_csv(e5_csv)
                e5 = e5[e5["split"].str.startswith("test")]
                fig, ax = plt.subplots(figsize=(6, 3.4))
                colors = ["#8A8A86" if m in train.index else "#111111" for m in e5["method"]]
                bars = ax.bar(e5["method"], e5["success_rate_mean"] * 100, color=colors, width=0.55)
                ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
                ax.set_ylabel("reached the goal (%)")
                ax.set_ylim(0, 110)
                ax.set_title("Brand-new maps the AI never saw (E5)")
                figure(fig)

    st.markdown('<p class="step">When to use which</p>', unsafe_allow_html=True)
    w1, w2 = st.columns(2, gap="medium")
    w1.markdown('<div class="vs"><h3>Plan it when…</h3><ul><li>you have a map and know how moves work</li>'
                '<li>you need the shortest route, guaranteed</li><li>maps change or are brand new</li>'
                '</ul></div>', unsafe_allow_html=True)
    w2.markdown('<div class="vs"><h3>Learn it when…</h3><ul><li>nobody can write down the rules</li>'
                '<li>the goal is more than "shortest" (energy, safety)</li>'
                '<li>you can afford long training on similar worlds</li></ul></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- results
with tab_res:
    step_label("06 — What the experiments found")
    available = {p: v for p, v in EXPERIMENTS.items() if any(FIGURES.glob(f"{p}*.png"))}
    if not available:
        st.info("No results yet. Run: python main.py experiment all, then python main.py figures")
    else:
        prefix = st.radio("Experiment", list(available), horizontal=True, format_func=lambda p: available[p][0])
        st.markdown(f'<div class="takeaway">{available[prefix][1]}</div>', unsafe_allow_html=True)
        files = sorted(FIGURES.glob(f"{prefix}*.png"))
        cols = st.columns(2)
        for i, f in enumerate(files):
            cols[i % 2].image(str(f), caption=f.stem.replace("_", " "))
    summary_dir = RESULTS / "summary"
    if summary_dir.exists():
        with st.expander("Summary tables (mean, 95% confidence interval)"):
            pick = st.selectbox("Table", sorted(p.name for p in summary_dir.glob("*.csv")))
            df = pd.read_csv(summary_dir / pick)
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Download CSV", df.to_csv(index=False), file_name=pick)
