"""Chapters 12-16: experiments and results, pipeline/dashboard/tests, rebuild recipe, viva questions, reference."""

from __future__ import annotations

import pandas as pd

from guide_lib import box, fig, pre, qa, src, summary, table


def math(text: str, caption: str | None = None) -> str:
    return pre(text, caption, cls="math")


def _ms(v):
    return f"{v:.3g}"


def ch12() -> str:
    e1 = summary("e1_summary")
    e1t = e1[e1.grid_size == 100][["density", "planner", "wall_time_ms_mean", "expansions_mean", "peak_memory_kib_mean", "optimality_ratio_mean"]]
    e1t.columns = ["density", "planner", "time (ms)", "expansions", "peak memory (KiB)", "cost / optimal"]
    e2 = summary("e2_summary")
    e2t = e2[e2.grid_size == 100][["density", "variant", "expansions_mean", "optimality_ratio_mean", "fraction_optimal"]]
    e2t.columns = ["density", "heuristic", "expansions", "cost / optimal", "fraction optimal"]
    e3 = summary("e3_summary")
    e3c = e3[(e3.grid_size == 50) & (e3.change_interval.isin([1, 5, 20]))][["change_interval", "method", "computation_ms_mean", "total_expansions_mean", "success_f_mean", "path_cost_ratio_mean"]]
    e3c.columns = ["change every k", "method", "ms / episode", "expansions / episode", "success", "path / static optimum"]
    e3r = e3[(e3.grid_size == 10)][["change_interval", "method", "success_f_mean", "path_cost_ratio_mean", "computation_per_step_ms_mean"]]
    e3r.columns = ["change every k", "method", "success", "path / static optimum", "ms / decision"]
    e4 = summary("e4_summary")
    e4t = e4[e4.grid_size == 10][["slip_prob", "method", "success_rate_mean", "mean_return_mean", "mean_collisions_mean"]]
    e4t.columns = ["slip", "method", "success", "return", "collisions"]
    e5 = summary("e5_summary")[["method", "split", "success_rate_mean", "success_rate_ci_low", "success_rate_ci_high", "optimality_ratio_mean", "n"]]
    e5.columns = ["method", "maps", "success", "CI low", "CI high", "cost / optimal (successes)", "n"]
    e6 = summary("e6_rl_summary")
    ql = e6[e6.algorithm == "q_learning"].pivot(index="learning_rate", columns="grid_size", values="converged_f_mean").reset_index()
    ql.columns = ["alpha \\ N"] + [str(c) for c in ql.columns[1:]]
    dq = e6[e6.algorithm != "q_learning"][["algorithm", "grid_size", "learning_rate", "converged_f_mean", "final_success_rate_mean"]]
    dq.columns = ["algorithm", "N", "lr", "converged fraction", "final success"]
    tr = pd.read_csv(__import__("guide_lib").ROOT / "outputs/results/train_dqn/train_dqn_summary.csv")[
        ["algorithm", "train_seed", "device", "train_time_s", "env_steps", "episodes", "final_success_rate", "converged"]]
    return """
<h1 id="ch12"><span class="num">12</span>The experiments and what they found</h1>
<p>Each experiment has a config in <code>configs/experiments/</code>, a runner in <code>src/pathrl/experiments/</code>, raw results
in <code>outputs/results/&lt;experiment&gt;/</code> and figures in <code>outputs/figures/</code>. All tables in this chapter are read from the
generated summary CSVs when the guide is built. Hardware: Apple-silicon Mac, 18 CPU cores, Python 3.11.16, PyTorch 2.15;
DQN training on the MPS GPU, all planning and inference timed on the CPU.</p>
""" + table(pd.DataFrame([
        ["E1", "Static scaling", "N in {10..100} x density {0.15, 0.30} x {BFS, Dijkstra, A*, D* Lite}", "30 maps", "How does planning cost grow with map size?"],
        ["E2", "Heuristic quality", "A* with h = 0, Manhattan, 1.5 x Manhattan on E1's maps", "30 maps", "What does a better or inflated heuristic buy?"],
        ["train_dqn", "Generalist agents", "3 seeds x {DQN, Double DQN}, 500k steps, 1,000 maps, mixed conditions", "3 seeds", "Train the policies E3-E5 evaluate"],
        ["E3", "Dynamic obstacles", "N in {10,20,30,50}, change every k in {20,10,5,2,1}; A* on_change / on_block, D* Lite; RL at N = 10", "30 maps", "Repair cost vs change rate; the crossover H*"],
        ["E4", "Slip", "N in {10, 20} x slip {0, 0.1, 0.2}; planners, VI, Q-Learning, generalists", "30 maps x 20 episodes", "Robustness to stochastic transitions"],
        ["E5", "Generalisation", "500 unseen + 500 training maps (static, N = 10)", "3 seeds", "Do learned policies transfer to new maps?"],
        ["E6", "Convergence", "VI (all N, densities, slips); Q-Learning (all N x 5 alphas); DQN/DDQN (N in {10,20} x 4 lrs)", "10 / 5 / 3 seeds", "How fast and how reliably does each converge?"],
    ], columns=["ID", "Name", "Conditions", "Replication", "Question"])) + """

<h2>12.1 E1: static scaling</h2>
""" + table(e1t, "N = 100 (mean over 30 maps)", fmt={"time (ms)": _ms, "expansions": lambda v: f"{v:,.0f}", "peak memory (KiB)": lambda v: f"{v:,.0f}", "cost / optimal": lambda v: f"{v:g}"}) + """
<ul>
<li>All planners found optimal paths on every map (4-connected unit costs make BFS optimal too).</li>
<li>A* was fastest everywhere; BFS and Dijkstra expand identical sets (same order with unit costs).</li>
<li>D* Lite's first plan was 7.7-60x slower than A*: its first backward search expands every cell with f &le; C*.
Dense maps prune that set, so the gap shrinks at density 0.30.</li>
</ul>
""" + fig("e1_runtime_vs_grid", "E1 wall time per query (log scale), mean and 95% CI over 30 maps.") + fig("e1_expansions_vs_grid", "E1 node expansions per query.") + """

<h2>12.2 E2: heuristic quality</h2>
""" + table(e2t, "A* at N = 100", fmt={"expansions": lambda v: f"{v:,.0f}", "cost / optimal": lambda v: f"{v:.3f}", "fraction optimal": lambda v: f"{v:.2f}"}) + """
<p>Inflating the heuristic cut expansions by 60-71% at N = 100 but made paths 6.6-8.3% longer on average, and none of the
weighted paths was optimal at N = 100. The worst single path was 1.204 x optimal, inside the theoretical bound of 1.5.</p>
""" + fig("e2_heuristic_quality", "E2: expansions and path quality for three heuristics.") + """

<h2>12.3 Training the generalist agents</h2>
""" + table(tr, "train_dqn_summary.csv (final_success_rate = greedy success on 50 held-out validation maps)",
            fmt={"train_time_s": lambda v: f"{v:,.0f}", "final_success_rate": lambda v: f"{v:.2f}"}) + """
<p>No generalist met the convergence criterion (success &ge; 0.9). DQN rows trained under CPU contention (see 10.3).</p>
""" + fig("train_dqn_validation_curves", "Greedy validation success during training.") + """

<h2>12.4 E3: dynamic obstacles and the crossover</h2>
""" + table(e3c, "Classical planners at N = 50", fmt={"ms / episode": _ms, "expansions / episode": lambda v: f"{v:,.0f}",
                                                       "success": lambda v: f"{v:.2f}", "path / static optimum": lambda v: f"{v:.2f}"}) + """
<p>Replanning cost grows with the change rate and with N. "Replan when blocked" was cheapest in every condition while
keeping 100% success. D* Lite was slowest in wall time in all 20 classical conditions, even where it did fewer expansions.</p>
""" + fig("e3_planning_cost_vs_change_rate", "E3: planning + repair time per episode vs change rate, for each grid size.") + table(
        e3r, "N = 10: planners and trained agents on the same maps and episode seeds",
        fmt={"success": lambda v: f"{v:.3f}", "path / static optimum": lambda v: f"{v:.2f}", "ms / decision": lambda v: f"{v:.4f}"}) + """
<p>The agents' success <i>rises</i> as obstacles move faster (0.69-0.73 at k = 20, 0.92-0.93 at k = 1) and every failure is a
truncation at the step limit, consistent with the aliasing-loop hypothesis of Chapter 9.</p>
""" + fig("e3_methods_vs_change_rate", "E3 at 10 x 10: computation, success and path quality vs change rate.") + """
<h3>Crossover</h3>
<p>Per decision, inference (≈0.099 ms) cost more than every planner at N = 10, so H* = never in all 30 conditions. The figure
plots both cumulative cost curves up to 10<sup>7</sup> episodes: the RL curves never meet the planners'.</p>
""" + fig("e3_cumulative_cost_crossover", "Cumulative computation vs deployment horizon (log-log).") + fig(
        "e3_per_decision_cost_vs_grid", "Per-decision cost vs grid size: planners (measured) vs the global-view CNN forward pass (untrained network, cost only). "
        "The trained local-view network is the same at every N (≈0.099 ms per decision, measured in E3).") + box("warn", "Reading the per-decision figure correctly", """
<p>For N &ge; 20, a forward pass is cheaper than a D* Lite repair in this implementation, so a crossover against D* Lite is
arithmetically possible there. It would need a policy that actually succeeds at that size with near-optimal paths, and none
was trained. That is <b>pending</b>, not a finding.</p>""") + """

<h2>12.5 E4: stochastic transitions</h2>
""" + table(e4t, "N = 10 (30 maps x 20 shared episode seeds)", fmt={"success": lambda v: f"{v:.3f}", "return": lambda v: f"{v:.1f}", "collisions": lambda v: f"{v:.3f}"}) + """
<ul>
<li>Planners and VI reached the goal in every episode: they replan (or act from V*) after each slip.</li>
<li>VI, knowing the slip model, earned the highest return in every slip condition by choosing routes with fewer wall contacts.</li>
<li>Per-map Q-Learning (trained on the test map itself, not zero-shot) came close to VI at N = 10.</li>
<li>The generalists did better <i>with</i> slip (DQN 0.34 to 0.81), again consistent with perturbations breaking loops.</li>
</ul>
""" + fig("e4_success_vs_slip", "E4: success, return and collisions vs slip probability.") + """

<h2>12.6 E5: generalisation</h2>
""" + table(e5, "N = 10, static maps", fmt={"success": lambda v: f"{v:.3f}", "CI low": lambda v: "" if pd.isna(v) else f"{v:.3f}",
                                                  "CI high": lambda v: "" if pd.isna(v) else f"{v:.3f}",
                                                  "cost / optimal (successes)": lambda v: "" if pd.isna(v) else f"{v:.2f}"}) + """
<p>Generalisation gap of about 0.10-0.13; even on their own training maps no agent exceeded 0.6 success. When they succeed,
paths are near-optimal (&le; 11% longer), so failures are loops, not detours. Train/test map overlap was measured as 0.</p>
""" + fig("e5_generalization", "E5: greedy success on seen vs unseen maps (95% CI over 3 training seeds).", "70%") + """

<h2>12.7 E6: convergence and learning-rate sensitivity</h2>
""" + table(ql, "Q-Learning: fraction of 5 runs that converged (fixed map, 3,000 episodes)") + """
<p>Larger alpha converged faster and on larger grids; the usable range narrows with N, and nothing converged at N = 100.
Convergence was all-or-nothing per cell: a learning rate converged for all five seeds or for none.</p>
""" + fig("e6_q_learning_heatmap", "E6 Q-Learning: episodes to convergence and final success, learning rate x grid size.") + table(
        dq, "DQN / Double DQN on a fixed map (3 seeds; 40k steps at N = 10, 80k at N = 20)",
        fmt={"converged fraction": lambda v: f"{v:.2f}", "final success": lambda v: f"{v:.2f}", "lr": lambda v: f"{v:g}"}) + """
<p>DQN converged in all 12 runs at N = 10 and none at N = 20. Value Iteration converged in all 360 runs; its sweeps grow roughly
linearly with N without slip, and jump to ~1,300-1,400 with slip on dense maps (Chapter 7).</p>
""" + fig("e6_value_iteration", "E6 Value Iteration: sweeps and solve time.") + fig("e6_convergence_vs_grid", "E6: episodes and time to convergence vs grid size (converged runs only).") + """

<h2>12.8 Limitations you must state</h2>
<ul>
<li>Trained agents exist only at N = 10; RL results for N &ge; 30 are pending (compute budget).</li>
<li>All implementations are Python; wall-time rankings are implementation-specific (expansion counts are the hardware-independent complement).</li>
<li>Few RL seeds (3), so wide intervals; no DQN-vs-Double-DQN claim is made.</li>
<li>One hardware platform; absolute times will differ elsewhere.</li>
<li>Information asymmetry: planners and VI see the whole map (VI also the slip model); the agents see 7 x 7.</li>
<li>Three hypotheses are untested: aliasing loops, gamma-limited VI convergence, and a compiled D* Lite closing the gap.</li>
</ul>
"""


def ch13() -> str:
    return """
<h1 id="ch13"><span class="num">13</span>Figures, report, dashboard and tests</h1>

<h2>13.1 From raw CSV to figures and RESULTS.md</h2>
<p><code>python main.py figures</code> runs <code>make_figures.py</code>: it loads each raw CSV, computes per-condition summaries with
<code>summarize</code> (mean, t-interval, std, n), saves them to <code>outputs/results/summary/</code>, and draws every figure.
Missing experiments are listed as <i>pending</i> instead of crashing. <code>python main.py report</code> runs <code>make_report.py</code>,
which formats those summary tables into <code>docs/RESULTS.md</code>. No number in RESULTS.md is typed by hand.</p>
""" + src("src/pathrl/visualization/plots.py", "line_ci") + """
<p>Plotting conventions: a colour-blind-safe palette (Okabe-Ito), error bars for 95% CIs, log axes for quantities that span
orders of magnitude, and the information assumption written in titles where methods see different things.</p>

<h2>13.2 The Streamlit dashboard</h2>
<p><code>streamlit run src/pathrl/dashboard/app.py</code> opens a four-tab web app:</p>
<ol>
<li><b>Environment simulator</b>: pick grid size, density, dynamics, slip and a planner; run an episode and replay it step by step.</li>
<li><b>Algorithm comparison</b>: every planner on the same map, with costs, expansions and expanded-cell pictures.</li>
<li><b>RL training (E6)</b>: train Q-Learning, DQN, Double DQN or run VI live with a progress bar and a convergence verdict.</li>
<li><b>Research results</b>: every generated figure and summary table.</li>
</ol>
<p>Streamlit re-runs the script top to bottom on every interaction; widgets (<code>st.selectbox</code>, <code>st.slider</code>,
<code>st.button</code>) return their current values. Live training runs are logged to a separate CSV (overridable with the
<code>PATHRL_DASHBOARD_LOG</code> environment variable, so tests never pollute real results).</p>

<h2>13.3 The test suite</h2>
<p>158 tests (111 test functions, some parametrised) run with <code>pytest</code> in about a minute. They check behaviour, not just
that code runs:</p>
""" + table(pd.DataFrame([
        ["test_setup.py", "10", "config layering, typed overrides, unknown keys fail, seeding reproducibility, hardware snapshot"],
        ["test_environment.py", "36", "collision rules, slip distribution, empirical slip rate, obstacle rules over long runs, observations, truncation, Gymnasium check_env"],
        ["test_classical.py", "19", "known shortest paths, all planners optimal on 4-connected maps, BFS not cost-optimal with diagonals, weighted-A* bound, heuristic admissibility/consistency, counter invariants"],
        ["test_dstar_lite.py", "9", "initial plan optimal; repaired plans match fresh A* under random changes; state preserved; changes detected without update()"],
        ["test_value_q_learning.py", "11", "VI values equal shortest-path returns; VI model matches environment by Monte Carlo; contraction; Q-Learning update rule and learning; convergence criterion"],
        ["test_dqn.py", "10", "network shape; buffer overwrite; DQN vs Double DQN targets exactly as defined; target copy; act does not learn; checkpoint round trip; disjoint map sets; slow end-to-end learning"],
        ["test_evaluation.py", "9", "navigation follows optimal paths; replanning; on_block replans no more than on_change; statistics; provenance"],
        ["test_analysis.py", "3", "crossover formula and table; inference scaling"],
        ["test_dashboard.py", "3", "dashboard loads; buttons run; live Q-Learning run (Streamlit AppTest)"],
        ["test_visualization.py", "1", "grid drawing"],
    ], columns=["File", "Functions", "What it verifies"])) + src("tests/test_dqn.py", "test_dqn_and_double_dqn_targets_differ_exactly_as_defined", title="Example test: the one formula that separates DQN from Double DQN")


def ch14() -> str:
    return """
<h1 id="ch14"><span class="num">14</span>Build it yourself: a stage-by-stage recipe</h1>
<p>The project was built in twelve stages, each ending with passing tests before the next began. Follow the same order;
each stage only needs the ones before it.</p>
""" + table(pd.DataFrame([
        ["1", "Setup", "venv (Python 3.11), src layout, pyproject, requirements; default.yaml; load_config with dotted overrides; derive_seeds; hardware snapshot", "config/seed tests pass"],
        ["2", "Environment core", "EnvConfig + validation; dynamics (actions, perpendicular, sample_delta, attempt_move); rewards; GridMap + generate_map + BFS reachability", "collision and slip tests"],
        ["3", "Gym environment", "obstacle controllers; GridWorldEnv (reset/step, two RNG streams, global/local observations, render); register with Gymnasium", "check_env passes"],
        ["4", "Classical search", "GridGraph on attempt_move; SearchStats; Planner ABC; BFS, Dijkstra, A* (+ heuristics, tie-breaking, weights)", "optimality vs Dijkstra on random maps"],
        ["5", "D* Lite", "g/rhs, keys with k_m, UpdateVertex, ComputeShortestPath, path extraction, 3x3 repair, rounded keys", "random-repair equivalence to A*"],
        ["6", "VI + Q-Learning", "vectorised VI on the env's own model; Q-Learning with epsilon schedule, truncation bootstrap; convergence criteria", "VI = shortest-path returns; Q learns small map"],
        ["7", "DQN", "QNetwork, ReplayBuffer, DQN/DoubleDQN (targets, Huber, Adam, clipping, target sync), checkpoints, device selection", "target formula test; tiny-corridor learning"],
        ["8", "Evaluation", "profiling.measure; rollout; navigation (on_change/on_block); scenarios; save_results with provenance", "navigation tests"],
        ["9", "Experiments", "train_dqn, E1-E6 runners with configs, seed families, parallel_map", "small debug configs run end to end"],
        ["10", "Analysis", "statistics (t, Wilson), crossover, make_figures", "figures render, pending listed"],
        ["11", "Dashboard", "Streamlit app, 4 tabs", "AppTest tests"],
        ["12", "Docs", "methodology, architecture, complexity, generated RESULTS.md, report", "every number traceable"],
    ], columns=["Stage", "Module", "Build", "Done when"])) + """

<h2>14.1 Commands, in order</h2>
""" + pre("""
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -e .
pytest -m "not slow"                        # fast tests while developing; plain `pytest` for everything

python main.py hardware                     # record the machine
python main.py experiment e1                # serial, clean timings
python main.py experiment e2
python main.py experiment train_dqn         # GPU if available; needed by E3-E5
python main.py experiment e3                # serial
python main.py experiment e4 --workers 8
python main.py experiment e5
python main.py experiment e6 --workers 12
python main.py figures
python main.py report
streamlit run src/pathrl/dashboard/app.py
""", "Full pipeline") + """
<p>Always try a small version first: <code>python main.py experiment e1 --set num_seeds=3 'e1.grid_sizes=[10,20]'</code>
(quote list values in zsh).</p>

<h2>14.2 Pitfalls we actually hit (and how they were fixed)</h2>
""" + table(pd.DataFrame([
        ["D* Lite stopped early on 8-connected maps", "float keys: sums of sqrt 2 in different orders differ in the last bits", "round keys to 9 decimals"],
        ["VI sweep counts were huge and meaningless", "walled-off pockets decay slowly and dominated the residual", "update only goal-connected cells"],
        ["DQN training very slow on CPU", "profiled: conv kernels slow on this CPU build (~18 ms per learn step)", "train on MPS GPU, infer on CPU"],
        ["Global-view DQN did not generalise (0.00-0.04)", "a full-map CNN must memorise each map", "model selection on validation maps chose the 7x7 local view and 500k steps"],
        ["A test's 8-connected BFS example was wrong", "hand-made expectation", "replaced by measurement over 40 random maps"],
        ["Dashboard test wrote into real results", "shared log path", "PATHRL_DASHBOARD_LOG environment variable"],
        ["Figure crashed on log scale", "every crossover was 'never' (no finite values)", "annotate 'never' instead of plotting"],
        ["Training times differed 2x between identical-cost algorithms", "CPU contention from parallel experiments", "disclosed; run timing-sensitive jobs alone"],
        ["zsh error on --set 'x=[a,b]'", "zsh treats [] as a glob", "quote list overrides"],
    ], columns=["Symptom", "Cause", "Fix"])) + box("check", "Habits that make the project trustworthy", """
<ul><li>Write a test that would fail if the idea were wrong (e.g. compare D* Lite against A* after random changes).</li>
<li>Never drop failed or non-convergent runs; report them.</li>
<li>Separate training, inference and repair cost; disclose hardware and concurrency.</li>
<li>Choose models on validation data only; keep test seeds disjoint.</li>
<li>Generate every table and figure from CSVs by code; mark missing results as pending.</li>
<li>Label hypotheses as hypotheses.</li></ul>""")


def ch15() -> str:
    return """
<h1 id="ch15"><span class="num">15</span>Viva and interview questions, with answers</h1>
<h2>15.1 Search</h2>
""" + qa([
        ("Why is BFS optimal on the 4-connected grid but not with diagonals?",
         "BFS minimises the number of moves. With unit costs that equals path cost; with sqrt 2 diagonals a path with more moves can be cheaper."),
        ("What is the difference between admissible and consistent heuristics?",
         "Admissible: never overestimates the remaining cost (guarantees optimal A*). Consistent: h(a) &le; c(a,b) + h(b) on every edge, so g is final when a node is popped and closed nodes never reopen. Consistent implies admissible."),
        ("Why is Manhattan not admissible with 8-connectivity?",
         "One diagonal step (cost sqrt 2 &asymp; 1.41) covers what Manhattan counts as 2, so Manhattan can overestimate. Octile distance is the correct heuristic."),
        ("What does weighted A* guarantee?", "With weight w &ge; 1 and a consistent h, the path cost is at most w times optimal, usually with far fewer expansions (E2: 60-71% fewer at N = 100, 6.6-8.3% longer paths)."),
        ("What is lazy deletion and why is it needed?", "heapq has no decrease-key; instead a new entry is pushed and stale entries are skipped when popped (they belong to closed nodes or have outdated keys)."),
        ("Why does tie-breaking matter in A*?", "Many cells share the same f on open grids; preferring smaller h drives the search toward the goal and can cut expansions by an order of magnitude."),
    ]) + "<h2>15.2 D* Lite</h2>" + qa([
        ("Why does D* Lite search backwards?", "g then means cost-to-goal, which stays valid when the agent moves; only the query's start changes, handled by the k_m offset."),
        ("Explain g, rhs and consistency.", "rhs is a one-step lookahead min over neighbours of c + g. A cell is consistent when g = rhs; only inconsistent cells are queued, so a change only triggers work where values actually change."),
        ("What is k_m for?", "When the agent moves, h(start, s) changes for every queued cell. Adding the distance moved to k_m keeps old keys valid lower bounds without re-keying the queue; stale keys are fixed when popped."),
        ("Why was D* Lite slower than A* replanning in your experiments?", "Its first search expands every cell with f &le; C*, and each repair operation (vertex updates, keys, heap traffic) costs more in Python. It did fewer expansions in 12 of 20 dynamic conditions but was slower in wall time in all 20."),
    ]) + "<h2>15.3 MDPs and RL</h2>" + qa([
        ("Write the Bellman optimality equation.", "V*(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma V*(s')]."),
        ("Why does Value Iteration converge?", "The Bellman update is a gamma-contraction in the max-norm, so the distance to V* shrinks by at least a factor gamma per sweep."),
        ("Model-based vs model-free?", "Model-based (VI) uses P and R directly; model-free (Q-Learning, DQN) learns only from sampled transitions."),
        ("What makes Q-Learning off-policy?", "Its target uses max over next actions (the greedy policy) while the behaviour policy explores (epsilon-greedy)."),
        ("Terminated vs truncated: why does it matter?", "A truncated state is not terminal, so the target must still bootstrap gamma max Q(s',.). Treating truncation as termination biases values."),
        ("What does gamma do?", "Discounts future rewards (0.99 here); it makes infinite sums finite and sets the effective planning horizon (~1/(1-gamma) = 100 steps)."),
    ]) + "<h2>15.4 Deep RL</h2>" + qa([
        ("Why experience replay?", "Breaks the correlation between consecutive samples and reuses data; stabilises SGD."),
        ("Why a target network?", "Keeps the regression target fixed between syncs, so the network is not chasing a target that moves every step."),
        ("What is overestimation bias and how does Double DQN reduce it?", "max over noisy estimates is biased upward. Double DQN selects the action with the online net and evaluates it with the target net."),
        ("Why Huber loss and reward scaling?", "Huber limits gradient size for large TD errors; scaling rewards by 0.01 keeps errors in Huber's quadratic region and does not change the optimal policy."),
        ("Why did the local 7x7 view generalise better than the global view?", "Local patterns look the same across maps; a global-view CNN must learn each map. Chosen on validation maps: 0.00-0.04 (global) vs 0.22 (local) in pilots."),
        ("Why might the trained agents loop?", "A memoryless policy with a 7x7 view sees identical observations in different places (perceptual aliasing). All failures were truncations, and perturbations increased success: consistent with loops, but untested."),
    ]) + "<h2>15.5 Methodology</h2>" + qa([
        ("What is the crossover horizon and why can it be 'never'?", "H* = C_train/(c_plan - c_infer). If inference per episode costs at least as much as planning, the RL cost line never falls below the planning line."),
        ("Why Student t instead of 1.96?", "n is small (3-30); t accounts for estimating the standard deviation and widens the interval accordingly."),
        ("What is the unit of replication?", "The seed (independent map and run), not episodes or steps within a seed, which are correlated."),
        ("What are common random numbers?", "Giving compared methods the same random draws (same episode seed, fixed draws per step) so differences come from the methods, reducing variance."),
        ("How did you prevent test-set leakage?", "Disjoint seed families for training, validation and test maps; model selection only on validation; E5 measured train/test overlap = 0."),
        ("Is the comparison fair?", "Not like-for-like in information: planners see the whole map and VI the slip model, while agents see 7x7. This is stated with every comparison; it reflects how each method would be deployed."),
        ("Why report expansions as well as time?", "Expansions are hardware-independent; time includes implementation overhead. They disagreed for D* Lite, which is itself a finding."),
    ])


def ch16() -> str:
    return """
<h1 id="ch16"><span class="num">16</span>Glossary, commands and references</h1>
<h2>16.1 Glossary</h2>
""" + table(pd.DataFrame([
        ["Admissible heuristic", "never overestimates the remaining cost"],
        ["Bootstrapping", "updating an estimate from other estimates (e.g. max Q(s',.))"],
        ["Bellman residual", "max change of V in one sweep; VI stops when it is below theta"],
        ["C*", "optimal path cost"],
        ["Common random numbers", "same random draws for compared methods, to reduce variance"],
        ["Consistent heuristic", "h(a) &le; c(a,b) + h(b) for every edge"],
        ["Contraction", "a map that brings any two points closer by a fixed factor; guarantees VI converges"],
        ["Crossover horizon H*", "episodes after which train-once-then-infer becomes cheaper than planning"],
        ["Epsilon-greedy", "act randomly with probability epsilon, greedily otherwise"],
        ["Expansion", "removing a node from the frontier and examining its neighbours"],
        ["g, h, f", "cost so far, estimated cost to go, and their sum"],
        ["Generalist", "one policy trained across many maps and conditions, evaluated zero-shot"],
        ["Greedy evaluation", "running the learned policy without exploration"],
        ["Huber loss", "squared error for small errors, absolute error for large ones"],
        ["k_m", "D* Lite's key offset that absorbs agent movement"],
        ["Lazy deletion", "leaving stale heap entries and skipping them when popped"],
        ["MDP", "states, actions, transition probabilities, rewards and a discount factor"],
        ["Off-policy", "learning about one policy (greedy) while following another (exploratory)"],
        ["Perceptual aliasing", "different states producing identical observations"],
        ["Relaxation", "lowering g(v) when a cheaper route through u is found"],
        ["rhs", "D* Lite's one-step-lookahead cost estimate"],
        ["Replay buffer", "store of past transitions sampled at random for training"],
        ["Slip", "the action is replaced by a perpendicular one with probability p"],
        ["Target network", "frozen copy of the Q-network used to compute TD targets"],
        ["TD error", "target minus current estimate in a temporal-difference update"],
        ["Truncation", "episode cut off by the step limit (not a terminal state)"],
        ["Wilson interval", "a confidence interval for proportions that stays inside [0, 1]"],
    ], columns=["Term", "Meaning"])) + """
<h2>16.2 Command cheat sheet</h2>
""" + pre("""
pytest                                    # all 158 tests        pytest -m "not slow"   # skip slow DQN tests
python main.py hardware                   # hardware/software snapshot
python main.py demo --config configs/dynamic.yaml --steps 80 --seed 4
python main.py plan --set env.grid_size=30 --seed 1
python main.py experiment <train_dqn|e1|e2|e3|e4|e5|e6> [--workers N] [--set key=value ...]
python main.py figures                    # summary tables + figures
python main.py report                     # docs/RESULTS.md
streamlit run src/pathrl/dashboard/app.py --server.port 8502
python docs/guide/build_guide.py          # rebuild this PDF
""") + """
<h2>16.3 Where to look in the repository</h2>
<ul>
<li><code>docs/methodology.md</code>: precise environment rules, convergence definitions, hardware protocol, model selection, experiment matrix, concurrency disclosure.</li>
<li><code>docs/complexity_analysis.md</code>: theoretical complexity of every algorithm, and the measured D* Lite f &le; C* finding.</li>
<li><code>docs/RESULTS.md</code>: every measured number (generated). <code>docs/report.md</code>: interpretation and limitations.</li>
<li><code>outputs/results/*/*.meta.json</code>: config, hardware and git commit behind each CSV.</li>
</ul>
<h2>16.4 References</h2>
<ul>
<li>Hart, P. E., Nilsson, N. J., &amp; Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. <i>IEEE Transactions on Systems Science and Cybernetics</i>, 4(2). doi:10.1109/TSSC.1968.300136</li>
<li>Koenig, S., &amp; Likhachev, M. (2002). D* Lite. <i>AAAI-02</i>.</li>
<li>Koenig, S., Likhachev, M., &amp; Furcy, D. (2004). Lifelong Planning A*. <i>Artificial Intelligence</i>, 155. doi:10.1016/j.artint.2003.12.001</li>
<li>Watkins, C. J. C. H., &amp; Dayan, P. (1992). Q-learning. <i>Machine Learning</i>, 8. doi:10.1007/BF00992698</li>
<li>Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. <i>Nature</i>, 518. doi:10.1038/nature14236</li>
<li>van Hasselt, H., Guez, A., &amp; Silver, D. (2016). Deep reinforcement learning with double Q-learning. <i>AAAI-16</i>. arXiv:1509.06461</li>
<li>Sutton, R. S., &amp; Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i> (2nd ed.). MIT Press.</li>
</ul>
<p class="small">Suggested reading order for a beginner: Sutton &amp; Barto chapters 3-4 (MDPs, dynamic programming) and 6 (TD,
Q-Learning); Hart et al. for A*; Koenig &amp; Likhachev for D* Lite; Mnih et al. and van Hasselt et al. for deep Q-learning.</p>
"""


def chapters() -> list[str]:
    return [ch12(), ch13(), ch14(), ch15(), ch16()]
