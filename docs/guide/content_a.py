"""Chapters 1-6: orientation, toolbox, architecture, environment, classical search, D* Lite."""

from __future__ import annotations

import live_examples as L
from guide_lib import box, fig, pre, src, table


def math(text: str, caption: str | None = None) -> str:
    return pre(text, caption, cls="math")


def ch1() -> str:
    return """
<h1 id="ch1"><span class="num">1</span>Start here</h1>
<p class="lead">This guide explains every idea, algorithm, file and decision in the
<b>DynamicPathfinding-RL</b> project, in the order you need to learn them. It is written so that
someone who knows basic Python, but has never studied path planning or reinforcement learning,
could understand the project completely and rebuild it from an empty folder.</p>

<h2>1.1 The question the project answers</h2>
<p>A robot (the <i>agent</i>) must cross a grid from a start cell to a goal cell. Some cells are walls,
some obstacles move, and sometimes the robot's wheels slip. There are two very different ways to
decide where to go:</p>
<ul>
<li><b>Classical planning</b> (BFS, Dijkstra, A*, D* Lite): look at the map and <i>search</i> for a
shortest path. Searching costs computation <i>every time</i> the world changes.</li>
<li><b>Reinforcement learning</b> (Q-Learning, DQN, Double DQN): let the robot <i>practise</i> for a
long time until it learns a policy, a function from "what I see" to "which way to move". After
training, choosing a move is a single cheap function evaluation.</li>
</ul>
<p>So RL pays a large cost once (training), and planning pays a small cost many times. The project asks
the precise version of "which is better?":</p>
""" + math("""
C_RL(H)       = C_train + H * c_infer          (train once, then infer for H episodes)
C_planning(H) = H * c_plan                     (plan/repair in every one of H episodes)

Crossover horizon:   H* = C_train / (c_plan - c_infer)     if c_plan > c_infer
                     H* = "never"                         otherwise""",
                   "The cost model at the centre of the project") + """
<p>H* is the number of episodes after which training once becomes cheaper in total than planning
every time. It is a statement about <b>computation only</b>. A cheap method that fails to reach the goal
has not won, so success rate and path quality are always reported next to it.</p>

<h2>1.2 The answer the experiments gave</h2>
""" + box("concept", "Measured result (10 x 10 grids, where RL agents were trained)", """
<p>H* = <b>never</b> in all 30 tested conditions. A forward pass of the trained network costs about
0.1 ms per decision, more than an A* replan (0.003-0.010 ms) and slightly more than a D* Lite
repair (0.083 ms). The learned policies also take 2.5-6.7x longer paths. The planners reached the
goal in 100% of episodes; the trained agents in 69-93% with moving obstacles and 35-41% on unseen
static maps. Details and all caveats are in Chapter 12.</p>""") + """
<p>A negative result like this is still a result. The project's value is that it measures the question
fairly and reproducibly: same maps, same random numbers, separated training and inference costs,
confidence intervals, and every number traceable to the code and machine that produced it.</p>

<h2>1.3 How to read this guide</h2>
<ul>
<li>Chapters 2-3 give the toolbox and the architecture: how the code is organised and why.</li>
<li>Chapters 4-9 teach one topic each (the environment, classical search, D* Lite, MDPs and Value
Iteration, Q-Learning, deep Q-networks). Each starts with the <b>concept</b>, then shows the
<b>real code</b> from the repository, then lists what can go wrong.</li>
<li>Chapters 10-11 cover measurement: running episodes, timing, statistics, convergence and the
crossover analysis.</li>
<li>Chapter 12 explains each experiment E1-E6 and what it found, with the generated figures.</li>
<li>Chapter 13 covers the figures/report pipeline, the dashboard and the tests.</li>
<li>Chapter 14 is a step-by-step recipe to rebuild the project yourself, including every bug we hit.</li>
<li>Chapter 15 is a bank of viva/interview questions with answers; Chapter 16 has a glossary,
commands and references.</li>
</ul>
""" + box("why", "About the code listings", """
<p>Every listing in grey is extracted automatically from the repository when this PDF is built
(file name and line numbers are in its caption). Every number in a purple <i>Worked example</i> box
is computed by running the project's code at build time. Nothing was retyped, so the guide matches
the code exactly.</p>""") + """
<h2>1.4 The project at a glance</h2>
""" + table(__import__("pandas").DataFrame([
        ["Environment", "N x N grid; N in {10,20,30,50,70,100}; wall density 0.15 / 0.30; static, random-walk or oscillating obstacles; slip 0 / 0.1 / 0.2; 4-connected (8-connected ablation)"],
        ["Rewards", "-1 per step, -50 per collision (agent stays put), +100 at the goal"],
        ["Observations", "global 3 x N x N tensor, or local 3 x 7 x 7 egocentric window with a goal compass"],
        ["Classical", "BFS, Dijkstra, A* (Manhattan / octile, weighted), D* Lite (incremental)"],
        ["Model-based", "Value Iteration (knows the exact transition model, including slip)"],
        ["Model-free RL", "tabular Q-Learning; DQN and Double DQN with a CNN"],
        ["Experiments", "E1 static scaling, E2 heuristics, E3 dynamic obstacles + crossover, E4 slip, E5 generalisation, E6 convergence and learning-rate sensitivity"],
        ["Stack", "Python 3.11, NumPy, pandas, SciPy, PyTorch (MPS GPU for training), Gymnasium, Matplotlib, Streamlit, pytest"],
        ["Size", "about 6,200 lines including 158 tests"],
    ], columns=["Part", "What it contains"]))


def ch2() -> str:
    return """
<h1 id="ch2"><span class="num">2</span>The toolbox: Python and libraries you need</h1>
<p>This chapter lists every language feature and library the project uses, with just enough explanation
to read the code. If you know one already, skip it.</p>

<h2>2.1 Environment and packaging</h2>
<ul>
<li><b>Virtual environment.</b> An isolated Python installation per project, so library versions do not
clash. The project uses <code>uv</code> (a fast pip replacement) to create <code>.venv</code> with Python 3.11,
because the Mac's system Python was 3.9.</li>
<li><b>src layout.</b> Library code lives in <code>src/pathrl/</code>, not at the top level. Tests then import
the <i>installed</i> package, which catches packaging mistakes. <code>pip install -e .</code> ("editable")
installs it so edits take effect without reinstalling.</li>
<li><b>pyproject.toml</b> declares the package name, Python version and pytest settings.
<b>requirements.txt</b> lists minimum versions; <b>requirements-lock.txt</b> pins the exact versions that
produced the results (reproducibility).</li>
</ul>

<h2>2.2 Python features used everywhere</h2>
<ul>
<li><b>Type hints</b> (<code>def plan(self, occupancy: np.ndarray, start: Cell) -&gt; PlanResult</code>) document
what goes in and out. <code>from __future__ import annotations</code> lets hints refer to classes defined later.</li>
<li><b>Dataclasses</b> (<code>@dataclass</code>) generate <code>__init__</code>, <code>__repr__</code> and equality for
record-like classes. <code>frozen=True</code> makes instances immutable, used for configurations and maps so a
shared object cannot be changed by accident. <code>__post_init__</code> runs validation after construction.</li>
<li><b>Abstract base classes</b> (<code>ABC</code>, <code>@abstractmethod</code>) define an interface that
subclasses must implement: <code>Planner.plan</code> and <code>LearnedPolicy.act</code>.</li>
<li><b>Enums</b> (<code>Outcome.MOVED / COLLIDED / REACHED_GOAL</code>) name a fixed set of values.</li>
<li><b>Generators</b> (<code>yield</code>) produce values lazily: <code>GridGraph.neighbors</code> yields one legal
neighbour at a time.</li>
<li><b>Dictionaries and sets</b> as the search bookkeeping (<code>g</code>, <code>parent</code>,
<code>closed</code>). Cells are tuples <code>(row, col)</code> because tuples are hashable.</li>
</ul>

<h2>2.3 Data structures for search</h2>
<ul>
<li><code>collections.deque</code>: a double-ended queue with O(1) <code>append</code> and <code>popleft</code>. BFS uses it
as a FIFO queue.</li>
<li><code>heapq</code>: a binary min-heap stored in a list. <code>heappush</code> and <code>heappop</code> are O(log n).
Dijkstra, A* and D* Lite use it as the priority queue. heapq has no "decrease-key", so the code uses
<b>lazy deletion</b> (push a new entry; skip stale ones when popped).</li>
<li><code>itertools.count()</code>: an increasing counter used as a tie-breaker inside heap entries, so equal
priorities come out in insertion order (deterministic results) and Python never has to compare two
cells.</li>
</ul>

<h2>2.4 NumPy essentials</h2>
<ul>
<li>A grid is a 2-D boolean array: <code>walls[r, c]</code> is True for a wall. <code>occupied[cell]</code> works
with a tuple index.</li>
<li><b>Random generators</b>: <code>np.random.default_rng(seed)</code> returns a <code>Generator</code>. The project
never uses the global <code>np.random.*</code> functions in library code (Chapter 3).</li>
<li><b>Vectorisation</b>: Value Iteration updates all states at once with array arithmetic
(<code>(prob * (rew + gamma * V[nxt])).sum(axis=2)</code>), which is far faster than Python loops.</li>
<li><code>np.pad</code> builds the local view with out-of-grid cells marked as obstacles.</li>
</ul>

<h2>2.5 pandas, SciPy, Matplotlib</h2>
<ul>
<li><b>pandas</b>: every experiment returns a list of dict rows, turned into a <code>DataFrame</code> and saved
as CSV. <code>groupby</code> then computes per-condition statistics.</li>
<li><b>SciPy</b>: <code>scipy.stats.t.ppf</code> gives the Student t quantile for confidence intervals.</li>
<li><b>Matplotlib</b>: all figures; log scales for quantities spanning orders of magnitude.</li>
</ul>

<h2>2.6 PyTorch essentials</h2>
<ul>
<li><b>Tensor</b>: an n-dimensional array that can live on CPU or GPU (<code>.to(device)</code>). On Apple
silicon the GPU backend is <b>MPS</b>.</li>
<li><b>nn.Module</b>: a network. Layers used: <code>Conv2d</code>, <code>ReLU</code>, <code>AdaptiveAvgPool2d</code>,
<code>Flatten</code>, <code>Linear</code>.</li>
<li><b>Autograd</b>: <code>loss.backward()</code> computes gradients; <code>optimizer.step()</code> updates weights;
<code>optimizer.zero_grad()</code> clears old gradients.</li>
<li><code>@torch.no_grad()</code> turns off gradient tracking for inference (faster, less memory).
<code>.eval()</code>/<code>.train()</code> switch layer behaviour modes.</li>
<li><code>torch.save</code>/<code>torch.load</code> store checkpoints (weights plus metadata).</li>
</ul>

<h2>2.7 Gymnasium</h2>
<p>Gymnasium is the standard RL environment interface. An environment defines an
<code>action_space</code> and <code>observation_space</code>, and two methods:</p>
""" + pre("""
obs, info = env.reset(seed=..., options=...)
obs, reward, terminated, truncated, info = env.step(action)
""", "The Gymnasium API") + """
<p><code>terminated</code> means the episode ended for a reason inside the task (goal reached).
<code>truncated</code> means it was cut off from outside (step limit). The difference matters for learning:
a truncated state is <i>not</i> terminal, so its future value must still be counted (Chapter 8).
<code>gymnasium.utils.env_checker.check_env</code> verifies an environment follows the API; the tests run it.</p>

<h2>2.8 Other tools</h2>
<ul>
<li><b>argparse</b>: the command-line interface in <code>main.py</code>.</li>
<li><b>PyYAML</b>: configuration files.</li>
<li><b>multiprocessing / ProcessPoolExecutor</b> with the <code>spawn</code> start method: runs independent
experiment cells in parallel processes.</li>
<li><b>tracemalloc</b>, <b>time.perf_counter</b>, <b>gc</b>: memory and time measurement (Chapter 10).</li>
<li><b>pytest</b>: tests in <code>tests/</code>; <code>@pytest.mark.slow</code> marks long ones.</li>
<li><b>Streamlit</b>: the interactive dashboard (a Python script that becomes a web page).</li>
<li><b>git</b>: version control. Every result file records the commit that produced it.</li>
</ul>
"""


def ch3() -> str:
    return """
<h1 id="ch3"><span class="num">3</span>Architecture, configuration and seeding</h1>

<h2>3.1 Repository layout</h2>
""" + pre("""
main.py                       command-line entry point (hardware, demo, plan, experiment, figures, report)
configs/
  default.yaml                every base parameter (env, rewards, RL hyperparameters, convergence, paths)
  static.yaml dynamic.yaml stochastic.yaml      scenario presets
  experiments/*.yaml          per-experiment overrides (train_dqn, e1 ... e6)
src/pathrl/
  utils/          config.py (layered YAML), seeding.py, hardware.py
  environment/    config.py, dynamics.py, rewards.py, map_generator.py, obstacles.py, grid_world.py
  algorithms/     grid_graph.py, base_planner.py, base_policy.py, __init__.py (make_planner)
    classical/              bfs.py, dijkstra.py, astar.py, dstar_lite.py
    model_based/            value_iteration.py
    reinforcement_learning/ q_learning.py, network.py, replay_buffer.py, dqn.py
  evaluation/     profiling.py, rollout.py, navigation.py, scenarios.py, results.py
  analysis/       statistics.py, convergence.py, crossover.py
  visualization/  grid_visualizer.py, plots.py, demo.py, planner_demo.py
  experiments/    common.py, train_dqn.py, e1 ... e6, make_figures.py, make_report.py
  dashboard/      app.py (Streamlit)
tests/            pytest suite (158 tests)
outputs/          results/ (CSV + .meta.json), figures/, models/ (checkpoints), logs/
docs/             methodology, architecture, complexity analysis, RESULTS (generated), report
""", "Directory tree") + """
<p>The code is organised in <b>layers</b>. Each layer only uses the layers below it:</p>
""" + pre("""
  6  dashboard, report          app.py, make_report.py, make_figures.py
  5  analysis, visualization    statistics, convergence, crossover, plots
  4  evaluation                 rollout (policies), navigation (planners), profiling, results I/O
  3  algorithms                 classical planners | Value Iteration | Q-Learning, DQN
  2  environment                GridWorld, maps, obstacles, dynamics, rewards
  1  configuration              YAML -> dict -> frozen dataclasses; seeding; hardware snapshot
""", "Layers (higher layers depend on lower ones, never the reverse)") + """
""" + box("why", "Why layers?", """
<p>The single most important design rule is that <b>every algorithm uses the same environment
rules</b>. The planners' graph (<code>GridGraph</code>) is built on the environment's own
<code>attempt_move</code>; Value Iteration's model is built from the environment's own
<code>action_outcomes</code> and <code>compute_reward</code>. So no algorithm can be "right" about a world that
differs from the one it is tested in. That is the basis of a fair comparison.</p>""") + """

<h2>3.2 Layered configuration</h2>
<p>Every experimental parameter lives in YAML, never hard-coded. A run is fully described by three
layers, applied in order:</p>
<ol>
<li><code>configs/default.yaml</code>: all base values;</li>
<li>an experiment file such as <code>configs/experiments/e1_static_scaling.yaml</code>, which overrides only what it changes;</li>
<li>command-line overrides such as <code>--set env.grid_size=50</code>.</li>
</ol>
""" + src("src/pathrl/utils/config.py", "deep_merge") + src("src/pathrl/utils/config.py", "load_config") + """
<p>Note the deliberate failure on unknown keys: a typo like <code>env.grid_szie=50</code> raises
<code>KeyError</code> instead of silently adding a useless key and running the wrong experiment.</p>
<p>The <code>env</code> section is then converted into a <b>frozen, validated dataclass</b>, so an invalid value
fails immediately rather than halfway through a 30-seed run:</p>
""" + src("src/pathrl/environment/config.py", "EnvConfig", short_doc=True) + """

<h2>3.3 Seeding and reproducibility</h2>
<p>Randomness appears in map generation, slip, obstacle movement, exploration, network
initialisation and replay sampling. To make every run repeatable:</p>
<ol>
<li><b>No global randomness in library code.</b> Each component receives its own
<code>np.random.Generator</code>. Adding a random call in one module therefore cannot shift the numbers seen
by another.</li>
<li><b>Independent seeds come from <code>SeedSequence.spawn</code></b>, which guarantees statistically independent
streams. (Seeds 0, 1, 2 used directly can produce correlated streams in some generators.)</li>
<li><b>Disjoint seed families</b> for different purposes: the master seed plus an offset. Training maps use
offset 1000, validation 3000, E1/E2 10000, E3 20000, E4 30000, E5 40000, E6 50000. So no test map is
ever a training map; E5 checks this and measures an overlap of 0.</li>
</ol>
""" + src("src/pathrl/utils/seeding.py", "derive_seeds") + src("src/pathrl/experiments/common.py", "run_seeds") + box(
        "example", "Worked example: derived seeds",
        f"<p><code>derive_seeds(0, 3)</code> returns <code>{L.stats_examples()['seeds']}</code>: three unrelated 32-bit "
        "seeds from master seed 0. Running the same command again gives the same three numbers.</p>") + """
<h3>Common random numbers (CRN)</h3>
<p>When comparing two algorithms, you want them to face the <i>same</i> luck, so that differences come from
the algorithms rather than the dice. The environment draws a <b>fixed amount of randomness per step</b>
(two numbers for slip, one per obstacle per update) whether or not it is used. With the same episode
seed, A*, D* Lite and DQN then see the same slip sequence and the same obstacle proposals. This
variance-reduction technique is called <i>common random numbers</i>.</p>
""" + box("warn", "What determinism cannot guarantee", """
<p>PyTorch on GPUs (including MPS) lacks deterministic implementations of some kernels, so DQN
training is repeatable in procedure but not always bit-for-bit. The project asks for deterministic
algorithms where available and discloses the rest in the methodology.</p>""") + """

<h2>3.4 Data flow of one experiment</h2>
""" + pre("""
python main.py experiment e3
   |  load_config(default.yaml + e3_dynamic_obstacles.yaml + --set ...)
   v
experiments/e3_dynamic_obstacles.run(cfg)
   |  run_seeds(cfg, offset=20000)  -> 30 test-map seeds
   |  for each (N, change interval, seed): generate_map -> GridWorldEnv
   |      planners: run_planner_episode(...)   (evaluation/navigation.py)
   |      RL:       run_policy_episode(...)    (evaluation/rollout.py), checkpoints loaded on CPU
   v
save_results(df, cfg, "e3_dynamic_obstacles", "e3_raw")
   -> outputs/results/e3_dynamic_obstacles/e3_raw.csv
   -> e3_raw.meta.json  {resolved config, hardware snapshot, git commit, timestamp}

python main.py figures   -> outputs/results/summary/*.csv  +  outputs/figures/*.png
python main.py report    -> docs/RESULTS.md   (numbers formatted by code, never typed)
""", "From command to report") + src("src/pathrl/evaluation/results.py", "save_results")


def ch4() -> str:
    return """
<h1 id="ch4"><span class="num">4</span>The GridWorld environment</h1>
<p>One environment, <code>GridWorldEnv</code>, is shared by every planner and learner. This chapter explains
its rules and then walks through the code that implements them.</p>

<h2>4.1 Rules of the world</h2>
<ul>
<li><b>Grid.</b> N x N cells, coordinates <code>(row, col)</code> with row 0 at the top, so "UP" decreases the row.</li>
<li><b>Walls.</b> Exactly round(density x N<sup>2</sup>) static walls, never on start or goal. Maps are rejected
and resampled until the goal is reachable from the start.</li>
<li><b>Actions.</b> 4-connected: UP, DOWN, LEFT, RIGHT, each costing 1. The 8-connected ablation adds
diagonals costing sqrt 2, and forbids "corner cutting" between two blocked cells by default.</li>
<li><b>Collisions.</b> Moving into a wall, an obstacle or off the grid is a collision: the agent stays where it
is and receives -50. Collisions do not end the episode (by default).</li>
<li><b>Slip.</b> With probability p, the chosen action is replaced by one of the two perpendicular actions
(each with probability p/2). A slip never reverses the agent.</li>
<li><b>Moving obstacles.</b> <i>Random walk</i>: every k steps each obstacle picks uniformly among stay/up/down/left/right.
<i>Oscillating corridor</i>: obstacles slide back and forth across the shortest path on a fixed schedule.
Obstacles never enter a wall, the goal, the agent's cell or another obstacle.</li>
<li><b>Rewards.</b> +100 on reaching the goal, -50 for a collision, -1 per normal step.</li>
<li><b>Episode end.</b> <i>Terminated</i> when the goal is reached; <i>truncated</i> after N<sup>2</sup> steps.</li>
</ul>

<h2>4.2 One step, in order</h2>
<ol>
<li>Slip: maybe replace the action.</li>
<li>Move: apply the collision rules.</li>
<li>Reward: exactly one outcome per step (goal / collision / move).</li>
<li>Obstacles: if the step count is a multiple of k, obstacles move <i>after</i> the agent.</li>
<li>Observe: build the observation tensor; <code>info</code> reports the cells that changed.</li>
</ol>
""" + src("src/pathrl/environment/grid_world.py", "GridWorldEnv.step") + """

<h2>4.3 Movement, slip and collisions: <code>dynamics.py</code></h2>
<p>The transition model is written <b>once</b>, here. The environment calls <code>sample_delta</code> (draw one
outcome); Value Iteration calls <code>action_outcomes</code> (list all outcomes with probabilities). Because both
use the same functions, the model VI reasons about is exactly what the environment executes.</p>
""" + src("src/pathrl/environment/dynamics.py", "perpendicular") + src("src/pathrl/environment/dynamics.py", "action_outcomes") + \
        src("src/pathrl/environment/dynamics.py", "sample_delta") + src("src/pathrl/environment/dynamics.py", "attempt_move") + """
<h2>4.4 Rewards</h2>
""" + src("src/pathrl/environment/rewards.py", "compute_reward") + """
<p>Diagonal steps are scaled by their cost (sqrt 2), so with 8-connectivity "maximise reward" and "minimise path
cost" are the same objective. That keeps RL returns and planner path costs comparable.</p>

<h2>4.5 Maps</h2>
<p>A <code>GridMap</code> is the immutable description of one scenario: walls, start, goal and the initial
obstacle layout. Its arrays are made read-only so a map shared by several algorithms can never be
modified in place. Two algorithms given the same <code>GridMap</code> face exactly the same problem.</p>
""" + src("src/pathrl/environment/map_generator.py", "generate_map") + src("src/pathrl/environment/map_generator.py", "bfs_distances") + """
<p>Reachability is checked with 4-connected moves on the static walls. A 4-connected path is also a valid
8-connected path, so the same maps serve the 8-connected ablation. Moving obstacles can block a route
temporarily, never permanently, because they keep moving.</p>

<h2>4.6 Moving obstacles</h2>
""" + src("src/pathrl/environment/obstacles.py", "RandomWalkObstacles") + """
<p><code>update</code> returns only the <b>net</b> changes: a cell vacated by one obstacle and filled by another in the
same update is not a change. D* Lite uses exactly this list to know which edges to repair.</p>

<h2>4.7 Observations: global and local</h2>
<p>Learners receive a 3-channel image: channel 0 obstacles, channel 1 agent, channel 2 goal.</p>
<ul>
<li><b>Global</b>: 3 x N x N, the whole map.</li>
<li><b>Local</b>: 3 x 7 x 7 centred on the agent. Cells outside the grid look like obstacles. If the goal is
outside the window it is drawn on the window border in its direction, like a compass. The agent knows
<i>where</i> the goal is, but not the map beyond its view.</li>
</ul>
""" + src("src/pathrl/environment/grid_world.py", "GridWorldEnv._observe_local") + box("why", "Why the trained agents use the local view", """
<p>A CNN that sees the whole 10 x 10 map must learn a different picture for every map. In pilot runs on
held-out validation maps it reached only 0.00-0.04 success. A local view looks the same in every map
("wall to my left, goal up-right"), so it generalises better: 0.22 validation success after 150k steps in the
pilot, and 0.34-0.46 after the final 500k-step training. The choice was made on validation maps,
never on test maps (documented in methodology section 4).</p>""") + """

<h2>4.8 Two independent random streams</h2>
""" + src("src/pathrl/environment/grid_world.py", "GridWorldEnv.reset") + """
<p>Slip and obstacle movement each get their own generator, both derived from the episode seed. Slipping
therefore never changes which way the obstacles move, and vice versa: another application of CRN.</p>
""" + box("check", "Check your understanding", """
<p>1. Why does <code>sample_delta</code> draw <code>side</code> even when no slip happens?
2. Why is reachability checked with 4-connected moves even for the 8-connected ablation?
3. In the local view, how does the agent know where a far-away goal is?</p>
<p class="small">Answers: (1) to use a fixed amount of randomness per step, for common random numbers;
(2) a 4-path is also an 8-path, so one map set serves both; (3) the goal is drawn on the window border in its direction.</p>""")


def ch5() -> str:
    pc = L.planner_comparison()
    hn = L.heuristic_numbers()
    rows = __import__("pandas").DataFrame(pc["rows"])
    return """
<h1 id="ch5"><span class="num">5</span>Grids as graphs and classical search</h1>

<h2>5.1 A grid is a graph</h2>
<p>A <b>graph</b> is a set of vertices (here: free cells) and edges (legal moves between them), each with a
cost. Path planning on a grid is <b>shortest-path search</b> on this graph. With 4-connectivity every cell
has at most 4 edges of cost 1; with 8-connectivity up to 8, diagonals costing sqrt 2. V = number of cells,
E = number of edges, and E &le; 8V, which matters for complexity.</p>
<p><code>GridGraph</code> defines the edges for every planner. It is built on the environment's
<code>attempt_move</code>, so a planner can never find a "path" the environment would refuse to execute.</p>
""" + src("src/pathrl/algorithms/grid_graph.py", "GridGraph.neighbors") + src("src/pathrl/algorithms/grid_graph.py", "GridGraph.path_cost") + """
<h3>Measuring work: <code>SearchStats</code></h3>
<p>Wall-clock time depends on the computer. The planners also count <b>machine-independent</b> work:</p>
""" + src("src/pathrl/algorithms/base_planner.py", "SearchStats", short_doc=False) + """
<p>An <b>expansion</b> is removing a cell from the frontier and examining its neighbours: the standard unit of
search effort. Reporting both expansions and time lets readers see when they disagree (they do, for D* Lite).</p>

<h2>5.2 Breadth-first search (BFS)</h2>
<p>BFS explores cells in order of <i>number of moves</i> from the start, level by level, using a FIFO queue.
The first time a cell is reached is via a fewest-moves route, so BFS records the parent and never revisits it.</p>
<ul>
<li>Finds the path with the <b>fewest moves</b>. That is the cheapest path only when all moves cost the same
(true for 4-connected grids, false with sqrt 2 diagonals: the tests found 3 of 40 random 8-connected maps where BFS
is cost-suboptimal).</li>
<li>Time O(V + E), memory O(V).</li>
</ul>
""" + src("src/pathrl/algorithms/classical/bfs.py", "BFS.plan") + src("src/pathrl/algorithms/base_planner.py", "Planner.reconstruct", short_doc=True) + """

<h2>5.3 Dijkstra's algorithm</h2>
<p>Dijkstra handles any non-negative edge costs. It keeps <b>g(n)</b>, the cheapest known cost from the start
to n, and always expands the frontier cell with the smallest g. Because costs are non-negative, nothing found
later can make that cell cheaper, so its g is <b>final</b> when it is popped ("settled"). It stops when the goal
is popped.</p>
<p><b>Relaxation</b>: when expanding u with neighbour v, if g(u) + c(u,v) &lt; g(v), update g(v) and v's parent.</p>
""" + math("""
Time:   O(E log V) with a binary heap   (on a grid E <= 8V, so O(V log V))
Memory: O(V)""") + """
<p><b>Lazy deletion.</b> Python's heapq cannot lower the priority of an entry already in the heap. When a cheaper
route to v is found, the code pushes a new entry and leaves the old one; when an old entry is popped later
it is skipped because v is already closed. The heap can then hold up to E entries, which is why the bound is
O(E log E) = O(E log V).</p>
""" + src("src/pathrl/algorithms/classical/dijkstra.py", "Dijkstra.plan") + """

<h2>5.4 A* search</h2>
<p>A* adds a <b>heuristic</b> h(n), an estimate of the remaining cost from n to the goal, and orders the heap by</p>
""" + math("f(n) = g(n) + w * h(n)        (w = 1 for standard A*)") + """
<ul>
<li><b>Admissible</b> h never overestimates the true remaining cost. Then the first time the goal is popped,
its path is optimal.</li>
<li><b>Consistent</b> (monotone) h satisfies h(a) &le; c(a,b) + h(b) for every edge. Then every node's g is final
when popped, so a closed node never needs reopening. Consistent implies admissible.</li>
<li><b>h = 0</b> turns A* into Dijkstra. A better-informed h (closer to the true cost) expands fewer nodes.</li>
<li><b>Weighted A*</b> (w &gt; 1) is inadmissible: usually far fewer expansions, with cost guaranteed &le; w x optimal.
E2 measures this trade-off.</li>
</ul>
<h3>Heuristics on grids</h3>
""" + math(f"""
Manhattan (4-connected):  h = |dr| + |dc|
Octile    (8-connected):  h = max(|dr|,|dc|) + (sqrt2 - 1) * min(|dr|,|dc|)
Euclidean:                h = sqrt(dr^2 + dc^2)

From (0,0) to (3,7):  Manhattan = {hn['manhattan']},  octile = {hn['octile']:.4f},  Euclidean = {hn['euclid']:.4f}""") + """
<p>Manhattan is the exact distance on an empty 4-connected grid, so it is admissible and consistent there.
With 8-connectivity it overestimates (one sqrt 2 diagonal replaces two straight moves), so the project uses octile,
the exact empty-grid distance for 8-connected moves.</p>
<h3>Tie-breaking</h3>
<p>On open grids many cells share the same f. The heap entries are <code>(f, h, counter, cell)</code>: among equal f,
the cell with smaller h (closer to the goal) comes first. This choice can change expansions by an order of
magnitude, so it is part of the algorithm specification and must be reported.</p>
""" + src("src/pathrl/algorithms/classical/astar.py", "AStar.plan") + box("example", "Worked example: every planner on the same 10 x 10 map", """
<p>The map below (S start, G goal, # wall) is solved by each planner. Path costs agree (all optimal on this
4-connected map); the work differs.</p>""" + pre(pc["ascii"], "demo map") + table(rows, fmt={"path cost": lambda v: f"{v:g}"}) + """
<p>BFS, Dijkstra and A* with h = 0 expand exactly the same cells: with unit costs they explore in the same
order. The Manhattan heuristic focuses the search toward the goal. In the pictures, <code>*</code> is the path and
<code>x</code> every other cell the planner expanded. On this map A* expanded <i>only</i> the 19 cells of its own path
(goal-directed tie-breaking never had to back up), while BFS expanded 62:</p>""" + "<div class='two'>" + pre(pc["renders"]["BFS"], "BFS: path * and expanded cells x") + pre(pc["renders"]["A* (Manhattan)"], "A* (Manhattan): path * and expanded cells x") + "</div>" + """
<p>D* Lite's first plan expands more than A*: it searches backwards from the goal and, on its first search,
expands every cell whose f is no more than the optimal cost (Chapter 6).</p>""") + """

<h2>5.5 Summary of the classical algorithms</h2>
""" + table(__import__("pandas").DataFrame([
        ["BFS", "FIFO queue", "fewest moves (optimal only for equal costs)", "O(V+E)", "O(V)"],
        ["Dijkstra", "min-heap on g", "optimal for non-negative costs", "O(E log V)", "O(V)"],
        ["A*", "min-heap on g+h", "optimal if h admissible", "O(E log V) worst case; usually far less", "O(V)"],
        ["Weighted A*", "min-heap on g+w h", "cost &le; w x optimal", "usually much less than A*", "O(V)"],
        ["D* Lite", "min-heap on 2-part key", "optimal on the current map; repairs incrementally", "first search ~ A*-like; repairs local", "O(V)"],
    ], columns=["Algorithm", "Frontier", "Guarantee", "Time", "Memory"]))


def ch6() -> str:
    d = L.dstar_repair()
    return """
<h1 id="ch6"><span class="num">6</span>Incremental search: D* Lite</h1>
<p>When obstacles move, a plan can become invalid. The simple fix is to <b>replan from scratch</b> with A*. D* Lite
(Koenig and Likhachev, 2002) instead <b>repairs</b> its previous search, re-examining only cells affected by
the change. It builds on Lifelong Planning A* (LPA*).</p>

<h2>6.1 Searching backwards</h2>
<p>D* Lite searches from the <b>goal</b> toward the agent. So g(s) means "cost from s to the goal". The goal never
moves, but the agent does; searching backwards means that when the agent moves, all g-values stay valid and
only the "start" of the query changes.</p>

<h2>6.2 Two values per cell: g and rhs</h2>
""" + math("""
g(s)    current estimate of the cost from s to the goal
rhs(s)  one-step lookahead:  rhs(s) = min over neighbours s' of [ c(s, s') + g(s') ],   rhs(goal) = 0

s is CONSISTENT        when g(s) == rhs(s)
s is OVER-consistent   when g(s) >  rhs(s)   (a cheaper route appeared, e.g. an obstacle vanished)
s is UNDER-consistent  when g(s) <  rhs(s)   (its route got dearer, e.g. an obstacle appeared)""", "The core idea") + """
<p>Only inconsistent cells are in the priority queue. When the map changes, only cells whose edges changed get
their rhs recomputed (<code>UpdateVertex</code>). Inconsistency then spreads only as far as the change actually matters;
values far away are kept from the previous search. That is what "incremental" means.</p>
<ul>
<li>Over-consistent cell popped: set g = rhs (accept the improvement), then update its neighbours.</li>
<li>Under-consistent cell popped: set g = infinity (forget the stale value), then update it and its neighbours so they
look for new routes.</li>
</ul>

<h2>6.3 The priority key and k_m</h2>
""" + math("""
key(s) = [ min(g(s), rhs(s)) + h(start, s) + k_m ,   min(g(s), rhs(s)) ]      compared lexicographically""") + """
<p>The first component plays the role of A*'s f; the second breaks ties. When the agent moves, h(start, s) changes for
<i>every</i> queued cell. Re-keying the whole queue would be expensive, so D* Lite adds the distance moved to a
running offset k<sub>m</sub>. Old keys remain valid lower bounds; when a cell with an out-of-date key is popped
(k_old &lt; k_new) it is simply re-inserted with its correct key.</p>
<p>The search stops when the agent's cell is consistent and its key is no larger than the smallest key in the queue.</p>

<h2>6.4 The code</h2>
""" + src("src/pathrl/algorithms/classical/dstar_lite.py", "DStarLite.plan") + src("src/pathrl/algorithms/classical/dstar_lite.py", "DStarLite._key") + \
        src("src/pathrl/algorithms/classical/dstar_lite.py", "DStarLite._update_vertex") + \
        src("src/pathrl/algorithms/classical/dstar_lite.py", "DStarLite._compute_shortest_path") + """
<h3>Three implementation details worth understanding</h3>
<ol>
<li><b>Rounded keys.</b> Sums of sqrt 2 added in different orders differ in the last bits (8.242640687119286 vs
...284). Two mathematically equal keys then compare unequal and the termination test can stop the search early.
This bug actually occurred in 8-connected tests; rounding keys to 9 decimals fixed it.</li>
<li><b>3 x 3 repair neighbourhood.</b> A changed cell alters its own edges and, with diagonals, the corner-cutting
rule of edges between its neighbours, so every cell in the surrounding 3 x 3 block is updated.</li>
<li><b>Self-healing.</b> <code>plan()</code> also diffs the occupancy grid against its own copy, so the result stays correct
even if a caller forgets to call <code>update()</code>.</li>
</ol>
""" + box("example", "Worked example: a repair after the map changes", f"""
<p>On the demo map, D* Lite's first plan costs {d['first_cost']:g} with {d['first_exp']} expansions. The agent walks
to {d['pos']}; then cell {d['blocked']} on its path becomes blocked. D* Lite repairs: new cost {d['rep_cost']:g},
{d['rep_exp']} expansions, {d['rep_updates']} vertex updates, still {d['inits']} initialisation (no restart).
A* from scratch at the same position: cost {d['astar_cost']:g}, {d['astar_exp']} expansions.</p>""" +
                                                                            pre(d["render"], "repaired path (*) from the agent's new position S") + """
<p>Note what this shows: on a small map, repairing is <i>not</i> automatically less work than replanning.
D* Lite pays for bookkeeping (vertex updates, key recomputation). Its advantage depends on where the change is
and how much of the old search remains valid. E3 measures this systematically.</p>""") + box("warn", "Measured lesson from the experiments", """
<p>D* Lite's <b>first</b> search expands every cell with f &le; C* (the optimal cost), more than A*, which stops as
soon as the goal is popped and benefits from tie-breaking toward the goal. In E3, D* Lite did fewer expansions than
A*-replanning in 12 of 20 dynamic conditions (mostly frequent changes), yet it was slower in wall time in all 20,
because each operation (g/rhs bookkeeping, keys, heap traffic) costs more in Python. Expansion count and wall time
can rank algorithms in opposite orders, so report both.</p>""") + """
<h2>6.5 How we know it is correct</h2>
<p>The tests (<code>tests/test_dstar_lite.py</code>) apply more than 100 random map changes and check after each repair that
D* Lite's path cost equals a fresh A* search on the same map, that the path is valid, and that the search state is
preserved (no silent re-initialisation).</p>
"""


def chapters() -> list[str]:
    return [ch1(), ch2(), ch3(), ch4(), ch5(), ch6()]
