"""Chapters 7-11: MDPs and Value Iteration, Q-Learning, DQN, evaluation, statistics/convergence/crossover."""

from __future__ import annotations

import pandas as pd

import live_examples as L
from guide_lib import box, pre, src, table


def math(text: str, caption: str | None = None) -> str:
    return pre(text, caption, cls="math")


def ch7() -> str:
    vi = L.value_iteration_example()
    return """
<h1 id="ch7"><span class="num">7</span>Markov decision processes and Value Iteration</h1>

<h2>7.1 The MDP</h2>
<p>Reinforcement learning describes decision problems as a <b>Markov decision process</b> (MDP), a tuple
(S, A, P, R, gamma):</p>
<ul>
<li><b>S</b>, states: here, the agent's cell (on a fixed map).</li>
<li><b>A</b>, actions: up, down, left, right.</li>
<li><b>P(s' | s, a)</b>, transition probabilities: where action a in state s leads. Slip makes this stochastic.</li>
<li><b>R(s, a, s')</b>, rewards: -1, -50 or +100.</li>
<li><b>gamma</b>, discount factor (0.99): a reward t steps in the future is worth gamma<sup>t</sup> today.</li>
</ul>
<p>"Markov" means the future depends only on the current state, not on how you got there. That holds for the
agent's cell on a static map. With moving obstacles the true state also includes every obstacle's position;
treating the cell alone as the state then makes the problem only <i>partially observable</i>, a limitation
stated whenever tabular methods meet dynamics.</p>

<h2>7.2 Returns, policies and value functions</h2>
""" + math("""
Return from time t:       G_t = r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ...
Policy:                   pi(a | s)   (here deterministic: one action per state)
State value:              V^pi(s)   = E[ G_t | s_t = s, following pi ]
Action value:             Q^pi(s,a) = E[ G_t | s_t = s, a_t = a, then following pi ]
Optimal values:           V*(s) = max_pi V^pi(s),   Q*(s,a) = max_pi Q^pi(s,a)
Optimal policy:           pi*(s) = argmax_a Q*(s,a)""") + """
<h2>7.3 The Bellman optimality equation</h2>
<p>The value of a state equals the best action's expected immediate reward plus the discounted value of where
you land:</p>
""" + math("""
V*(s) = max_a  sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V*(s') ]
Q*(s,a) =      sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * max_{a'} Q*(s',a') ]""") + """
<h2>7.4 Value Iteration</h2>
<p>Value Iteration turns the Bellman equation into an update and applies it to every state repeatedly
("sweeps") until the values stop changing:</p>
""" + math("""
V_{k+1}(s) = max_a  sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') * (1 - done) ]

Convergence:   residual_k = max_s | V_{k+1}(s) - V_k(s) |  <  theta   (theta = 1e-6)""") + """
<p><b>Why it converges.</b> The Bellman update is a <b>gamma-contraction</b> in the max-norm: applying it to two
value functions brings them at least a factor gamma closer. So the error shrinks geometrically, and when the residual
is below theta the values are within theta x gamma/(1 - gamma) of V*. The contraction also explains slow cases: with
gamma = 0.99 the error shrinks by only 1% per sweep in the worst case, so situations where the policy can cycle
(for example, risky moves next to walls under slip) need on the order of ln(error/theta)/ln(1/gamma) &asymp; 1,400 sweeps.
E6 observed ~1,300-1,400 sweeps exactly in such conditions (dense maps with slip). That explanation is a hypothesis,
not something the experiments tested directly.</p>
<h3>The vectorised implementation</h3>
<p>The model is built once as arrays of shape (S, A, K): for each state and action, the K &le; 3 possible outcomes
(intended move, slip left, slip right) with next-state index, probability, reward and done flag. Each sweep is then one
array expression.</p>
""" + src("src/pathrl/algorithms/model_based/value_iteration.py", "ValueIteration._build_model", short_doc=True) + \
        src("src/pathrl/algorithms/model_based/value_iteration.py", "ValueIteration.solve") + """
<p>Only cells <b>connected to the goal</b> are updated. Walled-off pockets can never succeed; their values only decay
slowly, and including them would make "sweeps to convergence" measure those dead pockets instead of the navigation
problem. (This was found during development: the sweep count was dominated by unreachable cells until they were excluded.)</p>
""" + box("example", "Worked example: Value Iteration on a 5 x 5 map", f"""
<p>Values V*(s) (G = goal, ### = wall) and the greedy policy, without slip and with slip 0.2.</p>
<div class="two"><div>{pre(vi[0.0]['values'], f"slip 0: V*  ({vi[0.0]['sweeps']} sweeps)")}{pre(vi[0.0]['policy'], "slip 0: greedy policy")}</div>
<div>{pre(vi[0.2]['values'], f"slip 0.2: V*  ({vi[0.2]['sweeps']} sweeps)")}{pre(vi[0.2]['policy'], "slip 0.2: greedy policy")}</div></div>
<p>Without slip the start's value is {vi[0.0]['start_value']:.2f}: the optimal path has 8 moves, so
V = -(1 + gamma + ... + gamma^6) + 100 gamma^7 = {vi[0.0]['start_value']:.2f}. With slip, values fall sharply near walls
(a slip into a wall costs -50) and convergence takes {vi[0.2]['sweeps']} sweeps instead of {vi[0.0]['sweeps']}: the
same contraction argument at work.</p>""") + box("warn", "Information assumption", """
<p>Value Iteration is given the exact model P and R, including the slip probability. Model-free learners never see it.
Whenever VI is compared with Q-Learning or DQN, this advantage must be stated. In E4, VI achieved the highest return
in every slip condition, which is what knowing the model buys you.</p>""")


def ch8() -> str:
    td = L.td_update_example()
    return """
<h1 id="ch8"><span class="num">8</span>Model-free learning: tabular Q-Learning</h1>

<h2>8.1 Learning without a model</h2>
<p>Value Iteration needs P and R. A <b>model-free</b> learner only experiences transitions (s, a, r, s') by acting,
and learns from those samples. Q-Learning (Watkins and Dayan, 1992) learns Q*(s, a) directly with a
<b>temporal-difference (TD)</b> update:</p>
""" + math("""
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_{a'} Q(s',a') * (1 - terminated)  -  Q(s,a) ]
                           \\_____________ TD target ____________________/
                           \\_______________________ TD error _______________________/""") + """
<ul>
<li><b>alpha</b>, the learning rate: how far to move toward the target each time (E6 tests 0.01-0.5).</li>
<li><b>Bootstrapping</b>: the target uses the current estimate Q(s', .) instead of waiting for the full return.</li>
<li><b>Off-policy</b>: the target uses max (the greedy policy) even though the agent explores.</li>
</ul>
""" + box("example", "Worked example: one TD update (executed through QLearning.update)", f"""
<p>Q(s,a) = {td['q_before']}, reward r = {td['r']}, best next value max Q(s',.) = {td['max_next']}, gamma = 0.99, alpha = 0.1.</p>
<p>Target = {td['r']} + 0.99 x {td['max_next']} = {td['target']:.2f}. TD error = {td['target']:.2f} - {td['q_before']} = {td['td']:.2f}.
New Q(s,a) = {td['q_before']} + 0.1 x {td['td']:.2f} = <b>{td['q_after']:.3f}</b>.</p>""") + """
<h2>8.2 Exploration: epsilon-greedy</h2>
<p>If the agent always takes its current best action it may never discover a better one. With probability epsilon it
acts randomly, otherwise greedily. Epsilon decays linearly from 1.0 to 0.05 over the first 80% of training
(<code>linear_schedule</code>). During training, ties between equal Q-values are broken randomly; with an all-zero initial
table a fixed tie-break would always pick action 0 and bias exploration.</p>
""" + src("src/pathrl/algorithms/base_policy.py", "linear_schedule") + """
<h2>8.3 Terminated vs truncated (a classic bug)</h2>
<p>If an episode is cut off by the step limit, the last state is <i>not</i> terminal: the agent could have continued.
So the target must still include gamma max Q(s', .). Only reaching the goal sets the future value to zero. Treating
truncation as termination teaches the agent that the step limit is a cliff, and is a common, silent error.</p>
""" + src("src/pathrl/algorithms/reinforcement_learning/q_learning.py", "QLearning.update") + \
        src("src/pathrl/algorithms/reinforcement_learning/q_learning.py", "QLearning.train") + """
<h2>8.4 Convergence in theory and in practice</h2>
<p>Q-Learning provably converges to Q* if every state-action pair is visited infinitely often and the learning rates
decrease appropriately (sum of alpha infinite, sum of alpha<sup>2</sup> finite). In practice we use a constant alpha and a finite
budget, so convergence is judged empirically by the criterion in Chapter 11. E6 found the usable alpha range
narrowing as grids grow: at N = 100 no setting converged within 3,000 episodes, because the goal reward must propagate back
along a path of at least 2(N - 1) = 198 moves (start and goal are opposite corners in E6) by bootstrapping, one step per update.</p>
<p><b>Limitations</b>: the table has N<sup>2</sup> x 4 entries and is tied to one map: it cannot generalise to a new map. That
is why the generalist agents need a neural network.</p>
"""


def ch9() -> str:
    layers = pd.DataFrame(L.network_layers(7))
    totals = pd.DataFrame(L.network_totals())
    return """
<h1 id="ch9"><span class="num">9</span>Deep Q-Networks: DQN and Double DQN</h1>

<h2>9.1 From a table to a function</h2>
<p>A table cannot represent Q for maps it has never seen. <b>Function approximation</b> replaces the table with a
parametrised function Q(s, a; theta), here a convolutional neural network (CNN) that takes the observation image and
outputs one Q-value per action. Similar-looking situations then get similar values: that is how generalisation happens.</p>

<h2>9.2 The network</h2>
<p>A <b>convolution</b> slides small learned filters (3 x 3 here) over the image, detecting local patterns such as "wall to
the left". Stacking convolutions builds larger patterns. ReLU (max(0, x)) adds non-linearity. Adaptive average pooling
fixes the spatial size so the final fully connected layers have a known input size.</p>
""" + src("src/pathrl/algorithms/reinforcement_learning/network.py", "QNetwork") + table(layers, "Layer-by-layer parameters for the 7 x 7 local view used by the trained agents") + \
        table(totals, "Total parameters by input size (stride 2 starts above 12 x 12, which is why 20 x 20 is smaller than 10 x 10)") + """
<h2>9.3 Why naive deep Q-learning is unstable, and the fixes</h2>
<p>Plugging a network into the Q-learning update tends to diverge. DQN (Mnih et al., 2015) combined several fixes:</p>
<ol>
<li><b>Experience replay.</b> Consecutive transitions are strongly correlated (the agent walks a path). Storing them
in a large buffer and training on random mini-batches breaks the correlation and reuses each experience many times.</li>
<li><b>Target network.</b> The target y uses a frozen copy of the network, refreshed every 2,000 steps, so the target
does not move with every gradient step ("chasing your own tail").</li>
<li><b>Huber loss</b> (smooth L1): quadratic for small errors, linear for large ones, so occasional big surprises
(a -50 collision) cannot produce huge gradients.</li>
<li><b>Gradient clipping</b> (norm 10) and the <b>Adam</b> optimiser for stable steps.</li>
<li><b>Reward scaling</b> (x 0.01 inside the learner). Multiplying all rewards by a positive constant multiplies Q* by
the same constant, so the optimal policy is unchanged, but TD errors stay near the Huber loss's quadratic region.
Reported returns are always unscaled.</li>
</ol>
""" + math("""
Loss(theta) = Huber( Q(s,a; theta) - y ),   averaged over a mini-batch of 64 transitions from the replay buffer

DQN target:         y = r + gamma * max_{a'} Q_target(s', a')                         (1 - terminated)
Double DQN target:  y = r + gamma * Q_target(s', argmax_{a'} Q_online(s', a'))        (1 - terminated)""") + """
<h2>9.4 Overestimation and Double DQN</h2>
<p>The max over noisy estimates is biased upward: if Q-values have random errors, the action that happens to be
overestimated is the one selected <i>and</i> counted. Double DQN (van Hasselt, Guez and Silver, 2016) decouples the two:
the online network <i>selects</i> the best next action, the target network <i>evaluates</i> it. In this codebase that is
the <b>only</b> difference between the two algorithms:</p>
""" + src("src/pathrl/algorithms/reinforcement_learning/dqn.py", "DQN.compute_targets") + src("src/pathrl/algorithms/reinforcement_learning/dqn.py", "DQN.learn_step") + """
<h2>9.5 The replay buffer</h2>
""" + src("src/pathrl/algorithms/reinforcement_learning/replay_buffer.py", "ReplayBuffer", short_doc=True) + """
<p>Observations are 0/1 grids, so they are stored as <code>uint8</code> (4x smaller than float32) and converted when sampled.
Note what is stored: <code>terminated</code>, not <code>truncated</code>, for the reason in Chapter 8.</p>

<h2>9.6 The training loop</h2>
""" + src("src/pathrl/algorithms/reinforcement_learning/dqn.py", "DQN.train", short_doc=True) + """
<p>Key schedule (final generalist training): 500,000 environment steps; learning starts after 2,000 steps; one
gradient step every 4 environment steps (124,501 gradient steps in total); target sync every 2,000 steps; epsilon decays
from 1.0 to 0.05 over the first 100,000 steps. Each episode samples a training map and a condition (static or random-walk
obstacles with k in {1, 2, 5, 10, 20}; slip in {0, 0.1, 0.2}), via <code>ScenarioSampler</code>. Training ran on the Apple GPU
(MPS); checkpoints are always loaded on the <b>CPU</b> for inference, so inference cost is measured on the same hardware as
the planners.</p>
""" + box("warn", "Why the trained agents fail: perceptual aliasing (hypothesis)", """
<p>The policy is <b>memoryless</b> and sees only 7 x 7 cells. Two different places can look identical; if the greedy action
from each leads to the other, the agent loops until the step limit. In E3 every RL failure was a truncation at the step
limit, never a terminal event, and success <i>increased</i> when the world was perturbed (faster obstacles, or slip), which
breaks such loops. This is consistent with aliasing loops, but loop counts were not measured, so it remains a hypothesis.
Remedies to test: recurrent (LSTM/GRU) policies, frame stacking, or a larger view.</p>""") + """
<h2>9.7 Checkpoints</h2>
""" + src("src/pathrl/algorithms/reinforcement_learning/dqn.py", "DQN.save") + src("src/pathrl/algorithms/reinforcement_learning/dqn.py", "DQN.load")


def ch10() -> str:
    return """
<h1 id="ch10"><span class="num">10</span>Evaluation machinery</h1>
<p>Algorithms are only half the project; the other half is measuring them fairly. This chapter covers how episodes are
run and timed.</p>

<h2>10.1 Running a policy: <code>rollout.py</code></h2>
<p>A learned policy acts once per step. Inference time is measured around <code>policy.act</code> only, never around the
environment step. <code>act</code> never learns (enforced by the <code>LearnedPolicy</code> interface), so the measured time is pure
inference.</p>
""" + src("src/pathrl/evaluation/rollout.py", "run_policy_episode") + """
<h2>10.2 Running a planner as an agent: <code>navigation.py</code></h2>
<p>A planner produces a path; an agent must follow it in a world that changes. The agent follows the current plan one
cell per step and asks for a new plan according to a <b>replanning policy</b>:</p>
<ul>
<li><b>on_change</b>: replan whenever the map changed or the agent is off its plan (after a slip or collision). Always holds
an optimal plan for the current map.</li>
<li><b>on_block</b>: replan only when one of the next 3 plan cells is now blocked, or the agent is off-plan. A lazy baseline
that does less work but may follow a no-longer-optimal plan. It was the cheapest strategy in every E3 condition.</li>
</ul>
<p>Every change is passed to <code>planner.update()</code> so D* Lite always knows what changed. If no path exists at the moment
(obstacles can temporarily cut the goal off) the agent takes the legal move that most reduces the heuristic distance; these
are counted as <code>blocked_steps</code>. Costs are split into the <b>initial plan</b> and later <b>repairs</b>.</p>
""" + src("src/pathrl/evaluation/navigation.py", "run_planner_episode", short_doc=True) + """
<h2>10.3 Timing and memory: <code>profiling.py</code></h2>
""" + src("src/pathrl/evaluation/profiling.py", "measure", short_doc=True) + """
<ul>
<li><code>time.perf_counter()</code> is a high-resolution monotonic clock, the right tool for short intervals.</li>
<li>The <b>median</b> of repeated runs ignores one-off slow runs (a background process, cache warm-up).</li>
<li>The <b>garbage collector is disabled</b> while timing so a random GC pause is not billed to the algorithm.</li>
<li><b>Memory is measured in a separate run</b> under <code>tracemalloc</code>, because tracing every allocation slows code down
2-5x and would inflate the timing. tracemalloc sees Python allocations only, not memory inside PyTorch.</li>
</ul>
""" + box("warn", "Timing hygiene: what can corrupt a measurement", """
<p>Other processes competing for the CPU. In this project the DQN agents trained while 12 E6 worker processes were
running and took 982-1,300 s; the Double DQN agents trained alone and took 531-532 s. The algorithms differ only in the
target formula, so the gap is contention, not algorithm. This is disclosed rather than hidden. Rules adopted: timing-sensitive
experiments (E1, E3) run serially, one torch thread, and parallel runs are only compared with each other.</p>""") + """
<h2>10.4 Scenario sampling and map sets</h2>
""" + src("src/pathrl/evaluation/scenarios.py", "ScenarioSampler") + src("src/pathrl/evaluation/scenarios.py", "make_map_set") + """
<h2>10.5 Parallel execution</h2>
""" + src("src/pathrl/experiments/common.py", "parallel_map") + """
<p><code>spawn</code> starts fresh interpreter processes (safer with PyTorch than <code>fork</code>), and each worker is limited to
one BLAS/torch thread so N workers do not oversubscribe the CPU.</p>
"""


def ch11() -> str:
    st = L.stats_examples()
    ci = st["ci"]
    return """
<h1 id="ch11"><span class="num">11</span>Statistics, convergence and crossover analysis</h1>

<h2>11.1 Confidence intervals</h2>
<p>Each experimental condition is repeated on independent seeds (30 maps for most classical results, 3-5 seeds for
RL runs). The <b>unit of replication is the seed</b>, never individual steps or episodes within one seed, which are
correlated. For a metric measured once per seed:</p>
""" + math("""
95% CI of the mean:   mean  +/-  t_{0.975, n-1} * s / sqrt(n)

s = sample standard deviation (ddof = 1);  t = Student t quantile
n = 3: t = 4.303     n = 5: t = 2.776     n = 30: t = 2.045     (normal approximation: 1.96)""") + """
<p>Student's t is used instead of 1.96 because n is small; with n = 3 the interval is more than twice as wide as the normal
approximation would suggest. This is why the project does not claim a difference between DQN and Double DQN: their
intervals overlap with n = 3.</p>
""" + src("src/pathrl/analysis/statistics.py", "mean_ci") + box("example", "Worked example", f"""
<p>Sample {st['sample']}: mean {ci['mean']:.2f}, s = {ci['std']:.3f}, n = 5, so the 95% CI is
[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}].</p>""") + """
<h3>Proportions: the Wilson interval</h3>
<p>For a success rate p from n trials, the textbook interval p &plusmn; z sqrt(p(1-p)/n) collapses to zero width at p = 0 or 1 and
can leave [0, 1]. The Wilson interval does neither.</p>
""" + src("src/pathrl/analysis/statistics.py", "wilson_interval", short_doc=True) + box("example", "Worked example", f"""
<p>0 successes out of 20: Wilson 95% CI [{st['wilson_0_of_20'][0]:.3f}, {st['wilson_0_of_20'][1]:.3f}], not [0, 0].
18 out of 20: [{st['wilson_18_of_20'][0]:.3f}, {st['wilson_18_of_20'][1]:.3f}].</p>""") + """

<h2>11.2 What "converged" means (one definition per family)</h2>
""" + table(pd.DataFrame([
        ["Q-Learning, DQN, Double DQN", "3 consecutive greedy evaluations, each with success >= 0.9 and spread of mean return <= 5% of its magnitude",
         "training episode of the 3rd (confirming) evaluation"],
        ["Value Iteration", "Bellman residual max|V_{k+1} - V_k| < 1e-6", "sweeps"],
        ["BFS, Dijkstra, A*", "terminate exactly; nothing converges", "n/a"],
        ["D* Lite", "after each repair all relevant cells are consistent (g = rhs)", "reported as repair work"],
    ], columns=["Family", "Criterion", "Reported as"])) + src("src/pathrl/analysis/convergence.py", "rl_convergence", short_doc=True) + """
<ul>
<li>Evaluations are <b>greedy</b> (no exploration), so they measure the learned policy, not luck.</li>
<li>The reported episode is the <b>confirming</b> one, not the first good window: a single lucky evaluation does not count.</li>
<li><b>Non-convergent runs are kept and reported</b>, never dropped. Dropping them would make an algorithm look better than it is.</li>
<li>Converging and staying good are different things: E6 found runs that met the criterion and later degraded
(Double DQN, N = 10, lr = 1e-4: 3/3 converged, mean final success 0.67). So final performance is reported separately.</li>
</ul>

<h2>11.3 The crossover analysis</h2>
""" + src("src/pathrl/analysis/crossover.py", "crossover_horizon") + box("example", "Worked example", f"""
<p>If training costs 500 s, planning costs 3 ms per episode and inference 1 ms, then
H* = 500 / (0.003 - 0.001) = {st['h_example']:,.0f} episodes. If inference cost 3 ms or more, H* would be "never",
no matter how cheap training was.</p>""") + """
<p>In the experiments, inference was the dearer side at N = 10 (≈0.1 ms per decision vs 0.003-0.083 ms for the planners),
so the gap c_plan - c_infer was negative and H* = never in every condition. The measured RL agents also needed more
steps per episode (longer paths), widening the per-episode gap further.</p>
""" + src("src/pathrl/analysis/crossover.py", "measure_inference_scaling", short_doc=True) + """
<p>The forward-pass cost does not depend on the weights, so an <i>untrained</i> network of the same architecture measures the
true inference cost at grid sizes where no agent was trained. It says nothing about whether a trained network would succeed
there; the guide and report keep these two things separate.</p>
"""


def chapters() -> list[str]:
    return [ch7(), ch8(), ch9(), ch10(), ch11()]
