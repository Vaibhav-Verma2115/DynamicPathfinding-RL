# Classical Pathfinding vs. Model-Free Deep RL in Dynamic Grid Worlds

**Comparative performance analysis: computation, success, and the crossover horizon**

Every number in this report is copied from [RESULTS.md](RESULTS.md), the generated summary
tables in `outputs/results/summary/`, or the raw CSVs in `outputs/results/` (each saved with its
resolved config, a hardware snapshot and the git commit). Claims marked **(hypothesis)** were
*not* tested by an experiment in this repository. Hardware: Apple-silicon Mac (macOS 26.4.1,
arm64, 18 CPU cores), Python 3.11.16, PyTorch 2.15.0. DQN training used the MPS GPU. All planning
and inference costs were measured on the CPU.

---

## Abstract

We asked when the cumulative cost of repeated classical replanning exceeds the cost of training a
reinforcement-learning (RL) policy once and then running it, i.e. the crossover horizon
H* = C_train / (c_plan − c_infer). On 4-connected grid worlds with static, randomly moving and
stochastic (slip) dynamics, we compared BFS, Dijkstra, A*, D* Lite and Value Iteration with
tabular Q-Learning, DQN and Double DQN.

**Within the tested range (generalist RL policies at 10 × 10), no crossover exists.** One CNN
forward pass (≈0.1 ms) costs more than an A* replan per decision (0.003–0.010 ms) and slightly
more than a D* Lite repair (0.083 ms). The learned policies also take 2.5–6.7× longer paths, so
H* = "never" in all 30 (change rate × planner × RL method) conditions, even before counting the
531–1300 s of training. The planners, which see the whole map, succeeded in 100% of episodes.
The generalist policies, which see a 7 × 7 local window, succeeded in 69–93% under moving
obstacles and 35–41% on unseen static maps.

The measurements do show where the cost comparison could turn. D* Lite's per-decision cost grows
faster with grid size than a CNN forward pass, and exceeds it from N = 20. However, no policy was
trained at those sizes, so whether a policy could *succeed* there is untested. The convergence
study (E6) shows tabular Q-Learning's learning-rate sensitivity sharpening with grid size: at
N = 100 no learning rate converged within 3000 episodes. DQN converged on a fixed 10 × 10 map but
not on a fixed 20 × 20 map within 80k steps.

## 1. Introduction

Classical graph search finds optimal paths when the map is known, but it must do some work every
time the world changes. Model-free RL moves that work to training, after which a decision is one
network evaluation. That raises an amortisation question, which this project makes measurable:

```
C_RL(H)       = C_train + H · c_infer
C_planning(H) = H · c_plan        (c_plan includes the initial plan and every repair)
H*            = C_train / (c_plan − c_infer)   if c_plan > c_infer, otherwise "never"
```

H* is a statement about *computation only*. A cheaper method that fails more often is not better,
so success rate and path quality are always reported next to it.

**Fairness and information access.** The two families do not receive the same information, and
every comparison should be read with this in mind:

| Method | Information at decision time |
|---|---|
| BFS, Dijkstra, A*, D* Lite | full current occupancy grid (global, exact) |
| Value Iteration | full grid **and** the slip model (exact MDP) |
| Q-Learning (E4, E6) | trained on the *same* map it is evaluated on (per-map, not zero-shot) |
| DQN / Double DQN generalists (E3–E5) | 7 × 7 egocentric window + goal direction, trained on 1000 other maps |

The local view was chosen by model selection on validation maps, not by the test results
(methodology §4). A global-view CNN reached only 0.00–0.04 validation success in pilot runs.

## 2. Methods (summary; full specification in [methodology.md](methodology.md))

- **Environment.** N × N grid, 4-connected, unit step cost. Rewards: −1 per step, −50 for a
  collision (the agent stays in place), +100 at the goal. Episodes are truncated after N² steps
  (2N² in E4). Dynamic obstacles take random walks every k steps (E3: k ∈ {20, 10, 5, 2, 1},
  density 0.05). Slip moves the agent perpendicular to the intended direction with probability p.
- **Planners.** A* (Manhattan, tie-break on smaller h) and D* Lite (Koenig & Likhachev, 2002;
  keys rounded to 9 decimals). Under dynamics, A* either *replans on every change* or *replans
  only when the next cell is blocked*. D* Lite repairs incrementally.
- **Learning.** Value Iteration with residual θ = 1e-6. Tabular Q-Learning. DQN / Double DQN with
  a CNN of 1.7 M parameters at N = 10, a 100k uint8 replay buffer, Huber loss, Adam and hard target
  updates, reward scale 0.01.
- **Generalists** (shared by E3–E5): 3 seeds × {DQN, Double DQN}, 500k environment steps each,
  sampled from 1000 training maps × {static, random walk (k ∈ {1, 2, 5, 10, 20})} × slip
  {0, 0.1, 0.2}.
- **Convergence (RL):** 3 consecutive greedy evaluations, each with success ≥ 0.9 and spread of
  mean return ≤ 5% of its magnitude. Non-convergent runs are kept and counted.
- **Statistics:** mean and Student-t 95% CI over independent maps or seeds. Train, validation and
  test maps come from disjoint seed families (train/test overlap measured as 0).

## 3. Results (static and stochastic)

### 3.1 E1 — static scaling

All four planners returned optimal paths on every map (cost / optimal = 1.0, n = 30 per cell).
At N = 100:

| density | A* | BFS | Dijkstra | D* Lite (initial) |
|---|---|---|---|---|
| 0.15 time (ms) | 1.09 | 7.69 | 9.93 | 65.6 |
| 0.15 expansions | 710 | 8,494 | 8,494 | 6,622 |
| 0.30 time (ms) | 2.27 | 6.04 | 7.75 | 17.5 |
| 0.30 expansions | 1,550 | 6,835 | 6,835 | 2,027 |

A* was the fastest planner at every size and density. D* Lite's *first* plan cost 7.7–60× more wall
time than A* (7.7× at density 0.30, N = 100; 60× at density 0.15, N = 100). On open maps it expanded
far more cells, which matches the measured property that its initial backward search expands every
cell with f ≤ C* ([complexity_analysis.md](complexity_analysis.md)). Denser maps prune that set,
which is why the gap shrinks at density 0.30. BFS and Dijkstra expanded identical sets
(4-connected, unit costs).

### 3.2 E2 — heuristic quality

Inflating the heuristic to 1.5 × Manhattan reduced A*'s expansions at N = 100 from 710 to 282
(density 0.15) and from 1,550 to 444 (density 0.30). It also raised path cost to 1.083× and
1.066× the optimum, and the fraction of optimal paths fell from 0.90–0.93 at N = 10 to 0.00 at
N = 100. With h = 0, A* expands exactly Dijkstra's set. The inflated heuristic stayed within its
theoretical bound (≤ 1.5 × C*) on every map: the worst single path was 1.204 × optimal.

### 3.3 E4 — stochastic transitions

| N | slip | A* (replan) return | Value Iteration return | A* collisions | VI collisions | DQN / DDQN success |
|---|---|---|---|---|---|---|
| 10 | 0.0 | 92.0 | 92.0 | 0 | 0 | 0.344 / 0.433 |
| 10 | 0.2 | 56.0 | 73.7 | 0.672 | 0.275 | 0.813 / 0.873 |
| 20 | 0.2 | 13.5 | 55.6 | 1.25 | 0.332 | not run |

- Every planner and Value Iteration reached the goal in all episodes, because each replans after
  every slip. Value Iteration, which knows the slip model, earned higher returns under slip by
  choosing routes with fewer wall contacts.
- Tabular Q-Learning, trained on the evaluation map, came close to VI at N = 10. At N = 20 with
  slip 0.1 its return was 27.0 [−36.9, 90.9]: on one of its 30 maps all 20 episodes failed,
  while the other 29 maps succeeded every time.
- **The generalists succeeded *more* often with slip (0.34 → 0.81 for DQN) than without it.**
  This is the opposite of the classical trend. The same pattern appears in E3 (§4.2).

## 4. Results (dynamic obstacles and crossover)

### 4.1 E3 — classical repair cost

Replanning cost grows with change rate and with N. At N = 50 with obstacles moving every step:

| method | ms / episode | expansions / episode | ms / decision | success | path / static optimum |
|---|---|---|---|---|---|
| A* (replan when blocked) | 0.532 | 366 | 0.0099 | 1.0 | 1.15 |
| A* (replan on change) | 3.78 | 2,695 | 0.072 | 1.0 | 1.12 |
| D* Lite | 81.0 | 1,595 | 1.61 | 1.0 | 1.20 |

**D* Lite did 41% fewer expansions than A* replanning on every change, yet took 21× more wall
time.** In this implementation, the cost of D* Lite is dominated by per-operation overhead (g/rhs
bookkeeping, key recomputation, priority-queue updates), not by search effort. D* Lite was slower
than both A* variants in all 20 classical E3 conditions. In expansions, it beat A*-on-change in 12 of
20 conditions: k ≤ 2 at N = 10 and 20, k ≤ 5 at N = 50, and every k at N = 30. In the other 8,
where changes were rarer, its larger initial search (§3.1) made it expand more. At high change rates, expansion counts and wall
time rank the two algorithms in *opposite* orders, which is why both are reported. Whether a compiled implementation would close the gap
**(hypothesis)** is not tested here.

### 4.2 E3 — generalist policies under moving obstacles (N = 10)

| change every k steps | 20 | 10 | 5 | 2 | 1 |
|---|---|---|---|---|---|
| DQN success | 0.689 | 0.844 | 0.867 | 0.922 | 0.922 |
| Double DQN success | 0.733 | 0.800 | 0.856 | 0.911 | 0.933 |
| DQN path / static optimum | 6.66 | 4.96 | 3.82 | 3.01 | 2.91 |
| planners' success | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

n = 90 episodes (30 maps × 3 training seeds). **Every** failed RL episode ended by truncation at
the 100-step limit. None ended in a terminal collision, since collisions are non-terminal here.
Success *rises* as the world changes faster. Read with E4 (slip helps) and E5 (static maps are
worst), the consistent picture is that the deterministic greedy policies get stuck in loops.
A memoryless policy with a 7 × 7 view cannot tell apart cells whose neighbourhoods look the same
(perceptual aliasing). Any perturbation, whether a moving obstacle or a slip, can break the loop.
**(hypothesis)** We did not measure loop frequency directly. A test would count revisited states
in failed episodes, or add a recurrent or memory-augmented policy.

### 4.3 E3 — crossover

| quantity (N = 10) | value |
|---|---|
| C_train, Double DQN (uncontended) | 531–532 s per seed |
| C_train, DQN (trained under CPU contention, see methodology §5) | 982–1300 s per seed |
| c_infer per decision (CNN forward pass, CPU) | ≈0.099 ms |
| c_plan per decision: A* blocked / A* on change / D* Lite (k = 1) | 0.0028 / 0.0097 / 0.083 ms |
| c_infer per episode | 2.0–5.3 ms |
| c_plan per episode | 0.022–0.76 ms |
| **H\*** | **never**, in all 30 conditions |

At this scale the network is more expensive per decision than every planner. On top of that, the
learned policy takes more decisions per episode (longer paths), so the per-episode gap is larger
still. Training cost never comes into play. The cumulative-cost figure plots both cost curves up
to H = 10⁷ episodes: the RL line stays above every planner line throughout and runs parallel to
them on the log–log plot, so they never meet.

**Where the comparison could change (cost only).** Forward-pass times of the same CNN
architecture (untrained, measured on CPU) at larger N, compared with measured planner costs per
decision at k = 1:

| N | CNN forward pass | D* Lite repair | A* replan on change |
|---|---|---|---|
| 10 | 0.127 ms | 0.083 ms | 0.0097 ms |
| 20 | 0.091 ms | 0.269 ms | 0.020 ms |
| 30 | 0.152 ms | 0.574 ms | 0.037 ms |
| 50 | 0.347 ms | 1.61 ms | 0.072 ms |

For N ≥ 20, a forward pass is cheaper per decision than a D* Lite repair in this implementation.
It is never cheaper than A* (either replanning policy) at any measured size. A crossover against
D* Lite is therefore arithmetically *possible* at N ≥ 20. It would require a policy that succeeds
at those sizes with paths not much longer than optimal. No such policy was trained, and E6 shows
fixed-map DQN failing to converge at N = 20. This is **pending**, not a finding.
(The N = 20 network has fewer parameters than the N = 10 one because stride-2 convolutions start
above 12 × 12 inputs.)

### 4.4 E5 — zero-shot generalisation (N = 10, static, 500 unseen maps)

| method | seen training maps | unseen test maps | cost / optimal on successes |
|---|---|---|---|
| DQN | 0.451 [0.318, 0.583] | 0.354 [0.298, 0.410] | 1.11 |
| Double DQN | 0.532 [0.485, 0.579] | 0.405 [0.350, 0.459] | 1.08 |
| A* | — | 1.000 | 1.00 |

(n = 3 training seeds for RL.) The generalisation gap is about 0.10–0.13. Even on the maps it was
trained on, no generalist exceeded 0.6 success. When the agents do succeed, their paths are close
to optimal (≤ 11% longer), so failures are loops rather than long detours. None of the six
generalists met the convergence criterion during training (final validation success 0.34–0.46).
Double DQN scored higher than DQN in every E5 cell, but the confidence intervals overlap and
n = 3, so we do not claim a difference.

## 5. Results — E6 convergence and learning-rate sensitivity

**Value Iteration.** Every one of the 360 runs converged (residual < 1e-6). Deterministic sweeps
grow roughly linearly with N (19.1 at N = 10 to 199 at N = 100, density 0.15). With slip on
denser maps, sweeps jumped to ~1,300–1,420 (e.g. density 0.30, slip 0.1, N ≥ 50). The bound
ln(Δ₀/θ)/ln(1/γ) gives ≈1,375 sweeps for γ = 0.99, θ = 1e-6 and an initial gap Δ₀ ≈ 1. This
suggests convergence there is limited by the γ-contraction rate (cyclic policies under slip) rather
than by path length **(hypothesis)**.

**Q-Learning** (fixed map per seed, 3000 episodes, 5 seeds; fraction of runs converged):

| α \ N | 10 | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|
| 0.01 | 1.0 | 0 | 0 | 0 | 0 | 0 |
| 0.05 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
| 0.1 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 |
| 0.3 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |

Larger learning rates converged faster and on larger grids. The usable range shrinks with grid
size. Episodes to convergence at α = 0.5 grew from 160 (N = 10) to 1,815 (N = 70), and **no
setting converged at N = 100** within the budget. Convergence was all-or-nothing per cell: each
learning rate converged for all five seeds or for none.

**DQN / Double DQN** (fixed map, 3 seeds; 40k steps at N = 10, 80k at N = 20):

- N = 10: DQN converged in 12/12 runs across all four learning rates. Double DQN converged in 7/12
  and never at lr = 5e-3.
- N = 20: DQN converged in 0/12 runs. Double DQN converged in 5/12, but its *final* success was
  ≤ 0.33 for every learning rate.

A run can pass the criterion and later degrade. For example, Double DQN at N = 10 with lr = 1e-4
converged in 3/3 runs, yet its mean final success was 0.67. Convergence episode and final
performance are therefore reported separately. E6 DQN wall-clock times (410–1050 s per run) include
12-worker CPU contention and are comparable only within E6.

## 6. Discussion

1. **On small grids, classical search wins on every axis we measured:** cost per decision, success,
   path quality and zero setup cost. The amortisation argument for RL needs c_infer < c_plan, and
   at N = 10 a CNN forward pass is 10–35× slower than A*.
2. **Incremental repair is not automatically cheaper.** D* Lite's fewer expansions did not
   translate into lower wall time, either in Python or for its initial plan. A* "replan when
   blocked" was the cheapest strategy in every dynamic condition and still kept 100% success, at
   the cost of slightly longer paths at high change rates (1.20 vs 1.09 at N = 30, k = 1).
3. **The deciding factor for RL here was capability, not cost.** Even with unlimited deployment
   episodes, a local-view memoryless policy at 35–53% success on static maps is not a substitute
   for a planner. The measured success gains under perturbation point to memory, or a larger
   receptive field, as the next thing to try. That is a research direction, not a result.
4. **Value Iteration is the strongest method under slip** when the model is known: it had the
   highest return in every slip condition, and fewer collisions than either planner. Per-map
   Q-Learning had slightly fewer collisions in two conditions but lower returns. The model-free methods are only
   relevant when that model is unavailable.

## 7. Limitations

- **One grid size for trained generalists (N = 10).** Every RL-vs-planner comparison is at N = 10.
  DQN results at N ≥ 30 (E3, E5, E6) are **pending**, because of the compute budget.
- **Python implementations.** Wall-time rankings reflect this codebase. Expansion counts are
  reported as a machine-independent complement, and they disagree with the wall-time ranking for
  D* Lite.
- **Training-time confound.** DQN generalists trained under CPU contention and Double DQN
  generalists trained alone (methodology §5). This affects C_train but not H*, which is "never"
  for both.
- **Small n for RL seeds** (3 per algorithm). CIs with n = 3 are wide, and we do not claim
  DQN-vs-Double-DQN differences.
- **Single hardware platform** (Apple-silicon CPU for timing, MPS for training). Absolute times
  will differ elsewhere; the ratios are more transferable but were not tested on other hardware.
- **Information asymmetry** (§1) favours the planners. That is by design, since it reflects
  realistic deployment, but it means E3–E5 are not a like-for-like algorithm comparison.
- **Provenance.** The `.meta.json` files record `git_commit`. E1 and E2 were produced from commit
  `f64899b` with an uncommitted working tree (`-dirty`); their code paths are unchanged in later
  commits.
- **Hypotheses not tested:** perceptual-aliasing loops as the cause of RL failures (§4.2),
  γ-limited VI convergence under slip (§5), and a compiled D* Lite closing the wall-time gap (§4.1).

## 8. Future work

- Train generalists at N ∈ {20, 30, 50} to test the D* Lite cost crossover that §4.3 shows is
  arithmetically possible.
- A recurrent (LSTM/GRU) or frame-stacked policy, to test the looping hypothesis directly. Also log
  state-revisit counts in failed episodes.
- A compiled (C++/Numba) D* Lite and A*, to separate algorithmic from interpreter effects.
- Batched or GPU inference over many agents, which lowers the amortised cost per decision of the
  CNN. A* has no comparable batching gain.
- 8-connected and weighted-cost grids, where BFS is no longer cost-optimal (verified in the tests:
  3 of 40 random maps).

## References

- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2). https://doi.org/10.1109/TSSC.1968.300136
- Koenig, S., & Likhachev, M. (2002). D* Lite. *AAAI-02*. https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf
- Koenig, S., Likhachev, M., & Furcy, D. (2004). Lifelong Planning A*. *Artificial Intelligence*, 155. https://doi.org/10.1016/j.artint.2003.12.001
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8. https://doi.org/10.1007/BF00992698
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518. https://doi.org/10.1038/nature14236
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI-16*. https://arxiv.org/abs/1509.06461
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). http://incompleteideas.net/book/the-book-2nd.html
