"""E6 - Convergence analysis and hyperparameter (learning-rate) sensitivity.

Value Iteration: Bellman residual vs sweeps, sweeps-to-theta and solve time, across grid size,
density and slip. Reported in sweeps, NOT episodes.
Q-Learning / DQN / Double DQN: greedy evaluation curves during training on one fixed map, and
episodes-to-convergence under the criterion in analysis/convergence.py. Non-convergent runs
are kept (converged = False).

Outputs (outputs/results/e6_convergence/):
  e6_vi_runs.csv, e6_vi_residuals.csv
  e6_rl_runs.csv          one row per (algorithm, grid, lr, seed): convergence + timing summary
  e6_rl_evaluations.csv   greedy evaluation curves (success / return vs episodes)
  e6_rl_training.csv      training curves aggregated per 10 episodes
"""

from __future__ import annotations

import pandas as pd

from pathrl.algorithms.model_based import ValueIteration
from pathrl.algorithms.reinforcement_learning import DQN, DQNConfig, DoubleDQN, QLearning
from pathrl.analysis.convergence import ConvergenceCriterion, bellman_convergence, rl_convergence
from pathrl.environment import GridWorldEnv, generate_map
from pathrl.evaluation.results import save_results
from pathrl.evaluation.rollout import evaluate_policy
from pathrl.evaluation.scenarios import ScenarioSampler
from pathrl.experiments.common import env_config, parallel_map, run_seeds
from pathrl.utils.seeding import make_rng

RL_ALGOS = {"dqn": DQN, "double_dqn": DoubleDQN}


def _training_curve(episodes: pd.DataFrame, block: int = 10) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["block"] = (episodes["episode"] - 1) // block
    out = episodes.groupby("block").agg(episode=("episode", "max"), env_steps=("env_steps", "max"),
                                        mean_return=("return", "mean"), success_rate=("success", "mean"))
    return out.reset_index(drop=True)


def _rl_task(args) -> dict:
    cfg, algo, n, lr, seed_index, map_seed = args
    e = cfg["e6"]
    section = e["q_learning"] if algo == "q_learning" else e["dqn"]
    ec = env_config(cfg, grid_size=n, obstacle_density=section["density"])
    m = generate_map(ec, make_rng(map_seed))
    env, eval_env = GridWorldEnv(ec, grid_map=m), GridWorldEnv(ec, grid_map=m)
    evaluator = lambda p: evaluate_policy(p, eval_env, section["eval_episodes"], seed=1)
    if algo == "q_learning":
        q = cfg["rl"]["q_learning"]
        agent = QLearning(n, alpha=lr, gamma=cfg["rl"]["gamma"], epsilon_start=q["epsilon_start"],
                          epsilon_end=q["epsilon_end"], epsilon_decay_fraction=q["epsilon_decay_fraction"], seed=map_seed)
        log = agent.train(env, section["episodes"], seed=map_seed, eval_interval=section["eval_interval"],
                          eval_episodes=section["eval_episodes"])
        budget = {"episode_budget": section["episodes"]}
    else:
        dcfg = DQNConfig.from_config(cfg, learning_rate=lr, total_steps=int(section["total_steps"][n]),
                                     device=section["device"], learning_starts=section["learning_starts"],
                                     target_update_interval=section["target_update_interval"])
        agent = RL_ALGOS[algo]((3, n, n), 4, dcfg, seed=map_seed)
        log = agent.train(ScenarioSampler([env]), seed=map_seed, eval_interval=section["eval_interval"], evaluator=evaluator)
        budget = {"step_budget": dcfg.total_steps}
    evals, curve = log.evaluations_df(), _training_curve(log.episodes_df())
    key = {"algorithm": algo, "grid_size": n, "learning_rate": lr, "seed_index": seed_index}
    for df in (evals, curve):
        for k, v in reversed(key.items()):
            df.insert(0, k, v)
    conv = rl_convergence(evals, ConvergenceCriterion.from_config(cfg))
    run = {**key, "map_seed": map_seed, "train_time_s": log.train_time_s, "eval_time_s": log.eval_time_s,
           "env_steps": log.env_steps, "episodes": len(log.episodes), "gradient_steps": log.gradient_steps,
           **budget, **conv}
    return {"run": run, "evals": evals, "curve": curve}


def _vi_task(args) -> dict:
    cfg, n, density, slip, seed_index, map_seed, keep_curve = args
    ec = env_config(cfg, grid_size=n, obstacle_density=density, slip_prob=slip)
    m = generate_map(ec, make_rng(map_seed))
    vi = ValueIteration(ec, gamma=cfg["rl"]["gamma"], theta=cfg["rl"]["value_iteration"]["theta"]).solve(m.walls, m.goal)
    key = {"grid_size": n, "density": density, "slip_prob": slip, "seed_index": seed_index}
    run = {**key, "solve_time_s": vi.solve_time_s, "sweeps": vi.sweeps,
           **bellman_convergence(vi.residuals, cfg["rl"]["value_iteration"]["theta"])}
    curve = pd.DataFrame({**key, "sweep": range(1, len(vi.residuals) + 1), "residual": vi.residuals}) if keep_curve else None
    return {"run": run, "curve": curve}


def run(cfg: dict, workers: int = 1, parts=("vi", "q_learning", "dqn")) -> dict:
    e = cfg["e6"]
    out = {}
    if "vi" in parts:
        v = e["value_iteration"]
        seeds = run_seeds(cfg, v["seeds"], offset=e["map_offset"])
        tasks = [(cfg, n, d, slip, i, s, i < v["residual_curve_seeds"]) for n in v["grid_sizes"]
                 for d in v["densities"] for slip in v["slip_probs"] for i, s in enumerate(seeds)]
        res = parallel_map(_vi_task, tasks, 1, desc="E6 value iteration")  # serial: solve times are reported
        out["vi"] = pd.DataFrame([r["run"] for r in res])
        save_results(out["vi"], cfg, "e6_convergence", "e6_vi_runs")
        save_results(pd.concat([r["curve"] for r in res if r["curve"] is not None]), cfg, "e6_convergence", "e6_vi_residuals")

    tasks = []
    if "q_learning" in parts:
        q = e["q_learning"]
        seeds = run_seeds(cfg, q["seeds"], offset=e["map_offset"])
        tasks += [(cfg, "q_learning", n, lr, i, s) for n in q["grid_sizes"] for lr in q["learning_rates"]
                  for i, s in enumerate(seeds)]
    if "dqn" in parts:
        d = e["dqn"]
        seeds = run_seeds(cfg, d["seeds"], offset=e["map_offset"])
        tasks += [(cfg, a, n, lr, i, s) for a in d["algorithms"] for n in d["grid_sizes"]
                  for lr in d["learning_rates"] for i, s in enumerate(seeds)]
    if tasks:
        # Longest runs first so parallel workers finish together
        tasks.sort(key=lambda t: (t[1] == "q_learning", -t[2]))
        res = parallel_map(_rl_task, tasks, workers, desc="E6 RL runs")
        out["rl"] = pd.DataFrame([r["run"] for r in res])
        extra = {"workers": workers, "note": "train times measured with parallel workers, 1 torch thread each"}
        save_results(out["rl"], cfg, "e6_convergence", "e6_rl_runs", extra=extra)
        save_results(pd.concat([r["evals"] for r in res]), cfg, "e6_convergence", "e6_rl_evaluations", extra=extra)
        save_results(pd.concat([r["curve"] for r in res]), cfg, "e6_convergence", "e6_rl_training", extra=extra)
    return out
