"""Train the generalist DQN / Double DQN agents (shared by E3, E4, E5).

Training data: `n_train_maps` maps generated from a dedicated seed family, combined with a
mixture of obstacle dynamics, change intervals and slip probabilities. Validation maps
come from another seed family and are used only for convergence curves. Test maps for
E3-E5 come from yet other families, so they are never seen during training.

Outputs (outputs/models/ and outputs/results/train_dqn/):
  generalist_N{N}_{algo}_seed{i}.pt            checkpoint
  train_dqn_episodes.csv / _evaluations.csv    learning curves
  train_dqn_summary.csv                        training cost per run (wall time, env/gradient steps, device)
"""

from __future__ import annotations

import pandas as pd
import torch

from pathrl.algorithms.reinforcement_learning import DQN, DQNConfig, DoubleDQN
from pathrl.algorithms.reinforcement_learning.dqn import resolve_device
from pathrl.analysis.convergence import ConvergenceCriterion, rl_convergence
from pathrl.environment import GridWorldEnv
from pathrl.evaluation.results import save_results
from pathrl.evaluation.rollout import evaluate_policy
from pathrl.evaluation.scenarios import ScenarioSampler, make_map_set
from pathrl.experiments.common import env_config, models_dir

ALGORITHMS = {"dqn": DQN, "double_dqn": DoubleDQN}


def training_envs(cfg: dict) -> list[GridWorldEnv]:
    t = cfg["train_dqn"]
    envs = []
    for dynamics in t["dynamics"]:
        intervals = t["change_intervals"] if dynamics != "static" else [0]
        for interval in intervals:
            for slip in t["slip_probs"]:
                ec = env_config(cfg, dynamics=dynamics, change_interval=interval, slip_prob=slip)
                envs.append(GridWorldEnv(ec, map_seed=0))
    return envs


def train_maps(cfg: dict):
    t = cfg["train_dqn"]
    return make_map_set(env_config(cfg, dynamics="random_walk", change_interval=1),
                        t["n_train_maps"], seed=cfg["seed"] + t["train_map_offset"])


def checkpoint_path(cfg: dict, algo: str, index: int):
    e = cfg["env"]
    return models_dir(cfg) / f"generalist_N{e['grid_size']}_{e['observation']}_{algo}_seed{index}.pt"


def run(cfg: dict, workers: int = 1) -> pd.DataFrame:
    t = cfg["train_dqn"]
    maps = train_maps(cfg)
    val_cfg = env_config(cfg, dynamics="static")
    val_maps = make_map_set(val_cfg, t["n_validation_maps"], seed=cfg["seed"] + t["validation_map_offset"])
    val_env = GridWorldEnv(val_cfg, grid_map=val_maps[0])
    sampler = ScenarioSampler(training_envs(cfg), maps)
    criterion = ConvergenceCriterion.from_config(cfg)
    n = cfg["env"]["grid_size"]
    obs_shape = val_env.observation_space.shape

    episodes, evaluations, summary = [], [], []
    for algo in t["algorithms"]:
        for index in range(t["training_seeds"]):
            seed = cfg["seed"] + 100 * index
            dqn_cfg = DQNConfig.from_config(cfg)
            agent = ALGORITHMS[algo](obs_shape, 4, dqn_cfg, seed=seed)
            evaluator = lambda p: evaluate_policy(p, val_env, t["validation_eval_episodes"], seed=7, maps=val_maps)
            print(f"Training {algo} seed {index} on {resolve_device(dqn_cfg.device)} for {dqn_cfg.total_steps} steps")
            log = agent.train(sampler, seed=seed, eval_interval=cfg["rl"]["evaluation"]["eval_interval"],
                              evaluator=evaluator)
            agent.save(checkpoint_path(cfg, algo, index))
            ep, ev = log.episodes_df(), log.evaluations_df()
            for df in (ep, ev):
                df.insert(0, "algorithm", algo)
                df.insert(1, "train_seed", index)
            episodes.append(ep)
            evaluations.append(ev)
            conv = rl_convergence(ev, criterion)
            summary.append({"algorithm": algo, "train_seed": index, "grid_size": n,
                            "observation": cfg["env"]["observation"],
                            "device": resolve_device(dqn_cfg.device), "train_time_s": log.train_time_s,
                            "eval_time_s": log.eval_time_s, "env_steps": log.env_steps,
                            "gradient_steps": log.gradient_steps, "episodes": len(ep),
                            "n_train_maps": len(maps), "n_training_conditions": len(sampler.envs),
                            "parameters": sum(p.numel() for p in agent.online.parameters()), **conv})
            print(f"  done: {log.train_time_s:.0f}s, final validation success {conv['final_success_rate']:.2f}")

    save_results(pd.concat(episodes), cfg, "train_dqn", "train_dqn_episodes")
    save_results(pd.concat(evaluations), cfg, "train_dqn", "train_dqn_evaluations")
    out = pd.DataFrame(summary)
    save_results(out, cfg, "train_dqn", "train_dqn_summary",
                 extra={"torch_threads": torch.get_num_threads()})
    return out
