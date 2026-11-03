"""DQN (Mnih et al., 2015) and Double DQN (van Hasselt, Guez & Silver, 2016).

Both learn Q(s, a; theta) with a CNN by minimising the Huber loss between Q(s, a) and a target y.
The ONLY difference between them is how y is computed for non-terminal transitions:

    DQN:        y = r + gamma * max_a'  Q_target(s', a')
    Double DQN: y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

In DQN the same (noisy) network both picks and scores the best next action, so noise that
happens to inflate one action's value is selected AND counted. That makes the max biased
upwards ("overestimation"). Double DQN lets the online network pick the action and the
target network score it, which reduces this bias. Whether that improves results here is
measured, not assumed.

Other ingredients:
  * experience replay (replay_buffer.py) for decorrelated mini-batches
  * target network: a copy of the online net, refreshed every `target_update_interval` steps,
    so the target y does not move every gradient step
  * epsilon-greedy exploration with linear decay
  * Huber (smooth L1) loss: quadratic for small errors, linear for large ones, so a few
    -50 collision surprises cannot produce huge gradients
  * Adam optimiser, gradient-norm clipping

Training and acting are separate: `act` runs the network in inference mode only.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from pathrl.algorithms.base_policy import LearnedPolicy, TrainingLog, linear_schedule
from pathrl.algorithms.reinforcement_learning.network import QNetwork
from pathrl.algorithms.reinforcement_learning.replay_buffer import ReplayBuffer
from pathrl.evaluation.scenarios import ScenarioSampler
from pathrl.utils.seeding import set_global_seed

# MPS lacks deterministic versions of a few kernels. That is disclosed in the methodology,
# so the per-call warning is silenced.
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def resolve_device(device: str) -> str:
    """'auto' -> 'mps' (Apple GPU) or 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DQNConfig:
    learning_rate: float = 5e-4
    gamma: float = 0.99
    total_steps: int = 200_000
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5
    max_grad_norm: float = 10.0
    device: str = "auto"

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "DQNConfig":
        return cls(**{**cfg["rl"]["dqn"], "gamma": cfg["rl"]["gamma"], **overrides})


class DQN(LearnedPolicy):
    name = "dqn"
    double = False

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int = 4,
                 config: DQNConfig | None = None, seed: int = 0):
        self.config = config or DQNConfig()
        self.obs_shape, self.n_actions, self.seed = tuple(obs_shape), n_actions, seed
        set_global_seed(seed)  # network initialisation is part of the run's seed
        self.device = torch.device(resolve_device(self.config.device))
        self.online = QNetwork(obs_shape[-1], n_actions, obs_shape[0]).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- acting

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(x)[0].cpu().numpy()

    @torch.no_grad()
    def act(self, obs, info=None) -> int:
        self.online.eval()
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    # -------------------------------------------------------------- learning

    @torch.no_grad()
    def compute_targets(self, rewards, next_obs, terminated) -> torch.Tensor:
        """TD targets y for a batch. This is the only place DQN and Double DQN differ."""
        if self.double:
            best = self.online(next_obs).argmax(dim=1, keepdim=True)     # online network selects
            next_q = self.target(next_obs).gather(1, best).squeeze(1)   # target network evaluates
        else:
            next_q = self.target(next_obs).max(dim=1).values            # target selects and evaluates
        return rewards + self.config.gamma * (1.0 - terminated) * next_q

    def learn_step(self, buffer: ReplayBuffer) -> float:
        obs, actions, rewards, next_obs, terminated = (
            torch.as_tensor(x, device=self.device) for x in buffer.sample(self.config.batch_size))
        self.online.train()
        targets = self.compute_targets(rewards, next_obs, terminated)
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)  # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def train(self, sampler: ScenarioSampler, seed: int, eval_interval: int = 100,
              evaluator: Callable[["DQN"], dict] | None = None,
              progress: Callable[[float], None] | None = None) -> TrainingLog:
        """Train for config.total_steps environment steps.

        `evaluator(policy) -> dict` is called every `eval_interval` episodes for convergence
        curves. Its wall time is recorded separately and excluded from train_time_s.
        """
        cfg, log = self.config, TrainingLog()
        buffer = ReplayBuffer(cfg.buffer_size, self.obs_shape, seed=seed)
        # Separate streams: exploration/scenario choice vs. per-episode environment seeds
        rng_ss, episode_ss = np.random.SeedSequence(seed).spawn(2)
        rng, episode_rng = np.random.default_rng(rng_ss), np.random.default_rng(episode_ss)
        decay_steps = cfg.epsilon_decay_fraction * cfg.total_steps

        def new_episode():
            env, options = sampler.sample(rng)
            obs, info = env.reset(seed=int(episode_rng.integers(2**31)), options=options)
            return env, obs, info

        episode, losses = 1, []
        env, obs, info = new_episode()
        ep_return, t_start = 0.0, time.perf_counter()
        for step in range(1, cfg.total_steps + 1):
            epsilon = linear_schedule(cfg.epsilon_start, cfg.epsilon_end, decay_steps, step - 1)
            action = int(rng.integers(self.n_actions)) if rng.random() < epsilon else self.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)  # truncation is not terminal
            obs, ep_return = next_obs, ep_return + reward

            if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
                losses.append(self.learn_step(buffer))
                log.gradient_steps += 1
            if step % cfg.target_update_interval == 0:
                self.sync_target()

            if terminated or truncated:
                log.episodes.append({"episode": episode, "env_steps": step, "return": ep_return,
                                     "success": info["outcome"] == "reached_goal", "steps": env.steps,
                                     "epsilon": epsilon, "loss": float(np.mean(losses)) if losses else np.nan})
                losses = []
                if evaluator is not None and episode % eval_interval == 0:
                    log.train_time_s += time.perf_counter() - t_start
                    t_eval = time.perf_counter()
                    ev = evaluator(self)
                    log.eval_time_s += time.perf_counter() - t_eval
                    log.evaluations.append({"episode": episode, "env_steps": step,
                                            "train_time_s": log.train_time_s, **ev})
                    t_start = time.perf_counter()
                episode += 1
                env, obs, info = new_episode()
                ep_return = 0.0
            if progress is not None and step % 1000 == 0:
                progress(step / cfg.total_steps)

        log.train_time_s += time.perf_counter() - t_start
        log.env_steps = cfg.total_steps
        if evaluator is not None and (not log.evaluations or log.evaluations[-1]["episode"] != episode - 1):
            t_eval = time.perf_counter()
            ev = evaluator(self)
            log.eval_time_s += time.perf_counter() - t_eval
            log.evaluations.append({"episode": episode - 1, "env_steps": cfg.total_steps,
                                    "train_time_s": log.train_time_s, **ev})
        return log

    # ------------------------------------------------------------ checkpoints

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": self.name, "state_dict": self.online.state_dict(),
                    "obs_shape": self.obs_shape, "n_actions": self.n_actions,
                    "config": asdict(self.config), "seed": self.seed}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQN":
        """Load a checkpoint. Default device is CPU: inference benchmarks run on the same CPU as the planners."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        algo = DoubleDQN if ckpt["algorithm"] == DoubleDQN.name else DQN
        agent = algo(ckpt["obs_shape"], ckpt["n_actions"], DQNConfig(**{**ckpt["config"], "device": device}),
                     seed=ckpt["seed"])
        agent.online.load_state_dict(ckpt["state_dict"])
        agent.sync_target()
        return agent


class DoubleDQN(DQN):
    name = "double_dqn"
    double = True
