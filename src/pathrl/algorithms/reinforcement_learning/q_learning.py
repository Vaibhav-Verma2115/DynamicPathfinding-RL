"""Tabular Q-Learning (Watkins & Dayan, 1992).

After each transition (s, a, r, s'):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') * (1 - terminated) - Q(s,a) ]
                               \\_______________________ TD target ________________/

* Model-free: it never looks at P or R, only at sampled transitions.
* Off-policy: the target uses max_a' (the greedy policy) while behaviour is epsilon-greedy.
* Bootstrapping on truncation: if an episode is cut off by the step limit, s' is NOT
  terminal, so the target still includes gamma * max Q(s'). Only reaching the goal zeroes it.

State = agent cell (an N*N x A table). That is a complete state only when the map is
fixed, i.e. static obstacles on a single map. Under moving obstacles it is partially
observable, a stated limitation of tabular Q-Learning in this project.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pathrl.algorithms.base_policy import LearnedPolicy, TrainingLog, linear_schedule
from pathrl.environment import GridWorldEnv
from pathrl.evaluation.rollout import evaluate_policy


class QLearning(LearnedPolicy):
    name = "q_learning"

    def __init__(self, grid_size: int, n_actions: int = 4, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay_fraction: float = 0.8,
                 seed: int = 0):
        self.grid_size, self.n_actions = grid_size, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_fraction = epsilon_start, epsilon_end, epsilon_decay_fraction
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((grid_size * grid_size, n_actions))

    def _state(self, cell) -> int:
        return cell[0] * self.grid_size + cell[1]

    def _greedy(self, s: int) -> int:
        # Break ties randomly while learning: with all-zero initial Q, a fixed tie-break
        # would always pick action 0 and bias early exploration.
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def act(self, obs, info) -> int:
        return int(np.argmax(self.Q[self._state(info["agent_pos"])]))  # deterministic greedy

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> float:
        """One Q-Learning update. Returns the TD error."""
        target = r + (0.0 if terminated else self.gamma * self.Q[s_next].max())
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return td_error

    def train(self, env: GridWorldEnv, episodes: int, seed: int, eval_interval: int = 100,
              eval_episodes: int = 20, progress=None) -> TrainingLog:
        log = TrainingLog()
        env.reset(seed=seed)  # seed the env streams once; later resets continue them
        decay_episodes = self.epsilon_decay_fraction * episodes
        for ep in range(1, episodes + 1):
            t0 = time.perf_counter()
            epsilon = linear_schedule(self.epsilon_start, self.epsilon_end, decay_episodes, ep - 1)
            _, info = env.reset()
            s = self._state(info["agent_pos"])
            ep_return, terminated, truncated = 0.0, False, False
            while not (terminated or truncated):
                a = int(self.rng.integers(self.n_actions)) if self.rng.random() < epsilon else self._greedy(s)
                _, r, terminated, truncated, info = env.step(a)
                s_next = self._state(info["agent_pos"])
                self.update(s, a, r, s_next, terminated)
                s, ep_return = s_next, ep_return + r
            log.train_time_s += time.perf_counter() - t0
            log.env_steps += env.steps
            log.episodes.append({"episode": ep, "env_steps": log.env_steps, "return": ep_return,
                                 "success": bool(terminated and info["outcome"] == "reached_goal"),
                                 "steps": env.steps, "epsilon": epsilon})
            if ep % eval_interval == 0 or ep == episodes:
                t1 = time.perf_counter()
                ev = evaluate_policy(self, env, eval_episodes, seed=seed + ep)
                log.eval_time_s += time.perf_counter() - t1
                log.evaluations.append({"episode": ep, "env_steps": log.env_steps,
                                        "train_time_s": log.train_time_s, **ev})
                env.reset(seed=seed + 10_000 + ep)  # evaluation consumed env randomness; re-seed training stream
            if progress:
                progress(ep / episodes)
        return log

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Q=self.Q, grid_size=self.grid_size, alpha=self.alpha, gamma=self.gamma)

    @classmethod
    def load(cls, path) -> "QLearning":
        data = np.load(path)
        agent = cls(int(data["grid_size"]), data["Q"].shape[1], float(data["alpha"]), float(data["gamma"]))
        agent.Q = data["Q"]
        return agent
