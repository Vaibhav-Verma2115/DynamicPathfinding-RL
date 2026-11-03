"""Value Iteration: model-based dynamic programming over the known MDP.

Bellman optimality update, applied to every state at once ("synchronous sweep"):

    V_{k+1}(s) = max_a  sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') * (1 - done) ]

P and R come from the environment's own transition functions (`action_outcomes`,
`attempt_move`, `compute_reward`). So VI knows exactly the dynamics the environment
executes, including slip. This full-model access is an information assumption that
must be stated whenever VI is compared with model-free learners.

Convergence: the update is a gamma-contraction in the max-norm, so the Bellman
residual  max_s |V_{k+1}(s) - V_k(s)|  shrinks by at least a factor gamma per sweep.
We stop when it falls below theta. This is VI's convergence criterion. It is NOT the
episode-based criterion used for RL, because VI never runs episodes.

State = agent cell on a FIXED occupancy grid. With moving obstacles the true state also
includes obstacle positions, so VI here is exact only for static maps.

Only cells connected to the goal are updated. Cells walled off from the goal can never
succeed: their values only decay towards -1/(1-gamma) by a factor gamma per sweep, so
including them would make "sweeps to convergence" measure those dead pockets instead of
the navigation problem. Their V is reported as 0 and they are excluded from the residual.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from collections import deque

from pathrl.algorithms.base_policy import LearnedPolicy
from pathrl.environment.config import EnvConfig
from pathrl.environment.dynamics import action_deltas, action_outcomes, attempt_move, move_cost
from pathrl.environment.rewards import Outcome, compute_reward


class ValueIteration(LearnedPolicy):
    name = "value_iteration"

    def __init__(self, env_cfg: EnvConfig, gamma: float = 0.99, theta: float = 1e-6, max_sweeps: int = 100_000):
        self.cfg = env_cfg
        self.gamma, self.theta, self.max_sweeps = gamma, theta, max_sweeps
        self.V: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.residuals: list[float] = []
        self.sweeps = 0
        self.converged = False
        self.solve_time_s = 0.0

    def _build_model(self, occupancy: np.ndarray, goal):
        """Arrays of shape (S, A, K): next-state index, probability, reward, done flag."""
        n = occupancy.shape[0]
        deltas = action_deltas(self.cfg.connectivity)
        n_states, n_actions, k = n * n, len(deltas), 3
        nxt = np.zeros((n_states, n_actions, k), dtype=np.int64)
        prob = np.zeros((n_states, n_actions, k))
        rew = np.zeros((n_states, n_actions, k))
        done = np.zeros((n_states, n_actions, k))
        for r in range(n):
            for c in range(n):
                s = r * n + c
                if occupancy[r, c] or (r, c) == goal:
                    nxt[s] = s  # walls are unreachable, goal is absorbing: both keep V = 0
                    continue
                for a in range(n_actions):
                    for j, (p, delta) in enumerate(action_outcomes(deltas, a, self.cfg.slip_prob)):
                        cell, collided = attempt_move((r, c), delta, occupancy, self.cfg.allow_corner_cutting)
                        outcome = (Outcome.REACHED_GOAL if cell == goal
                                   else Outcome.COLLIDED if collided else Outcome.MOVED)
                        nxt[s, a, j] = cell[0] * n + cell[1]
                        prob[s, a, j] = p
                        rew[s, a, j] = compute_reward(outcome, self.cfg.rewards, move_cost(delta, self.cfg.diagonal_cost))
                        done[s, a, j] = outcome is Outcome.REACHED_GOAL or (collided and self.cfg.terminate_on_collision)
        connected = self._connected_to(goal, occupancy, deltas)
        active = np.array([(s // n, s % n) in connected and (s // n, s % n) != goal for s in range(n_states)])
        return nxt, prob, rew, done, active

    def _connected_to(self, goal, occupancy, deltas) -> set:
        """Cells with a legal move sequence to the goal (grid moves are symmetric, so BFS from the goal)."""
        seen, queue = {goal}, deque([goal])
        while queue:
            cell = queue.popleft()
            for delta in deltas:
                nxt, collided = attempt_move(cell, delta, occupancy, self.cfg.allow_corner_cutting)
                if not collided and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def solve(self, occupancy: np.ndarray, goal) -> "ValueIteration":
        t0 = time.perf_counter()
        self.n = occupancy.shape[0]
        nxt, prob, rew, done, active = self._build_model(occupancy, goal)
        V = np.zeros(self.n * self.n)
        self.residuals, self.converged = [], False
        for sweep in range(1, self.max_sweeps + 1):
            Q = (prob * (rew + self.gamma * (1.0 - done) * V[nxt])).sum(axis=2)
            V_new = np.where(active, Q.max(axis=1), 0.0)
            residual = float(np.abs(V_new - V).max())
            V = V_new
            self.residuals.append(residual)
            if residual < self.theta:
                self.converged = True
                break
        self.sweeps = sweep
        self.V, self.Q = V, Q
        self.solve_time_s = time.perf_counter() - t0
        return self

    def value(self, cell) -> float:
        return float(self.V[cell[0] * self.n + cell[1]])

    def act(self, obs, info) -> int:
        r, c = info["agent_pos"]
        return int(np.argmax(self.Q[r * self.n + c]))

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, V=self.V, Q=self.Q, residuals=np.array(self.residuals))
