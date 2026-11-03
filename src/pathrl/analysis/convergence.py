"""Convergence definitions, one per algorithm family (they are deliberately different).

RL (Q-Learning, DQN, Double DQN): episode-based, judged by GREEDY evaluations
(no exploration) taken every `eval_interval` training episodes.

    A run has converged at evaluation k if evaluations k-W+1 ... k (W consecutive windows) all have
      (1) greedy success rate >= success_threshold, and
      (2) spread of mean return, max - min, <= return_tolerance * max(1, |mean of those returns|).
    Episodes-to-convergence = the training episode of evaluation k, the first point at which
    the criterion is CONFIRMED, i.e. after W stable windows. If no such k exists, the run is
    reported as NON-CONVERGENT (never dropped).

Value Iteration: Bellman residual max|V_{k+1} - V_k| < theta. The result is reported as sweeps,
not episodes.

Classical search (BFS, Dijkstra, A*): terminates exactly. There is nothing to "converge".
D* Lite: after a repair, all relevant states are locally consistent (g = rhs). Its cost is
reported as repair work, not convergence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConvergenceCriterion:
    success_threshold: float = 0.9
    windows: int = 3
    return_tolerance: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> "ConvergenceCriterion":
        return cls(**cfg["rl"]["convergence"])


def rl_convergence(evals: pd.DataFrame, criterion: ConvergenceCriterion) -> dict:
    """Apply the RL criterion to an evaluations table (columns: episode, success_rate, mean_return,
    and optionally env_steps, train_time_s). Returns converged flag plus episode/steps/time at convergence."""
    result = {"converged": False, "episodes_to_convergence": np.nan, "steps_to_convergence": np.nan,
              "time_to_convergence_s": np.nan, "final_success_rate": np.nan, "final_mean_return": np.nan}
    if evals.empty:
        return result
    result["final_success_rate"] = float(evals["success_rate"].iloc[-1])
    result["final_mean_return"] = float(evals["mean_return"].iloc[-1])
    w = criterion.windows
    success = evals["success_rate"].to_numpy()
    returns = evals["mean_return"].to_numpy()
    for k in range(w - 1, len(evals)):
        window_s, window_r = success[k - w + 1 : k + 1], returns[k - w + 1 : k + 1]
        stable = (window_r.max() - window_r.min()) <= criterion.return_tolerance * max(1.0, abs(window_r.mean()))
        if (window_s >= criterion.success_threshold).all() and stable:
            row = evals.iloc[k]
            result.update(converged=True, episodes_to_convergence=int(row["episode"]),
                          steps_to_convergence=float(row.get("env_steps", np.nan)),
                          time_to_convergence_s=float(row.get("train_time_s", np.nan)))
            break
    return result


def bellman_convergence(residuals: list[float], theta: float) -> dict:
    below = [i for i, r in enumerate(residuals) if r < theta]
    return {"converged": bool(below), "sweeps_to_convergence": below[0] + 1 if below else np.nan,
            "final_residual": residuals[-1] if residuals else np.nan}
