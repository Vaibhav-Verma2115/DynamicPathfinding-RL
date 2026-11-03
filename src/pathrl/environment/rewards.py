"""Reward function.

Every step has exactly one outcome, and the outcome alone decides the reward:

    REACHED_GOAL -> rewards.goal                      (+100)
    COLLIDED     -> rewards.collision                 (-50, the agent stays in place)
    MOVED        -> rewards.step * move_cost          (-1 per cardinal step, -sqrt(2) per diagonal)

Diagonal steps are scaled by their cost so that, with 8-connectivity, maximising
reward and minimising the path cost a planner reports are the same objective.
"""

from __future__ import annotations

from enum import Enum

from pathrl.environment.config import RewardConfig


class Outcome(str, Enum):
    MOVED = "moved"
    COLLIDED = "collided"
    REACHED_GOAL = "reached_goal"


def compute_reward(outcome: Outcome, rewards: RewardConfig, step_cost: float = 1.0) -> float:
    if outcome is Outcome.REACHED_GOAL:
        return rewards.goal
    if outcome is Outcome.COLLIDED:
        return rewards.collision
    return rewards.step * step_cost
