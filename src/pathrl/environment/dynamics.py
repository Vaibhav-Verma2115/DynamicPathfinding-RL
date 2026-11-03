"""Agent movement: action tables, slip (stochastic transitions) and collision rules.

Coordinates are (row, col) with row 0 at the top, so UP decreases the row.

The transition model is written once here and used by both the environment
(`sample_delta`, which draws one outcome) and later by Value Iteration
(`action_outcomes`, which lists every outcome with its probability). Because both
call the same functions, the model a planner reasons about is exactly the dynamics
the environment executes.
"""

from __future__ import annotations

import numpy as np

Cell = tuple[int, int]
Delta = tuple[int, int]

ACTIONS_4: tuple[Delta, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTIONS_8: tuple[Delta, ...] = ACTIONS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")


def action_deltas(connectivity: int) -> tuple[Delta, ...]:
    if connectivity == 4:
        return ACTIONS_4
    if connectivity == 8:
        return ACTIONS_8
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def perpendicular(delta: Delta) -> tuple[Delta, Delta]:
    """The two moves at +/-90 degrees: rotating (dr, dc) gives (-dc, dr) and (dc, -dr).

    RIGHT (0,1) -> UP (-1,0) and DOWN (1,0); UP_RIGHT (-1,1) -> UP_LEFT and DOWN_RIGHT.
    A slip never reverses the agent.
    """
    dr, dc = delta
    return (-dc, dr), (dc, -dr)


def action_outcomes(deltas: tuple[Delta, ...], action: int, slip_prob: float) -> list[tuple[float, Delta]]:
    """All possible executed moves for `action`, as (probability, delta) pairs summing to 1."""
    intended = deltas[action]
    if slip_prob == 0.0:
        return [(1.0, intended)]
    left, right = perpendicular(intended)
    return [(1.0 - slip_prob, intended), (slip_prob / 2, left), (slip_prob / 2, right)]


def sample_delta(
    deltas: tuple[Delta, ...], action: int, slip_prob: float, rng: np.random.Generator
) -> tuple[Delta, bool]:
    """Draw the executed move. Returns (delta, slipped).

    Both random numbers are drawn on every call, even when slip_prob is 0, so each
    step uses a fixed amount of randomness. Two methods run with the same seed then
    see the same slip sequence ("common random numbers"), which makes paired
    comparisons between algorithms less noisy.
    """
    u = rng.random()
    side = int(rng.integers(2))
    if u < slip_prob:
        return perpendicular(deltas[action])[side], True
    return deltas[action], False


def move_cost(delta: Delta, diagonal_cost: float) -> float:
    return diagonal_cost if delta[0] != 0 and delta[1] != 0 else 1.0


def attempt_move(
    pos: Cell, delta: Delta, occupied: np.ndarray, allow_corner_cutting: bool = False
) -> tuple[Cell, bool]:
    """Apply the collision rules. Returns (new_position, collided).

    A move collides, and the agent stays where it is, if the target cell is
    outside the grid or occupied by a wall or a moving obstacle. A diagonal move
    also collides when corner cutting is disallowed and either orthogonal neighbour
    it would squeeze past is occupied.
    """
    n_rows, n_cols = occupied.shape
    r, c = pos
    dr, dc = delta
    nr, nc = r + dr, c + dc
    if not (0 <= nr < n_rows and 0 <= nc < n_cols) or occupied[nr, nc]:
        return pos, True
    if dr != 0 and dc != 0 and not allow_corner_cutting:
        if occupied[r + dr, c] or occupied[r, c + dc]:
            return pos, True
    return (nr, nc), False
