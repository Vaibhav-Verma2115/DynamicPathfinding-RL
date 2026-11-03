"""CNN Q-network: observation tensor (3 x H x W) -> one Q-value per action.

    channels: 0 obstacles, 1 agent, 2 goal
    conv 3x3 (32) -> ReLU -> conv 3x3 (64, stride s) -> ReLU -> conv 3x3 (64, stride s) -> ReLU
    -> adaptive average pool to at most 12 x 12 -> flatten -> FC 256 -> ReLU -> FC n_actions

Stride s is 1 for small inputs (<= 12, e.g. 10x10 grids and 7x7 local views) and 2 otherwise,
which keeps the fully connected layer manageable at 100x100. Padding keeps border cells
visible to the filters.
"""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_side: int, n_actions: int = 4, in_channels: int = 3):
        super().__init__()
        stride = 1 if obs_side <= 12 else 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=stride, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=stride, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            side = self.features(torch.zeros(1, in_channels, obs_side, obs_side)).shape[-1]
        pooled = min(side, 12)
        self.pool = nn.AdaptiveAvgPool2d(pooled)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * pooled * pooled, 256), nn.ReLU(), nn.Linear(256, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))
