"""
Q-Network: a simple multi-layer perceptron for DQN.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Multi-layer perceptron that approximates the Q-function.

    Architecture: input_dim -> [hidden_dims] -> output_dim
    Activations:  ReLU after every hidden layer; no BN.

    Args:
        input_dim:   Size of the state observation vector.
        output_dim:  Number of discrete actions.
        hidden_dims: List of hidden layer widths (e.g. [128, 128]).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
