"""
TD3 Actor and Twin Critic Networks.

TD3ActorNetwork:  Identical to DDPG's ActorNetwork — deterministic policy with tanh output.
TD3CriticNetwork: Two independent Q-networks (Q1, Q2) in a single module.
                  During critic training, both Q-networks are updated.
                  During actor training, only Q1 is used (avoids correlated overestimation).

The twin-critic design (Fujimoto et al. 2018) reduces the positive bias in the
policy improvement step that causes DDPG to overestimate Q-values:
    - target_Q = r + gamma * min(Q1_target, Q2_target)
Using the minimum of the two target networks propagates more conservative
(lower) Q-value estimates through the Bellman backup.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    """Build a fully-connected MLP with ReLU hidden activations."""
    layers: List[nn.Module] = []
    in_features = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        in_features = h
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class TD3ActorNetwork(nn.Module):
    """
    Deterministic actor for TD3 (same architecture as DDPG actor).

    Maps state -> bounded continuous action via tanh.

    Args:
        input_dim:    Dimensionality of the state vector.
        action_dim:   Dimensionality of the continuous action vector.
        hidden_dims:  List of hidden layer sizes.
        action_scale: Scales the tanh output to [-action_scale, action_scale].
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_scale = action_scale
        self.net = _build_mlp(input_dim, hidden_dims, action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for i, module in enumerate(self.net.modules()):
            if isinstance(module, nn.Linear):
                is_last = (i == len(list(self.net.modules())) - 1)
                if is_last:
                    nn.init.uniform_(module.weight, -3e-3, 3e-3)
                else:
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute deterministic action.

        Args:
            x: State tensor of shape [batch_size, input_dim].

        Returns:
            Action tensor of shape [batch_size, action_dim],
            elements in [-action_scale, action_scale].
        """
        return torch.tanh(self.net(x)) * self.action_scale


class TD3CriticNetwork(nn.Module):
    """
    Twin Q-network critic for TD3.

    Contains two independent Q-networks (Q1 and Q2) to mitigate the
    overestimation bias inherent in single-critic actor-critic methods.

    During Bellman backup:   use min(Q1, Q2) from target critics.
    During actor update:     use only Q1 (avoids correlated Q1/Q2 gradient).

    Args:
        state_dim:   Dimensionality of the state vector.
        action_dim:  Dimensionality of the action vector.
        hidden_dims: Hidden layer sizes for both Q-networks.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        sa_dim = state_dim + action_dim
        self.q1_net = _build_mlp(sa_dim, hidden_dims, 1)
        self.q2_net = _build_mlp(sa_dim, hidden_dims, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for net in [self.q1_net, self.q2_net]:
            for i, module in enumerate(net.modules()):
                if isinstance(module, nn.Linear):
                    is_last = (i == len(list(net.modules())) - 1)
                    if is_last:
                        nn.init.uniform_(module.weight, -3e-3, 3e-3)
                    else:
                        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values for a batch of (state, action) pairs.

        Used during critic updates: both Q1 and Q2 losses are computed simultaneously.

        Args:
            state:  State tensor of shape [batch_size, state_dim].
            action: Action tensor of shape [batch_size, action_dim].

        Returns:
            Tuple (Q1, Q2), each of shape [batch_size, 1].
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q1 only (used during actor update to avoid double-counting).

        Args:
            state:  State tensor of shape [batch_size, state_dim].
            action: Action tensor of shape [batch_size, action_dim].

        Returns:
            Q1 tensor of shape [batch_size, 1].
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)
