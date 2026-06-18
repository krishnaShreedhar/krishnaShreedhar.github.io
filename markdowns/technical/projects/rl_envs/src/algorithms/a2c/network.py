"""
Actor-Critic Network for A2C.

Shared MLP backbone with separate actor (policy logits) and critic (state value) heads.
The shared backbone allows the policy and value function to learn common feature
representations, which improves sample efficiency for on-policy learning.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone actor-critic network.

    Architecture:
        input -> [shared MLP trunk with ReLU] -> actor_head (logits) + critic_head (value)

    The shared trunk extracts state features used by both heads.
    Actor head: single linear layer producing action_dim logits (for Categorical distribution).
    Critic head: single linear layer producing a scalar state value V(s).
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        """
        Initialize the shared actor-critic network.

        Args:
            input_dim: Dimensionality of the input state vector.
            action_dim: Number of discrete actions (logits output size).
            hidden_dims: List of hidden layer sizes for the shared trunk.
                         E.g. [128, 128] creates two hidden layers of 128 units each.
        """
        super().__init__()

        # Build the shared trunk as a sequential stack of Linear + ReLU blocks
        trunk_layers: List[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)

        # Actor head: maps trunk output -> action logits (unnormalized log-probabilities)
        self.actor_head = nn.Linear(in_features, action_dim)

        # Critic head: maps trunk output -> scalar state value V(s)
        self.critic_head = nn.Linear(in_features, 1)

        # Weight initialization: orthogonal for trunk, smaller scale for output heads
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply orthogonal initialization to all linear layers."""
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared trunk and both heads.

        Args:
            x: Input state tensor of shape [batch_size, input_dim].

        Returns:
            Tuple of:
                action_logits: Raw logits of shape [batch_size, action_dim].
                               Pass through Categorical(logits=...) for a policy distribution.
                value:         Estimated state value of shape [batch_size, 1].
        """
        features = self.trunk(x)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value
