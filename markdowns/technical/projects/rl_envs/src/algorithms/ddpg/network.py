"""
DDPG Actor and Critic Networks.

Actor:  Deterministic policy pi(s) -> action in [-action_scale, action_scale].
        Bounded by tanh to keep actions in a valid continuous range.

Critic: Action-value function Q(s, a) -> scalar Q-value.
        State and action are concatenated and fed through a shared MLP.
        This avoids the need to enumerate all actions and works naturally
        for continuous action spaces.
"""

from typing import List

import torch
import torch.nn as nn


def _build_mlp_body(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    """Build MLP hidden layers with ReLU activations (no output layer)."""
    layers: List[nn.Module] = []
    in_features = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        in_features = h
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """
    Deterministic actor: maps state -> bounded continuous action.

    Architecture:
        state -> [MLP hidden layers with ReLU] -> Linear -> Tanh -> scaled output

    The tanh output is scaled by ``action_scale`` so that the network always
    produces actions in the range ``[-action_scale, action_scale]``.

    Args:
        input_dim:    Dimensionality of the state vector.
        action_dim:   Dimensionality of the (continuous) action vector.
        hidden_dims:  List of hidden layer sizes.
        action_scale: Scales the tanh output.  Default 1.0.
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
        self.body = _build_mlp_body(input_dim, hidden_dims)
        last_hidden = hidden_dims[-1]
        self.output_layer = nn.Linear(last_hidden, action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.body.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a deterministic action.

        Args:
            x: State tensor of shape [batch_size, input_dim].

        Returns:
            Action tensor of shape [batch_size, action_dim],
            each element in the range [-action_scale, action_scale].
        """
        features = self.body(x)
        return torch.tanh(self.output_layer(features)) * self.action_scale


class CriticNetwork(nn.Module):
    """
    Action-value critic Q(s, a) -> scalar.

    State and action are concatenated at the input so the network can model
    the interaction between the two.  The output is a single Q-value.

    Architecture:
        [state | action] -> [MLP hidden layers with ReLU] -> Linear -> Q-value

    Args:
        state_dim:   Dimensionality of the state vector.
        action_dim:  Dimensionality of the (continuous) action vector.
        hidden_dims: List of hidden layer sizes.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.body = _build_mlp_body(state_dim + action_dim, hidden_dims)
        last_hidden = hidden_dims[-1]
        self.output_layer = nn.Linear(last_hidden, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.body.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the action-value Q(s, a).

        Args:
            state:  State tensor of shape [batch_size, state_dim].
            action: Action tensor of shape [batch_size, action_dim].

        Returns:
            Q-value tensor of shape [batch_size, 1].
        """
        x = torch.cat([state, action], dim=-1)
        features = self.body(x)
        return self.output_layer(features)
