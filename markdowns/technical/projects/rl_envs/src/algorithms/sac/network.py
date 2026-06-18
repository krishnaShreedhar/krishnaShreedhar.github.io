"""SAC network architectures: Gaussian actor and twin soft Q-network critic."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    """Build a multi-layer perceptron with ReLU activations."""
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class SACActorNetwork(nn.Module):
    """Gaussian stochastic policy network for SAC.

    Outputs a mean and log_std for a diagonal Gaussian. Actions are sampled
    using the reparameterization trick and then squashed through tanh.
    The log-probability includes the tanh correction term.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk
        trunk_layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h))
            trunk_layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std of the Gaussian policy.

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            mean: Action mean of shape (batch, action_dim).
            log_std: Clamped log standard deviation of shape (batch, action_dim).
        """
        features = self.trunk(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action using the reparameterization trick with tanh squashing.

        The log-probability accounts for the tanh change of variables:
            log pi(a) = log pi(u) - sum_i log(1 - tanh(u_i)^2)
        where u is the pre-tanh Gaussian sample.

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            action: Tanh-squashed action of shape (batch, action_dim).
            log_prob: Log probability of each action, shape (batch, 1).
            mean: Deterministic mean action (tanh applied), shape (batch, action_dim).
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()

        # Reparameterization: u = mean + std * eps, eps ~ N(0, I)
        eps = torch.randn_like(mean)
        u = mean + std * eps

        # Tanh squashing
        action = torch.tanh(u)

        # Log probability of the Gaussian at u
        log_prob_gaussian = (
            -0.5 * ((u - mean) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        ).sum(dim=-1, keepdim=True)

        # Tanh correction: subtract log(1 - tanh(u)^2)
        # Numerically stable version: 2*(log2 - u - softplus(-2u))
        tanh_correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob_gaussian - tanh_correction

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class SACCriticNetwork(nn.Module):
    """Twin soft Q-networks for SAC.

    Two independent Q-networks that share the same interface but have
    separate weights. Using the minimum of the two reduces overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        # Q1 network
        self.q1 = _build_mlp(input_dim, hidden_dims, 1)
        # Q2 network (independent weights)
        self.q2 = _build_mlp(input_dim, hidden_dims, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both networks.

        Args:
            state: State tensor of shape (batch, state_dim).
            action: Action tensor of shape (batch, action_dim).

        Returns:
            q1: Q-values from network 1, shape (batch, 1).
            q2: Q-values from network 2, shape (batch, 1).
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
