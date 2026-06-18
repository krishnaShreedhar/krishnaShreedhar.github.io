"""
PPO Actor-Critic Network with separate actor and critic MLPs.

Unlike A2C's shared backbone, PPO uses independent networks for the policy and
the value function. This avoids gradient interference between the two objectives
and is particularly beneficial for fine-tuning tasks (e.g., RLHF) where the
policy must remain close to a reference distribution.

For discrete action spaces the actor outputs logits fed into a Categorical distribution.
For continuous action spaces the actor outputs (mean, log_std) fed into a Normal distribution.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    """Helper to build a fully-connected MLP with ReLU activations."""
    layers: List[nn.Module] = []
    in_features = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        in_features = h
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class PPOActorCriticNetwork(nn.Module):
    """
    Separate actor and critic networks for PPO.

    Actor (discrete):   MLP -> action_dim logits
    Actor (continuous): MLP -> action_dim means; log_std is a learnable parameter
    Critic:             MLP -> 1 (state value V(s))

    Args:
        input_dim:         Dimensionality of the state observation.
        action_dim:        Number of actions (discrete) or action dimensions (continuous).
        hidden_dims:       Hidden layer sizes shared by both actor and critic MLPs.
        action_space_type: ``"discrete"`` or ``"continuous"``.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_space_type: str = "discrete",
    ) -> None:
        super().__init__()

        if action_space_type not in ("discrete", "continuous"):
            raise ValueError(
                f"action_space_type must be 'discrete' or 'continuous', got '{action_space_type}'"
            )

        self.action_space_type = action_space_type
        self.action_dim = action_dim

        # Actor network
        if action_space_type == "discrete":
            self.actor = _build_mlp(input_dim, action_dim, hidden_dims)
        else:
            # For continuous: output mean; log_std is a separate learnable parameter
            self.actor = _build_mlp(input_dim, action_dim, hidden_dims)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network (independent MLP)
        self.critic = _build_mlp(input_dim, 1, hidden_dims)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Orthogonal initialisation for all linear layers."""
        for net in [self.actor, self.critic]:
            for i, module in enumerate(net.modules()):
                if isinstance(module, nn.Linear):
                    # Last layer uses a smaller gain to keep initial actions near uniform
                    is_last = (i == len(list(net.modules())) - 1)
                    gain = 0.01 if is_last else nn.init.calculate_gain("relu")
                    nn.init.orthogonal_(module.weight, gain=gain)
                    nn.init.zeros_(module.bias)

    def _get_distribution(self, x: torch.Tensor):
        """Return the action distribution for a batch of states."""
        if self.action_space_type == "discrete":
            logits = self.actor(x)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.actor(x)
            std = self.log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)

    def get_action_and_value(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute associated quantities.

        Used during rollout collection (forward pass without requiring specific actions).

        Args:
            x: State tensor of shape [batch_size, input_dim].

        Returns:
            action:   Sampled action(s), shape [batch_size] (discrete) or [batch_size, action_dim].
            log_prob: Log-probability of the sampled action(s), shape [batch_size].
            entropy:  Policy entropy, shape [batch_size].
            value:    Critic estimate V(s), shape [batch_size, 1].
        """
        dist = self._get_distribution(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # For continuous Normal distributions, sum log_probs and entropies over action dims
        if self.action_space_type == "continuous":
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        value = self.critic(x)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log-probability and entropy of existing actions.

        Used during mini-batch updates to recompute log_probs under the current policy.

        Args:
            x:       State tensor of shape [batch_size, input_dim].
            actions: Previously taken actions of shape [batch_size] (discrete)
                     or [batch_size, action_dim] (continuous).

        Returns:
            log_prob: Log-probability of the given actions, shape [batch_size].
            entropy:  Policy entropy, shape [batch_size].
            value:    Critic estimate V(s), shape [batch_size, 1].
        """
        dist = self._get_distribution(x)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        if self.action_space_type == "continuous":
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        value = self.critic(x)
        return log_prob, entropy, value
