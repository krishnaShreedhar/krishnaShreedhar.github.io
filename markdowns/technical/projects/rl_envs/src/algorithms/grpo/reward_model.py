"""GRPO reward model: a learned preference-based reward function.

The reward model is trained on preference pairs using the Bradley-Terry model.
It maps (state/context embedding, action/response embedding) -> scalar score.
"""

from typing import List

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


class GRPORewardModel(nn.Module):
    """Learned reward model for GRPO using Bradley-Terry preference modelling.

    Takes a (state, action) pair — where state is the question/context embedding
    and action is the response embedding — and outputs a scalar reward score.

    The model is trained by contrasting preferred (winner) responses against
    dispreferred (loser) responses via the Bradley-Terry loss:
        L = -log(sigma(r_winner - r_loser))

    This reward signal is then used by the GRPO agent during group advantage
    computation. In a full LLM setting, the embeddings would come from a
    pre-trained encoder; here we use fixed-dimension vectors for the tutorial.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> None:
        """Initialise the reward model.

        Args:
            state_dim: Dimensionality of the state/context embedding.
            action_dim: Dimensionality of the action/response embedding.
            hidden_dims: List of hidden layer sizes.
        """
        super().__init__()
        input_dim = state_dim + action_dim
        # Final layer outputs a single scalar (the reward score)
        self.network = _build_mlp(input_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict a scalar reward for the given (state, action) pair.

        Args:
            state: Context embedding of shape (batch, state_dim).
            action: Response embedding of shape (batch, action_dim).

        Returns:
            reward: Scalar reward scores of shape (batch, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def compute_preference_loss(
        self,
        winner_state: torch.Tensor,
        winner_action: torch.Tensor,
        loser_state: torch.Tensor,
        loser_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Bradley-Terry preference loss.

        Encourages the model to assign a higher reward to the preferred
        (winner) response than to the dispreferred (loser) response.

            L = -mean(log(sigma(r_winner - r_loser)))

        Args:
            winner_state: Context embeddings for preferred responses, (batch, state_dim).
            winner_action: Response embeddings for preferred responses, (batch, action_dim).
            loser_state: Context embeddings for dispreferred responses, (batch, state_dim).
            loser_action: Response embeddings for dispreferred responses, (batch, action_dim).

        Returns:
            loss: Scalar preference loss tensor.
        """
        r_winner = self.forward(winner_state, winner_action)   # (batch, 1)
        r_loser = self.forward(loser_state, loser_action)      # (batch, 1)

        # Bradley-Terry: P(winner > loser) = sigma(r_winner - r_loser)
        # Loss: negative log likelihood
        loss = -F.logsigmoid(r_winner - r_loser).mean()
        return loss
