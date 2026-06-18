"""GRPO (Group Relative Policy Optimization) agent implementation.

Use case: VLM / LLM fine-tuning (e.g. DeepSeek-R1 style RLHF).

Key innovation over PPO:
  - No critic network required.
  - Advantage is estimated by comparing G outputs for the *same* input:
        A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + eps)
  - This is much cheaper than maintaining a full value function.
  - A KL penalty keeps the updated policy close to a frozen reference policy.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.base_agent import BaseAgent
from core.logger import RLLogger


def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class _PolicyNetwork(nn.Module):
    """Simple MLP policy that maps state embeddings to action logits."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.network = _build_mlp(input_dim, hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits for the given state embedding.

        Args:
            x: State/context embedding of shape (batch, input_dim).

        Returns:
            logits: Action logits of shape (batch, action_dim).
        """
        return self.network(x)

    def log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for given (state, action) pairs.

        Args:
            x: State embedding of shape (batch, input_dim).
            actions: Integer action indices of shape (batch,).

        Returns:
            log_probs: Log probabilities of shape (batch,).
        """
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)


class GRPOAgent(BaseAgent):
    """Group Relative Policy Optimization agent.

    The agent maintains:
      - A learnable policy network
      - A frozen reference policy (copy of initial weights) for KL regularisation
      - An in-memory experience buffer that is cleared after each update

    Data collection pattern:
      1. For each question/state, call collect_group() to generate G responses.
      2. Obtain rewards (from environment or reward model) for each response.
      3. Call store_group() to compute group-relative advantages and buffer them.
      4. Call update() periodically to run n_epochs of clipped policy gradient.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        logger: RLLogger,
    ) -> None:
        super().__init__(config, device, logger)

        net_cfg = config["network"]
        train_cfg = config["training"]

        input_dim: int = int(net_cfg["input_dim"])
        action_dim: int = int(net_cfg["action_dim"])
        hidden_dims: List[int] = list(net_cfg["hidden_dims"])

        # ------------------------------------------------------------------ #
        # Networks
        # ------------------------------------------------------------------ #
        self.policy_net = _PolicyNetwork(input_dim, action_dim, hidden_dims).to(device)

        # Reference policy: frozen copy of initial weights (never updated)
        import copy
        self._ref_policy = copy.deepcopy(self.policy_net).to(device)
        for p in self._ref_policy.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Optimiser
        # ------------------------------------------------------------------ #
        self._optimizer = optim.Adam(
            self.policy_net.parameters(), lr=float(train_cfg["learning_rate"])
        )

        # ------------------------------------------------------------------ #
        # Hyperparameters
        # ------------------------------------------------------------------ #
        self._group_size: int = int(train_cfg["group_size"])
        self._clip_epsilon: float = float(train_cfg["clip_epsilon"])
        self._n_epochs: int = int(train_cfg["n_epochs"])
        self._kl_coef: float = float(train_cfg["kl_coef"])
        self._batch_size: int = int(train_cfg["batch_size"])
        self._gamma: float = float(train_cfg.get("gamma", 1.0))

        # ------------------------------------------------------------------ #
        # In-memory experience buffer (cleared after each update call)
        # ------------------------------------------------------------------ #
        # Each entry: (state, action, advantage, old_log_prob)
        self._buffer_states: List[np.ndarray] = []
        self._buffer_actions: List[int] = []
        self._buffer_advantages: List[float] = []
        self._buffer_old_log_probs: List[float] = []

        self.logger.info(
            f"GRPOAgent initialised | input_dim={input_dim} action_dim={action_dim} "
            f"group_size={self._group_size} clip_eps={self._clip_epsilon} kl_coef={self._kl_coef}"
        )

    # ---------------------------------------------------------------------- #
    # BaseAgent interface
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select a single action for the given state.

        For data collection call collect_group() instead to get G responses.

        Args:
            state: Current state/context embedding.
            training: If True, sample from softmax; if False, take argmax.

        Returns:
            action: Integer action index.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            if training:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
            else:
                action = logits.argmax(dim=-1).item()
        self.policy_net.train()
        return int(action)

    def collect_group(
        self, state: np.ndarray
    ) -> Tuple[List[int], List[float]]:
        """Generate G responses for a single state (question).

        All G responses are sampled independently from the current policy.

        Args:
            state: State/context embedding of shape (input_dim,).

        Returns:
            actions: List of G sampled action indices.
            log_probs: List of G corresponding log probabilities.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Expand to (G, input_dim)
        state_batch = state_tensor.expand(self._group_size, -1)

        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_batch)               # (G, action_dim)
            probs = F.softmax(logits, dim=-1)                   # (G, action_dim)
            actions_tensor = torch.multinomial(probs, num_samples=1).squeeze(1)  # (G,)
            log_probs_tensor = F.log_softmax(logits, dim=-1).gather(
                1, actions_tensor.unsqueeze(1)
            ).squeeze(1)                                         # (G,)
        self.policy_net.train()

        actions = actions_tensor.cpu().tolist()
        log_probs = log_probs_tensor.cpu().tolist()
        self.logger.debug(f"collect_group | actions={actions} log_probs={log_probs}")
        return actions, log_probs

    def store_group(
        self,
        state: np.ndarray,
        actions: List[int],
        rewards: List[float],
        log_probs: List[float],
    ) -> None:
        """Compute group-relative advantages and buffer the experience.

        Group-relative advantage normalises rewards within the group:
            A_i = (r_i - mean(rewards)) / (std(rewards) + 1e-8)

        This eliminates the need for a learned value baseline.

        Args:
            state: State/context embedding (shared for all G responses).
            actions: List of G action indices.
            rewards: List of G scalar rewards.
            log_probs: List of G log probabilities at collection time.
        """
        rewards_arr = np.array(rewards, dtype=np.float32)
        mean_r = rewards_arr.mean()
        std_r = rewards_arr.std()
        advantages = (rewards_arr - mean_r) / (std_r + 1e-8)

        self.logger.debug(
            f"store_group | mean_reward={mean_r:.4f} std_reward={std_r:.4f} "
            f"advantages={advantages.tolist()}"
        )

        for i in range(len(actions)):
            self._buffer_states.append(state)
            self._buffer_actions.append(actions[i])
            self._buffer_advantages.append(float(advantages[i]))
            self._buffer_old_log_probs.append(log_probs[i])

    def update(self) -> Dict[str, float]:
        """Update policy using clipped group-relative policy gradient.

        Runs n_epochs over the buffered experience. Clears the buffer afterwards.

        Returns:
            Dictionary with keys: clip_loss, kl_penalty, mean_advantage, mean_reward.
        """
        n = len(self._buffer_states)
        if n == 0:
            self.logger.warning("update called with empty buffer — skipping.")
            return {"clip_loss": 0.0, "kl_penalty": 0.0, "mean_advantage": 0.0, "mean_reward": 0.0}

        states_np = np.stack(self._buffer_states, axis=0)             # (N, input_dim)
        actions_np = np.array(self._buffer_actions, dtype=np.int64)   # (N,)
        advantages_np = np.array(self._buffer_advantages, dtype=np.float32)  # (N,)
        old_lp_np = np.array(self._buffer_old_log_probs, dtype=np.float32)   # (N,)

        states_t = torch.FloatTensor(states_np).to(self.device)
        actions_t = torch.LongTensor(actions_np).to(self.device)
        advantages_t = torch.FloatTensor(advantages_np).to(self.device)
        old_lp_t = torch.FloatTensor(old_lp_np).to(self.device)

        # Precompute reference policy log-probs (constant across epochs)
        with torch.no_grad():
            ref_log_probs_t = self._ref_policy.log_prob(states_t, actions_t)

        total_clip_loss = 0.0
        total_kl = 0.0

        for epoch in range(self._n_epochs):
            # Mini-batch iteration over the full buffer
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self._batch_size):
                idx = indices[start: start + self._batch_size]
                s = states_t[idx]
                a = actions_t[idx]
                adv = advantages_t[idx]
                old_lp = old_lp_t[idx]
                ref_lp = ref_log_probs_t[idx]

                new_log_probs = self.policy_net.log_prob(s, a)

                # Clipped policy gradient (PPO-style)
                ratio = torch.exp(new_log_probs - old_lp)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon
                )
                clip_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                # Approximate KL: E[log pi_ref - log pi_new]
                kl = (ref_lp - new_log_probs).mean()

                loss = clip_loss + self._kl_coef * kl

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_clip_loss += clip_loss.item()
                total_kl += kl.item()

        num_batches = max(
            1,
            self._n_epochs * (n // self._batch_size + (1 if n % self._batch_size else 0)),
        )
        avg_clip = total_clip_loss / num_batches
        avg_kl = total_kl / num_batches
        mean_advantage = float(advantages_np.mean())
        group_reward_std = float(advantages_np.std())

        self.logger.info(
            f"update | clip_loss={avg_clip:.4f} kl_penalty={avg_kl:.4f} "
            f"mean_advantage={mean_advantage:.4f} group_reward_std={group_reward_std:.4f}"
        )

        # Clear buffer for next collection round
        self._buffer_states.clear()
        self._buffer_actions.clear()
        self._buffer_advantages.clear()
        self._buffer_old_log_probs.clear()

        return {
            "clip_loss": avg_clip,
            "kl_penalty": avg_kl,
            "mean_advantage": mean_advantage,
            "mean_reward": group_reward_std,
        }

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return named network modules for checkpointing.

        Note: The reference policy is intentionally excluded — it is fixed
        and should not be saved/loaded as a checkpoint.

        Returns:
            Dictionary with key: policy.
        """
        return {"policy": self.policy_net}
