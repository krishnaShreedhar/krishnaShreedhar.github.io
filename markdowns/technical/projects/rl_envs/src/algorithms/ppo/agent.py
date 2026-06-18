"""
PPO (Proximal Policy Optimization) Agent.

Use case: VLM finetuning (RLHF-style training) and general policy learning.

Algorithm overview:
    1. Collect n_steps transitions using the current policy (rollout phase).
    2. Compute Generalized Advantage Estimation (GAE) for every timestep.
    3. Normalize advantages across the rollout.
    4. For n_epochs epochs, shuffle the rollout into mini-batches and perform:
        - Clipped surrogate objective (PPO-Clip) to limit policy change.
        - Value function MSE loss.
        - Entropy bonus to maintain exploration.
    5. Clear the rollout buffer and repeat.

Key properties:
    - On-policy: old data is discarded after n_epochs of reuse.
    - Clipped surrogate avoids large, destabilizing policy updates without
      requiring an explicit KL constraint.
    - GAE interpolates between Monte-Carlo (lambda=1) and TD(0) (lambda=0).
    - Well-suited for RLHF because the clip prevents the policy from drifting
      too far from the reference model within one update phase.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.logger import RLLogger
from algorithms.ppo.network import PPOActorCriticNetwork


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent with GAE and mini-batch clipped updates.

    Config keys expected:
        network.input_dim          (int)   - state feature dimension
        network.action_dim         (int)   - number of actions / action dimensions
        network.hidden_dims        (list)  - hidden layer sizes for actor and critic MLPs
        network.action_space_type  (str)   - "discrete" or "continuous"
        training.learning_rate     (float) - Adam learning rate
        training.gamma             (float) - discount factor
        training.gae_lambda        (float) - GAE lambda (0=TD, 1=MC)
        training.clip_epsilon      (float) - PPO clipping threshold (e.g. 0.2)
        training.n_epochs          (int)   - number of mini-batch passes per rollout
        training.n_steps           (int)   - rollout length before an update
        training.batch_size        (int)   - mini-batch size for gradient updates
        training.entropy_coef      (float) - weight on entropy bonus
        training.value_loss_coef   (float) - weight on critic MSE loss
        training.max_grad_norm     (float) - gradient clipping threshold
    """

    def __init__(self, config: Dict, device: torch.device, logger: RLLogger) -> None:
        super().__init__(config, device, logger)

        # ── Hyperparameters ──────────────────────────────────────────────────
        self._lr: float = config["training"]["learning_rate"]
        self._gamma: float = config["training"]["gamma"]
        self._gae_lambda: float = config["training"]["gae_lambda"]
        self._clip_epsilon: float = config["training"]["clip_epsilon"]
        self._n_epochs: int = config["training"]["n_epochs"]
        self._n_steps: int = config["training"]["n_steps"]
        self._batch_size: int = config["training"]["batch_size"]
        self._entropy_coef: float = config["training"]["entropy_coef"]
        self._value_loss_coef: float = config["training"]["value_loss_coef"]
        self._max_grad_norm: float = config["training"]["max_grad_norm"]

        input_dim: int = config["network"]["input_dim"]
        action_dim: int = config["network"]["action_dim"]
        hidden_dims: List[int] = config["network"]["hidden_dims"]
        action_space_type: str = config["network"].get("action_space_type", "discrete")

        # ── Network ──────────────────────────────────────────────────────────
        self.network = PPOActorCriticNetwork(
            input_dim, action_dim, hidden_dims, action_space_type
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self._lr)

        # ── Rollout buffer ────────────────────────────────────────────────
        self._states: List[np.ndarray] = []
        self._actions: List[Any] = []
        self._rewards: List[float] = []
        self._values: List[float] = []
        self._log_probs: List[float] = []
        self._dones: List[bool] = []

        self.logger.info(
            f"PPOAgent initialised | lr={self._lr} gamma={self._gamma} "
            f"gae_lambda={self._gae_lambda} clip_epsilon={self._clip_epsilon} "
            f"n_epochs={self._n_epochs} n_steps={self._n_steps} "
            f"batch_size={self._batch_size} action_space={action_space_type}"
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> Any:
        """
        Sample from the policy distribution (training) or act greedily (eval).

        Stores the action, log_prob, and value in the rollout buffer.

        Args:
            state:    State observation as a 1-D numpy array.
            training: Stochastic sampling when True; greedy/mean when False.

        Returns:
            Selected action (int for discrete, np.ndarray for continuous).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor, log_prob_tensor, _, value_tensor = self.network.get_action_and_value(
                state_tensor
            )

        if not training:
            # Greedy: use mode of distribution
            if self.network.action_space_type == "discrete":
                logits = self.network.actor(state_tensor)
                action_tensor = logits.argmax(dim=-1)
                log_prob_tensor = torch.distributions.Categorical(logits=logits).log_prob(
                    action_tensor
                )
            else:
                action_tensor = self.network.actor(state_tensor)
                log_prob_tensor = torch.zeros(1, device=self.device)

        log_prob: float = log_prob_tensor.item()
        value: float = value_tensor.item()

        if self.network.action_space_type == "discrete":
            action: Any = action_tensor.item()
        else:
            action = action_tensor.cpu().numpy().squeeze()

        self._log_probs.append(log_prob)
        self._values.append(value)

        self.logger.debug(
            f"select_action | state_norm={np.linalg.norm(state):.4f} "
            f"action={action} log_prob={log_prob:.4f} value={value:.4f}"
        )
        return action

    # ── Rollout storage ───────────────────────────────────────────────────────

    def store_step(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Append a transition to the rollout buffer.

        ``select_action`` must have been called first to populate log_prob and value.

        Args:
            state:      State at this step.
            action:     Action taken.
            reward:     Scalar reward received.
            next_state: Resulting next state (for GAE bootstrap).
            done:       Episode termination flag.
        """
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)

        self.logger.debug(
            f"store_step | reward={reward:.4f} done={done} "
            f"buffer_len={len(self._rewards)}"
        )

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update over the collected rollout.

        Steps:
            1. Compute GAE advantages and returns.
            2. Normalize advantages.
            3. For n_epochs epochs, shuffle and iterate mini-batches:
                - Recompute log_probs and entropy under current policy.
                - Compute clipped surrogate loss.
                - Compute value loss and entropy bonus.
                - Backpropagate and clip gradients.
            4. Compute and log approximate KL divergence.
            5. Clear the rollout buffer.

        Returns:
            Dict with keys: policy_loss, value_loss, entropy, approx_kl, clip_fraction.
        """
        if len(self._rewards) == 0:
            self.logger.warning("update() called with empty rollout buffer.")
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_fraction": 0.0,
            }

        T = len(self._rewards)

        # Convert rollout to tensors
        states_np = np.array(self._states, dtype=np.float32)
        states_t = torch.FloatTensor(states_np).to(self.device)
        old_log_probs_t = torch.FloatTensor(self._log_probs).to(self.device)   # [T]
        old_values_t = torch.FloatTensor(self._values).to(self.device)          # [T]
        rewards_t = torch.FloatTensor(self._rewards).to(self.device)            # [T]
        dones_t = torch.FloatTensor(self._dones).to(self.device)                # [T]

        if self.network.action_space_type == "discrete":
            actions_t = torch.LongTensor(self._actions).to(self.device)         # [T]
        else:
            actions_t = torch.FloatTensor(np.array(self._actions)).to(self.device)  # [T, action_dim]

        # ── GAE computation ───────────────────────────────────────────────
        advantages_t = torch.zeros(T, device=self.device)
        last_gae = 0.0

        # Bootstrap the value of the state after the last step
        if not self._dones[-1]:
            last_state = torch.FloatTensor(self._states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, _, last_value_tensor = self.network.get_action_and_value(last_state)
            next_value: float = last_value_tensor.item()
        else:
            next_value = 0.0

        # Extend values array with the bootstrap value for delta computation
        values_extended = torch.cat([old_values_t, torch.tensor([next_value], device=self.device)])

        for t in reversed(range(T)):
            not_done = 1.0 - dones_t[t].item()
            delta = (
                rewards_t[t].item()
                + self._gamma * values_extended[t + 1].item() * not_done
                - old_values_t[t].item()
            )
            last_gae = delta + self._gamma * self._gae_lambda * not_done * last_gae
            advantages_t[t] = last_gae

        returns_t = advantages_t + old_values_t  # [T]

        # ── Normalize advantages ──────────────────────────────────────────
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        advantages_t = (advantages_t - adv_mean) / adv_std

        # ── Mini-batch update for n_epochs ────────────────────────────────
        all_policy_losses: List[float] = []
        all_value_losses: List[float] = []
        all_entropies: List[float] = []
        all_approx_kls: List[float] = []
        all_clip_fracs: List[float] = []

        indices = np.arange(T)
        for epoch in range(self._n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self._batch_size):
                end = min(start + self._batch_size, T)
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Recompute under current policy
                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    mb_states, mb_actions
                )
                new_values = new_values.squeeze(-1)

                # PPO clipped surrogate loss
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon)
                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages
                ).mean()

                # Value function loss
                value_loss = self._value_loss_coef * nn.functional.mse_loss(
                    new_values, mb_returns
                )

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -self._entropy_coef * entropy.mean()

                total_loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self._max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self._clip_epsilon).float().mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.mean().item())
                all_approx_kls.append(approx_kl)
                all_clip_fracs.append(clip_frac)

            self.logger.debug(
                f"PPO epoch {epoch + 1}/{self._n_epochs} done | "
                f"approx_kl={np.mean(all_approx_kls):.4f}"
            )

        # ── Aggregate metrics ─────────────────────────────────────────────
        mean_policy_loss = float(np.mean(all_policy_losses))
        mean_value_loss = float(np.mean(all_value_losses))
        mean_entropy = float(np.mean(all_entropies))
        mean_approx_kl = float(np.mean(all_approx_kls))
        mean_clip_frac = float(np.mean(all_clip_fracs))

        self.logger.info(
            f"PPO update | policy_loss={mean_policy_loss:.4f} "
            f"value_loss={mean_value_loss:.4f} "
            f"entropy={mean_entropy:.4f} "
            f"approx_kl={mean_approx_kl:.4f} "
            f"clip_fraction={mean_clip_frac:.4f} "
            f"rollout_len={T}"
        )

        if mean_approx_kl > 0.1:
            self.logger.warning(
                f"High approximate KL divergence: {mean_approx_kl:.4f}. "
                "Consider reducing learning rate or n_epochs."
            )

        self._clear_rollout()

        return {
            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,
            "entropy": mean_entropy,
            "approx_kl": mean_approx_kl,
            "clip_fraction": mean_clip_frac,
        }

    # ── Readiness check ───────────────────────────────────────────────────────

    def ready_to_update(self) -> bool:
        """Return True once the rollout buffer has accumulated n_steps transitions."""
        return len(self._rewards) >= self._n_steps

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return named modules for checkpointing."""
        return {"actor_critic": self.network}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _clear_rollout(self) -> None:
        """Reset all rollout storage lists."""
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._log_probs.clear()
        self._dones.clear()
        self.logger.debug("Rollout buffer cleared.")
