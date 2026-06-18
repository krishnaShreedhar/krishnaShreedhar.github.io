"""SAC (Soft Actor-Critic) agent implementation.

Use case: Autonomous vehicle control.

SAC is an off-policy, maximum entropy reinforcement learning algorithm.
Key innovations:
  - Entropy-regularised objective: maximise reward + alpha * entropy
  - Twin Q-networks to reduce overestimation bias
  - Automatic temperature (alpha) tuning via a dual objective
"""

import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.logger import RLLogger
from core.replay_buffer import ReplayBuffer
from algorithms.sac.network import SACActorNetwork, SACCriticNetwork


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent with automatic entropy tuning.

    Implements the full SAC algorithm including:
      - Off-policy data collection into a replay buffer
      - Twin soft Q-network critic updates
      - Stochastic actor update via reparameterization
      - Automatic temperature (alpha) tuning
      - Polyak-averaged target critic network
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
        buf_cfg = config["buffer"]

        state_dim: int = net_cfg["state_dim"]
        action_dim: int = net_cfg["action_dim"]
        hidden_dims = list(net_cfg["hidden_dims"])
        self._action_scale: float = float(net_cfg.get("action_scale", 1.0))
        log_std_min: float = float(net_cfg.get("log_std_min", -20.0))
        log_std_max: float = float(net_cfg.get("log_std_max", 2.0))

        # ------------------------------------------------------------------ #
        # Networks
        # ------------------------------------------------------------------ #
        self.actor = SACActorNetwork(
            state_dim, action_dim, hidden_dims, log_std_min, log_std_max
        ).to(device)

        self.critic = SACCriticNetwork(state_dim, action_dim, hidden_dims).to(device)

        # Target critic: kept as an EMA of the live critic
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Temperature (alpha) parameter
        # ------------------------------------------------------------------ #
        self._auto_entropy: bool = bool(train_cfg.get("auto_entropy", True))
        if self._auto_entropy:
            target_entropy = train_cfg.get("target_entropy", -float(action_dim))
        else:
            target_entropy = float(train_cfg.get("target_entropy", -float(action_dim)))
        self._target_entropy: float = float(target_entropy)

        # log_alpha is the learnable parameter (ensures alpha > 0)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=float(train_cfg["alpha_lr"])
        )

        # ------------------------------------------------------------------ #
        # Optimisers
        # ------------------------------------------------------------------ #
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(train_cfg["actor_lr"])
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=float(train_cfg["critic_lr"])
        )

        # ------------------------------------------------------------------ #
        # Hyperparameters
        # ------------------------------------------------------------------ #
        self._gamma: float = float(train_cfg["gamma"])
        self._tau: float = float(train_cfg["tau"])
        self._batch_size: int = int(train_cfg["batch_size"])
        self._replay_start: int = int(train_cfg["replay_start"])

        # ------------------------------------------------------------------ #
        # Replay buffer
        # ------------------------------------------------------------------ #
        self._replay_buffer = ReplayBuffer(
            capacity=int(buf_cfg["capacity"]),
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        self.logger.info(
            f"SACAgent initialised | state_dim={state_dim} action_dim={action_dim} "
            f"auto_entropy={self._auto_entropy} target_entropy={self._target_entropy}"
        )

    # ---------------------------------------------------------------------- #
    # BaseAgent interface
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select an action for the given state.

        Args:
            state: Current environment observation.
            training: If True, sample stochastically; if False, use deterministic mean.

        Returns:
            action: Numpy array of shape (action_dim,) scaled to [-action_scale, action_scale].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            if training:
                action_tensor, log_prob, _ = self.actor.sample(state_tensor)
            else:
                _, _, mean_action = self.actor.sample(state_tensor)
                action_tensor = mean_action
                log_prob = torch.zeros(1, 1, device=self.device)

        self.actor.train()

        action = action_tensor.squeeze(0).cpu().numpy()
        action = action * self._action_scale

        self.logger.debug(
            f"select_action | state_norm={float(np.linalg.norm(state)):.4f} "
            f"action={action} log_prob={log_prob.item():.4f}"
        )
        return action

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Whether the episode ended.
        """
        # Scale action back to [-1, 1] before storage
        stored_action = action / self._action_scale
        self._replay_buffer.push(state, stored_action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Perform one SAC update step (critic, actor, temperature).

        Returns:
            Dictionary with keys: critic_loss, actor_loss, alpha_loss, alpha, mean_log_pi.
            Returns zeroed metrics if the buffer has insufficient data.
        """
        if len(self._replay_buffer) < self._replay_start:
            self.logger.debug(
                f"update skipped | buffer={len(self._replay_buffer)} < replay_start={self._replay_start}"
            )
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.log_alpha.exp().item(),
                "mean_log_pi": 0.0,
            }

        batch = self._replay_buffer.sample(self._batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        alpha = self.log_alpha.exp().detach()

        # ------------------------------------------------------------------ #
        # Critic update
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            # Soft Bellman target: r + gamma * (V_target - alpha * H_target)
            target_q = rewards + self._gamma * (min_q_target - alpha * next_log_pi) * (1.0 - dones)

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ #
        # Actor update
        # ------------------------------------------------------------------ #
        sampled_actions, log_pi, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # Maximise soft value: E[Q - alpha * log_pi]
        actor_loss = (alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------------ #
        # Temperature (alpha) update
        # ------------------------------------------------------------------ #
        if self._auto_entropy:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self._target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # ------------------------------------------------------------------ #
        # Soft update of critic target
        # ------------------------------------------------------------------ #
        self._soft_update_target()

        alpha_val = self.log_alpha.exp().item()
        mean_log_pi = log_pi.mean().item()

        self.logger.info(
            f"update | critic_loss={critic_loss.item():.4f} "
            f"actor_loss={actor_loss.item():.4f} "
            f"alpha_loss={alpha_loss.item():.4f}"
        )
        self.logger.debug(f"update | alpha={alpha_val:.4f} mean_log_pi={mean_log_pi:.4f}")

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha_val,
            "mean_log_pi": mean_log_pi,
        }

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return all named networks for checkpointing.

        Returns:
            Dictionary with keys: actor, critic, critic_target.
        """
        return {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
        }

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _soft_update_target(self) -> None:
        """Polyak-average the live critic weights into the target critic."""
        for target_param, live_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self._tau * live_param.data + (1.0 - self._tau) * target_param.data
            )
