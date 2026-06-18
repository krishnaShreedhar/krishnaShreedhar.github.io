"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent.

Use case: Autonomous vehicle control with continuous action spaces.

Three key improvements over DDPG:
    1. Twin critics:        Two Q-networks; Bellman target uses min(Q1, Q2).
                            Eliminates positive Q-value overestimation bias.
    2. Delayed policy update: Actor and target networks updated every policy_delay
                              critic steps. Ensures the value function is more
                              accurate before the policy is updated.
    3. Target policy smoothing: Gaussian noise is added to the target action,
                              then clipped.  This regularises the Q-function
                              by preventing the actor from exploiting narrow
                              Q-function peaks.

Reference: Fujimoto et al. (2018) "Addressing Function Approximation Error in
           Actor-Critic Methods"
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.replay_buffer import ReplayBuffer
from core.logger import RLLogger
from algorithms.td3.network import TD3ActorNetwork, TD3CriticNetwork


class TD3Agent(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient agent.

    Reimplements the full DDPG pattern (not subclassed from DDPGAgent) with
    the three TD3 improvements applied.

    Config keys expected:
        network.state_dim       (int)   - state feature dimension
        network.action_dim      (int)   - action vector dimension
        network.hidden_dims     (list)  - hidden layer sizes
        network.action_scale    (float) - tanh output scale (max action magnitude)
        training.actor_lr       (float) - learning rate for the actor
        training.critic_lr      (float) - learning rate for the twin critic
        training.gamma          (float) - discount factor
        training.tau            (float) - soft update coefficient
        training.batch_size     (int)   - mini-batch size
        training.replay_start   (int)   - minimum buffer size before updates begin
        training.policy_noise   (float) - std of smoothing noise for target action
        training.noise_clip     (float) - clip range for smoothing noise
        training.policy_delay   (int)   - critic updates per actor update
        buffer.capacity         (int)   - replay buffer size
    """

    def __init__(self, config: Dict, device: torch.device, logger: RLLogger) -> None:
        super().__init__(config, device, logger)

        # ── Hyperparameters ──────────────────────────────────────────────────
        self._actor_lr: float = config["training"]["actor_lr"]
        self._critic_lr: float = config["training"]["critic_lr"]
        self._gamma: float = config["training"]["gamma"]
        self._tau: float = config["training"]["tau"]
        self._batch_size: int = config["training"]["batch_size"]
        self._replay_start: int = config["training"]["replay_start"]
        self._policy_noise: float = config["training"]["policy_noise"]
        self._noise_clip: float = config["training"]["noise_clip"]
        self._policy_delay: int = config["training"]["policy_delay"]

        state_dim: int = config["network"]["state_dim"]
        action_dim: int = config["network"]["action_dim"]
        hidden_dims: List[int] = config["network"]["hidden_dims"]
        self._action_scale: float = config["network"].get("action_scale", 1.0)
        buffer_capacity: int = config["buffer"]["capacity"]

        self._state_dim = state_dim
        self._action_dim = action_dim

        # Internal step counter: tracks how many critic updates have occurred
        self._update_step: int = 0

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor = TD3ActorNetwork(state_dim, action_dim, hidden_dims, self._action_scale).to(device)
        self.actor_target = TD3ActorNetwork(state_dim, action_dim, hidden_dims, self._action_scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3CriticNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = TD3CriticNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks are never directly trained
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._critic_lr)

        # ── Replay buffer ─────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        self.logger.info(
            f"TD3Agent initialised | actor_lr={self._actor_lr} critic_lr={self._critic_lr} "
            f"gamma={self._gamma} tau={self._tau} policy_delay={self._policy_delay} "
            f"policy_noise={self._policy_noise} noise_clip={self._noise_clip} "
            f"action_scale={self._action_scale} buffer_capacity={buffer_capacity}"
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute the deterministic actor output.

        In TD3, exploration noise is typically added externally by the training
        loop (Gaussian noise), not inside the agent, unlike DDPG's OU noise.
        This keeps the agent interface clean and separates concerns.

        Args:
            state:    State observation as a 1-D numpy array.
            training: Flag passed for interface compatibility (no internal noise added).

        Returns:
            Clipped continuous action of shape [action_dim].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
        self.actor.train()

        action: np.ndarray = action_tensor.cpu().numpy().squeeze()
        action = np.clip(action, -self._action_scale, self._action_scale)

        self.logger.debug(
            f"select_action | state_norm={np.linalg.norm(state):.4f} action={action}"
        )
        return action

    # ── Replay buffer push ────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.

        Args:
            state:      Current state.
            action:     Continuous action taken (may include external exploration noise).
            reward:     Scalar reward.
            next_state: Resulting next state.
            done:       Episode termination flag.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.logger.debug(
            f"push | buffer_size={len(self.replay_buffer)} reward={reward:.4f} done={done}"
        )

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Perform one TD3 update step.

        Each call increments the internal step counter.
        The critic is always updated; the actor and target networks are only
        updated every ``policy_delay`` critic steps.

        Returns:
            Dict with keys: critic_loss, actor_loss, mean_q1, policy_delay_step.
            ``actor_loss`` is 0.0 when the policy update is skipped.
        """
        if len(self.replay_buffer) < self._replay_start:
            self.logger.debug(
                f"update skipped: buffer={len(self.replay_buffer)} < "
                f"replay_start={self._replay_start}"
            )
            return {"critic_loss": 0.0, "actor_loss": 0.0, "mean_q1": 0.0, "policy_delay_step": 0}

        self._update_step += 1
        batch = self.replay_buffer.sample(self._batch_size)
        states = batch["states"]            # [B, state_dim]
        actions = batch["actions"]          # [B, action_dim]
        rewards = batch["rewards"]          # [B, 1]
        next_states = batch["next_states"]  # [B, state_dim]
        dones = batch["dones"]              # [B, 1]

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target action
            noise = torch.randn_like(actions) * self._policy_noise
            noise = noise.clamp(-self._noise_clip, self._noise_clip)

            smoothed_target_action = (
                self.actor_target(next_states) + noise
            ).clamp(-self._action_scale, self._action_scale)

            # Twin critics: use the minimum to compute the Bellman target
            q1_t, q2_t = self.critic_target(next_states, smoothed_target_action)
            target_q = rewards + self._gamma * torch.min(q1_t, q2_t) * (1.0 - dones)

        q1, q2 = self.critic(states, actions)
        # Sum of MSE losses for both Q-networks
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.critic.parameters()
            if p.grad is not None
        ) ** 0.5
        self.logger.debug(f"critic grad_norm={critic_grad_norm:.4f}")
        self.critic_optimizer.step()

        # ── Delayed policy update ─────────────────────────────────────────
        actor_loss_val = 0.0
        if self._update_step % self._policy_delay == 0:
            # Actor loss: maximise Q1(s, pi(s)) — use only Q1 to avoid correlation
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            actor_loss_val = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.actor.parameters()
                if p.grad is not None
            ) ** 0.5
            self.logger.debug(f"actor grad_norm={actor_grad_norm:.4f}")
            self.actor_optimizer.step()

            # Soft update both target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            self.logger.info(
                f"TD3 update (step={self._update_step}) | "
                f"critic_loss={critic_loss.item():.4f} "
                f"actor_loss={actor_loss_val:.4f} "
                f"mean_q1={q1.mean().item():.4f}"
            )
        else:
            self.logger.debug(
                f"skipped_policy_update | update_step={self._update_step} "
                f"policy_delay={self._policy_delay}"
            )
            self.logger.info(
                f"TD3 update (step={self._update_step}) | "
                f"critic_loss={critic_loss.item():.4f} "
                f"actor_update=skipped"
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
            "mean_q1": q1.mean().item(),
            "policy_delay_step": self._update_step,
        }

    # ── Soft update helper ────────────────────────────────────────────────────

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        """Polyak averaging: target = tau*online + (1-tau)*target."""
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(
                self._tau * online_param.data + (1.0 - self._tau) * target_param.data
            )

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return all four networks for checkpointing."""
        return {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
        }
