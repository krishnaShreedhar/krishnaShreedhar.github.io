"""
DDPG (Deep Deterministic Policy Gradient) Agent.

Use case: Autonomous vehicle control with continuous action spaces.

Algorithm overview:
    - Off-policy: stores all transitions in a replay buffer.
    - Deterministic policy: actor outputs a single continuous action (no sampling).
    - Ornstein-Uhlenbeck (OU) noise is added during training for temporally
      correlated exploration, mimicking inertial systems (vehicles, robot arms).
    - Target networks: slowly-updated copies of actor and critic that provide
      stable TD targets, preventing oscillations during training.
    - Soft update: theta_target = tau*theta + (1-tau)*theta_target (tau << 1).

Reference: Lillicrap et al. (2016) "Continuous control with deep reinforcement learning"
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.replay_buffer import ReplayBuffer
from core.logger import RLLogger
from algorithms.ddpg.network import ActorNetwork, CriticNetwork


class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient agent for continuous action spaces.

    Config keys expected:
        network.state_dim       (int)   - state feature dimension
        network.action_dim      (int)   - action vector dimension
        network.hidden_dims     (list)  - hidden layer sizes
        network.action_scale    (float) - tanh output scale (max action magnitude)
        training.actor_lr       (float) - learning rate for the actor
        training.critic_lr      (float) - learning rate for the critic
        training.gamma          (float) - discount factor
        training.tau            (float) - soft update coefficient (e.g. 0.005)
        training.batch_size     (int)   - mini-batch size for updates
        training.replay_start   (int)   - minimum buffer size before updates begin
        training.policy_noise   (float) - OU noise sigma (exploration scale)
        buffer.capacity         (int)   - maximum replay buffer size
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

        state_dim: int = config["network"]["state_dim"]
        action_dim: int = config["network"]["action_dim"]
        hidden_dims: List[int] = config["network"]["hidden_dims"]
        self._action_scale: float = config["network"].get("action_scale", 1.0)
        buffer_capacity: int = config["buffer"]["capacity"]

        self._state_dim = state_dim
        self._action_dim = action_dim

        # ── Networks (online + target copies) ───────────────────────────────
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims, self._action_scale).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dims, self._action_scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks are never directly optimised
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

        # ── OU noise state ────────────────────────────────────────────────
        self._ou_theta: float = 0.15          # mean-reversion speed
        self._ou_mu: np.ndarray = np.zeros(action_dim)
        self._ou_sigma: float = self._policy_noise
        self._ou_state: np.ndarray = np.zeros(action_dim)
        self._init_ou_noise()

        self.logger.info(
            f"DDPGAgent initialised | actor_lr={self._actor_lr} critic_lr={self._critic_lr} "
            f"gamma={self._gamma} tau={self._tau} batch_size={self._batch_size} "
            f"replay_start={self._replay_start} action_scale={self._action_scale} "
            f"buffer_capacity={buffer_capacity}"
        )

    # ── OU Noise ──────────────────────────────────────────────────────────────

    def _init_ou_noise(self) -> None:
        """Reset OU noise to the zero state."""
        self._ou_state = np.copy(self._ou_mu)
        self.logger.debug("OU noise initialised / reset.")

    def _sample_ou_noise(self) -> np.ndarray:
        """
        Advance and sample from the Ornstein-Uhlenbeck process.

        dx = theta*(mu - x)*dt + sigma*dW
        With dt=1 and dW ~ N(0,1).
        """
        noise = self._ou_state.copy()
        dx = (
            self._ou_theta * (self._ou_mu - self._ou_state)
            + self._ou_sigma * np.random.randn(self._action_dim)
        )
        self._ou_state = self._ou_state + dx
        return noise

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Choose an action using the deterministic actor, plus OU noise during training.

        Args:
            state:    State observation as a 1-D numpy array.
            training: If True, add OU exploration noise; if False, pure policy output.

        Returns:
            Clipped continuous action of shape [action_dim].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
        self.actor.train()

        action: np.ndarray = action_tensor.cpu().numpy().squeeze()

        if training:
            noise = self._sample_ou_noise()
            action = action + noise
            self.logger.debug(
                f"select_action | state_norm={np.linalg.norm(state):.4f} "
                f"raw_action={action - noise} noise_norm={np.linalg.norm(noise):.4f} "
                f"noisy_action={action}"
            )
        else:
            self.logger.debug(
                f"select_action (eval) | state_norm={np.linalg.norm(state):.4f} "
                f"action={action}"
            )

        action = np.clip(action, -self._action_scale, self._action_scale)
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
            action:     Continuous action taken.
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
        Perform one DDPG update step (critic then actor, then soft-update targets).

        Returns early if the replay buffer has fewer transitions than replay_start.

        Returns:
            Dict with keys: critic_loss, actor_loss, mean_q, noise_scale.
        """
        if len(self.replay_buffer) < self._replay_start:
            self.logger.debug(
                f"update skipped: buffer={len(self.replay_buffer)} < "
                f"replay_start={self._replay_start}"
            )
            return {"critic_loss": 0.0, "actor_loss": 0.0, "mean_q": 0.0, "noise_scale": self._ou_sigma}

        batch = self.replay_buffer.sample(self._batch_size)
        states = batch["states"]           # [B, state_dim]
        actions = batch["actions"]         # [B, action_dim]
        rewards = batch["rewards"]         # [B, 1]
        next_states = batch["next_states"] # [B, state_dim]
        dones = batch["dones"]             # [B, 1]

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = rewards + self._gamma * self.critic_target(next_states, target_actions) * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Log gradient norm before clipping for debugging
        critic_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.critic.parameters()
            if p.grad is not None
        ) ** 0.5
        self.logger.debug(f"critic grad_norm={critic_grad_norm:.4f}")
        self.critic_optimizer.step()

        # ── Actor update ──────────────────────────────────────────────────
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.actor.parameters()
            if p.grad is not None
        ) ** 0.5
        self.logger.debug(f"actor grad_norm={actor_grad_norm:.4f}")
        self.actor_optimizer.step()

        # ── Soft update targets ───────────────────────────────────────────
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # ── Metrics ───────────────────────────────────────────────────────
        mean_q: float = current_q.mean().item()
        critic_loss_val: float = critic_loss.item()
        actor_loss_val: float = actor_loss.item()

        self.logger.info(
            f"DDPG update | critic_loss={critic_loss_val:.4f} "
            f"actor_loss={actor_loss_val:.4f} "
            f"mean_q={mean_q:.4f} "
            f"noise_scale={self._ou_sigma:.4f}"
        )

        return {
            "critic_loss": critic_loss_val,
            "actor_loss": actor_loss_val,
            "mean_q": mean_q,
            "noise_scale": self._ou_sigma,
        }

    # ── Soft update helper ────────────────────────────────────────────────────

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        """
        Polyak averaging: target = tau*online + (1-tau)*target.

        Args:
            online: The network being trained.
            target: The slowly-updated target network.
        """
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
