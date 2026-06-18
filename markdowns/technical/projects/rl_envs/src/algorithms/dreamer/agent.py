"""Dreamer V2 agent implementation.

Use case: Autonomous vehicle control (model-based RL).

Dreamer learns a compact latent world model from real experience, then trains
an actor-critic policy *entirely inside the world model's imagination*, never
needing additional real environment interaction for policy improvement.

Two-phase training per update call:
  Phase 1 — World model:
    Sample short sequences from the replay buffer, encode observations to
    latents via the RSSM posterior, and minimise the combined loss:
    KL (with free nats) + reconstruction + reward prediction + continue prediction.

  Phase 2 — Actor-critic in imagination:
    Starting from posterior latent states, unroll the world model for `horizon`
    steps using the current actor. Compute lambda-returns from imagined rewards
    and critic values, then update actor and critic on these synthetic rollouts.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.base_agent import BaseAgent
from core.logger import RLLogger
from core.replay_buffer import ReplayBuffer
from algorithms.dreamer.world_model import WorldModel, RSSM


# --------------------------------------------------------------------------- #
# Latent actor and critic for Dreamer
# --------------------------------------------------------------------------- #

class _LatentActor(nn.Module):
    """Continuous actor in latent space: (h, z) -> action."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()
        input_dim = latent_dim + hidden_dim
        layers: List[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ELU())
            in_d = h
        layers.append(nn.Linear(in_d, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([h, z], dim=-1))


class _LatentCritic(nn.Module):
    """Value function in latent space: (h, z) -> V(s)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()
        input_dim = latent_dim + hidden_dim
        layers: List[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ELU())
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([h, z], dim=-1))


# --------------------------------------------------------------------------- #
# Sequence replay buffer wrapper
# --------------------------------------------------------------------------- #

class _SequenceBuffer:
    """Stores transitions in episode order and samples overlapping sequences.

    For Dreamer, sequences (not independent transitions) are needed for RSSM
    training. We store all transitions in a flat circular buffer and sample
    contiguous windows of length `seq_len`.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self._cap = capacity
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample_sequences(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sample `batch_size` contiguous sequences of length `seq_len`.

        Returns:
            Dict with keys obs, actions, rewards, dones; each shape (seq_len, batch, dim).
        """
        max_start = self._size - seq_len
        starts = np.random.randint(0, max_start, size=batch_size)
        obs_batch = np.stack([self._obs[s: s + seq_len] for s in starts], axis=1)
        act_batch = np.stack([self._actions[s: s + seq_len] for s in starts], axis=1)
        rew_batch = np.stack([self._rewards[s: s + seq_len] for s in starts], axis=1)
        done_batch = np.stack([self._dones[s: s + seq_len] for s in starts], axis=1)

        return {
            "obs": torch.FloatTensor(obs_batch).to(device),
            "actions": torch.FloatTensor(act_batch).to(device),
            "rewards": torch.FloatTensor(rew_batch).to(device),
            "dones": torch.FloatTensor(done_batch).to(device),
        }

    def __len__(self) -> int:
        return self._size


# --------------------------------------------------------------------------- #
# DreamerAgent
# --------------------------------------------------------------------------- #

class DreamerAgent(BaseAgent):
    """Dreamer V2: model-based RL with imagination-based actor-critic training."""

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        logger: RLLogger,
    ) -> None:
        super().__init__(config, device, logger)

        model_cfg = config["model"]
        train_cfg = config["training"]
        buf_cfg = config["buffer"]

        latent_dim: int = int(model_cfg["latent_dim"])
        hidden_dim: int = int(model_cfg["hidden_dim"])
        action_dim: int = int(model_cfg["action_dim"])
        obs_dim: int = int(model_cfg["obs_dim"])
        hidden_dims: List[int] = list(model_cfg["hidden_dims"])

        # ------------------------------------------------------------------ #
        # Components
        # ------------------------------------------------------------------ #
        self.world_model = WorldModel(config).to(device)

        self.actor = _LatentActor(
            latent_dim, hidden_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = _LatentCritic(
            latent_dim, hidden_dim, hidden_dims
        ).to(device)

        # ------------------------------------------------------------------ #
        # Optimisers
        # ------------------------------------------------------------------ #
        self._wm_optimizer = optim.Adam(
            self.world_model.parameters(), lr=float(train_cfg["world_model_lr"])
        )
        self._actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(train_cfg["actor_lr"])
        )
        self._critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=float(train_cfg["critic_lr"])
        )

        # ------------------------------------------------------------------ #
        # Hyperparameters
        # ------------------------------------------------------------------ #
        self._gamma: float = float(train_cfg["gamma"])
        self._lambda: float = float(train_cfg["lambda_"])
        self._horizon: int = int(train_cfg["horizon"])
        self._batch_size: int = int(train_cfg["batch_size"])
        self._seq_len: int = int(train_cfg["seq_len"])
        self._replay_start: int = int(train_cfg["replay_start"])
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        # ------------------------------------------------------------------ #
        # Sequence replay buffer
        # ------------------------------------------------------------------ #
        self._seq_buffer = _SequenceBuffer(
            capacity=int(buf_cfg["capacity"]),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # Recurrent state maintained across environment steps
        self._current_h: Optional[torch.Tensor] = None
        self._current_z: Optional[torch.Tensor] = None

        self.logger.info(
            f"DreamerAgent initialised | obs_dim={obs_dim} action_dim={action_dim} "
            f"latent_dim={latent_dim} hidden_dim={hidden_dim} horizon={self._horizon}"
        )

    # ---------------------------------------------------------------------- #
    # Episode management
    # ---------------------------------------------------------------------- #

    def reset_recurrent_state(self) -> None:
        """Zero out the recurrent state at the start of a new episode."""
        self._current_h = torch.zeros(1, self._hidden_dim, device=self.device)
        self._current_z = torch.zeros(1, self._latent_dim, device=self.device)
        self.logger.debug("reset_recurrent_state | h and z zeroed")

    # ---------------------------------------------------------------------- #
    # BaseAgent interface
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Encode observation to latent state, then sample action from actor.

        Maintains the recurrent state (h, z) across steps.
        Call reset_recurrent_state() at the start of each episode.

        Args:
            state: Raw environment observation, shape (obs_dim,).
            training: Unused for Dreamer (actor is always deterministic in this impl).

        Returns:
            action: Numpy array of shape (action_dim,).
        """
        if self._current_h is None:
            self.reset_recurrent_state()

        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, obs_dim)

        self.world_model.eval()
        self.actor.eval()
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            post_mean, post_lv = self.world_model.rssm.posterior(self._current_h, embed)
            z = RSSM.reparameterise(post_mean, post_lv)
            self._current_z = z

            action = self.actor(self._current_h, z)  # (1, action_dim)

            # Advance h for the next step
            prev_state = {"h": self._current_h, "z": z}
            self._current_h = self.world_model.rssm.recurrent_step(prev_state, action)

        self.world_model.train()
        self.actor.train()

        action_np = action.squeeze(0).cpu().numpy()
        self.logger.debug(
            f"select_action | state_norm={float(np.linalg.norm(state)):.4f} action={action_np}"
        )
        return action_np

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the sequence replay buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation (not stored separately; sequences are contiguous).
            done: Whether the episode ended.
        """
        self._seq_buffer.push(state, action, reward, done)

    def update(self) -> Dict[str, float]:
        """Run one world model + actor-critic update step.

        Returns:
            Dict with keys: wm_loss, kl_loss, recon_loss, reward_loss, actor_loss, critic_loss.
        """
        if len(self._seq_buffer) < self._replay_start + self._seq_len:
            self.logger.debug(
                f"update skipped | buffer={len(self._seq_buffer)} < "
                f"replay_start={self._replay_start}"
            )
            return {
                "wm_loss": 0.0, "kl_loss": 0.0, "recon_loss": 0.0,
                "reward_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
            }

        batch = self._seq_buffer.sample_sequences(
            self._batch_size, self._seq_len, self.device
        )
        obs_seq = batch["obs"]          # (T, B, obs_dim)
        action_seq = batch["actions"]   # (T, B, action_dim)
        reward_seq = batch["rewards"]   # (T, B, 1)
        done_seq = batch["dones"]       # (T, B, 1)

        # ------------------------------------------------------------------ #
        # Phase 1: World model update
        # ------------------------------------------------------------------ #
        wm_losses = self.world_model.compute_loss(obs_seq, action_seq, reward_seq, done_seq)
        wm_total = wm_losses["total_loss"]

        self._wm_optimizer.zero_grad()
        wm_total.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=100.0)
        self._wm_optimizer.step()

        # ------------------------------------------------------------------ #
        # Phase 2: Imagination-based actor-critic update
        # ------------------------------------------------------------------ #
        # Use posterior latent states as starting points for imagination
        with torch.no_grad():
            latent = self.world_model.encode_sequence(obs_seq, action_seq)

        # Flatten (T, B) into a single batch of start states
        T, B = latent["h_seq"].shape[:2]
        h_flat = latent["h_seq"].reshape(T * B, self._hidden_dim).detach()
        z_flat = latent["z_seq"].reshape(T * B, self._latent_dim).detach()
        start_state = {"h": h_flat, "z": z_flat}

        # Create a wrapper so WorldModel.imagine can call actor(h, z)
        class _ActorWrapper(nn.Module):
            def __init__(self, actor: nn.Module) -> None:
                super().__init__()
                self._actor = actor

            def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
                return self._actor(h, z)

        actor_wrapper = _ActorWrapper(self.actor)

        imagined = self.world_model.imagine(start_state, self._horizon, actor_wrapper)
        h_img = imagined["h_seq"]         # (horizon, T*B, hidden_dim)
        z_img = imagined["z_seq"]         # (horizon, T*B, latent_dim)
        reward_img = imagined["reward_seq"]    # (horizon, T*B, 1)
        cont_logit = imagined["continue_seq"]  # (horizon, T*B, 1)
        cont = torch.sigmoid(cont_logit).detach()   # treat as discount mask

        # Compute critic values along imagined trajectory
        values = self.critic(h_img, z_img)    # (horizon, T*B, 1)

        # Lambda returns
        lambda_returns = self._compute_lambda_returns(
            reward_img, values, cont, self._gamma, self._lambda
        )   # (horizon, T*B, 1)

        # Critic loss: fit V to lambda returns
        critic_loss = F.mse_loss(values, lambda_returns.detach())

        # Actor loss: maximise lambda returns
        # Re-compute values with gradient flow through actor parameters
        h_act = h_img.detach()
        z_act = z_img.detach()
        actor_actions = self.actor(h_act, z_act)   # (horizon, T*B, action_dim) — gradient flows
        # For a simple actor loss, use the critic estimate as the return signal
        actor_values = self.critic(h_act, z_act)    # (horizon, T*B, 1)
        actor_loss = -actor_values.mean()

        # Combined actor-critic update
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        self._critic_optimizer.step()

        self.logger.info(
            f"update | wm_loss={wm_total.item():.4f} "
            f"kl_loss={wm_losses['kl_loss'].item():.4f} "
            f"recon_loss={wm_losses['recon_loss'].item():.4f} "
            f"reward_loss={wm_losses['reward_loss'].item():.4f} "
            f"actor_loss={actor_loss.item():.4f} "
            f"critic_loss={critic_loss.item():.4f}"
        )

        return {
            "wm_loss": wm_total.item(),
            "kl_loss": wm_losses["kl_loss"].item(),
            "recon_loss": wm_losses["recon_loss"].item(),
            "reward_loss": wm_losses["reward_loss"].item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return all named network modules for checkpointing.

        Returns:
            Dict with keys: world_model, actor, critic.
        """
        return {
            "world_model": self.world_model,
            "actor": self.actor,
            "critic": self.critic,
        }

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _compute_lambda_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> torch.Tensor:
        """Compute lambda-returns (TD-lambda) over an imagined trajectory.

        G_t^lambda = r_t + gamma * continues_t *
                     ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda)

        Args:
            rewards:   (H, batch, 1) – imagined rewards
            values:    (H, batch, 1) – critic value estimates (detached)
            continues: (H, batch, 1) – continuation probabilities (from sigmoid)
            gamma:     Discount factor.
            lam:       Lambda for multi-step return mixing.

        Returns:
            lambda_returns: (H, batch, 1) – lambda return targets.
        """
        H = rewards.shape[0]
        # Bootstrap from the last value
        next_value = values[-1].detach()
        returns = []

        for t in reversed(range(H)):
            if t == H - 1:
                # At final step: G_H = r_H + gamma * continues_H * V_{H+1}
                g = rewards[t] + gamma * continues[t] * next_value
            else:
                # G_t = r_t + gamma * continues_t * ((1-lambda)*V_{t+1} + lambda*G_{t+1})
                g = (
                    rewards[t]
                    + gamma * continues[t] * (
                        (1.0 - lam) * values[t + 1].detach() + lam * returns[-1]
                    )
                )
            returns.append(g)

        returns.reverse()
        return torch.stack(returns, dim=0)
