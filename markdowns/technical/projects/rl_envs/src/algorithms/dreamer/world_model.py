"""Dreamer V2 world model components.

Architecture overview:
  ObsEncoder  -> encodes raw observations to embedding vectors
  RSSM        -> Recurrent State Space Model:
                   * h_t (deterministic): GRU hidden state
                   * z_t (stochastic):    Gaussian latent sampled from prior/posterior
  ObsDecoder  -> reconstructs observations from (h, z)
  RewardPredictor   -> predicts reward from (h, z)
  ContinuePredictor -> predicts episode continuation (binary) from (h, z)
  WorldModel  -> assembles all components and exposes training/imagination APIs
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def _build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _gaussian_kl(
    mean1: torch.Tensor, log_var1: torch.Tensor,
    mean2: torch.Tensor, log_var2: torch.Tensor,
) -> torch.Tensor:
    """KL divergence KL(N1 || N2) element-wise, then summed over the latent dim."""
    var1 = log_var1.exp()
    var2 = log_var2.exp()
    kl = 0.5 * (
        log_var2 - log_var1
        + (var1 + (mean1 - mean2).pow(2)) / (var2 + 1e-8)
        - 1.0
    )
    return kl.sum(dim=-1)  # (batch,) or (T, batch)


# --------------------------------------------------------------------------- #
# RSSM
# --------------------------------------------------------------------------- #

class RSSM(nn.Module):
    """Recurrent State Space Model.

    State representation:
      h_t  – deterministic part: hidden state of a GRU cell
      z_t  – stochastic part: Gaussian latent (prior or posterior)

    Transitions:
      Prior:     p(z_t | h_t)           -- used during imagination
      Posterior: q(z_t | h_t, embed_t)  -- used during training (with real obs)
      Recurrent: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    """

    def __init__(
        self,
        obs_embed_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        """Initialise the RSSM.

        Args:
            obs_embed_dim: Dimensionality of the encoded observation embedding.
            action_dim: Dimensionality of the action vector.
            latent_dim: Dimensionality of the stochastic latent z.
            hidden_dim: Dimensionality of the GRU hidden state h.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input to GRU: concatenation of previous z and previous action
        gru_input_dim = latent_dim + action_dim
        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_dim)

        # Prior: h -> (mean, log_var) of z
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean + log_var
        )

        # Posterior: (h, obs_embed) -> (mean, log_var) of z
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the prior distribution p(z | h).

        Args:
            h: Deterministic hidden state, shape (..., hidden_dim).

        Returns:
            mean: Prior mean, shape (..., latent_dim).
            log_var: Prior log variance (clamped), shape (..., latent_dim).
        """
        out = self.prior_net(h)
        mean, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 4.0)
        return mean, log_var

    def posterior(
        self, h: torch.Tensor, obs_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior distribution q(z | h, obs_embed).

        Args:
            h: Deterministic hidden state, shape (..., hidden_dim).
            obs_embed: Encoded observation embedding, shape (..., obs_embed_dim).

        Returns:
            mean: Posterior mean, shape (..., latent_dim).
            log_var: Posterior log variance (clamped), shape (..., latent_dim).
        """
        inp = torch.cat([h, obs_embed], dim=-1)
        out = self.posterior_net(inp)
        mean, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 4.0)
        return mean, log_var

    def recurrent_step(
        self, prev_state: Dict[str, torch.Tensor], prev_action: torch.Tensor
    ) -> torch.Tensor:
        """Advance the GRU by one step.

        Args:
            prev_state: Dict with keys "h" (hidden_dim,) and "z" (latent_dim,).
                        Batch dimensions are leading.
            prev_action: Previous action, shape (batch, action_dim).

        Returns:
            h_t: New GRU hidden state, shape (batch, hidden_dim).
        """
        z = prev_state["z"]
        h = prev_state["h"]
        gru_inp = torch.cat([z, prev_action], dim=-1)
        h_t = self.gru_cell(gru_inp, h)
        return h_t

    @staticmethod
    def reparameterise(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick."""
        std = (log_var * 0.5).exp()
        eps = torch.randn_like(std)
        return mean + std * eps

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return a zero initial state dict."""
        return {
            "h": torch.zeros(batch_size, self.hidden_dim, device=device),
            "z": torch.zeros(batch_size, self.latent_dim, device=device),
        }


# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #

class ObsEncoder(nn.Module):
    """Encodes raw observations to a fixed-size embedding vector."""

    def __init__(self, obs_dim: int, embed_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.network = _build_mlp(obs_dim, hidden_dims, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation.

        Args:
            obs: Raw observation tensor, shape (..., obs_dim).

        Returns:
            embed: Embedding tensor, shape (..., embed_dim).
        """
        return self.network(obs)


class ObsDecoder(nn.Module):
    """Decodes a latent state (h, z) back to observation space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        obs_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()
        input_dim = latent_dim + hidden_dim
        self.network = _build_mlp(input_dim, hidden_dims, obs_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct observation from latent state.

        Args:
            h: Deterministic hidden state, shape (..., hidden_dim).
            z: Stochastic latent, shape (..., latent_dim).

        Returns:
            obs_recon: Reconstructed observation, shape (..., obs_dim).
        """
        return self.network(torch.cat([h, z], dim=-1))


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #

class RewardPredictor(nn.Module):
    """Predicts the expected reward from a latent state (h, z)."""

    def __init__(
        self, latent_dim: int, hidden_dim: int, hidden_dims: List[int]
    ) -> None:
        super().__init__()
        input_dim = latent_dim + hidden_dim
        self.network = _build_mlp(input_dim, hidden_dims, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward.

        Args:
            h: Deterministic hidden state, shape (..., hidden_dim).
            z: Stochastic latent, shape (..., latent_dim).

        Returns:
            reward: Predicted reward, shape (..., 1).
        """
        return self.network(torch.cat([h, z], dim=-1))


class ContinuePredictor(nn.Module):
    """Predicts a binary continue / done signal from a latent state (h, z).

    Outputs a logit (not a probability). Use BCEWithLogitsLoss during training
    and sigmoid when sampling.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = latent_dim + hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict continuation logit.

        Args:
            h: Deterministic hidden state, shape (..., hidden_dim).
            z: Stochastic latent, shape (..., latent_dim).

        Returns:
            logit: Continue logit, shape (..., 1). Positive = continue, negative = done.
        """
        return self.network(torch.cat([h, z], dim=-1))


# --------------------------------------------------------------------------- #
# WorldModel
# --------------------------------------------------------------------------- #

class WorldModel(nn.Module):
    """Dreamer V2 world model.

    Combines all sub-components into a single trainable module:
      encoder, RSSM, decoder, reward predictor, continue predictor.

    Training loss (compute_loss):
      L_total = kl_scale * max(KL - free_nats, 0)
                + L_recon  (MSE on observations)
                + L_reward (MSE on rewards)
                + L_continue (BCE on done flags)

    Imagination (imagine):
      Unrolls the RSSM forward for `horizon` steps using an external actor,
      collecting predicted (h, z, reward, continue) for actor-critic training.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        train_cfg = config["training"]

        obs_dim: int = int(model_cfg["obs_dim"])
        action_dim: int = int(model_cfg["action_dim"])
        latent_dim: int = int(model_cfg["latent_dim"])
        hidden_dim: int = int(model_cfg["hidden_dim"])
        embed_dim: int = int(model_cfg["embed_dim"])
        hidden_dims: List[int] = list(model_cfg["hidden_dims"])

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self._kl_scale: float = float(train_cfg.get("kl_scale", 1.0))
        self._free_nats: float = float(train_cfg.get("free_nats", 3.0))

        self.encoder = ObsEncoder(obs_dim, embed_dim, hidden_dims)
        self.rssm = RSSM(embed_dim, action_dim, latent_dim, hidden_dim)
        self.decoder = ObsDecoder(latent_dim, hidden_dim, obs_dim, hidden_dims)
        self.reward_predictor = RewardPredictor(latent_dim, hidden_dim, hidden_dims)
        self.continue_predictor = ContinuePredictor(latent_dim, hidden_dim)

    # ---------------------------------------------------------------------- #

    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode a sequence of observations and actions into latent trajectories.

        Args:
            obs_seq: Observations, shape (T, batch, obs_dim).
            action_seq: Actions, shape (T, batch, action_dim).
                        action_seq[t] is the action *before* obs_seq[t+1].
                        Use zero-padding for the action at t=0.

        Returns:
            Dict with keys:
              "h_seq":    shape (T, batch, hidden_dim)
              "z_seq":    shape (T, batch, latent_dim)
              "post_mean": shape (T, batch, latent_dim)
              "post_lv":   shape (T, batch, latent_dim)
              "prior_mean":shape (T, batch, latent_dim)
              "prior_lv":  shape (T, batch, latent_dim)
        """
        T, B, _ = obs_seq.shape
        device = obs_seq.device

        state = self.rssm.initial_state(B, device)
        embeds = self.encoder(obs_seq)   # (T, batch, embed_dim)

        h_list, z_list = [], []
        post_mean_list, post_lv_list = [], []
        prior_mean_list, prior_lv_list = [], []

        for t in range(T):
            # Prior: p(z_t | h_t) — uses h already updated by recurrent_step
            prior_mean, prior_lv = self.rssm.prior(state["h"])
            # Posterior: q(z_t | h_t, embed_t) — conditions on observed embedding
            post_mean, post_lv = self.rssm.posterior(state["h"], embeds[t])
            # Sample z from posterior (for training)
            z_t = RSSM.reparameterise(post_mean, post_lv)

            h_list.append(state["h"])
            z_list.append(z_t)
            post_mean_list.append(post_mean)
            post_lv_list.append(post_lv)
            prior_mean_list.append(prior_mean)
            prior_lv_list.append(prior_lv)

            # Advance recurrent state using current z and the action taken at t
            state = {"h": state["h"], "z": z_t}
            state["h"] = self.rssm.recurrent_step(state, action_seq[t])

        return {
            "h_seq": torch.stack(h_list, dim=0),
            "z_seq": torch.stack(z_list, dim=0),
            "post_mean": torch.stack(post_mean_list, dim=0),
            "post_lv": torch.stack(post_lv_list, dim=0),
            "prior_mean": torch.stack(prior_mean_list, dim=0),
            "prior_lv": torch.stack(prior_lv_list, dim=0),
        }

    # ---------------------------------------------------------------------- #

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        done_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the Dreamer world model training loss.

        Args:
            obs_seq:    (T, batch, obs_dim)  – raw observations
            action_seq: (T, batch, action_dim) – actions taken
            reward_seq: (T, batch, 1)  – observed rewards
            done_seq:   (T, batch, 1)  – 1.0 if episode ended, 0.0 otherwise

        Returns:
            Dict with keys: total_loss, kl_loss, recon_loss, reward_loss, continue_loss.
            All are scalar tensors.
        """
        latent = self.encode_sequence(obs_seq, action_seq)
        h = latent["h_seq"]   # (T, B, hidden_dim)
        z = latent["z_seq"]   # (T, B, latent_dim)

        # ---- Reconstruction loss ----------------------------------------- #
        obs_pred = self.decoder(h, z)       # (T, B, obs_dim)
        recon_loss = F.mse_loss(obs_pred, obs_seq)

        # ---- Reward prediction loss --------------------------------------- #
        reward_pred = self.reward_predictor(h, z)   # (T, B, 1)
        reward_loss = F.mse_loss(reward_pred, reward_seq)

        # ---- Continue prediction loss ------------------------------------- #
        continue_logit = self.continue_predictor(h, z)   # (T, B, 1)
        # continue label = 1 - done
        continue_label = 1.0 - done_seq
        continue_loss = F.binary_cross_entropy_with_logits(continue_logit, continue_label)

        # ---- KL loss with free nats --------------------------------------- #
        kl = _gaussian_kl(
            latent["post_mean"], latent["post_lv"],
            latent["prior_mean"], latent["prior_lv"],
        )   # (T, B)
        # Free nats: do not penalise KL below free_nats threshold
        kl_loss = torch.clamp(kl.mean(), min=self._free_nats) * self._kl_scale

        total_loss = kl_loss + recon_loss + reward_loss + continue_loss

        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
        }

    # ---------------------------------------------------------------------- #

    def imagine(
        self,
        start_state: Dict[str, torch.Tensor],
        horizon: int,
        actor: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Unroll the world model for `horizon` steps using the current actor.

        All operations are in latent space — no real environment interaction.

        Args:
            start_state: Dict {"h": (batch, hidden_dim), "z": (batch, latent_dim)}.
            horizon: Number of imagination steps.
            actor: Policy network; call signature actor(h, z) -> action_tensor.

        Returns:
            Dict with keys:
              "h_seq":       (horizon, batch, hidden_dim)
              "z_seq":       (horizon, batch, latent_dim)
              "reward_seq":  (horizon, batch, 1)
              "continue_seq":(horizon, batch, 1)  logits
        """
        state = {k: v.detach() for k, v in start_state.items()}

        h_list, z_list, rew_list, cont_list = [], [], [], []

        for _ in range(horizon):
            h, z = state["h"], state["z"]

            # Actor decides action in latent space
            with torch.no_grad():
                action = actor(h, z)   # (batch, action_dim)

            # Predict reward and continue from current latent state
            reward = self.reward_predictor(h, z)          # (batch, 1)
            continue_logit = self.continue_predictor(h, z)  # (batch, 1)

            h_list.append(h)
            z_list.append(z)
            rew_list.append(reward)
            cont_list.append(continue_logit)

            # Advance recurrent state
            h_next = self.rssm.recurrent_step(state, action)
            prior_mean, prior_lv = self.rssm.prior(h_next)
            z_next = RSSM.reparameterise(prior_mean, prior_lv)

            state = {"h": h_next, "z": z_next}

        return {
            "h_seq": torch.stack(h_list, dim=0),
            "z_seq": torch.stack(z_list, dim=0),
            "reward_seq": torch.stack(rew_list, dim=0),
            "continue_seq": torch.stack(cont_list, dim=0),
        }
