"""
A2C (Advantage Actor-Critic) Agent.

Use case: Path planning in grid-world environments.

Algorithm overview:
    1. Collect a rollout of n_steps transitions using the current policy.
    2. Compute n-step returns and advantages: A_t = G_t - V(s_t).
    3. Update actor and critic jointly using the combined loss.
    4. Clear the rollout buffer and repeat.

Key properties:
    - On-policy: rollout data is discarded after each update.
    - Synchronous: single environment, single rollout per update.
    - Shared network backbone: reduces parameters, promotes feature reuse.
    - Entropy bonus: encourages exploration by penalising overconfident policies.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.base_agent import BaseAgent
from core.logger import RLLogger
from algorithms.a2c.network import ActorCriticNetwork


class A2CAgent(BaseAgent):
    """
    Synchronous Advantage Actor-Critic agent.

    Rollout storage uses Python lists that are cleared after every update.
    The agent exposes a ``store_step`` method so the training loop can push
    transitions without coupling itself to buffer internals.

    Config keys expected:
        network.input_dim       (int)   - state feature dimension
        network.action_dim      (int)   - number of discrete actions
        network.hidden_dims     (list)  - hidden layer sizes for the shared trunk
        training.learning_rate  (float) - Adam learning rate
        training.gamma          (float) - discount factor
        training.n_steps        (int)   - rollout length before an update
        training.entropy_coef   (float) - weight on entropy bonus (encourages exploration)
        training.value_loss_coef(float) - weight on critic MSE loss
        training.max_grad_norm  (float) - gradient clipping threshold
    """

    def __init__(self, config: Dict, device: torch.device, logger: RLLogger) -> None:
        super().__init__(config, device, logger)

        # ── Hyperparameters ──────────────────────────────────────────────────
        self._lr: float = config["training"]["learning_rate"]
        self._gamma: float = config["training"]["gamma"]
        self._n_steps: int = config["training"]["n_steps"]
        self._entropy_coef: float = config["training"]["entropy_coef"]
        self._value_loss_coef: float = config["training"]["value_loss_coef"]
        self._max_grad_norm: float = config["training"]["max_grad_norm"]

        input_dim: int = config["network"]["input_dim"]
        action_dim: int = config["network"]["action_dim"]
        hidden_dims: List[int] = config["network"]["hidden_dims"]

        # ── Network ──────────────────────────────────────────────────────────
        self.network = ActorCriticNetwork(input_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self._lr)

        # ── Rollout storage (cleared after each update) ───────────────────
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._values: List[torch.Tensor] = []
        self._log_probs: List[torch.Tensor] = []
        self._dones: List[bool] = []

        self.logger.info(
            f"A2CAgent initialised | lr={self._lr} gamma={self._gamma} "
            f"n_steps={self._n_steps} entropy_coef={self._entropy_coef} "
            f"value_loss_coef={self._value_loss_coef}"
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Sample an action from the current policy (training) or take the greedy action (eval).

        Side effects when called:
            Appends ``log_prob`` and ``value`` to the rollout buffer so that
            ``store_step`` only needs to add reward and done signal.

        Args:
            state:    State observation as a 1-D numpy array.
            training: If True, sample stochastically; if False, take argmax.

        Returns:
            Selected action index.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(False):
            logits, value = self.network(state_tensor)

        dist = Categorical(logits=logits)

        if training:
            action_tensor = dist.sample()
        else:
            action_tensor = logits.argmax(dim=-1)

        log_prob = dist.log_prob(action_tensor)
        action: int = action_tensor.item()

        # Detach value from graph; gradients will be recomputed in update()
        self._log_probs.append(log_prob.detach())
        self._values.append(value.detach().squeeze())

        self.logger.debug(
            f"select_action | state_norm={np.linalg.norm(state):.4f} "
            f"action={action} log_prob={log_prob.item():.4f} "
            f"value={value.item():.4f}"
        )
        return action

    # ── Rollout storage ───────────────────────────────────────────────────────

    def store_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Append a single transition to the rollout buffer.

        The log_prob and value for this step must already have been set by
        ``select_action`` before this method is called.

        Args:
            state:      State at this step.
            action:     Action taken.
            reward:     Scalar reward received.
            next_state: Resulting next state (used only for bootstrap value).
            done:       Whether the episode terminated after this step.
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
        Perform one A2C update over the accumulated rollout.

        Steps:
            1. Bootstrap the last value if the episode is not done.
            2. Compute discounted n-step returns G_t backwards.
            3. Compute advantages A_t = G_t - V(s_t).
            4. Compute actor loss, critic loss, and entropy bonus.
            5. Backpropagate and clip gradients.
            6. Clear the rollout buffer.

        Returns:
            Dict with keys: actor_loss, critic_loss, entropy, mean_advantage.
        """
        if len(self._rewards) == 0:
            self.logger.warning("update() called with empty rollout buffer.")
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "mean_advantage": 0.0}

        T = len(self._rewards)

        # ── Re-compute logits and values with gradients enabled ───────────
        states_tensor = torch.FloatTensor(np.array(self._states)).to(self.device)
        actions_tensor = torch.LongTensor(self._actions).to(self.device)

        logits, values = self.network(states_tensor)          # [T, action_dim], [T, 1]
        values = values.squeeze(-1)                            # [T]
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)              # [T]
        entropy = dist.entropy().mean()                        # scalar

        # ── Bootstrap final value ─────────────────────────────────────────
        # If the last step was not terminal, bootstrap with V(s_T).
        if not self._dones[-1]:
            last_state = torch.FloatTensor(self._states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value = self.network(last_state)
            bootstrap_value: float = last_value.item()
        else:
            bootstrap_value = 0.0

        # ── Compute n-step returns backwards ─────────────────────────────
        returns = torch.zeros(T, device=self.device)
        G = bootstrap_value
        for t in reversed(range(T)):
            G = self._rewards[t] + self._gamma * G * (1.0 - float(self._dones[t]))
            returns[t] = G

        # ── Advantages ───────────────────────────────────────────────────
        advantages = returns - values.detach()
        mean_advantage: float = advantages.mean().item()

        # Warn if advantages contain NaN or are extremely large
        if torch.isnan(advantages).any():
            self.logger.warning("NaN detected in advantages — skipping update.")
            self._clear_rollout()
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "mean_advantage": 0.0}

        # ── Losses ───────────────────────────────────────────────────────
        # Actor loss: policy gradient with advantage baseline, minus entropy bonus
        actor_loss = -(log_probs * advantages.detach()).mean() - self._entropy_coef * entropy

        # Critic loss: mean-squared error between predicted value and target return
        critic_loss = self._value_loss_coef * nn.functional.mse_loss(values, returns.detach())

        total_loss = actor_loss + critic_loss

        # ── Optimisation step ─────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self._max_grad_norm)
        if grad_norm > self._max_grad_norm * 2:
            self.logger.warning(f"Large gradient norm before clipping: {grad_norm:.4f}")

        self.optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        actor_loss_val: float = actor_loss.item()
        critic_loss_val: float = critic_loss.item()
        entropy_val: float = entropy.item()

        self.logger.info(
            f"A2C update | actor_loss={actor_loss_val:.4f} "
            f"critic_loss={critic_loss_val:.4f} "
            f"entropy={entropy_val:.4f} "
            f"mean_advantage={mean_advantage:.4f} "
            f"rollout_len={T}"
        )

        self._clear_rollout()

        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss_val,
            "entropy": entropy_val,
            "mean_advantage": mean_advantage,
        }

    # ── Readiness check ───────────────────────────────────────────────────────

    def ready_to_update(self) -> bool:
        """
        Return True when the rollout buffer has enough data for an update.

        An update is triggered either when the buffer holds n_steps transitions
        or when the last stored transition ended the episode (done=True).
        """
        if len(self._rewards) >= self._n_steps:
            return True
        if self._dones and self._dones[-1]:
            return True
        return False

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
