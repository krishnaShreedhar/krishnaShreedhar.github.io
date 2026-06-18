"""
Double DQN agent.

Use case: improved path planning with reduced Q-value overestimation.
Reference: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2015.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from algorithms.dqn.agent import DQNAgent
from core.logger import RLLogger


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN — overrides only ``update()`` to decouple action selection
    from action evaluation.

    Vanilla DQN target:
        r + gamma * max_a Q_target(s', a)          <- same net selects AND evaluates

    Double DQN target:
        r + gamma * Q_target(s', argmax_a Q_online(s', a))
                               ^-- online net selects    ^-- target net evaluates

    This prevents the systematic overestimation bias caused by the max operator.
    """

    def update(self) -> Dict[str, float]:
        """
        Double DQN update.

        Returns {"loss": float, "epsilon": float, "mean_q": float}.
        """
        if len(self.buffer) < self.replay_start:
            return {"loss": 0.0, "epsilon": self.epsilon, "mean_q": 0.0}

        batch = self.buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"].long()    # [B, 1]
        rewards = batch["rewards"]           # [B, 1]
        next_states = batch["next_states"]
        dones = batch["dones"]               # [B, 1]

        # Current Q-values for taken actions
        q_values = self.q_net(states).gather(1, actions)  # [B, 1]

        with torch.no_grad():
            # --- Double DQN: use online net to SELECT the best action ---
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)  # [B, 1]

            # --- Use target net to EVALUATE that action ---
            double_dqn_q = self.q_target(next_states).gather(1, best_actions)   # [B, 1]

            # --- Vanilla DQN would use: self.q_target(next_states).max(1, keepdim=True).values ---
            vanilla_q = self.q_target(next_states).max(dim=1, keepdim=True).values
            bias_estimate = float((vanilla_q - double_dqn_q).mean().item())

            self.logger.debug(
                "double_dqn_update: mean_bias_vs_vanilla=%.6f (positive = DQN overestimates)",
                bias_estimate,
            )

            targets = rewards + self.gamma * double_dqn_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard update of target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
            self.logger.info(
                "Target network updated (Double DQN). step=%d", self._step_count
            )

        self._decay_epsilon()

        mean_q = float(q_values.mean().item())
        loss_val = float(loss.item())

        self.logger.debug(
            "double_dqn_update: step=%d loss=%.6f epsilon=%.4f mean_q=%.4f",
            self._step_count, loss_val, self.epsilon, mean_q,
        )
        return {"loss": loss_val, "epsilon": self.epsilon, "mean_q": mean_q}
