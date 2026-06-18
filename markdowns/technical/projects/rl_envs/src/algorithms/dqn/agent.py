"""
Deep Q-Network (DQN) agent.

Use case: path planning with continuous state and discrete action space.
Reference: Mnih et al., "Human-level control through deep reinforcement learning", 2015.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.logger import RLLogger
from core.replay_buffer import ReplayBuffer
from algorithms.dqn.network import QNetwork


class DQNAgent(BaseAgent):
    """
    DQN with experience replay and a periodically-updated target network.

    Key innovations over tabular Q-Learning:
    1. Experience replay: breaks temporal correlations in the data stream.
    2. Target network: stabilises the bootstrapped TD target.

    Config keys consumed:
        network.input_dim            - observation vector size
        network.output_dim           - number of discrete actions
        network.hidden_dims          - list of hidden layer widths
        training.batch_size          - mini-batch size for each gradient step
        training.learning_rate       - Adam learning rate
        training.gamma               - discount factor
        training.epsilon_start       - initial epsilon for exploration
        training.epsilon_end         - minimum epsilon
        training.epsilon_decay_steps - linear annealing schedule length
        training.target_update_freq  - hard-copy online->target every N steps
        training.replay_start        - minimum buffer size before learning starts
        buffer.capacity              - maximum replay buffer capacity
    """

    def __init__(self, config: Dict, device: torch.device, logger: RLLogger) -> None:
        super().__init__(config, device, logger)

        input_dim: int = int(config["network.input_dim"])
        output_dim: int = int(config["network.output_dim"])
        hidden_dims: List[int] = list(config["network.hidden_dims"])

        self.batch_size: int = int(config["training.batch_size"])
        self.gamma: float = float(config["training.gamma"])
        self.epsilon: float = float(config["training.epsilon_start"])
        self.epsilon_end: float = float(config["training.epsilon_end"])
        self.epsilon_decay_steps: int = int(config["training.epsilon_decay_steps"])
        self.target_update_freq: int = int(config["training.target_update_freq"])
        self.replay_start: int = int(config["training.replay_start"])
        buffer_capacity: int = int(config["buffer.capacity"])

        self.n_actions = output_dim
        self._step_count: int = 0

        # Online and target networks
        self.q_net = QNetwork(input_dim, output_dim, hidden_dims).to(device)
        self.q_target = QNetwork(input_dim, output_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=float(config["training.learning_rate"])
        )
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=input_dim,
            action_dim=1,
            device=device,
        )

        self.logger.info(
            "DQNAgent ready. input_dim=%d output_dim=%d hidden_dims=%s "
            "batch_size=%d lr=%.5f gamma=%.4f epsilon_start=%.3f replay_start=%d",
            input_dim, output_dim, hidden_dims,
            self.batch_size, float(config["training.learning_rate"]),
            self.gamma, self.epsilon, self.replay_start,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.rand() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals = self.q_net(t)
                action = int(q_vals.argmax(dim=1).item())

        self.logger.debug(
            "select_action: epsilon=%.4f action=%d training=%s", self.epsilon, action, training
        )
        return action

    def update(self) -> Dict[str, float]:
        """
        Sample a mini-batch and perform one gradient descent step.

        Returns {"loss": float, "epsilon": float, "mean_q": float}.
        """
        if len(self.buffer) < self.replay_start:
            return {"loss": 0.0, "epsilon": self.epsilon, "mean_q": 0.0}

        batch = self.buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"].long()        # [B, 1]
        rewards = batch["rewards"]               # [B, 1]
        next_states = batch["next_states"]
        dones = batch["dones"]                   # [B, 1]

        # Current Q-values for taken actions
        q_values = self.q_net(states).gather(1, actions)  # [B, 1]

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.q_target(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard update of target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
            self.logger.info(
                "Target network updated. step=%d", self._step_count
            )

        self._decay_epsilon()

        mean_q = float(q_values.mean().item())
        loss_val = float(loss.item())

        self.logger.debug(
            "update: step=%d loss=%.6f epsilon=%.4f mean_q=%.4f",
            self._step_count, loss_val, self.epsilon, mean_q,
        )
        return {"loss": loss_val, "epsilon": self.epsilon, "mean_q": mean_q}

    def get_model_dict(self) -> Dict[str, nn.Module]:
        return {"online": self.q_net, "target": self.q_target}

    # ------------------------------------------------------------------
    # DQN-specific
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Linear annealing of epsilon."""
        fraction = min(1.0, self._step_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon + fraction * (self.epsilon_end - float(
            self.config["training.epsilon_start"]
        ))
        self.epsilon = max(self.epsilon_end, self.epsilon)
