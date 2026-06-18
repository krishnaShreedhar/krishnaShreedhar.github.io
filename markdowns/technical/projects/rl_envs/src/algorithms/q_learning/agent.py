"""
Tabular Q-Learning agent.

Use case: path planning in GridWorld with a small, discrete state space.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from core.base_agent import BaseAgent
from core.logger import RLLogger


class QLearningAgent(BaseAgent):
    """
    Classic tabular Q-Learning (Watkins, 1989).

    State space is finite; the Q-function is stored as a 2-D numpy array
    of shape [n_states, n_actions].

    Update rule (TD(0)):
        Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])

    Epsilon is decayed multiplicatively after each update.

    Config keys consumed:
        network.n_states          - total number of discrete states
        network.n_actions         - total number of discrete actions
        training.alpha            - learning rate
        training.gamma            - discount factor
        training.epsilon_start    - initial exploration rate
        training.epsilon_end      - minimum exploration rate
        training.epsilon_decay    - multiplicative decay per step
    """

    def __init__(self, config: Dict, device: torch.device, logger: RLLogger) -> None:
        super().__init__(config, device, logger)

        n_states: int = int(config["network.n_states"])
        n_actions: int = int(config["network.n_actions"])

        self.alpha: float = float(config["training.alpha"])
        self.gamma: float = float(config["training.gamma"])
        self.epsilon: float = float(config["training.epsilon_start"])
        self.epsilon_end: float = float(config["training.epsilon_end"])
        self.epsilon_decay: float = float(config["training.epsilon_decay"])

        self.n_states = n_states
        self.n_actions = n_actions

        # Q-table initialised to zeros.
        self.q_table: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        self._last_experience: Optional[Dict] = None
        self._step_count: int = 0

        self.logger.info(
            "QLearningAgent ready. n_states=%d n_actions=%d alpha=%.4f gamma=%.4f "
            "epsilon_start=%.3f epsilon_end=%.3f epsilon_decay=%.5f",
            n_states, n_actions, self.alpha, self.gamma,
            self.epsilon, self.epsilon_end, self.epsilon_decay,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        *state* can be a flat integer index or an array; if an array is
        passed it is interpreted as the ravelled flat index (int cast of
        first element, useful when state is already a 1-element array).
        """
        state_idx = self._to_state_idx(state)

        if training and np.random.rand() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            action = int(np.argmax(self.q_table[state_idx]))

        self.logger.debug(
            "select_action: state_idx=%d epsilon=%.4f action=%d training=%s",
            state_idx, self.epsilon, action, training,
        )
        return action

    def update(self) -> Dict[str, float]:
        """
        Apply one TD(0) update using the most recent stored experience.

        Returns {"td_error": float, "epsilon": float}.
        """
        if self._last_experience is None:
            return {"td_error": 0.0, "epsilon": self.epsilon}

        s = self._to_state_idx(self._last_experience["state"])
        a = int(self._last_experience["action"])
        r = float(self._last_experience["reward"])
        s_next = self._to_state_idx(self._last_experience["next_state"])
        done = bool(self._last_experience["done"])

        # Bellman target
        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.q_table[s_next]))

        td_error = target - float(self.q_table[s, a])
        self.q_table[s, a] += self.alpha * td_error

        self.logger.debug(
            "update: s=%d a=%d r=%.4f s_next=%d done=%s td_error=%.6f",
            s, a, r, s_next, done, td_error,
        )

        self._decay_epsilon()
        self._step_count += 1

        return {"td_error": float(td_error), "epsilon": self.epsilon}

    def get_model_dict(self) -> Dict[str, nn.Module]:
        """No neural networks; returns empty dict."""
        return {}

    # ------------------------------------------------------------------
    # Q-Learning specific methods
    # ------------------------------------------------------------------

    def store_experience(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        """Buffer the latest (s, a, r, s', done) for use in update()."""
        self._last_experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

    def save(self, path: str) -> None:  # type: ignore[override]
        """Pickle the Q-table instead of using torch.save."""
        with open(path, "wb") as fh:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, fh)
        self.logger.info("Q-table saved. path=%s", path)

    def load(self, path: str) -> None:  # type: ignore[override]
        """Load Q-table from a pickle file."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_end)
        self.logger.info("Q-table loaded. path=%s epsilon=%.4f", path, self.epsilon)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Multiplicative epsilon decay with floor at epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @staticmethod
    def _to_state_idx(state) -> int:
        """Convert a state representation to an integer index."""
        if isinstance(state, (int, np.integer)):
            return int(state)
        arr = np.asarray(state).ravel()
        if arr.size == 1:
            return int(arr[0])
        # If a multi-element array is passed, treat it as a ravelled grid coordinate.
        return int(arr[0])
