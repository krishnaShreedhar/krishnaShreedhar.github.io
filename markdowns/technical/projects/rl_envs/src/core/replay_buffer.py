"""
Fixed-capacity circular replay buffer for off-policy RL algorithms.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """
    Pre-allocated circular replay buffer.

    Stores (state, action, reward, next_state, done) transitions.
    Samples are returned as torch.Tensors placed on the target device.

    Responsibilities (SRP):
    - Memory management (pre-allocation, circular overwrite).
    - Efficient random sampling.
    - Device placement of batches.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self._capacity = capacity
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device

        self._pos: int = 0
        self._size: int = 0

        # Pre-allocate storage as contiguous numpy arrays for speed.
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition. Overwrites oldest entry when full."""
        self._states[self._pos] = state
        # action may be a scalar int or a 1-D array; normalise to 1-D.
        self._actions[self._pos] = np.atleast_1d(action).astype(np.float32)
        self._rewards[self._pos] = float(reward)
        self._next_states[self._pos] = next_state
        self._dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample *batch_size* transitions uniformly at random.

        Returns a dict with keys:
            states, actions, rewards, next_states, dones
        All values are torch.Tensors on self._device.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self._size}."
            )
        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            "states": torch.tensor(self._states[idxs], dtype=torch.float32, device=self._device),
            "actions": torch.tensor(self._actions[idxs], dtype=torch.float32, device=self._device),
            "rewards": torch.tensor(self._rewards[idxs], dtype=torch.float32, device=self._device),
            "next_states": torch.tensor(self._next_states[idxs], dtype=torch.float32, device=self._device),
            "dones": torch.tensor(self._dones[idxs], dtype=torch.float32, device=self._device),
        }

    def __len__(self) -> int:
        return self._size
