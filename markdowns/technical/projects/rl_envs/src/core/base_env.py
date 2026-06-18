"""
Abstract base class for all RL environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from core.logger import RLLogger


class BaseEnv(ABC):
    """
    Abstract base for every RL environment in this library.

    Responsibilities (SRP):
    - Declares the environment interface (reset, step, render, close).
    - Provides a default no-op render and close implementation.
    - Exposes environment metadata as abstract properties.

    No fallbacks: subclasses must implement all abstract members.
    """

    def __init__(self, config: Dict, logger: RLLogger) -> None:
        self.config = config
        self.logger = logger
        self.logger.info(
            "BaseEnv initialised. max_episode_steps=%d", self.max_episode_steps
        )

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimensionality of the observation vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of actions (discrete) or action vector size (continuous)."""

    @property
    @abstractmethod
    def action_space_type(self) -> str:
        """Either 'discrete' or 'continuous'."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply *action* and return (observation, reward, done, info).
        """

    # ------------------------------------------------------------------
    # Default implementations
    # ------------------------------------------------------------------

    def render(self, mode: str = "headless") -> Optional[np.ndarray]:
        """
        Render the environment.

        Default implementation returns None (headless mode).
        Subclasses should override to support 'rgb_array' mode.
        """
        return None

    def close(self) -> None:
        """Release any resources held by the environment."""

    # ------------------------------------------------------------------
    # Shared properties
    # ------------------------------------------------------------------

    @property
    def max_episode_steps(self) -> int:
        """Maximum steps per episode, read from config."""
        return self.config.get("env.max_episode_steps", 500)
