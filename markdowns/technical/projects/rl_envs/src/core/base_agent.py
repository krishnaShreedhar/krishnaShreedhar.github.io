"""
Abstract base class for all RL agents.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from core.logger import RLLogger


class BaseAgent(ABC):
    """
    Abstract base for every RL agent in this library.

    Responsibilities (SRP):
    - Owns training/eval mode toggling across all sub-networks.
    - Owns checkpoint save/load logic.
    - Delegates action selection and update to concrete subclasses.

    No fallbacks: missing config keys raise KeyError at init time.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        logger: RLLogger,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger

        checkpoint_dir: str = config.get("training.checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.logger.info(
            "BaseAgent initialised. device=%s checkpoint_dir=%s",
            device,
            checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Any:
        """Choose an action given the current state."""

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Perform one learning update. Returns a dict of named scalar losses."""

    @abstractmethod
    def get_model_dict(self) -> Dict[str, nn.Module]:
        """Return a mapping of name -> nn.Module for all networks that should be saved/loaded."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def set_train_mode(self, training: bool) -> None:
        """Toggle all networks between train and eval mode."""
        for name, module in self.get_model_dict().items():
            if training:
                module.train()
            else:
                module.eval()
            self.logger.debug(
                "set_train_mode: module=%s mode=%s", name, "train" if training else "eval"
            )

    def save(self, path: str) -> None:
        """Save state dicts of all networks to *path*."""
        state = {name: mod.state_dict() for name, mod in self.get_model_dict().items()}
        torch.save(state, path)
        self.logger.info("Checkpoint saved. path=%s", path)

    def load(self, path: str) -> None:
        """Load state dicts for all networks from *path*."""
        state = torch.load(path, map_location=self.device)
        for name, mod in self.get_model_dict().items():
            if name in state:
                mod.load_state_dict(state[name])
                self.logger.info("Loaded state_dict for module=%s from %s", name, path)
            else:
                self.logger.warning("Module '%s' not found in checkpoint '%s'.", name, path)
