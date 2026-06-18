"""
Entry point for transformer_training tutorial.

Reads configs/transformer_training.yaml and runs training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.logging_utils import setup_logger
from src.transformer_training.trainer import Trainer

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "transformer_training.yaml"


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    log = setup_logger("transformer_training.main", config)
    log.info("Config loaded from %s", CONFIG_PATH)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    log.info("CUDA device: %s", torch.cuda.get_device_name(0))
    log.info("CUDA capability: %s", torch.cuda.get_device_capability(0))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
