"""
PPO (Proximal Policy Optimization) algorithm package.

Use case: VLM finetuning (RLHF-style) and general discrete/continuous control.
Implements PPO-Clip with Generalized Advantage Estimation (GAE) and mini-batch updates.
"""

from algorithms.ppo.network import PPOActorCriticNetwork
from algorithms.ppo.agent import PPOAgent

__all__ = ["PPOAgent", "PPOActorCriticNetwork"]
