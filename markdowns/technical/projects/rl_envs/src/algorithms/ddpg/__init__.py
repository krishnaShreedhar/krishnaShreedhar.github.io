"""
DDPG (Deep Deterministic Policy Gradient) algorithm package.

Use case: Autonomous vehicle control (continuous action spaces).
Off-policy actor-critic with deterministic policy and experience replay.
"""

from algorithms.ddpg.network import ActorNetwork, CriticNetwork
from algorithms.ddpg.agent import DDPGAgent

__all__ = ["DDPGAgent", "ActorNetwork", "CriticNetwork"]
