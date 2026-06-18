"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm package.

Use case: Autonomous vehicle control (continuous action spaces).
Three improvements over DDPG:
  1. Twin critics to reduce Q-value overestimation bias.
  2. Delayed policy updates for a more stable value function before actor updates.
  3. Target policy smoothing to prevent the actor from exploiting critic errors.
"""

from algorithms.td3.network import TD3ActorNetwork, TD3CriticNetwork
from algorithms.td3.agent import TD3Agent

__all__ = ["TD3Agent", "TD3ActorNetwork", "TD3CriticNetwork"]
