"""
A2C (Advantage Actor-Critic) algorithm package.

Use case: Path planning in grid-world environments.
On-policy, synchronous advantage actor-critic with shared backbone network.
"""

from algorithms.a2c.network import ActorCriticNetwork
from algorithms.a2c.agent import A2CAgent

__all__ = ["A2CAgent", "ActorCriticNetwork"]
