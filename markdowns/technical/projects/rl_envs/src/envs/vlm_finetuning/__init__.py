"""
VLM/LLM Finetuning RL Environment package.

Provides a minimal RLHF-style environment where a policy must select the
correct answer from multiple choices given a question embedding.
Suitable for training PPO or GRPO for LLM alignment tasks.
"""

from envs.vlm_finetuning.vlm_env import VLMEnv

__all__ = ["VLMEnv"]
