"""
Generic training loop that orchestrates agent-environment interaction.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.base_agent import BaseAgent
from core.base_env import BaseEnv
from core.logger import RLLogger


class Trainer:
    """
    Drives the training loop for any (BaseAgent, BaseEnv) pair.

    Responsibilities (SRP):
    - Episode execution (train and eval).
    - Periodic evaluation and checkpoint saving.
    - Collecting and returning training history.

    All hyper-parameters are read from the config dict; no CLI args.
    No fallbacks: missing required keys raise at construction time.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: BaseEnv,
        config: Dict,
        logger: RLLogger,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger

        # Read training schedule from config (KeyError if absent).
        self.max_episodes: int = int(config["training.max_episodes"])
        self.eval_interval: int = int(config.get("training.eval_interval", 100))
        self.eval_episodes: int = int(config.get("training.eval_episodes", 10))
        self.save_interval: int = int(config.get("training.save_interval", 200))
        self.render_mode: str = config.get("training.render_mode", "headless")
        self.checkpoint_dir: str = config.get("training.checkpoint_dir", "checkpoints")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger.info(
            "Trainer initialised. max_episodes=%d eval_interval=%d "
            "save_interval=%d render_mode=%s",
            self.max_episodes,
            self.eval_interval,
            self.save_interval,
            self.render_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns a history dict:
            {
                "episode_rewards": [...],
                "episode_steps":   [...],
                "eval_rewards":    [...],  # one entry per eval checkpoint
            }
        """
        history: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_steps": [],
            "eval_rewards": [],
        }

        global_step = 0

        for episode in range(1, self.max_episodes + 1):
            reward, steps = self._run_episode(training=True)
            global_step += steps

            history["episode_rewards"].append(reward)
            history["episode_steps"].append(steps)

            self.logger.log_episode(episode, reward, steps)

            # Periodic evaluation
            if episode % self.eval_interval == 0:
                eval_stats = self.evaluate(self.eval_episodes)
                history["eval_rewards"].append(eval_stats["mean_reward"])
                self.logger.info(
                    "[EVAL] episode=%d mean_reward=%.2f std_reward=%.2f mean_steps=%.1f",
                    episode,
                    eval_stats["mean_reward"],
                    eval_stats["std_reward"],
                    eval_stats["mean_steps"],
                )

            # Periodic checkpoint
            if episode % self.save_interval == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"checkpoint_ep{episode:05d}.pt"
                )
                self.agent.save(ckpt_path)

        self.logger.info("Training complete. total_episodes=%d", self.max_episodes)
        return history

    def evaluate(self, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Run *n_episodes* evaluation episodes (no learning).

        Returns {"mean_reward", "std_reward", "mean_steps"}.
        """
        if n_episodes is None:
            n_episodes = self.eval_episodes

        rewards: List[float] = []
        steps_list: List[int] = []

        for ep in range(n_episodes):
            reward, steps = self._run_episode(training=False)
            rewards.append(reward)
            steps_list.append(steps)
            self.logger.debug(
                "[EVAL_EP] eval_ep=%d reward=%.2f steps=%d", ep + 1, reward, steps
            )

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps_list)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(self, training: bool) -> Tuple[float, int]:
        """
        Execute one full episode.

        For online algorithms (e.g. Q-Learning) the agent exposes
        ``store_experience`` + ``update`` which are called each step.
        For offline algorithms the agent exposes ``push`` which is called
        each step, and ``update`` is called when the buffer is ready.

        Returns (total_reward, step_count).
        """
        self.agent.set_train_mode(training)

        obs = self.env.reset()
        total_reward: float = 0.0
        step_count: int = 0

        while True:
            action = self.agent.select_action(obs, training=training)
            next_obs, reward, done, info = self.env.step(action)

            total_reward += reward
            step_count += 1

            self.logger.debug(
                "[STEP] step=%d action=%s reward=%.4f done=%s",
                step_count,
                action,
                reward,
                done,
            )

            if self.render_mode == "rgb_array":
                self.env.render(mode="rgb_array")

            if training:
                # Support both online agents (store_experience + update)
                # and off-policy agents (push + update).
                if hasattr(self.agent, "store_experience"):
                    self.agent.store_experience(obs, action, reward, next_obs, done)
                elif hasattr(self.agent, "push"):
                    self.agent.push(obs, action, reward, next_obs, done)

                update_result = self.agent.update()

                if update_result:
                    # Warn on NaN losses.
                    for k, v in update_result.items():
                        import math
                        if math.isnan(v) or math.isinf(v):
                            self.logger.warning(
                                "NaN/Inf detected in loss. key=%s step=%d", k, step_count
                            )

            obs = next_obs

            if done or step_count >= self.env.max_episode_steps:
                break

        return total_reward, step_count
