"""
VLM/LLM Finetuning Environment for RLHF-style RL training.

This environment models a simplified multiple-choice QA task where a policy
(representing a language model) must select the correct answer from N choices
given a question embedding (representing the model's internal representation
of the question and context).

Design choices:
    - State: fixed 128-dimensional embedding vector per question.
             In production this would be the output of the LLM's encoder or
             a frozen embedding model. Here we use pre-computed synthetic vectors.
    - Action: integer index into the choice list (0..num_choices-1).
    - Episode: one question per episode (done=True after one step).
               This matches the RLHF setup where each prompt/completion pair
               is an independent episode.
    - Reward: +reward_correct for correct, +reward_wrong for wrong.
              No step penalty since the episode is one step long.

Connection to real RLHF:
    In production PPO-RLHF (e.g., InstructGPT / RLHF on LLMs):
        - State:  the tokenized prompt (here: embedding of the prompt).
        - Action: the full response token sequence (here: a single choice index).
        - Reward: a reward model score (here: +1/-0.5 based on correctness).
        - Policy: the LLM being fine-tuned (here: a small MLP).
    This tutorial environment isolates the RL mechanics without the LLM infrastructure.

Use with PPO (configs/algorithms/05_ppo.yaml):
    network.input_dim  = 128   (embedding_dim)
    network.action_dim = 4     (num_choices)
    training.n_steps   = 128   (collect 128 question episodes before each PPO update)
"""

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from core.base_env import BaseEnv
from core.logger import RLLogger


class VLMEnv(BaseEnv):
    """
    Multiple-choice QA environment for RLHF-style LLM finetuning.

    Each episode consists of a single step:
        - reset() draws a question and returns its embedding.
        - step(action) checks the answer and terminates.

    The question bank is either synthetic (random embeddings + random correct answers)
    or can be extended to load real QA data.

    Config keys expected:
        env.num_questions     (int)   - size of the question bank
        env.embedding_dim     (int)   - question embedding dimensionality (default 128)
        env.num_choices       (int)   - number of answer choices (default 4)
        env.reward_correct    (float) - reward for correct answer (default +1.0)
        env.reward_wrong      (float) - reward for wrong answer (default -0.5)
        env.use_synthetic_data(bool)  - if True, generate random question bank
        env.max_episode_steps (int)   - always 1 for this environment
    """

    # Fixed observation and action space constants
    _EMBEDDING_DIM: int = 128
    _NUM_CHOICES: int = 4

    def __init__(self, config: Dict, logger: RLLogger) -> None:
        super().__init__(config, logger)

        # ── Config ────────────────────────────────────────────────────────
        self._num_questions: int = config.get("env", {}).get("num_questions", 1000)
        self._embedding_dim: int = config.get("env", {}).get("embedding_dim", self._EMBEDDING_DIM)
        self._num_choices: int = config.get("env", {}).get("num_choices", self._NUM_CHOICES)
        self._reward_correct: float = config.get("env", {}).get("reward_correct", 1.0)
        self._reward_wrong: float = config.get("env", {}).get("reward_wrong", -0.5)
        use_synthetic: bool = config.get("env", {}).get("use_synthetic_data", True)

        # ── Build question bank ───────────────────────────────────────────
        if use_synthetic:
            self._question_embeddings, self._correct_answers = self._generate_synthetic_data()
        else:
            raise NotImplementedError(
                "Real QA data loading is not implemented. "
                "Set env.use_synthetic_data=true for tutorial use."
            )

        # ── Episode state ─────────────────────────────────────────────────
        self._current_question_idx: int = 0
        self._episode_done: bool = True  # force reset before first step

        # ── Rolling accuracy tracker (last 100 episodes) ──────────────────
        self._recent_correct: deque = deque(maxlen=100)
        self._total_episodes: int = 0

        self.logger.info(
            f"VLMEnv initialised | num_questions={self._num_questions} "
            f"embedding_dim={self._embedding_dim} num_choices={self._num_choices} "
            f"reward_correct={self._reward_correct} reward_wrong={self._reward_wrong} "
            f"use_synthetic={use_synthetic}"
        )

    # ── Synthetic data generation ─────────────────────────────────────────────

    def _generate_synthetic_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random question bank for tutorial purposes.

        Embeddings are L2-normalised random vectors so that different questions
        are spread across the unit hypersphere, giving the policy distinguishable
        observations.

        Returns:
            question_embeddings: shape [num_questions, embedding_dim], float32, normalised.
            correct_answers:     shape [num_questions], int, values in [0, num_choices).
        """
        rng = np.random.default_rng(seed=42)
        embeddings = rng.standard_normal((self._num_questions, self._embedding_dim)).astype(
            np.float32
        )
        # L2-normalise each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        correct_answers = rng.integers(0, self._num_choices, size=self._num_questions).astype(np.int64)

        self.logger.info(
            f"Synthetic question bank generated | "
            f"num_questions={self._num_questions} embedding_dim={self._embedding_dim} "
            f"choice_distribution={np.bincount(correct_answers).tolist()}"
        )
        return embeddings, correct_answers

    # ── BaseEnv properties ────────────────────────────────────────────────────

    @property
    def observation_dim(self) -> int:
        """Dimensionality of the question embedding (state space)."""
        return self._embedding_dim

    @property
    def action_dim(self) -> int:
        """Number of answer choices (action space size)."""
        return self._num_choices

    @property
    def action_space_type(self) -> str:
        """Discrete action space: one choice index per step."""
        return "discrete"

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Start a new episode by sampling a question uniformly at random.

        Returns:
            Question embedding of shape [embedding_dim], dtype float32.
        """
        self._current_question_idx = np.random.randint(0, self._num_questions)
        self._episode_done = False

        observation = self._question_embeddings[self._current_question_idx].copy()

        self.logger.debug(
            f"reset | question_idx={self._current_question_idx} "
            f"correct_answer={self._correct_answers[self._current_question_idx]} "
            f"obs_norm={np.linalg.norm(observation):.4f}"
        )
        return observation

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a single step: evaluate the chosen answer and terminate the episode.

        Args:
            action: Integer index of the chosen answer (0..num_choices-1).

        Returns:
            next_obs: Zero vector (episode is over; no meaningful next state).
            reward:   reward_correct if action matches ground truth, else reward_wrong.
            done:     Always True (one question = one episode).
            info:     Dict with keys: correct (bool), question_idx (int), correct_answer (int).
        """
        if self._episode_done:
            self.logger.warning(
                "step() called on a finished episode. Call reset() first."
            )

        correct_answer: int = int(self._correct_answers[self._current_question_idx])
        is_correct: bool = (int(action) == correct_answer)
        reward: float = self._reward_correct if is_correct else self._reward_wrong

        # Episode always terminates after one step (one question, one response)
        done = True
        self._episode_done = True

        # Next observation is a zero vector (irrelevant since episode is done)
        next_obs = np.zeros(self._embedding_dim, dtype=np.float32)

        # Update rolling accuracy
        self._recent_correct.append(float(is_correct))
        self._total_episodes += 1

        info: Dict = {
            "correct": is_correct,
            "question_idx": self._current_question_idx,
            "correct_answer": correct_answer,
        }

        self.logger.debug(
            f"step | question_idx={self._current_question_idx} "
            f"action={action} correct_answer={correct_answer} "
            f"is_correct={is_correct} reward={reward:.4f}"
        )

        # Log rolling accuracy summary at INFO every 100 episodes
        if self._total_episodes % 100 == 0 and len(self._recent_correct) > 0:
            rolling_accuracy = float(np.mean(self._recent_correct))
            self.logger.info(
                f"VLMEnv | episode={self._total_episodes} "
                f"rolling_accuracy_100={rolling_accuracy:.4f} "
                f"(chance={1.0/self._num_choices:.4f})"
            )

        return next_obs, reward, done, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, mode: str = "headless") -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Args:
            mode: ``"headless"`` returns None; ``"rgb_array"`` returns a dummy image.

        Returns:
            None for headless mode, or a placeholder uint8 array for rgb_array.
        """
        if mode == "headless":
            return None

        if mode == "rgb_array":
            # Create a simple 64x64 grey image as a placeholder visualisation
            # In a real implementation this might render question text as an image
            img = np.full((64, 64, 3), fill_value=200, dtype=np.uint8)
            # Encode the current question index as brightness
            brightness = int(255 * (self._current_question_idx / self._num_questions))
            img[16:48, 16:48] = brightness
            return img

        self.logger.warning(f"Unknown render mode '{mode}'. Returning None.")
        return None

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release environment resources (no-op for this lightweight environment)."""
        self.logger.info("VLMEnv closed.")
