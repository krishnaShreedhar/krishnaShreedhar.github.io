"""Autonomous Vehicle 2D navigation environment.

A simple continuous-control environment for a 2D vehicle navigating in a
unit square [0,1] x [0,1] while avoiding circular obstacles and reaching
a goal position.

State space (7 dims, all normalised):
  [x, y, vx, vy, heading, dist_to_goal, angle_to_goal]

Action space (2 dims in [-1, 1]):
  [steering_delta, acceleration]

Physics: simple Euler integration with clamped speed.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.base_env import BaseEnv
from core.logger import RLLogger


class VehicleEnv(BaseEnv):
    """2D autonomous vehicle navigation environment with obstacle avoidance.

    The vehicle starts at a random position (or fixed position) and must
    navigate to a goal while avoiding circular obstacles. Episode terminates
    on goal reached, collision, or step limit exceeded.
    """

    def __init__(self, config: Dict, logger: RLLogger) -> None:
        super().__init__(config, logger)

        env_cfg = config["env"]

        # ------------------------------------------------------------------ #
        # Environment parameters (from config, no hardcoded values)
        # ------------------------------------------------------------------ #
        self._num_obstacles: int = int(env_cfg["num_obstacles"])
        self._obstacle_radius: float = float(env_cfg["obstacle_radius"])
        self._goal_threshold: float = float(env_cfg["goal_threshold"])
        self._max_speed: float = float(env_cfg["max_speed"])
        self._max_steering: float = float(env_cfg["max_steering"])
        self._max_accel: float = float(env_cfg["max_accel"])
        self._dt: float = float(env_cfg["dt"])
        self._vehicle_radius: float = float(env_cfg["vehicle_radius"])
        self._reward_goal: float = float(env_cfg["reward_goal"])
        self._reward_collision: float = float(env_cfg["reward_collision"])
        self._reward_step: float = float(env_cfg["reward_step"])
        self._reward_progress_scale: float = float(env_cfg["reward_progress_scale"])
        self._random_start: bool = bool(env_cfg.get("random_start", True))
        self._random_goal: bool = bool(env_cfg.get("random_goal", True))

        # Internal vehicle state
        self._x: float = 0.0
        self._y: float = 0.0
        self._speed: float = 0.0
        self._heading: float = 0.0
        self._goal_x: float = 0.9
        self._goal_y: float = 0.9
        self._obstacles: List[Tuple[float, float, float]] = []  # (cx, cy, r)
        self._prev_dist_to_goal: float = 0.0
        self._step_count: int = 0
        self._out_of_bounds_penalty: float = float(env_cfg.get("out_of_bounds_penalty", -1.0))

        self.logger.info(
            f"VehicleEnv initialised | num_obstacles={self._num_obstacles} "
            f"max_speed={self._max_speed} max_steps={self.max_episode_steps}"
        )

    # ---------------------------------------------------------------------- #
    # BaseEnv properties
    # ---------------------------------------------------------------------- #

    @property
    def observation_dim(self) -> int:
        """Dimensionality of the observation vector."""
        return 7

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""
        return 2

    @property
    def action_space_type(self) -> str:
        """Action space type identifier."""
        return "continuous"

    # ---------------------------------------------------------------------- #
    # Core interface
    # ---------------------------------------------------------------------- #

    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode.

        Randomises vehicle starting position, goal position, and obstacles.

        Returns:
            Initial state vector of shape (7,).
        """
        self._step_count = 0

        # Place goal
        if self._random_goal:
            self._goal_x = float(np.random.uniform(0.7, 0.95))
            self._goal_y = float(np.random.uniform(0.7, 0.95))
        else:
            self._goal_x = 0.9
            self._goal_y = 0.9

        # Place vehicle far from goal
        if self._random_start:
            for _ in range(100):
                sx = float(np.random.uniform(0.05, 0.35))
                sy = float(np.random.uniform(0.05, 0.35))
                d = math.hypot(sx - self._goal_x, sy - self._goal_y)
                if d > 0.4:
                    self._x, self._y = sx, sy
                    break
            else:
                self._x, self._y = 0.1, 0.1
        else:
            self._x, self._y = 0.1, 0.1

        self._speed = 0.0
        self._heading = float(np.random.uniform(-math.pi, math.pi)) if self._random_start else 0.0

        # Place obstacles (avoid overlapping vehicle start and goal)
        self._obstacles = []
        attempts = 0
        while len(self._obstacles) < self._num_obstacles and attempts < 200:
            cx = float(np.random.uniform(0.1, 0.9))
            cy = float(np.random.uniform(0.1, 0.9))
            r = self._obstacle_radius
            # Keep clear of vehicle start and goal
            dist_start = math.hypot(cx - self._x, cy - self._y)
            dist_goal = math.hypot(cx - self._goal_x, cy - self._goal_y)
            if dist_start > r + self._vehicle_radius + 0.1 and dist_goal > r + 0.1:
                self._obstacles.append((cx, cy, r))
            attempts += 1

        self._prev_dist_to_goal = math.hypot(self._x - self._goal_x, self._y - self._goal_y)

        state = self._compute_state()
        self.logger.debug(
            f"reset | pos=({self._x:.2f},{self._y:.2f}) goal=({self._goal_x:.2f},{self._goal_y:.2f}) "
            f"n_obstacles={len(self._obstacles)}"
        )
        return state

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an action and advance the simulation by one time step.

        Args:
            action: Array [steering_delta, acceleration] in [-1, 1].

        Returns:
            state: New state vector of shape (7,).
            reward: Scalar reward.
            done: True if episode should end.
            info: Dict with diagnostic information.
        """
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)
        steering_delta = float(action[0])
        acceleration = float(action[1])

        out_of_bounds_occurred = False

        # ------------------------------------------------------------------ #
        # Physics update (Euler integration)
        # ------------------------------------------------------------------ #
        self._heading += steering_delta * self._max_steering * self._dt
        # Normalise heading to [-pi, pi]
        self._heading = math.atan2(math.sin(self._heading), math.cos(self._heading))

        self._speed = float(
            np.clip(
                self._speed + acceleration * self._max_accel * self._dt,
                0.0,
                self._max_speed,
            )
        )
        vx = self._speed * math.cos(self._heading)
        vy = self._speed * math.sin(self._heading)

        new_x = self._x + vx * self._dt
        new_y = self._y + vy * self._dt

        # Clip to arena and flag out-of-bounds
        if new_x < 0.0 or new_x > 1.0 or new_y < 0.0 or new_y > 1.0:
            out_of_bounds_occurred = True

        self._x = float(np.clip(new_x, 0.0, 1.0))
        self._y = float(np.clip(new_y, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # Reward computation
        # ------------------------------------------------------------------ #
        dist_to_goal = math.hypot(self._x - self._goal_x, self._y - self._goal_y)
        delta_dist = self._prev_dist_to_goal - dist_to_goal
        self._prev_dist_to_goal = dist_to_goal

        reward = self._reward_step
        reward += delta_dist * self._reward_progress_scale

        if out_of_bounds_occurred:
            reward += self._out_of_bounds_penalty

        done = False
        info: Dict = {
            "x": self._x,
            "y": self._y,
            "speed": self._speed,
            "dist_to_goal": dist_to_goal,
            "step": self._step_count,
        }

        # Goal check
        if dist_to_goal < self._goal_threshold:
            reward += self._reward_goal
            done = True
            info["outcome"] = "goal_reached"
            self.logger.info(f"step | GOAL REACHED at step {self._step_count}")

        # Collision check
        elif self._check_collision():
            reward += self._reward_collision
            done = True
            info["outcome"] = "collision"
            self.logger.debug(f"step | collision at pos=({self._x:.2f},{self._y:.2f})")

        # Max steps
        elif self._step_count >= self.max_episode_steps:
            done = True
            info["outcome"] = "max_steps"

        state = self._compute_state()
        self.logger.debug(
            f"step | pos=({self._x:.2f},{self._y:.2f}) vx={vx:.3f} vy={vy:.3f} "
            f"action={action} reward={reward:.3f} done={done}"
        )
        return state, reward, done, info

    def render(self, mode: str = "headless") -> Optional[np.ndarray]:
        """Render the environment.

        Args:
            mode: "headless" returns None; "rgb_array" returns a matplotlib image.

        Returns:
            None in headless mode; numpy uint8 array (H, W, 3) in rgb_array mode.
        """
        if mode == "headless":
            return None

        if mode == "rgb_array":
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from io import BytesIO

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal")

                # Draw obstacles
                for (cx, cy, r) in self._obstacles:
                    circ = patches.Circle((cx, cy), r, color="gray", alpha=0.7)
                    ax.add_patch(circ)

                # Draw goal
                goal_circ = patches.Circle(
                    (self._goal_x, self._goal_y), self._goal_threshold,
                    color="green", alpha=0.5, label="goal"
                )
                ax.add_patch(goal_circ)

                # Draw vehicle as arrow
                ax.annotate(
                    "", xy=(
                        self._x + 0.04 * math.cos(self._heading),
                        self._y + 0.04 * math.sin(self._heading),
                    ),
                    xytext=(self._x, self._y),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                )
                vehicle_circ = patches.Circle(
                    (self._x, self._y), self._vehicle_radius, color="blue", alpha=0.6
                )
                ax.add_patch(vehicle_circ)

                ax.set_title(f"Step {self._step_count} | speed={self._speed:.3f}")

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=80)
                plt.close(fig)
                buf.seek(0)

                import PIL.Image
                img = PIL.Image.open(buf)
                return np.array(img)[:, :, :3]
            except ImportError:
                self.logger.warning("render(rgb_array) requires matplotlib and PIL")
                return None

        self.logger.warning(f"render | unknown mode '{mode}' — returning None")
        return None

    def close(self) -> None:
        """No-op: no resources to clean up."""
        self.logger.debug("VehicleEnv.close() called")

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _compute_state(self) -> np.ndarray:
        """Compute the normalised 7-dimensional state vector.

        State components:
          [x, y, vx, vy, heading/pi, dist_to_goal, angle_to_goal/pi]

        All components are in approximately [-1, 1].

        Returns:
            state: Shape (7,) float32 array.
        """
        vx = self._speed * math.cos(self._heading)
        vy = self._speed * math.sin(self._heading)
        dist_to_goal = math.hypot(self._x - self._goal_x, self._y - self._goal_y)
        angle_to_goal = math.atan2(
            self._goal_y - self._y, self._goal_x - self._x
        )

        state = np.array(
            [
                self._x * 2.0 - 1.0,              # x in [-1, 1]
                self._y * 2.0 - 1.0,              # y in [-1, 1]
                vx / (self._max_speed + 1e-8),    # vx normalised
                vy / (self._max_speed + 1e-8),    # vy normalised
                self._heading / math.pi,           # heading in [-1, 1]
                dist_to_goal,                      # distance (unbounded but small)
                angle_to_goal / math.pi,           # angle in [-1, 1]
            ],
            dtype=np.float32,
        )
        return state

    def _check_collision(self) -> bool:
        """Check whether the vehicle overlaps with any obstacle.

        Returns:
            True if a collision is detected, False otherwise.
        """
        for (cx, cy, r) in self._obstacles:
            dist = math.hypot(self._x - cx, self._y - cy)
            if dist < r + self._vehicle_radius:
                return True
        return False
