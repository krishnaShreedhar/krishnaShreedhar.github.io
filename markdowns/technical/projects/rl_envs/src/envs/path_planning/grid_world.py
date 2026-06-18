"""
GridWorld environment for path planning experiments.

An NxN grid with a movable agent, a goal cell, and optional obstacles.
The agent receives a sparse reward for reaching the goal and a small
step penalty at every other step.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.base_env import BaseEnv
from core.logger import RLLogger


# Action constants
_ACTION_UP = 0
_ACTION_DOWN = 1
_ACTION_LEFT = 2
_ACTION_RIGHT = 3

_DELTA: Dict[int, Tuple[int, int]] = {
    _ACTION_UP:    (-1,  0),
    _ACTION_DOWN:  ( 1,  0),
    _ACTION_LEFT:  ( 0, -1),
    _ACTION_RIGHT: ( 0,  1),
}


class GridWorldEnv(BaseEnv):
    """
    NxN GridWorld for discrete path planning.

    State encoding:
        Normalised 4-D vector: [row/N, col/N, goal_row/N, goal_col/N]

    Actions (discrete):
        0 = up, 1 = down, 2 = left, 3 = right

    Rewards:
        +reward_goal   on reaching the goal
        +reward_wall   on hitting a wall or obstacle (negative value)
        +reward_step   every other step (negative, step penalty)

    Episode termination:
        - Agent reaches the goal.
        - max_episode_steps exceeded.

    Config keys consumed:
        env.grid_size          (int,   default 10)
        env.num_obstacles      (int,   default 10)
        env.max_episode_steps  (int,   default 200)
        env.reward_goal        (float, default  10.0)
        env.reward_wall        (float, default  -1.0)
        env.reward_step        (float, default  -0.01)
        env.random_start       (bool,  default True)
        env.random_goal        (bool,  default True)
        env.start_pos          (list [row, col], used if random_start=False)
        env.goal_pos           (list [row, col], used if random_goal=False)
        env.obstacles          (list of [row, col], optional)
    """

    def __init__(self, config: Dict, logger: RLLogger) -> None:
        super().__init__(config, logger)

        self._grid_size: int = int(config.get("env.grid_size", 10))
        self._num_obstacles: int = int(config.get("env.num_obstacles", 10))
        self._reward_goal: float = float(config.get("env.reward_goal", 10.0))
        self._reward_wall: float = float(config.get("env.reward_wall", -1.0))
        self._reward_step: float = float(config.get("env.reward_step", -0.01))
        self._random_start: bool = bool(config.get("env.random_start", True))
        self._random_goal: bool = bool(config.get("env.random_goal", True))

        # Fixed positions (used when random_start/goal is False)
        self._fixed_start: Optional[Tuple[int, int]] = None
        self._fixed_goal: Optional[Tuple[int, int]] = None

        start_pos = config.get("env.start_pos", None)
        if start_pos is not None:
            self._fixed_start = (int(start_pos[0]), int(start_pos[1]))

        goal_pos = config.get("env.goal_pos", None)
        if goal_pos is not None:
            self._fixed_goal = (int(goal_pos[0]), int(goal_pos[1]))

        # Obstacles: either from config or generated fresh each episode.
        obstacle_list = config.get("env.obstacles", None)
        self._fixed_obstacles: Optional[List[Tuple[int, int]]] = None
        if obstacle_list is not None:
            self._fixed_obstacles = [(int(r), int(c)) for r, c in obstacle_list]

        # Runtime state (populated on reset())
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._goal_pos: Tuple[int, int] = (0, 0)
        self._obstacles: List[Tuple[int, int]] = []
        self._step_count: int = 0

        self.logger.info(
            "GridWorldEnv ready. grid_size=%dx%d num_obstacles=%d "
            "reward_goal=%.2f reward_wall=%.2f reward_step=%.4f",
            self._grid_size, self._grid_size, self._num_obstacles,
            self._reward_goal, self._reward_wall, self._reward_step,
        )

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return 4  # [row/N, col/N, goal_row/N, goal_col/N]

    @property
    def action_dim(self) -> int:
        return 4  # up, down, left, right

    @property
    def action_space_type(self) -> str:
        return "discrete"

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment; randomise agent, goal, and obstacles."""
        self._step_count = 0

        # Generate obstacles first so we don't place agent/goal on them.
        if self._fixed_obstacles is not None:
            self._obstacles = list(self._fixed_obstacles)
        else:
            self._obstacles = self._generate_obstacles()

        occupied = set(self._obstacles)

        # Agent start position
        if not self._random_start and self._fixed_start is not None:
            self._agent_pos = self._fixed_start
        else:
            self._agent_pos = self._random_free_cell(occupied)
        occupied.add(self._agent_pos)

        # Goal position
        if not self._random_goal and self._fixed_goal is not None:
            self._goal_pos = self._fixed_goal
        else:
            self._goal_pos = self._random_free_cell(occupied)

        obs = self._encode_state()
        self.logger.debug(
            "reset: agent=%s goal=%s obstacles=%d",
            self._agent_pos, self._goal_pos, len(self._obstacles),
        )
        self._log_grid()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply *action* and return (obs, reward, done, info)."""
        self._step_count += 1
        row, col = self._agent_pos
        dr, dc = _DELTA[action]
        new_row, new_col = row + dr, col + dc

        hit_wall = not self._in_bounds(new_row, new_col)
        hit_obstacle = not hit_wall and (new_row, new_col) in set(self._obstacles)

        if hit_wall or hit_obstacle:
            # Agent stays in place; penalise.
            reward = self._reward_wall
            done = False
            info = {"event": "wall" if hit_wall else "obstacle"}
        else:
            self._agent_pos = (new_row, new_col)
            reached_goal = (self._agent_pos == self._goal_pos)
            if reached_goal:
                reward = self._reward_goal
                done = True
                info = {"event": "goal"}
            else:
                reward = self._reward_step
                done = False
                info = {"event": "step"}

        obs = self._encode_state()

        self.logger.debug(
            "step: action=%d agent=%s reward=%.4f done=%s event=%s",
            action, self._agent_pos, reward, done, info.get("event"),
        )
        return obs, reward, done, info

    def render(self, mode: str = "headless") -> Optional[np.ndarray]:
        """
        Render the grid.

        - 'headless': returns None.
        - 'rgb_array': returns an H x W x 3 uint8 numpy array via matplotlib.
        """
        if mode == "headless":
            return None
        if mode == "rgb_array":
            return self._render_rgb()
        return None

    def close(self) -> None:
        self.logger.debug("GridWorldEnv closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_state(self) -> np.ndarray:
        """Return normalised 4-D observation vector."""
        N = float(self._grid_size)
        ar, ac = self._agent_pos
        gr, gc = self._goal_pos
        return np.array([ar / N, ac / N, gr / N, gc / N], dtype=np.float32)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._grid_size and 0 <= col < self._grid_size

    def _random_free_cell(self, occupied: set) -> Tuple[int, int]:
        """Sample a uniformly random cell that is not in *occupied*."""
        N = self._grid_size
        while True:
            r = int(np.random.randint(N))
            c = int(np.random.randint(N))
            if (r, c) not in occupied:
                return (r, c)

    def _generate_obstacles(self) -> List[Tuple[int, int]]:
        """Generate *num_obstacles* randomly placed obstacles."""
        N = self._grid_size
        obstacles: List[Tuple[int, int]] = []
        occupied: set = set()
        attempts = 0
        while len(obstacles) < self._num_obstacles and attempts < N * N:
            r = int(np.random.randint(N))
            c = int(np.random.randint(N))
            if (r, c) not in occupied:
                obstacles.append((r, c))
                occupied.add((r, c))
            attempts += 1
        return obstacles

    def _log_grid(self) -> None:
        """Log an ASCII representation of the grid at DEBUG level."""
        N = self._grid_size
        obstacle_set = set(self._obstacles)
        lines = []
        for r in range(N):
            row_chars = []
            for c in range(N):
                if (r, c) == self._agent_pos:
                    row_chars.append("A")
                elif (r, c) == self._goal_pos:
                    row_chars.append("G")
                elif (r, c) in obstacle_set:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            lines.append("".join(row_chars))
        self.logger.debug("Grid:\n%s", "\n".join(lines))

    def _render_rgb(self) -> np.ndarray:
        """Render grid as an RGB numpy array using matplotlib."""
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        N = self._grid_size
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#f0f0f0")

        # Draw grid lines
        for i in range(N + 1):
            ax.axhline(i, color="gray", linewidth=0.5)
            ax.axvline(i, color="gray", linewidth=0.5)

        # Obstacles
        for (r, c) in self._obstacles:
            rect = patches.Rectangle((c, N - r - 1), 1, 1, fc="#333333")
            ax.add_patch(rect)

        # Goal
        gr, gc = self._goal_pos
        rect = patches.Rectangle((gc, N - gr - 1), 1, 1, fc="#00cc44")
        ax.add_patch(rect)

        # Agent
        ar, ac = self._agent_pos
        rect = patches.Rectangle((ac, N - ar - 1), 1, 1, fc="#3366ff")
        ax.add_patch(rect)

        ax.set_title(f"GridWorld {N}x{N}  step={self._step_count}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        plt.close(fig)
        buf.seek(0)

        import struct
        import zlib
        # Parse PNG to get raw pixel data
        buf_bytes = buf.read()
        # Use matplotlib to re-read the image as array
        buf.seek(0)
        img = plt.imread(buf)  # H x W x 4 float [0,1]
        plt.close("all")
        return (img[:, :, :3] * 255).astype(np.uint8)
