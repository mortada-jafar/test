"""
2D Continuous Navigation Environment for Multi-Agent Social Learning
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Navigation2DEnv(gym.Env):
    """
    2D Navigation environment where agent moves to a goal position.

    State: [x, y, vx, vy, gx, gy]
    Action: [ax, ay] in [-1, 1]^2
    Reward: -||p - g||_2 (negative distance to goal)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        dt=0.1,
        v_max=2.0,
        world_size=10.0,
        goal_threshold=0.5,
        max_steps=200,
        random_goal=False,
        render_mode=None
    ):
        super().__init__()

        self.dt = dt
        self.v_max = v_max
        self.world_size = world_size
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.random_goal = random_goal
        self.render_mode = render_mode

        # State space: [x, y, vx, vy, gx, gy]
        self.observation_space = spaces.Box(
            low=np.array([-world_size, -world_size, -v_max, -v_max, -world_size, -world_size]),
            high=np.array([world_size, world_size, v_max, v_max, world_size, world_size]),
            dtype=np.float32
        )

        # Action space: [ax, ay]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Internal state
        self.state = None
        self.steps = 0
        self.goal = None

        # For rendering
        self.fig = None
        self.ax = None
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial position and velocity
        x = self.np_random.uniform(-self.world_size * 0.8, self.world_size * 0.8)
        y = self.np_random.uniform(-self.world_size * 0.8, self.world_size * 0.8)
        vx = self.np_random.uniform(-self.v_max * 0.5, self.v_max * 0.5)
        vy = self.np_random.uniform(-self.v_max * 0.5, self.v_max * 0.5)

        # Goal position
        if self.random_goal or self.goal is None:
            gx = self.np_random.uniform(-self.world_size * 0.7, self.world_size * 0.7)
            gy = self.np_random.uniform(-self.world_size * 0.7, self.world_size * 0.7)
            self.goal = np.array([gx, gy], dtype=np.float32)

        gx, gy = self.goal

        self.state = np.array([x, y, vx, vy, gx, gy], dtype=np.float32)
        self.steps = 0
        self.trajectory = [[x, y]]

        return self.state, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        x, y, vx, vy, gx, gy = self.state
        ax, ay = action

        # Update velocity with acceleration
        vx_new = vx + self.dt * ax
        vy_new = vy + self.dt * ay

        # Clip velocity
        vx_new = np.clip(vx_new, -self.v_max, self.v_max)
        vy_new = np.clip(vy_new, -self.v_max, self.v_max)

        # Update position
        x_new = x + self.dt * vx_new
        y_new = y + self.dt * vy_new

        # Keep within bounds
        x_new = np.clip(x_new, -self.world_size, self.world_size)
        y_new = np.clip(y_new, -self.world_size, self.world_size)

        # Update state
        self.state = np.array([x_new, y_new, vx_new, vy_new, gx, gy], dtype=np.float32)
        self.steps += 1
        self.trajectory.append([x_new, y_new])

        # Compute reward: negative distance to goal
        distance = np.sqrt((x_new - gx)**2 + (y_new - gy)**2)
        reward = -distance

        # Check termination
        terminated = distance < self.goal_threshold
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {"distance": distance}

    def get_partial_observation(self, obs_type="full"):
        """
        Get partial observation based on observation type.

        obs_type:
            - "position_only": [x, y, gx, gy] (no velocity)
            - "full": [x, y, vx, vy, gx, gy] (complete state)
        """
        if obs_type == "position_only":
            return np.array([self.state[0], self.state[1],
                           self.state[4], self.state[5]], dtype=np.float32)
        else:  # full
            return self.state.copy()

    def render(self):
        if self.render_mode is None:
            return None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()

        # Set limits
        self.ax.set_xlim(-self.world_size, self.world_size)
        self.ax.set_ylim(-self.world_size, self.world_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Draw goal
        goal_circle = plt.Circle(
            (self.state[4], self.state[5]),
            self.goal_threshold,
            color='green',
            alpha=0.3,
            label='Goal'
        )
        self.ax.add_patch(goal_circle)

        # Draw agent
        agent_circle = plt.Circle(
            (self.state[0], self.state[1]),
            0.3,
            color='blue',
            alpha=0.8,
            label='Agent'
        )
        self.ax.add_patch(agent_circle)

        # Draw velocity vector
        self.ax.arrow(
            self.state[0], self.state[1],
            self.state[2] * 0.3, self.state[3] * 0.3,
            head_width=0.2,
            head_length=0.2,
            fc='red',
            ec='red',
            alpha=0.6
        )

        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)

        # Add info
        distance = np.sqrt((self.state[0] - self.state[4])**2 +
                          (self.state[1] - self.state[5])**2)
        self.ax.set_title(f'Step: {self.steps}, Distance: {distance:.2f}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        else:  # rgb_array
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
