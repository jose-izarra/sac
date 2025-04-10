"""
+--------------------------------------------------------------------------------+
|  WARNING!!!                                                                    |
|  THIS IS JUST AN STUB FILE (TEMPLATE)                                          |
|  PROBABLY ALL LINES SHOULD BE CHANGED OR TOTALLY REPLACED IN ORDER TO GET A    |
|  WORKING FUNCTIONAL VERSION FOR YOUR ASSIGNMENT                                |
+--------------------------------------------------------------------------------+
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rocket import Rocket
import cv2


class RocketEnv(gym.Env):
    """
    A Gymnasium wrapper for the Rocket environment.

    The environment simulates a rocket that can either hover at a target point
    or perform a landing maneuver. The rocket has a rigid body with a thin rod
    and considers acceleration, angular acceleration, and air resistance.

    Tasks:
        - hover: Keep the rocket hovering at a target point while staying upright
        - landing: Land the rocket safely at a designated landing point
    """
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, task='hover', rocket_type='falcon', max_steps=800):
        super(RocketEnv, self).__init__()

        # Initialize the rocket simulation
        self.rocket = Rocket(max_steps=max_steps, task=task, rocket_type=rocket_type)

        # Define action space: 9 discrete actions combining thrust and nozzle angle
        self.action_space = spaces.Discrete(len(self.rocket.action_table))

        # Define observation space: 8 continuous values normalized by 100
        # [x, y, vx, vy, theta, vtheta, t, phi]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.pi/2], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max_steps, np.pi/2], dtype=np.float32),
            dtype=np.float32
        )

        self.task = task
        self.max_steps = max_steps
        self._frames = []

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset (unused)

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset the rocket simulation
        initial_state = self.rocket.reset()

        info = {}
        return initial_state, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Integer in [0, 8] representing the action to take

        Returns:
            observation: New state observation
            reward: Reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was artificially terminated
            info: Additional information
        """
        # Execute action in rocket simulation
        next_state, reward, done, info = self.rocket.step(action)

        # Check if episode should be truncated due to max steps
        truncated = self.rocket.step_id >= self.max_steps

        # Split done into terminated (natural end) and truncated (artificial end)
        terminated = done and not truncated

        return next_state, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode: Rendering mode (only "human" supported)

        Returns:
            None or rendered frames depending on mode
        """
        if mode == "human":
            frame_0, frame_1 = self.rocket.render(wait_time=1)
            return frame_0

    def close(self):
        """Clean up resources."""
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("File __name__ is set to: {}" .format(__name__))
    # Using the gym library to create the environment
    # env = gym.make('your_environment_name-v0')
    from stable_baselines3.common.env_checker import check_env
    env = RocketEnv()
    print('CHECK_ENV','OK' if check_env(env) is None else 'ERROR')
    print(env.observation_space)
    print(env.action_space)
    print(type(env).__name__)
