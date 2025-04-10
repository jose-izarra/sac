import os
import glob
import random
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import utils  
from rocket import Rocket
from rocket_env import RocketEnv

# Import gymnasium instead of gym.
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# This wrapper adapts the Gymnasium API to what SB3's DummyVecEnv expects.
class RocketGymWrapper(gym.Env):
    # Required metadata for gymnasium
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, task='land', max_steps=800, rocket_type='starship',
                 viewport_h=768, path_to_bg_img=None, continuous_action=False):
        super(RocketGymWrapper, self).__init__()
        # Instantiate the custom Rocket environment.
        self.rocket = Rocket(max_steps=max_steps, task=task, rocket_type=rocket_type,
                             viewport_h=viewport_h, path_to_bg_img=path_to_bg_img)
        # Observation space matches the flattened state (8 elements).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Save the number of possible actions (usually 9).
        self.num_actions = len(self.rocket.action_table)
        self.continuous_action = continuous_action
        
        if self.continuous_action:
            # Use a Box action space. The policy will output a 1-D action in [-1, 1].
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # Use the original discrete action space.
            self.action_space = spaces.Discrete(self.num_actions)
        
    def reset(self, seed=None, options=None):
        """
        Resets the Rocket environment.
        Returns:
            observation (np.array): The initial observation.
            info (dict): Additional info (empty here).
        """
        obs = self.rocket.reset()
        return obs, {}  # Gymnasium's reset returns (obs, info)
    
    def step(self, action):
        """
        Executes one step in the Rocket environment.
        Returns:
            obs (np.array): The new observation.
            reward (float): The reward for this step.
            terminated (bool): Whether the episode ended successfully.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """
        obs, reward, done, info = self.rocket.step(action)
        terminated = done 
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        return self.rocket.render(mode)

    def close(self):
        """
        Closes the environment.
        """
        return self.rocket.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

FROM_ZERO = False
SAVE_PLOT = False

if __name__ == '__main__':

    task = 'landing'

    max_m_episode = 20000
    max_steps = 800
    use_entropy = False
    alpha = 0.001       # Entropy regularization coefficient (for SAC)

    # Create the Gymnasium-compatible environment.
    env = RocketGymWrapper(task=task, max_steps=max_steps)
    # Wrap the environment with SB3's Monitor for logging.
    env = Monitor(env)
    
    # Folder to save checkpoints.
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # Total timesteps (approximate conversion: m episodes * max_steps per episode)
    total_timesteps = max_m_episode * max_steps

    # Choose algorithm. If use_entropy is True we use SAC; otherwise, A2C is used.
    if use_entropy:
        model = SAC("MlpPolicy", env, ent_coef=alpha, verbose=1, device=device)
    else:
        model = A2C("MlpPolicy", env, verbose=1, device=device)

    # Optionally load a previously saved checkpoint.
    ckpt_files = glob.glob(os.path.join(ckpt_folder, '*.zip'))
    if not FROM_ZERO and len(ckpt_files) > 0:
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        model = model.load(latest_ckpt, env=env)
        print(f"Loaded checkpoint from {latest_ckpt}")

    class SaveCheckpointCallback(BaseCallback):
        def __init__(self, save_freq, save_path, verbose=1):
            super(SaveCheckpointCallback, self).__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                checkpoint_path = os.path.join(self.save_path, f"ckpt_{self.n_calls}.zip")
                self.model.save(checkpoint_path)
                if self.verbose:
                    print(f"Saved checkpoint: {checkpoint_path}")
            return True

    checkpoint_callback = SaveCheckpointCallback(save_freq=10000, save_path=ckpt_folder)

    # Train using stable-baselines3's learn() method.
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model at the end of training.
    final_model_path = os.path.join(ckpt_folder, "final_model.zip")
    model.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    # Adjust the log file path as needed
    log_file = os.path.join(ckpt_folder, "monitor.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file, skiprows=1)
        rewards = df['r'].values
        plt.figure()
        plt.plot(rewards, label="Episode Reward")
        plt.plot(utils.moving_avg(rewards, N=50), label="Moving Avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(loc=2)
        plt.savefig(os.path.join(ckpt_folder, 'rewards_final.jpg'))
        plt.close()
        print(f"Reward plot saved to {ckpt_folder}")
