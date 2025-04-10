import numpy as np
import torch
import cv2
import time
from rocket import Rocket
from sac import SAC

"""
Soft Actor-Critic (SAC) Algorithm for Rocket Landing

Why SAC is better than other RL algorithms like A2C for rocket landing:

1. SAMPLE EFFICIENCY:
   SAC is an off-policy algorithm that uses a replay buffer to store and reuse past experiences,
   making it more sample-efficient than on-policy methods like A2C which can only learn from
   current trajectories. This allows SAC to make better use of the data it collects during training.

2. EXPLORATION-EXPLOITATION BALANCE:
   SAC incorporates entropy maximization into its objective function, encouraging exploration 
   while still optimizing for rewards. This is particularly important for rocket landing, where 
   discovering the optimal control policy requires both exploration to find good strategies and 
   exploitation to refine the landing technique.

3. STABILITY AND ROBUSTNESS:
   SAC uses two Q-functions and takes the minimum value to reduce overestimation bias,
   leading to more stable learning. This double Q-learning approach helps avoid the overly
   optimistic value estimates that can plague other algorithms, making SAC more reliable
   for critical tasks like rocket landing where stability is essential.

4. AUTOMATIC TEMPERATURE TUNING:
   SAC automatically adjusts its temperature parameter (alpha) which controls the 
   trade-off between exploration and exploitation. This adaptation mechanism helps
   the agent automatically find the right balance as learning progresses.

5. CONVERGENCE PROPERTIES:
   In practice, SAC often achieves better asymptotic performance than traditional 
   actor-critic methods, finding more optimal policies that can lead to more fuel-efficient
   and safer landing strategies.

6. HANDLING CONTINUOUS/DISCRETE ACTIONS:
   While SAC was designed for continuous action spaces, our implementation adapts it
   for the discrete actions in the rocket environment. This provides the benefits of SAC
   while working within the constraints of the pre-defined rocket control actions.
"""

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

if __name__ == '__main__':
    # Choose task: 'hover' or 'landing'
    task = 'landing'
    
    # Set up environment
    max_steps = 800
    env = Rocket(task=task, max_steps=max_steps)
    
    # Load the trained SAC model
    ckpt_folder = f"./{task}_sac_ckpt"
    
    # Initialize SAC agent with the same parameters used during training
    sac = SAC(
        input_dim=env.state_dims,
        action_dim=env.action_dims,
        hidden_layers=2,
        hidden_size=256,
        L=5,  # Positional mapping dimension
        automatic_entropy_tuning=True  # Automatically adjust exploration-exploitation balance
    )
    
    # Load model weights
    try:
        sac.load_model(ckpt_folder)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Run evaluation episodes
    num_episodes = 5
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        frames = []
        
        for step in range(max_steps):
            # Select action using the trained policy (deterministic for evaluation)
            # In SAC, we can easily switch between stochastic and deterministic policies
            # by setting evaluate=True, which is useful for exploration vs exploitation
            action = sac.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and episode reward
            state = next_state
            episode_reward += reward
            
            # Render and save frame
            frame_0, frame_1 = env.render()
            if frame_0 is not None:
                frames.append(frame_0)
            
            # Slow down rendering for better visualization
            time.sleep(0.01)
            
            if done:
                break
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.3f}, Steps: {step+1}")
        total_reward += episode_reward
        
        # Save video of this episode
        if frames:
            # Define the video codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"{ckpt_folder}/sac_{task}_episode_{episode+1}.mp4"
            video = cv2.VideoWriter(video_path, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
            
            # Write frames to video
            for frame in frames:
                video.write(frame[:,:,::-1])  # Convert BGR to RGB for proper video colors
            
            video.release()
            print(f"Video saved to {video_path}")
    
    # Print average performance
    avg_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.3f}")
    
    """
    Performance Comparison:
    
    When comparing SAC to A2C for rocket landing tasks, we typically observe:
    
    1. BETTER SAMPLE EFFICIENCY: SAC often requires fewer environment interactions
       to achieve the same level of performance, saving computational resources.
       
    2. HIGHER FINAL PERFORMANCE: SAC's policy usually converges to better landing
       strategies with higher rewards and more consistent successful landings.
       
    3. MORE ROBUST POLICIES: The learned policies tend to be more robust to
       different initial conditions and disturbances during landing.
       
    4. SMOOTHER CONTROL: SAC typically produces smoother control sequences
       that are more fuel-efficient and avoid abrupt changes in thrust and angle.
       
    These advantages make SAC a superior choice for rocket landing control,
    especially when sample efficiency and final performance are important.
    """ 