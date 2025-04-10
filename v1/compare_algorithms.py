import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import cv2
from rocket import Rocket
from policy import ActorCritic
from sac import SAC

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

def load_a2c_model(task='landing'):
    """Load the Actor-Critic (A2C) model"""
    ckpt_folder = os.path.join('./', task + '_ckpt')
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_folder) or len(os.listdir(ckpt_folder)) == 0:
        print(f"No A2C checkpoints found in {ckpt_folder}")
        return None
    
    # Find the latest checkpoint
    ckpts = [f for f in os.listdir(ckpt_folder) if f.endswith('.pt')]
    if not ckpts:
        print(f"No A2C checkpoints found in {ckpt_folder}")
        return None
    
    # Get the latest checkpoint
    latest_ckpt = sorted(ckpts)[-1]
    ckpt_path = os.path.join(ckpt_folder, latest_ckpt)
    
    # Setup environment to get state and action dimensions
    env = Rocket(task=task)
    
    # Initialize the Actor-Critic model
    model = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_G_state_dict'])
        episode_id = checkpoint['episode_id']
        print(f"Loaded A2C model from episode {episode_id}")
        return model, env
    except Exception as e:
        print(f"Error loading A2C model: {e}")
        return None, env

def load_sac_model(task='landing'):
    """Load the Soft Actor-Critic (SAC) model"""
    ckpt_folder = os.path.join('./', task + '_sac_ckpt')
    ckpt_path = os.path.join(ckpt_folder, 'sac_checkpoint.pt')
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"No SAC checkpoint found at {ckpt_path}")
        return None
    
    # Setup environment to get state and action dimensions
    env = Rocket(task=task)
    
    # Initialize the SAC model
    sac = SAC(
        input_dim=env.state_dims,
        action_dim=env.action_dims,
        hidden_layers=2,
        hidden_size=256,
        L=5
    )
    
    # Load the checkpoint
    try:
        sac.load_model(ckpt_folder)
        print(f"Loaded SAC model from {ckpt_path}")
        return sac, env
    except Exception as e:
        print(f"Error loading SAC model: {e}")
        return None, env

def evaluate_a2c(model, env, episodes=10, render=True, save_video=True):
    """Evaluate the A2C model"""
    rewards = []
    steps = []
    videos = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        frames = []
        
        while True:
            # Get action from model
            action, _, _ = model.get_action(state, deterministic=True)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and episode rewards
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render if requested
            if render:
                frame_0, frame_1 = env.render()
                if save_video and frame_0 is not None:
                    frames.append(frame_0)
            
            if done:
                break
        
        # Save episode statistics
        rewards.append(episode_reward)
        steps.append(step)
        
        # Save video if requested
        if save_video and frames:
            videos.append(frames)
            
        print(f"A2C Episode {episode+1}, Reward: {episode_reward:.3f}, Steps: {step}")
    
    return {
        'rewards': rewards,
        'steps': steps,
        'videos': videos,
        'avg_reward': np.mean(rewards),
        'avg_steps': np.mean(steps)
    }

def evaluate_sac(sac, env, episodes=10, render=True, save_video=True):
    """Evaluate the SAC model"""
    rewards = []
    steps = []
    videos = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        frames = []
        
        while True:
            # Get action from model
            action = sac.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and episode rewards
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render if requested
            if render:
                frame_0, frame_1 = env.render()
                if save_video and frame_0 is not None:
                    frames.append(frame_0)
            
            if done:
                break
        
        # Save episode statistics
        rewards.append(episode_reward)
        steps.append(step)
        
        # Save video if requested
        if save_video and frames:
            videos.append(frames)
            
        print(f"SAC Episode {episode+1}, Reward: {episode_reward:.3f}, Steps: {step}")
    
    return {
        'rewards': rewards,
        'steps': steps,
        'videos': videos,
        'avg_reward': np.mean(rewards),
        'avg_steps': np.mean(steps)
    }

def save_comparison_video(a2c_frames, sac_frames, output_path):
    """Save a side-by-side comparison video of both algorithms"""
    if not a2c_frames or not sac_frames:
        print("No frames to save for comparison")
        return
    
    # Get the frame shape
    h, w, c = a2c_frames[0].shape
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, (w*2, h))
    
    # Determine the maximum number of frames
    max_frames = min(len(a2c_frames), len(sac_frames))
    
    # Create and write frames
    for i in range(max_frames):
        # Create a side-by-side comparison
        combined_frame = np.hstack((a2c_frames[i], sac_frames[i]))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_frame, "A2C", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, "SAC", (w + 10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write frame to video
        video.write(combined_frame[:,:,::-1])  # Convert BGR to RGB
    
    video.release()
    print(f"Comparison video saved to {output_path}")

def plot_comparison(a2c_results, sac_results, output_path):
    """Plot a comparison of the two algorithms' performance"""
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    a2c_rewards = a2c_results['rewards']
    sac_rewards = sac_results['rewards']
    
    plt.plot(a2c_rewards, 'b-', label='A2C')
    plt.plot(sac_rewards, 'r-', label='SAC')
    plt.axhline(y=a2c_results['avg_reward'], color='b', linestyle='--', 
                label=f'A2C Avg: {a2c_results["avg_reward"]:.2f}')
    plt.axhline(y=sac_results['avg_reward'], color='r', linestyle='--', 
                label=f'SAC Avg: {sac_results["avg_reward"]:.2f}')
    
    plt.title('Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot steps
    plt.subplot(2, 1, 2)
    a2c_steps = a2c_results['steps']
    sac_steps = sac_results['steps']
    
    plt.plot(a2c_steps, 'b-', label='A2C')
    plt.plot(sac_steps, 'r-', label='SAC')
    plt.axhline(y=a2c_results['avg_steps'], color='b', linestyle='--', 
                label=f'A2C Avg: {a2c_results["avg_steps"]:.2f}')
    plt.axhline(y=sac_results['avg_steps'], color='r', linestyle='--', 
                label=f'SAC Avg: {sac_results["avg_steps"]:.2f}')
    
    plt.title('Steps per Episode Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")

if __name__ == '__main__':
    task = 'landing'  # 'hover' or 'landing'
    num_eval_episodes = 5
    results_dir = f"./algorithm_comparison_{task}"
    
    # Create results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load models
    a2c_model, a2c_env = load_a2c_model(task)
    sac_model, sac_env = load_sac_model(task)
    
    a2c_results = None
    sac_results = None
    
    # Evaluate models
    if a2c_model is not None:
        print("\nEvaluating A2C model...")
        a2c_results = evaluate_a2c(a2c_model, a2c_env, episodes=num_eval_episodes)
        print(f"A2C Average Reward: {a2c_results['avg_reward']:.3f}")
        print(f"A2C Average Steps: {a2c_results['avg_steps']:.1f}")
    
    if sac_model is not None:
        print("\nEvaluating SAC model...")
        sac_results = evaluate_sac(sac_model, sac_env, episodes=num_eval_episodes)
        print(f"SAC Average Reward: {sac_results['avg_reward']:.3f}")
        print(f"SAC Average Steps: {sac_results['avg_steps']:.1f}")
    
    # Create comparison if both models are available
    if a2c_results is not None and sac_results is not None:
        # Plot comparison
        plot_comparison(a2c_results, sac_results, os.path.join(results_dir, 'comparison_plot.png'))
        
        # Save comparison video of first episode
        if a2c_results['videos'] and sac_results['videos']:
            save_comparison_video(
                a2c_results['videos'][0], 
                sac_results['videos'][0],
                os.path.join(results_dir, 'comparison_video.mp4')
            )
        
        # Print conclusion
        print("\nAlgorithm Comparison Results:")
        print(f"A2C Average Reward: {a2c_results['avg_reward']:.3f}, Average Steps: {a2c_results['avg_steps']:.1f}")
        print(f"SAC Average Reward: {sac_results['avg_reward']:.3f}, Average Steps: {sac_results['avg_steps']:.1f}")
        
        if a2c_results['avg_reward'] > sac_results['avg_reward']:
            print("A2C performed better in terms of reward.")
        elif sac_results['avg_reward'] > a2c_results['avg_reward']:
            print("SAC performed better in terms of reward.")
        else:
            print("Both algorithms performed similarly in terms of reward.")
            
        if a2c_results['avg_steps'] < sac_results['avg_steps']:
            print("A2C completed episodes in fewer steps on average.")
        elif sac_results['avg_steps'] < a2c_results['avg_steps']:
            print("SAC completed episodes in fewer steps on average.")
        else:
            print("Both algorithms used similar numbers of steps.") 