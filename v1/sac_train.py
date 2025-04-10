import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import glob
from rocket import Rocket
from sac import SAC

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

FROM_ZERO = True  # Set to False to continue from a previous checkpoint
SAVE_PLOT = True

if __name__ == '__main__':
    # Choose task: 'hover' or 'landing'
    task = 'landing'
    
    # Set up environment
    max_m_episode = 800_000
    max_steps = 800
    env = Rocket(task=task, max_steps=max_steps)
    
    # Create directory for checkpoints
    ckpt_folder = os.path.join('./', task + '_sac_ckpt')
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    
    # Initialize episode tracking
    last_episode_id = 0
    REWARDS = []
    EVAL_REWARDS = []
    
    print(f"State dimensions: {env.state_dims}, Action dimensions: {env.action_dims}")
    
    # Initialize SAC agent
    sac = SAC(
        input_dim=env.state_dims,
        action_dim=env.action_dims,
        buffer_capacity=1000000,
        batch_size=256,
        hidden_layers=2,
        hidden_size=256,
        L=5,  # Positional mapping dimension
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        reward_scale=5.0,  # Higher reward scale can help with exploration
        automatic_entropy_tuning=True
    )
    
    # Load existing checkpoint if continuing training
    if not FROM_ZERO and len(glob.glob(os.path.join(ckpt_folder, 'sac_checkpoint.pt'))) > 0:
        print("Loading model from checkpoint...")
        sac.load_model(ckpt_folder)
        
        # Also load training history if available
        history_file = os.path.join(ckpt_folder, 'training_history.npz')
        if os.path.exists(history_file):
            history = np.load(history_file)
            REWARDS = history['rewards'].tolist()
            EVAL_REWARDS = history.get('eval_rewards', []).tolist()
            last_episode_id = len(REWARDS)
            print(f"Loaded training history with {last_episode_id} episodes")
    
    # Training loop
    for episode_id in range(last_episode_id, max_m_episode):
        state = env.reset()  # Reset environment for new episode
        episode_reward = 0
        
        # Episode loop
        for step_id in range(max_steps):
            # Select action from policy
            action = sac.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            sac.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update state and episode reward
            state = next_state
            episode_reward += reward
            
            # Render every 100 episodes
            if episode_id % 100 == 0:
                env.render()
            
            # Update SAC parameters
            sac.update_parameters()
            
            # End episode if done
            if done:
                break
        
        # Record episode reward
        REWARDS.append(episode_reward)
        
        # Print progress
        print(f'Episode {episode_id}, Reward: {episode_reward:.3f}, Steps: {step_id+1}')
        
        # Periodically perform evaluation
        if episode_id % 100 == 0:
            eval_reward = 0
            state = env.reset()
            
            # Evaluation episode
            for step_id in range(max_steps):
                action = sac.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                eval_reward += reward
                env.render()
                
                if done:
                    break
            
            EVAL_REWARDS.append(eval_reward)
            print(f'Evaluation Episode {episode_id}, Reward: {eval_reward:.3f}')
            
            # Save plot
            if SAVE_PLOT:
                plt.figure(figsize=(12, 6))
                
                # Plot training rewards
                plt.subplot(1, 2, 1)
                plt.plot(REWARDS)
                plt.plot(np.convolve(REWARDS, np.ones(50)/50, mode='valid'))
                plt.title('Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend(['Episode Reward', 'Moving Avg (50)'])
                
                # Plot evaluation rewards if we have any
                if len(EVAL_REWARDS) > 0:
                    plt.subplot(1, 2, 2)
                    eval_x = np.arange(0, episode_id+1, 100)[:len(EVAL_REWARDS)]
                    plt.plot(eval_x, EVAL_REWARDS)
                    plt.title('Evaluation Rewards')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                
                plt.tight_layout()
                plt.savefig(os.path.join(ckpt_folder, f'rewards_{episode_id:08d}.jpg'))
                plt.close()
            
            # Save model
            sac.save_model(ckpt_folder)
            
            # Save training history
            np.savez(
                os.path.join(ckpt_folder, 'training_history.npz'),
                rewards=np.array(REWARDS),
                eval_rewards=np.array(EVAL_REWARDS)
            ) 