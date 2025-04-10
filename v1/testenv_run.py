import torch
from rocket_env import RocketEnv
from policy import ActorCritic
import os
import glob

# Decide which device we want to run on
device = torch.device("cpu")

if __name__ == '__main__':
    # Choose task
    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800

    # Find latest checkpoint
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))
    if ckpt_dir:
        ckpt_dir = ckpt_dir[-1]  # last ckpt
        print(f"Loading checkpoint: {ckpt_dir}")

    # Create environment
    env = RocketEnv(task=task, max_steps=max_steps)
    print(f"Created environment with observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create and load policy network
    net = ActorCritic(input_dim=8, output_dim=env.action_space.n).to(device)
    if ckpt_dir and os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False, map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        print("Loaded checkpoint successfully")

    # Run simulation
    state, _ = env.reset()
    total_reward = 0

    print("\nStarting simulation...")
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        if terminated or truncated:
            print(f"\nEpisode finished after {step_id+1} steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
