import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

FROM_ZERO = False
SAVE_PLOT = False

if __name__ == '__main__':
    task = 'hover'   # 'hover' or 'landing'
    task = 'landing' # 'hover' or 'landing'

    max_m_episode = 800_000
    max_steps = 800

    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []
    print(env.state_dims, 'states', env.action_dims, 'actions')

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)

    if not FROM_ZERO and len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], weights_only=False, map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):
        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        net.entropy_buffer = []  # Reset entropy buffer for each episode

        for step_id in range(max_steps):
            # Get action and store entropy
            action, log_prob, value, entropy = net.get_action(state)
            net.entropy_buffer.append((action, log_prob, value, entropy))

            # Take step in environment
            state, reward, done, _ = env.step(action)

            # Store transition data
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)

            if episode_id % 100 == 1:
                env.render()

            if done or step_id == max_steps-1:
                # Get bootstrap value for incomplete episode
                _, _, value, _ = net.get_action(state)

                # Update policy with entropy regularization
                loss_info = net.update_ac(net, rewards, log_probs, values, masks, value, gamma=0.999)

                if episode_id % 100 == 1:
                    print(f"Losses - Actor: {loss_info['actor_loss']:.3f}, "
                          f"Critic: {loss_info['critic_loss']:.3f}, "
                          f"Entropy: {loss_info['entropy_loss']:.3f}")
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 1:
            if SAVE_PLOT:
                plt.figure()
                plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
                plt.legend(['episode reward', 'moving avg'], loc=2)
                plt.xlabel('m episode')
                plt.ylabel('reward')
                plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
                plt.close()

            torch.save({'episode_id': episode_id,
                       'REWARDS': REWARDS,
                       'model_G_state_dict': net.state_dict()},
                      os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))
