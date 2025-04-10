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

if __name__ == '__main__':
    task = 'hover'   # 'hover' or 'landing'
    task = 'landing' # 'hover' or 'landing'
    max_steps = 800

    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')

    print(f"{env.state_dims} states, {env.action_dims} actions")

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)

    # load the last ckpt
    checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1],
                          weights_only=False, map_location=device)
    net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    done = False

    while not done:
        # Get action and entropy (now properly unpacking all 4 return values)
        action, log_prob, value, entropy = net.get_action(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        env.render()
