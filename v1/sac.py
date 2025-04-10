import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ReplayBuffer for off-policy learning
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Same positional mapping from the original implementation
class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale

# MLP network for SAC
class SACNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=256, L=5):
        super(SACNetwork, self).__init__()
        
        self.mapping = PositionalMapping(input_dim=input_dim, L=L)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.mapping.output_dim, hidden_size))
        
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            
        self.layers.append(nn.Linear(hidden_size, output_dim))
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.mapping(x)
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Actor network for SAC (outputs mean and log_std for Gaussian policy)
class SACPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layers=2, hidden_size=256, L=5, log_std_min=-20, log_std_max=2):
        super(SACPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        self.network = SACNetwork(input_dim, action_dim * 2, hidden_layers, hidden_size, L)
        
    def forward(self, state):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        
        # Constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Convert to discrete action (since environment has discrete actions)
        action_probs = F.softmax(x_t, dim=-1)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Apply correction for Tanh squashing (not needed for discrete actions, but kept for future continuous versions)
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action_probs, log_prob

# Critic network for SAC (Q-function)
class SACQFunction(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layers=2, hidden_size=256, L=5):
        super(SACQFunction, self).__init__()
        
        self.network = SACNetwork(input_dim + action_dim, 1, hidden_layers, hidden_size, L)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q_value = self.network(x)
        return q_value

# Soft Actor-Critic main class
class SAC:
    def __init__(
        self, 
        input_dim,
        action_dim,
        buffer_capacity=1000000,
        batch_size=256,
        hidden_layers=2,
        hidden_size=256,
        L=5,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        reward_scale=1.0,
        automatic_entropy_tuning=True
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Initialize policy
        self.policy = SACPolicy(input_dim, action_dim, hidden_layers, hidden_size, L).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize Q functions and their targets
        self.q1 = SACQFunction(input_dim, action_dim, hidden_layers, hidden_size, L).to(device)
        self.q2 = SACQFunction(input_dim, action_dim, hidden_layers, hidden_size, L).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Initialize target networks with the same weights
        self.q1_target = SACQFunction(input_dim, action_dim, hidden_layers, hidden_size, L).to(device)
        self.q2_target = SACQFunction(input_dim, action_dim, hidden_layers, hidden_size, L).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Automatic entropy adjustment
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
            
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        if evaluate:
            # For evaluation, use the mean action
            with torch.no_grad():
                mean, _ = self.policy(state)
                action_probs = F.softmax(mean, dim=-1)
                action_id = torch.argmax(action_probs).item()
        else:
            # For training, sample from the policy
            with torch.no_grad():
                action_probs, _ = self.policy.sample(state)
                # Convert probabilities to action id for discrete action space
                action_id = torch.multinomial(action_probs, 1).item()
                
        return action_id
    
    def update_parameters(self):
        # Sample from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
            
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        # For discrete actions, convert to one-hot encoding
        action_one_hot = F.one_hot(action_batch.long(), num_classes=self.action_dim).float()
        
        with torch.no_grad():
            # Sample next action from policy
            next_action_probs, next_log_prob = self.policy.sample(next_state_batch)
            
            # Compute target Q value
            next_q1_target = self.q1_target(next_state_batch, next_action_probs)
            next_q2_target = self.q2_target(next_state_batch, next_action_probs)
            next_q_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_log_prob
            next_q_value = reward_batch * self.reward_scale + (1 - done_batch) * self.gamma * next_q_target
            
        # Compute current Q values
        q1_value = self.q1(state_batch, action_one_hot)
        q2_value = self.q2(state_batch, action_one_hot)
        
        # Compute Q loss
        q1_loss = F.mse_loss(q1_value, next_q_value)
        q2_loss = F.mse_loss(q2_value, next_q_value)
        
        # Update Q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        action_probs, log_prob = self.policy.sample(state_batch)
        q1 = self.q1(state_batch, action_probs)
        q2 = self.q2(state_batch, action_probs)
        min_q = torch.min(q1, q2)
        
        policy_loss = (self.alpha * log_prob - min_q).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (entropy coefficient)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # Soft update of the target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(
            {
                'policy_state_dict': self.policy.state_dict(),
                'q1_state_dict': self.q1.state_dict(),
                'q2_state_dict': self.q2.state_dict(),
                'q1_target_state_dict': self.q1_target.state_dict(),
                'q2_target_state_dict': self.q2_target.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
                'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
            },
            os.path.join(path, 'sac_checkpoint.pt')
        )
        
    def load_model(self, path):
        checkpoint = torch.load(os.path.join(path, 'sac_checkpoint.pt'), map_location=device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp() 