import random
import numpy as np
import torch
import torch.nn.functional as F
import utils
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    """
    Compute discounted returns with bootstrapping.

    Args:
        next_value: Bootstrap value for incomplete episode
        rewards: List of rewards for each step
        masks: List of masks (0 for terminal states, 1 otherwise)
        gamma: Discount factor

    Returns:
        List of discounted returns
    """
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Aug/2020)
    """

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


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping
    """
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=L)

        # Input layer
        k = 1
        self.add_module('linear'+str(k), nn.Linear(in_features=self.mapping.output_dim, out_features=hidden_size, bias=True))

        # Hidden layers
        for layer in range(hidden_layers):
            k += 1
            self.add_module('linear'+str(k), nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True))

        # Output layer
        k += 1
        self.add_module('linear'+str(k), nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True))

        self.layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        self.relu = nn.LeakyReLU(0.2)

        for child in self.named_children(): print(child)
        print(self.layers)

    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        for k in range(len(self.layers)-1):
            x = self.relu(self.layers[k](x))
        return self.layers[-1](x)


GAMMA = 0.99
ENTROPY_COEF = 0.01  # Entropy coefficient (β) controls exploration-exploitation trade-off

class ActorCritic(nn.Module):
    """
    Actor-Critic implementation with entropy regularization.

    The actor (policy network) outputs action probabilities and the critic (value network)
    estimates state values. Entropy regularization is added to encourage exploration.

    Args:
        input_dim: Dimension of state space
        output_dim: Dimension of action space (number of discrete actions)
        hidden_layers: Number of hidden layers in both actor and critic networks
        hidden_size: Number of units in each hidden layer
        L: Positional encoding parameter (0 for no encoding)
        learning_rate: Learning rate for RMSprop optimizer
    """
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        super().__init__()
        self.output_dim = output_dim

        # Actor and Critic networks
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim,
                        hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.critic = MLP(input_dim=input_dim, output_dim=1,
                         hidden_layers=hidden_layers, hidden_size=hidden_size, L=L)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Forward pass through both actor and critic networks.
        Returns action probabilities and state value estimate.
        """
        action_logits = self.actor(x)
        probs = self.softmax(action_logits)
        value = self.critic(x)
        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):
        """
        Select an action based on the current policy.

        Args:
            state: Current environment state
            deterministic: If True, select best action; if False, sample from distribution
            exploration: Probability of random action for epsilon-greedy exploration

        Returns:
            action_id: Selected action
            log_prob: Log probability of selected action
            value: Value estimate of current state
            entropy: Entropy of the action distribution (for regularization)
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        # Create a proper probability distribution
        dist = Categorical(probs)

        if deterministic:
            action_id = probs.argmax().item()
        else:
            if random.random() < exploration:  # epsilon-greedy exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action_id))
        entropy = dist.entropy()

        return action_id, log_prob, value, entropy

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA):
        """
        Update actor and critic networks using collected experience.

        Implements A2C with entropy regularization:
        - Actor Loss = -log_prob * advantage - β * entropy
        - Critic Loss = 0.5 * advantage²
        - Total Loss = Actor Loss + Critic Loss

        Args:
            network: ActorCritic network to update
            rewards: List of rewards from episode
            log_probs: List of log probabilities of taken actions
            values: List of value estimates
            masks: List of done masks (0 for terminal states)
            Qval: Bootstrap value for incomplete episode
            gamma: Discount factor
        """
        # Compute Q values and advantages
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        # Stack collected tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        # Get policy distribution and compute entropy
        advantage = Qvals - values

        # Compute losses with entropy regularization
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Add entropy bonus (entropy maximization encourages exploration)
        entropy_loss = -ENTROPY_COEF * torch.stack([step_output[3] for step_output in network.entropy_buffer]).mean()

        # Combined loss
        total_loss = actor_loss + critic_loss + entropy_loss

        # Optimize
        network.optimizer.zero_grad()
        total_loss.backward()
        network.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
