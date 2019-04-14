import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300, random_seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.reset_parameters()
        
        # batch norms
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        return None

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        return None

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        
        out = F.tanh(self.fc3(x))
        
        return out

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, fc1_units=400, fc2_units=300, random_seed=42):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): number of dimensions for input layer
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        
        out = self.fc3(x)
        
        return out

class ActorCritic(object):
    """Purpose: set up Actor and Critic Networks
    needed for each Agent.
    """
    
    def __init__(self, state_size, action_size,
                 num_agents, fc1_units, fc2_units,
                 random_seed):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, fc1_units, fc2_units, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, fc1_units, fc2_units, random_seed).to(device)
        
        # this is to account for shared Critic between agents
        critic_input_size = (state_size + action_size) * num_agents

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(critic_input_size, fc1_units, fc2_units, random_seed).to(device)
        self.critic_target = Critic(critic_input_size, fc1_units, fc2_units, random_seed).to(device)
        
        return None