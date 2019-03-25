import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import ActorCritic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

"""
Taken from: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgentDDPG():
    """This is an extension of Agent() class
    to accomodate more than 1 agent.
    This is based on the idea from https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf.
    In this case Critic is augmented with extra information about other agent's policies whereas Actor has information
    only about current agent.
    Agents also share Experience Replay Buffer.
    """
    
    def __init__(self, action_size, state_size, random_seed, num_agents):
        """Initialize Multi Agent DDPG.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed 
            num_agents (int): number of DDPG agents
        """
        
        # create two agents with their own instances of Actor and Critic
        models = [ActorCritic(state_size, action_size, num_agents, random_seed) for _ in range(num_agents)]
        
        # create agents
        self.agents = [Agent(model, action_size, random_seed, i) for i, model in enumerate(models)]
        
        # add number of agents
        self.num_agents = num_agents
        
        # create memory (shared between agents)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        return None
    
    def step(self, states, actions, rewards, next_states, dones):
        """Step function of the Agents.
        Params
        ======
            states (array): states of all agents
            next_states (array): next states of all agents
            actions (array): actions of all agents
            rewards (array): rewards of all agents
            dones (array): if done for all agents
        """
        
        # reshape arrays nicely
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        
        # add to shared replay memory 
        self.memory.add(states, actions, rewards, next_states, dones)
        
        return None
    
    def update(self):
        """Train agents.
        """
        
        # learn for each agent
        if len(self.memory) > BATCH_SIZE:
            
            # list of experiences for each agent
            experiences = [self.memory.sample() for _ in range(self.num_agents)]
            
            # use the same for all agents
            # e = self.memory.sample()
            # experiences = [e for _ in range(self.num_agents)]
            
            # each agent learns (loops over each agent in self.learn())
            self.learn(experiences, GAMMA)
        
        return None
    
    def learn(self, experiences, GAMMA):
        """learning of Agents.
        Params
        ======
            experiences (list): list of experiences sampled for each agent
            GAMMA (float): discount parameter for learning Q values
        
        """
        
        # each agent uses its own actor to calculate next_actions
        
        # empty lists to store actions
        next_actions_ = []
        actions_ = []
        
        # loop over each agent
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            
            # get agent_id
            agent_id = torch.tensor([i]).to(device)
            
            # extract agent i state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state) # predict action
            actions_.append(action)
            
            # extract agent i next state and get action via target actor network
            next_state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions_.append(next_action)
        
        # let each agent learn from his experiences
        for i, agent in enumerate(self.agents):
            agent.learn(experiences[i], GAMMA, next_actions_, actions_, i)
    
        return None
    
    def act(self, states, decay):
        """
        Given the state, return action for each agent.
        Params
        ======
            states (list): list of states for actions
            add_noise (bool): if add noise to actions
            decay (int): decay of noise
        """
        
        # list to store actions
        actions_ = []
        
        for agent, state in zip(self.agents, states):
            action = agent.act(state, decay)
            actions_.append(action)
        
        # reshape array to get 1 by 4 output
        out = np.array(actions_).reshape(1, -1)
        
        return out

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, model, action_size, random_seed, id_agent):
        """Initialize an Agent object.
        
        Params
        ======
            model (instance): class instance of Actor and Critic
            action_size (int): number of actions Agent can take
            randome_seed (int): integer for random seed generator
            id_agent (int): number of an agent
        """

        # Actor Network (w/ Target Network)
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        # Set weights for local and target actor, respectively, critic the same
        self.copy_weights(self.actor_target, self.actor_local)
        self.copy_weights(self.critic_target, self.critic_local)
        
        # agent id
        self.id_agent = id_agent

        return None
    
    def copy_weights(self, target, source):
        """Copy weights from source to target network,
        modified version of agent.soft_update()"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        return None

    def act(self, state, add_noise=True, decay=1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action = action + (1.0/decay) * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, next_actions, actions, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            next_actions (list): list of agents next actions
            actions (list): list of agents actions
            agent_id (int): agent_id, needed to distinguish between agents
        """
        states, actions, rewards, next_states, dones = experiences
        
        # convert agent_id to pytorch
        agent_id = torch.tensor([agent_id]).to(device)
        
        # put all next_actions into array
        actions_next = torch.cat(next_actions, dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i). Also take rewards only for the given agent_id
        Q_targets = rewards.index_select(1, agent_id) + (gamma * Q_targets_next * (1 - dones.index_select(1, agent_id)))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # huber loss
        #huber = torch.nn.SmoothL1Loss()
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach()) # use detach to make sure that we can pass through network
        
        # store critic_loss
        self.critic_loss = critic_loss.data.numpy()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # detach actions from others
        
        actions_pred = [a.unsqueeze(1) if i == self.id_agent else a.unsqueeze(1).detach() for i, a in enumerate(actions)]
        
        # put actions together
        actions_pred = torch.cat(actions_pred, dim=1).transpose(dim0=0, dim1=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # store loss
        self.actor_loss = actor_loss.data.numpy()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)