import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DQN
from utils import ReplayBuffer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 1
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

    
class Agent:
    def __init__(self, state_size, action_size, seed=SEED):
        """
        Initialize an Agent object.
        
        :params:
        
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.qnetwork_local = DQN(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = DQN(state_size, action_size, seed).to(DEVICE)
        
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.step_count = 0

    def act(self, state, eps=0.05):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)Di

        # Epsilon-greedy action selection
        if np.random.rand() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.step_count += 1
        
        if self.step_count % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Move experiences to DEVICE
        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        rewards = rewards.to(DEVICE)
        next_states = next_states.to(DEVICE)
        dones = dones.to(DEVICE)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        self.qnetwork_local.train()
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
        

        