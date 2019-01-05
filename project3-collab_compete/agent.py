import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import Actor, Critic
from utils import PrioritizedMemory, OUNoise


torch.manual_seed(0)  # set random seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0000   # L2 weight decay


class DDPGAgent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size

        np.random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = PrioritizedMemory(BUFFER_SIZE, random_seed)

    def act(self, state, add_noise=True):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        # Save experience and reward in replay buffer
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()

        state_tensor = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action_tensor = torch.tensor(action, dtype=torch.float, device=DEVICE)
        done_array = np.vstack(done).astype(np.uint8)
        reward = np.vstack(reward)

        with torch.no_grad():
            action_next = self.actor_target(next_state_tensor)
            Q_target_next = self.critic_target(next_state_tensor, action_next).cpu().data.numpy()
            Q_target = reward + (GAMMA * Q_target_next * (1 - done_array))

            Q_expected = self.critic_local(state_tensor, action_tensor).cpu().data.numpy()

            error = np.abs((Q_expected - Q_target)).squeeze()

        self.actor_target.train()
        self.critic_target.train()
        self.critic_local.train()

        for e, s, a, r, ns, d in zip(error, state, action, reward, next_state, done):
            self.memory.add(e, (s, a, r, ns, d))

        # Learn after aquiring enough experiences in memory
        if len(self.memory) > BATCH_SIZE:
            batch = self.memory.sample(BATCH_SIZE)
            self.learn(batch)

    def learn(self, batch):
        experiences, idxs, weights = batch

        weights = torch.tensor(weights, dtype=torch.float, device=DEVICE).unsqueeze(1)

        states, actions, rewards, next_states, dones = tuple(zip(*experiences))

        states = torch.tensor(states, dtype=torch.float, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float, device=DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float, device=DEVICE)
        dones = torch.tensor(np.vstack(dones).astype(np.uint8), dtype=torch.float, device=DEVICE)

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Update the critic
        Q_expected = self.critic_local(states, actions)
        critic_loss = torch.mean((Q_expected - Q_targets) ** 2 * weights)  # Weighted MSE loss.

        errors = torch.abs(Q_expected - Q_targets).cpu().data.numpy()

        for i, idx in enumerate(idxs):
            self.memory.update(idx, errors[i])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred) * weights
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def load(self, checkpoint):
        self.actor_local.load_state_dict(checkpoint.actor_local)
        self.critic_local.load_state_dict(checkpoint.critic_local)
        self.actor_target.load_state_dict(checkpoint.actor_target)
        self.critic_target.load_state_dict(checkpoint.critic_target)
