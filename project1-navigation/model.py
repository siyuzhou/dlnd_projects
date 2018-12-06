import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed=1):
        """Initialize a Deep-Q Network
        
        Params
        ======
            state_size (int): dimesion of state
            action_size (int): dimension of action
            seed (int): random seed
        """
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        """Return Q values of all actions for state"""
        x = state
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x