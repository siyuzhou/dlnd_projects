import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)  # set random seed


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[128]):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.hidden = nn.ModuleList([nn.Linear(hin, hout)
                                     for hin, hout in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.fc2 = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        for hidden_layer in self.hidden:
            x = F.relu(hidden_layer(x))
        return F.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed, state_layers=[128], action_layers=[128]):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_layers = nn.ModuleList([
            nn.Linear(hin, hout) for hin, hout in zip([state_size]+state_layers[:-1], state_layers)])
        self.action_layers = nn.ModuleList([
            nn.Linear(hin, hout) for hin, hout in zip([state_layers[-1]+action_size]+action_layers[:-1], action_layers)])
        self.fc = nn.Linear(action_layers[-1], 1)

    def forward(self, state, action):
        x = state
        for layer in self.state_layers:
            x = F.leaky_relu(layer(x))
        x = torch.cat((x, action), -1)
        for layer in self.action_layers:
            x = F.leaky_relu(layer(x))
        return self.fc(x)
