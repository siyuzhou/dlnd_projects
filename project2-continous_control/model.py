import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)  # set random seed


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[256]):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.hidden = nn.ModuleList([nn.Linear(hin, hout)
                                     for hin, hout in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.fc2 = nn.Linear(hidden_layers[-1], action_size)

        self.reset_parameters()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        for hidden_layer in self.hidden:
            x = F.relu(hidden_layer(x))
        return F.tanh(self.fc2(x))

    def reset_parameters(self):
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed, hidden_layers=[256, 128]):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_layers[0])
        self.hidden = nn.ModuleList([nn.Linear(hin, hout)
                                     for hin, hout in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.fc2 = nn.Linear(hidden_layers[-1], action_size)

        self.reset_parameters()

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(torch.cat((state, action), dim=1)))
        for hidden_layer in self.hidden:
            x = F.leaky_relu(hidden_layer(x))
        return self.fc2(x)

    def reset_parameters(self):
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
