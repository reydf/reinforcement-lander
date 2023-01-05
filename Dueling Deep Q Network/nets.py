#Neural Network module to be used for the agent.
import torch
import torch.nn as nn

import torch.nn.functional as F

class Nets(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(Nets, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.A = nn.Linear(128, 1)
        self.B = nn.Linear(128, n_actions)
        #self.layer3 = nn.Linear(128, n_actions)

    # There are four states, so there are four layers.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        A = self.A(x)
        B = self.B(x)
        ave_B = torch.mean(B, dim = 1, keepdim = True)
        Q = (A + (B - ave_B))
        return Q
