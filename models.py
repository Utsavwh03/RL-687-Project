# Implmenting the policy and value networks

import torch 
import torch.nn as nn 
import torch.optim as optim 

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x =self.fc2(x)
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)