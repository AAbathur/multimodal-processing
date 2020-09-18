import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return - (x - y)**2 + F.relu(x + y)

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)

        self.drop_value = drop
        self.dorp = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None 
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()
    
    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)
        return x