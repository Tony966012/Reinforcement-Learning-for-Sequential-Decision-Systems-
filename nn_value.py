
import torch
import torch.nn as nn
from src.nets.policy_mlp import ValueNet


class NNValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden=(128,128)):
        super().__init__()
        self.vnet = ValueNet(obs_dim, hidden)
    def forward(self, obs):
        return self.vnet(obs)

