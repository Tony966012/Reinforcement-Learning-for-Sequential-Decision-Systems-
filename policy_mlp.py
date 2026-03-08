
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128,128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.logits = nn.Linear(last, act_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        return self.logits(z)



    def dist(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(x)
        # Replace NaNs/Infs with 0, and clamp to a reasonable range
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)
        return torch.distributions.Categorical(logits=logits)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128,128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.v = nn.Linear(last, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.body(x)).squeeze(-1)


