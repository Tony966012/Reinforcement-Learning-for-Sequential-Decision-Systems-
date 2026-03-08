
"""Broad Learning System (BLS) critic: ridge regression on random features.
Reference idea: wide random mapping + ridge closed-form weights.
This module plugs into PPO as a drop-in value function approximator.
"""
import numpy as np
import torch


class BroadFeatures:
    def __init__(self, obs_dim, width=1024, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0, size=(obs_dim, width)).astype(np.float32)
        self.b = rng.normal(0, 1.0, size=(width,)).astype(np.float32)
        self.width = width
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim]
        x = obs.detach().cpu().numpy()
        z = x @ self.W + self.b
        # nonlinear map (ReLU)
        z = np.maximum(z, 0.0)
        return torch.from_numpy(z).to(obs.device)


class BLSValueCritic:
    def __init__(self, obs_dim, width=1024, ridge_lambda=1e-2, seed=0, device="cpu"):
        self.features = BroadFeatures(obs_dim, width, seed)
        self.ridge = ridge_lambda
        self.device = torch.device(device)
        self.Wout = torch.zeros((width,), device=self.device) # vector for scalar value


    def to(self, device):
        self.device = torch.device(device)
        self.Wout = self.Wout.to(self.device)
        return self


    def fit(self, obs: torch.Tensor, targets: torch.Tensor):
        # Closed-form ridge: w = (Z^T Z + λI)^-1 Z^T y
        Z = self.features(obs) # [B, W]
        ZtZ = Z.T @ Z # [W, W]
        ridgeI = torch.eye(ZtZ.shape[0], device=Z.device) * self.ridge
        Zty = Z.T @ targets
        # Solve linear system
        w = torch.linalg.solve(ZtZ + ridgeI, Zty)
        self.Wout = w.detach()


    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        Z = self.features(obs)
        return (Z @ self.Wout).squeeze(-1)


