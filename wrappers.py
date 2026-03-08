
import gymnasium as gym
import numpy as np


class NormalizeObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 1e-4
    def observation(self, obs):
        self.count += 1
        alpha = 1.0 / self.count
        self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
        self.obs_var = (1 - alpha) * self.obs_var + alpha * (obs - self.obs_mean) ** 2
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)


class ClipAction(gym.Wrapper):
    def step(self, action):
        if hasattr(self.action_space, 'clip'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return super().step(action)

