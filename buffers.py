
import torch


class RolloutBuffer:
    def __init__(self, size, obs_dim, device="cpu"):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act = torch.zeros((size,), dtype=torch.int64, device=device)
        self.rew = torch.zeros((size,), dtype=torch.float32, device=device)
        self.done = torch.zeros((size,), dtype=torch.float32, device=device)
        self.val = torch.zeros((size,), dtype=torch.float32, device=device)
        self.logp = torch.zeros((size,), dtype=torch.float32, device=device)
        self.adv = torch.zeros((size,), dtype=torch.float32, device=device)
        self.ret = torch.zeros((size,), dtype=torch.float32, device=device)


    def add(self, obs, act, rew, done, val, logp):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = done
        self.val[i] = val
        self.logp[i] = logp
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True


    def finish(self, last_val, gamma, lam):
        adv = 0.0
        for i in reversed(range(self.ptr)):
            mask = 1.0 - self.done[i]
            delta = self.rew[i] + gamma * last_val * mask - self.val[i]
            adv = delta + gamma * lam * mask * adv
            self.adv[i] = adv
            self.ret[i] = self.adv[i] + self.val[i]
            last_val = self.val[i]
        # normalize advantages
        a = self.adv[:self.ptr]
        self.adv[:self.ptr] = (a - a.mean()) / (a.std() + 1e-8)


    def get(self):
        n = self.ptr
        return (self.obs[:n], self.act[:n], self.logp[:n], self.adv[:n], self.ret[:n])


    def reset(self):
        self.ptr = 0
        self.full = False


