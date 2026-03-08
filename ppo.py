
import torch
import torch.nn as nn
import torch.optim as optim
from src.nets.policy_mlp import CategoricalPolicy
from src.critics.nn_value import NNValueCritic
from src.critics.bls import BLSValueCritic
from src.algos.buffers import RolloutBuffer


class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg, device="cpu"):
        self.device = torch.device(device)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("lam", 0.95))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.ent_coef = float(cfg.get("ent_coef", 0.01))
        self.vf_coef = float(cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))


        self.actor = CategoricalPolicy(obs_dim, act_dim).to(self.device)
        critic_type = cfg.get("critic", "nn")
        if critic_type == "bls":
            self.critic = BLSValueCritic(obs_dim, width=1024, ridge_lambda=1e-2, seed=cfg.get("seed",0), device=self.device)
            self._critic_is_bls = True
        else:
            self.critic = NNValueCritic(obs_dim).to(self.device)
            self._critic_is_bls = False


        self.opt_actor = optim.Adam(self.actor.parameters(), lr=float(cfg.get("lr_actor", 3e-4)))
        if not self._critic_is_bls:
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=float(cfg.get("lr_critic", 5e-4)))


        self.rollout = RolloutBuffer(cfg.get("rollout_len", 2048), obs_dim, device=self.device)
        self.update_epochs = cfg.get("update_epochs", 10)
        self.minibatch_size = cfg.get("minibatch_size", 256)


    @torch.no_grad()
    def act(self, obs_t):
        dist = self.actor.dist(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a)
        if self._critic_is_bls:
            v = self.critic(obs_t)
        else:
            v = self.critic(obs_t)
        return a, logp, v


    def update(self):
        (obs, act, old_logp, adv, ret) = self.rollout.get()
        n = obs.shape[0]
        idx = torch.randperm(n, device=self.device)
        for _ in range(self.update_epochs):
            for start in range(0, n, self.minibatch_size):
                j = idx[start:start+self.minibatch_size]
                dist = self.actor.dist(obs[j])
                logp = dist.log_prob(act[j])
                ratio = (logp - old_logp[j]).exp()
                surr1 = ratio * adv[j]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[j]
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = dist.entropy().mean()


                if self._critic_is_bls:
                    # fit BLS to targets each minibatch
                    self.critic.fit(obs[j], ret[j])
                    v_pred = self.critic(obs[j])
                    value_loss = 0.5 * (v_pred - ret[j]).pow(2).mean()
                    self.opt_actor.zero_grad()
                    loss = policy_loss - self.ent_coef * ent + self.vf_coef * value_loss.detach() * 0.0
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.opt_actor.step()
                else:
                    v_pred = self.critic(obs[j])
                    value_loss = 0.5 * (v_pred - ret[j]).pow(2).mean()
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent
                    self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.opt_actor.step(); self.opt_critic.step()


        self.rollout.reset()                    