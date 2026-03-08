
"""Potential-based reward shaping for LunarLander.
Phi(s) rewards being close to pad, upright, and slow.
shaped_r = r + gamma * Phi(s') - Phi(s) (Ng et al., 1999)
"""
import numpy as np


def potential(obs):
    # LunarLander obs layout: [x, y, vx, vy, theta, vtheta, legL, legR]
    x, y, vx, vy, th, vth, legL, legR = obs
    pos_term = - (abs(x) + abs(y-0))
    vel_term = - 0.5*(abs(vx)+abs(vy))
    orient_term = - 0.5*(abs(th) + 0.1*abs(vth))
    legs_term = 2.0 * (legL + legR) # standing gives positive potential
    return 0.5*pos_term + 0.5*vel_term + 0.3*orient_term + legs_term


def shaped_reward(r, obs, obs_next, gamma):
    return float(r + gamma * potential(obs_next) - potential(obs))


