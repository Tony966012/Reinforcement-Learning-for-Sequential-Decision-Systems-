
"""Heuristic controller for LunarLander-v2 from OpenAI Gym docs.
Used to generate expert demos or hybrid action suggestions.
"""
import math


def heuristic_controller(state):
    x, y, vx, vy, theta, vtheta, legL, legR = state
    # target angle points to center
    angle_targ = x * 0.5 + vx * 1.0
    angle_targ = max(min(angle_targ, 0.4), -0.4)
    hover_targ = 0.55 * abs(x) # target y-position
    angle = theta
    a = 0
    if legL or legR: # if on the ground, reduce angle
        a = 0
    elif angle > angle_targ + 0.05:
        a = 1 # fire right engine
    elif angle < angle_targ - 0.05:
        a = 3 # fire left engine
    elif vy < -0.1:
        a = 2 # fire main engine
    return a