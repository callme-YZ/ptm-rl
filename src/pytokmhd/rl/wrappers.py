"""
Compatibility wrappers for different RL frameworks.

Author: 小A 🤖
Date: 2026-03-16
"""

import gymnasium as gym
from gymnasium import spaces


class SB3CompatWrapper(gym.Wrapper):
    """
    Wrapper for Stable-Baselines3 compatibility.
    
    SB3 works with both Gym and Gymnasium, but this wrapper
    ensures consistent API usage.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        """Reset with SB3-compatible signature."""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Step with SB3-compatible signature."""
        return self.env.step(action)
