"""
Gym wrappers for compatibility with Stable-Baselines3.

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
"""

import gym
import numpy as np
from typing import Tuple, Dict, Any


class SB3CompatWrapper(gym.Wrapper):
    """
    Wrapper to make environment compatible with Stable-Baselines3.
    
    SB3 expects reset() to return only obs (not tuple),
    but Gym/Gymnasium standard is reset() returns (obs, info).
    
    This wrapper provides backward compatibility.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_info = {}
    
    def reset(self, **kwargs):
        """
        Reset environment.
        
        Returns:
            obs: Observation only (for SB3 compatibility)
        """
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs
    
    def step(self, action):
        """
        Step environment.
        
        Returns:
            obs, reward, done, info (standard Gym interface)
        """
        return self.env.step(action)
    
    @property
    def last_info(self) -> Dict[str, Any]:
        """Get info from last reset()."""
        return self._last_info
