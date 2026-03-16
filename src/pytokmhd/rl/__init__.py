"""
PyTokMHD RL Module - Reinforcement Learning Interface for Tearing Mode Control.

This module provides Gym-compatible environments for training RL agents
to control tearing modes in tokamak plasmas.

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
Phase: 5 (RL Interface)
Status: Week 1 Implementation
"""

from .env import MHDTearingControlEnv
from .wrappers import SB3CompatWrapper

__version__ = '0.1.0'
__all__ = ['MHDTearingControlEnv', 'SB3CompatWrapper']
