"""
Reinforcement Learning module for MHD Tearing Mode Control.

This module provides Gymnasium-compatible environments for training
RL agents to control tearing modes in tokamak plasmas.

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
Status: Phase 5 Step 2.5 - Gymnasium Migration + Parameterization

Public API
----------
MHDTearingControlEnv : Gymnasium environment
    Main RL environment with configurable equilibrium types

Example
-------
>>> from pytokmhd.rl import MHDTearingControlEnv
>>> 
>>> # Simple equilibrium (fast, for testing)
>>> env = MHDTearingControlEnv(equilibrium_type='simple')
>>> obs, info = env.reset()
>>> 
>>> # Realistic Solovev equilibrium (requires PyTokEq)
>>> env = MHDTearingControlEnv(equilibrium_type='solovev', R0=1.0, a=0.3)
>>> obs, info = env.reset()
"""

from .env import MHDTearingControlEnv

__all__ = ['MHDTearingControlEnv']
