"""
Test Environment Wrapper for IC Variation

Wraps MHDEnv3D to allow specifying IC parameters (n, m0, epsilon).

Author: 小A 🤖
Date: 2026-03-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import gymnasium as gym
from pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D
from pytokmhd.ic.ballooning_mode import create_ballooning_mode_ic, create_equilibrium_ic


class MHDEnv3DWithICParams(MHDEnv3D):
    """
    Extended MHDEnv3D that allows specifying IC parameters.
    
    Additional parameters:
    - mode_n: Toroidal mode number (default 5)
    - mode_m0: Poloidal mode number (default 2)
    - epsilon_ic: Perturbation amplitude (default 0.0001)
    """
    
    def __init__(self, mode_n=5, mode_m0=2, epsilon_ic=0.0001, **kwargs):
        self.mode_n = mode_n
        self.mode_m0 = mode_m0
        self.epsilon_ic = epsilon_ic
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        """Reset with custom IC parameters."""
        # Call parent's __init__ parts needed
        if seed is not None:
            np.random.seed(seed)
        
        # Create equilibrium + ballooning mode perturbation with custom params
        psi0, omega0, q_profile = create_equilibrium_ic(self.grid)
        psi1, omega1 = create_ballooning_mode_ic(
            self.grid,
            n=self.mode_n,
            m0=self.mode_m0,
            epsilon=self.epsilon_ic,
            q_profile=q_profile
        )
        
        # Combine equilibrium + perturbation
        self.psi = psi0 + psi1
        self.omega = omega0 + omega1
        
        # Compute normalization factors from initial condition
        from pytokmhd.physics.hamiltonian_3d import compute_hamiltonian_3d
        self.psi_max = np.max(np.abs(self.psi))
        self.omega_max = np.max(np.abs(self.omega))
        self.E0 = compute_hamiltonian_3d(self.psi, self.omega, self.grid)
        
        # Initialize step counter and energy tracker
        self.current_step = 0
        self.energy_prev = self.E0
        
        # Get initial observation
        obs = self._compute_observation()
        
        info = {
            'E0': self.E0,
            'psi_max': self.psi_max,
            'omega_max': self.omega_max,
            'q_profile': q_profile,
        }
        
        return obs, info


class SimplifiedObsWrapper(gym.ObservationWrapper):
    """
    Wrap MHDEnv3D to provide simplified observation.
    
    Copied from train_mhd_ppo_v1_4.py for consistency.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(50,), dtype=np.float32
        )
    
    def observation(self, obs_dict):
        """Extract simplified features from full observation."""
        psi = obs_dict['psi']
        omega = obs_dict['omega']
        energy = obs_dict['energy']
        max_psi = obs_dict['max_psi']
        max_omega = obs_dict['max_omega']
        
        # Statistical features
        features = [
            energy,
            max_psi,
            max_omega,
            np.mean(np.abs(psi)),
            np.mean(np.abs(omega)),
        ]
        
        # Radial profiles (average over θ, ζ)
        psi_r = np.mean(np.abs(psi), axis=(1, 2))  # (nr,)
        omega_r = np.mean(np.abs(omega), axis=(1, 2))  # (nr,)
        
        # Sample 8 radial points
        nr = psi_r.shape[0]
        r_indices = np.linspace(0, nr-1, 8, dtype=int)
        features.extend(psi_r[r_indices])
        features.extend(omega_r[r_indices])
        
        # Toroidal mode amplitudes (FFT along ζ axis)
        psi_fft = np.fft.rfft(psi, axis=2)  # (nr, nθ, nζ/2+1)
        omega_fft = np.fft.rfft(omega, axis=2)
        
        # Average over r, θ and take first 8 modes
        psi_modes = np.mean(np.abs(psi_fft), axis=(0, 1))[:8]
        omega_modes = np.mean(np.abs(omega_fft), axis=(0, 1))[:8]
        
        # Pad if needed
        psi_modes = np.pad(psi_modes, (0, max(0, 8-len(psi_modes))))
        omega_modes = np.pad(omega_modes, (0, max(0, 8-len(omega_modes))))
        
        features.extend(psi_modes)
        features.extend(omega_modes)
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
