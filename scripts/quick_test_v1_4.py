#!/usr/bin/env python3
"""Quick test of MHDEnv3D with wrapper."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym
from pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D


class SimplifiedObsWrapper(gym.ObservationWrapper):
    """Simplified observation wrapper."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )
    
    def observation(self, obs_dict):
        psi = obs_dict['psi']
        omega = obs_dict['omega']
        features = [
            obs_dict['energy'], obs_dict['max_psi'], obs_dict['max_omega'],
            np.mean(np.abs(psi)), np.mean(np.abs(omega)),
        ]
        psi_r = np.mean(np.abs(psi), axis=(1, 2))
        omega_r = np.mean(np.abs(omega), axis=(1, 2))
        nr = psi_r.shape[0]
        r_indices = np.linspace(0, nr-1, 8, dtype=int)
        features.extend(psi_r[r_indices])
        features.extend(omega_r[r_indices])
        psi_fft = np.fft.rfft(psi, axis=2)
        omega_fft = np.fft.rfft(omega, axis=2)
        psi_modes = np.mean(np.abs(psi_fft), axis=(0, 1))[:8]
        omega_modes = np.mean(np.abs(omega_fft), axis=(0, 1))[:8]
        psi_modes = np.pad(psi_modes, (0, max(0, 8-len(psi_modes))))
        omega_modes = np.pad(omega_modes, (0, max(0, 8-len(omega_modes))))
        features.extend(psi_modes)
        features.extend(omega_modes)
        while len(features) < 50:
            features.append(0.0)
        return np.array(features[:50], dtype=np.float32)


print("Creating environment...")
env = MHDEnv3D(
    grid_size=(16, 32, 16),  # Smaller grid for speed
    eta=1e-3,
    dt=0.005,
    max_steps=10,  # Just 10 steps
    I_max=0.5,
    n_coils=5,
)
env = SimplifiedObsWrapper(env)

print("Running 1 episode...")
obs, info = env.reset()
print(f"Obs shape: {obs.shape}, E0={info['E0']:.2e}")

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: reward={reward:.4f}, E/E0={info['energy']:.4f}")
    if terminated or truncated:
        break

print("✅ Test complete!")
