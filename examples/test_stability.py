"""Test numerical stability without external currents."""
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl')

import numpy as np
from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D

# Test with zero action
env = MHDEnv3D(grid_size=(32, 64, 32), dt=0.01, max_steps=10)
obs, info = env.reset(seed=42)
print(f"E0 = {info['E0']:.3e}")

for step in range(10):
    action = np.zeros(5)  # Zero action - no external current
    obs, reward, _, _, info = env.step(action)
    print(f"Step {step+1}: E = {info['energy']:.3e}, drift = {info['energy_drift']:.3e}")
    
    if not np.isfinite(info['energy']):
        print("INSTABILITY DETECTED!")
        break
