"""
Test environment with different initial conditions.

Verifies that the environment works correctly across a range of
initial island widths and RMP amplitudes.

Author: 小A 🤖
Date: 2026-03-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pytokmhd.rl import MHDTearingControlEnv


def test_initial_widths():
    """Test different initial island widths."""
    print("=" * 60)
    print("Testing Different Initial Island Widths")
    print("=" * 60)
    
    widths = [0.01, 0.03, 0.05, 0.07, 0.10]
    
    for w_0 in widths:
        env = MHDTearingControlEnv(w_0=w_0, max_steps=50)
        obs = env.reset()
        
        print(f"\nInitial width w_0 = {w_0:.3f}:")
        print(f"  Observed w: {obs[0]:.6f}")
        print(f"  Difference: {abs(obs[0] - w_0):.6f}")
        
        # Run a few steps
        for _ in range(10):
            action = np.array([0.0])  # No control
            obs, reward, done, info = env.step(action)
        
        print(f"  After 10 steps: w = {obs[0]:.6f}, gamma = {obs[1]:.6f}")


def test_rmp_amplitudes():
    """Test different RMP amplitudes."""
    print("\n" + "=" * 60)
    print("Testing Different RMP Amplitudes")
    print("=" * 60)
    
    amplitudes = [0.0, 0.05, 0.1]
    
    for A in amplitudes:
        env = MHDTearingControlEnv(w_0=0.05, max_steps=20)
        obs = env.reset()
        
        w_initial = obs[0]
        
        # Apply constant RMP
        for _ in range(20):
            action = np.array([A / 0.1])  # Scale to [-1, 1]
            obs, reward, done, info = env.step(action)
        
        w_final = obs[0]
        delta_w = w_final - w_initial
        
        print(f"\nRMP amplitude = {A:.2f}:")
        print(f"  Initial w: {w_initial:.6f}")
        print(f"  Final w:   {w_final:.6f}")
        print(f"  Change:    {delta_w:+.6f} ({delta_w/w_initial*100:+.1f}%)")


def test_long_rollout():
    """Test stability over long rollout."""
    print("\n" + "=" * 60)
    print("Testing Long-term Stability (100 steps)")
    print("=" * 60)
    
    env = MHDTearingControlEnv(w_0=0.03, max_steps=100)
    obs = env.reset()
    
    w_trajectory = [obs[0]]
    energy_trajectory = [obs[20]]
    
    for i in range(100):
        # Random policy
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        w_trajectory.append(obs[0])
        energy_trajectory.append(obs[20])
        
        if done:
            print(f"  Episode terminated at step {i+1}")
            break
    
    print(f"\n  Steps completed: {len(w_trajectory)-1}/100")
    print(f"  Final w: {w_trajectory[-1]:.6f}")
    print(f"  w range: [{min(w_trajectory):.6f}, {max(w_trajectory):.6f}]")
    
    # Energy drift
    E_initial = energy_trajectory[0]
    E_final = energy_trajectory[-1]
    drift = abs(E_final - E_initial) / abs(E_initial)
    print(f"  Energy drift: {drift:.2%}")
    
    # Check for NaN/Inf
    has_nan = any(np.isnan(w) for w in w_trajectory)
    has_inf = any(np.isinf(w) for w in w_trajectory)
    
    if not has_nan and not has_inf:
        print(f"  ✅ No NaN/Inf detected")
    else:
        print(f"  ❌ NaN/Inf detected!")
    
    return w_trajectory, energy_trajectory


if __name__ == '__main__':
    # Run all tests
    test_initial_widths()
    test_rmp_amplitudes()
    w_traj, e_traj = test_long_rollout()
    
    print("\n" + "=" * 60)
    print("All Tests Complete")
    print("=" * 60)
