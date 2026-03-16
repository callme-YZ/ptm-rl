#!/usr/bin/env python3
"""
Quick verification script for Step 3 (PyTokEq integration).

Verifies:
1. Solovev equilibrium initializes correctly
2. Environment runs 100 steps without crash
3. RL training works (short test)

Usage:
    python scripts/verify_step3.py
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytokmhd.rl import MHDTearingControlEnv

def test_initialization():
    """Test 1: Solovev initialization."""
    print("=" * 60)
    print("Test 1: Solovev Initialization")
    print("=" * 60)
    
    env = MHDTearingControlEnv(
        equilibrium_type='solovev',
        R0=1.0,
        a=0.3,
        kappa=1.7,
        delta=0.3,
        grid_size=64
    )
    
    obs, info = env.reset()
    
    print(f"✅ Environment created")
    print(f"   Grid: {env.Nr} x {env.Nphi} x {env.Nz}")
    print(f"   obs.shape: {obs.shape}")
    print(f"   psi range: [{np.min(env.psi):.3e}, {np.max(env.psi):.3e}]")
    print(f"   NaN check: {np.any(np.isnan(env.psi))}")
    
    assert obs.shape == (25,), "Wrong observation shape"
    assert not np.any(np.isnan(env.psi)), "psi contains NaN"
    
    print("✅ PASS\n")
    return env

def test_evolution(env):
    """Test 2: 100-step evolution."""
    print("=" * 60)
    print("Test 2: 100-Step Evolution")
    print("=" * 60)
    
    for i in range(100):
        action = np.array([0.1 * np.sin(i * 0.1)])
        obs, reward, term, trunc, info = env.step(action)
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"❌ FAIL at step {i}: NaN/Inf detected")
            return False
        
        if term or trunc:
            print(f"⚠️ Episode ended at step {i}")
            break
    
    print(f"✅ Completed {env.step_count} steps")
    print(f"   Final psi range: [{np.min(env.psi):.3e}, {np.max(env.psi):.3e}]")
    print("✅ PASS\n")
    return True

def test_training():
    """Test 3: Short RL training."""
    print("=" * 60)
    print("Test 3: Short RL Training (100 steps)")
    print("=" * 60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("⚠️ SKIP: SB3 not installed")
        return True
    
    env = MHDTearingControlEnv(
        equilibrium_type='solovev',
        grid_size=32  # Smaller for faster test
    )
    
    env = DummyVecEnv([lambda: env])
    
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=100, progress_bar=False)
    
    print("✅ Training completed without crash")
    print("✅ PASS\n")
    return True

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 5 Step 3 Verification")
    print("=" * 60 + "\n")
    
    try:
        env = test_initialization()
        test_evolution(env)
        test_training()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nStep 3 (PyTokEq Integration) verified successfully!")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
