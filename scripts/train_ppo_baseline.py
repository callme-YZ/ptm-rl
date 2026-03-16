#!/usr/bin/env python3
"""
PPO Baseline Training - Phase 5 Step 2

Train PPO agent on MHD tearing mode control environment.
Baseline: gamma=0.95, 10k timesteps pilot run.

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from pytokmhd.rl import MHDTearingControlEnv, SB3CompatWrapper
import numpy as np


def main():
    print("=" * 60)
    print("Phase 5 Step 2: PPO Baseline Training")
    print("=" * 60)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    base_env = MHDTearingControlEnv(
        Nr=64,
        Nz=128,
        dt=0.01,
        eta=1e-3,
        nu=1e-3,
        max_steps=200,
        w_0=0.01,  # Start with small initial island
        use_phase4_api=True  # Use real MHD solver
    )
    # Wrap for SB3 compatibility (reset() returns obs only)
    env = SB3CompatWrapper(base_env)
    print(f"✅ Environment created (with SB3 wrapper)")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Test environment
    print("\n[2/4] Testing environment...")
    obs = env.reset()
    print(f"✅ Reset successful")
    print(f"   Initial island width: {obs[0]:.6f}")
    print(f"   Diagnostics available: {env.last_info['diagnostics'] is not None}")
    
    # Create PPO model
    print("\n[3/4] Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.95,  # Baseline discount factor
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/ppo_baseline/",
        verbose=1,
        device='auto'  # Use GPU if available
    )
    print(f"✅ PPO model created")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Gamma: {model.gamma}")
    print(f"   Batch size: {model.batch_size}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save every 5k steps
        save_path='./checkpoints/ppo_baseline/',
        name_prefix='ppo_model',
        verbose=1
    )
    
    # Train
    print("\n[4/4] Training PPO (10k timesteps pilot)...")
    print("Monitor progress: tensorboard --logdir logs/ppo_baseline")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=10000,
            callback=checkpoint_callback,
            tb_log_name="gamma_0.95",
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print("=" * 60)
        
        # Save final model
        model_path = "./models/ppo_baseline_10k"
        model.save(model_path)
        print(f"\n✅ Model saved: {model_path}.zip")
        
        # Evaluation
        print("\n[Evaluation] Testing trained policy...")
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        w_trajectory = [obs[0]]
        
        done = False
        while not done and episode_length < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            w_trajectory.append(obs[0])
        
        print(f"\n✅ Evaluation episode:")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Episode length: {episode_length}")
        print(f"   Initial island width: {w_trajectory[0]:.6f}")
        print(f"   Final island width: {w_trajectory[-1]:.6f}")
        print(f"   Width change: {(w_trajectory[-1] - w_trajectory[0]):.6f}")
        
        # Check if learning occurred
        if episode_reward > -200:
            print("\n✅ Policy shows learning! (reward > -200)")
        else:
            print("\n⚠️  Policy may need more training (reward ≤ -200)")
        
        print("\n" + "=" * 60)
        print("Phase 5 Step 2 (Baseline) COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review tensorboard logs: tensorboard --logdir logs/ppo_baseline")
        print("2. Run gamma tuning: python scripts/train_ppo_gamma_sweep.py")
        print("3. Proceed to 100k full training")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current model...")
        model.save("./models/ppo_baseline_interrupted")
        print("✅ Model saved: ./models/ppo_baseline_interrupted.zip")
    
    except Exception as e:
        print(f"\n\n❌ Training failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
