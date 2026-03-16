#!/usr/bin/env python3
"""
PPO Baseline Training for MHD Tearing Mode Control.

Usage:
    python scripts/train_ppo_baseline.py [--equilibrium simple|solovev] [--total-timesteps N]

Author: 小A 🤖
Date: 2026-03-16
Status: Phase 5 Step 2.5 - Gymnasium Migration + Parameterization
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytokmhd.rl import MHDTearingControlEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def train_ppo_baseline(
    equilibrium_type='simple',
    total_timesteps=10000,
    gamma=0.95,
    learning_rate=3e-4,
    batch_size=256,
    checkpoint_freq=5000,
    log_dir='logs/ppo_baseline',
    model_save_path='models/ppo_baseline_10k.zip',
):
    """
    Train PPO baseline on MHD tearing mode control.
    
    Parameters
    ----------
    equilibrium_type : {'simple', 'solovev'}
        Equilibrium initialization type
    total_timesteps : int
        Total training timesteps
    gamma : float
        Discount factor
    learning_rate : float
        PPO learning rate
    batch_size : int
        Minibatch size
    checkpoint_freq : int
        Checkpoint save frequency
    log_dir : str
        Tensorboard log directory
    model_save_path : str
        Final model save path
    """
    print("=" * 60)
    print("PPO Baseline Training - MHD Tearing Mode Control")
    print("=" * 60)
    print(f"Equilibrium type: {equilibrium_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Gamma: {gamma}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Create environment with configuration
    env = MHDTearingControlEnv(
        equilibrium_type=equilibrium_type,
        grid_size=64,
        action_smoothing_alpha=0.3,
        max_psi_threshold=10.0,
        max_steps=200,
    )
    
    # Wrap for SB3
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Setup checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path='./checkpoints/ppo_baseline/',
        name_prefix='ppo_model'
    )
    
    # Train
    print("\n[Training] Starting PPO training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    print(f"\n✅ Model saved: {model_save_path}")
    model.save(model_save_path)
    
    # Evaluate
    print("\n[Evaluation] Testing trained policy...")
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        episode_length += 1
        if done[0]:
            break
    
    print(f"\n✅ Evaluation episode:")
    print(f"   Total reward: {episode_reward:.2f}")
    print(f"   Episode length: {episode_length}")
    
    if episode_reward > -200:
        print("\n✅ Policy shows learning! (reward > -200)")
    else:
        print("\n⚠️ Policy may need more training (reward < -200)")
    
    print("\n" + "=" * 60)
    print("Phase 5 Step 2 (Baseline) COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review tensorboard logs: tensorboard --logdir logs/ppo_baseline")
    print("2. Run gamma tuning: python scripts/train_ppo_gamma_sweep.py")
    print("3. Proceed to 100k full training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on MHD control')
    parser.add_argument(
        '--equilibrium',
        type=str,
        default='simple',
        choices=['simple', 'solovev'],
        help='Equilibrium type (default: simple)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=10000,
        help='Total training timesteps (default: 10000)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor (default: 0.95)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip model saving (for quick verification)'
    )
    
    args = parser.parse_args()
    
    # Override save path if no-save
    save_path = None if args.no_save else 'models/ppo_baseline_10k.zip'
    
    train_ppo_baseline(
        equilibrium_type=args.equilibrium,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        model_save_path=save_path if save_path else 'models/temp.zip'
    )
