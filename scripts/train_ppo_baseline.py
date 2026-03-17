#!/usr/bin/env python3
"""
PPO Baseline Training for MHD Tearing Mode Control.

Usage:
    python scripts/train_ppo_baseline.py [OPTIONS]
    
Options:
    --equilibrium {simple,solovev}  Equilibrium type (default: simple)
    --total-timesteps N             Total timesteps (default: 10000)
    --n-envs N                      Parallel environments (default: 1)
    --gamma GAMMA                   Discount factor (default: 0.95)
    --no-save                       Skip model saving

Author: 小A 🤖
Date: 2026-03-17
Status: Phase 5 Step 4 - 8-core Parallel Training
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytokmhd.rl import MHDTearingControlEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(equilibrium_type, rank=0):
    """
    Create environment factory for multiprocessing.
    
    Parameters
    ----------
    equilibrium_type : str
        'simple' or 'solovev'
    rank : int
        Process rank (for seed differentiation)
    
    Returns
    -------
    callable
        Environment factory function
    """
    def _init():
        env = MHDTearingControlEnv(equilibrium_type=equilibrium_type)
        return env
    return _init


def train_ppo_baseline(
    equilibrium_type='simple',
    total_timesteps=10000,
    n_envs=1,
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
    n_envs : int
        Number of parallel environments (CPU cores)
    gamma : float
        Discount factor
    learning_rate : float
        PPO learning rate
    batch_size : int
        Minibatch size
    """
    
    print("=" * 60)
    print("PPO Baseline Training - MHD Tearing Mode Control")
    print("=" * 60)
    print(f"Equilibrium type: {equilibrium_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Gamma: {gamma}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Create vectorized environment
    if n_envs == 1:
        # Single process
        env = MHDTearingControlEnv(equilibrium_type=equilibrium_type)
        vec_env = DummyVecEnv([lambda: env])
        print("Using single-process environment")
    else:
        # Multiprocessing
        vec_env = SubprocVecEnv([make_env(equilibrium_type, i) for i in range(n_envs)])
        print(f"Using {n_envs}-process parallel environments")
    
    # Create PPO model with evaluation callback
    from stable_baselines3.common.callbacks import EvalCallback
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Evaluation callback for monitoring
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
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
    if model_save_path:
        model.save(model_save_path)
        print(f"\n✅ Model saved: {model_save_path}")
    
    # Evaluate
    print("\n[Evaluation] Testing trained policy...")
    obs = vec_env.reset()
    episode_reward = 0
    episode_length = 0
    
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
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
    
    vec_env.close()
    
    print("\n" + "=" * 60)
    if total_timesteps >= 100000:
        print("Phase 5 Step 4 (100k Baseline) COMPLETE")
    else:
        print("Phase 5 Step 2 (10k Baseline) COMPLETE")
    print("=" * 60)


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
        '--n-envs',
        type=int,
        default=1,
        help='Number of parallel environments/CPU cores (default: 1)'
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
        help='Skip saving model'
    )
    
    args = parser.parse_args()
    
    model_path = None if args.no_save else 'models/ppo_baseline_10k.zip'
    if args.total_timesteps >= 1000000:
        model_path = 'models/ppo_baseline_1m.zip'
    elif args.total_timesteps >= 100000:
        model_path = 'models/ppo_baseline_100k.zip'
    
    train_ppo_baseline(
        equilibrium_type=args.equilibrium,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        gamma=args.gamma,
        model_save_path=model_path,
    )
