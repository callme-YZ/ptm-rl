"""
PPO Baseline Training for v2.0 MHD (Phase 4.2)

Author: 小A 🤖
Date: 2026-03-20

Train PPO agent on v2.0 Elsasser MHD environment.

Goal: Suppress ballooning mode (m=1,2) via RMP control.
"""

import numpy as np
import os
from datetime import datetime
import torch.nn as nn

# Gym and SB3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# v2.0 env
from mhd_elsasser_env import MHDElsasserEnv


def make_env(rank=0, seed=0):
    """Create and wrap environment
    
    Args:
        rank: Process rank (for parallel envs)
        seed: Random seed
        
    Returns:
        Wrapped env
    """
    def _init():
        env = MHDElsasserEnv(
            grid_shape=(16, 32, 16),  # Small for fast training
            n_coils=4,
            epsilon=0.323,
            eta=0.01,
            pressure_scale=0.2,
            dt_rl=0.01,
            steps_per_action=10,
            max_episode_steps=100
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo_baseline(
    total_timesteps=50000,
    eval_freq=5000,
    save_freq=10000,
    log_dir='./logs',
    model_dir='./models',
    run_name=None
):
    """Train PPO baseline on v2.0 MHD
    
    Args:
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency
        save_freq: Model checkpoint frequency
        log_dir: Logging directory
        model_dir: Model save directory
        run_name: Experiment name (auto-generated if None)
        
    Returns:
        model: Trained PPO model
    """
    # Create run name
    if run_name is None:
        run_name = f"v2_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Directories
    log_path = os.path.join(log_dir, run_name)
    model_path = os.path.join(model_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    print("=" * 60)
    print("PPO Baseline Training (v2.0 Phase 4.2)")
    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Total steps: {total_timesteps}")
    print(f"Log dir: {log_path}")
    print(f"Model dir: {model_path}\n")
    
    # Create training env
    print("Creating training environment...")
    env = DummyVecEnv([make_env(rank=0, seed=42)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval env (must match training env wrapper)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(rank=1, seed=123)])
    # Important: Wrap with VecNormalize to match training env
    # But don't update stats during eval (training=False)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, 
                           clip_obs=10.0, training=False)
    
    # PPO hyperparameters (based on v1.4 baseline)
    ppo_config = {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1,
        'tensorboard_log': log_path,
    }
    
    print("\nPPO Configuration:")
    for k, v in ppo_config.items():
        if k != 'policy':
            print(f"  {k}: {v}")
    
    # Policy network kwargs
    policy_kwargs = {
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
        'activation_fn': nn.ReLU,
    }
    ppo_config['policy_kwargs'] = policy_kwargs
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(env=env, **ppo_config)
    
    # Callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_path,
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_path, 'ppo_final')
    model.save(final_path)
    env.save(os.path.join(model_path, 'vec_normalize_final.pkl'))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Final model saved: {final_path}")
    print(f"Logs: {log_path}")
    
    return model, env


def evaluate_model(model_path, env, n_episodes=10):
    """Evaluate trained model
    
    Args:
        model_path: Path to saved model
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        results: Dict with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60 + "\n")
    
    # Load model
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    final_amplitudes = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if 'm1_amplitude' in info:
            final_amplitudes.append(info['m1_amplitude'])
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"length={step_count}, "
              f"final_m1={info.get('m1_amplitude', 0):.4f}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_final_amplitude': np.mean(final_amplitudes) if final_amplitudes else 0,
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")
    print(f"Mean final m1 amplitude: {results['mean_final_amplitude']:.4f}")
    
    return results


def quick_test_training():
    """Quick test: Train for 10k steps to verify setup
    
    Returns:
        model: Trained model (if successful)
    """
    print("=" * 60)
    print("Quick Training Test (10k steps)")
    print("=" * 60 + "\n")
    
    try:
        model, env = train_ppo_baseline(
            total_timesteps=10000,
            eval_freq=5000,
            save_freq=5000,
            run_name='v2_ppo_quick_test'
        )
        
        print("\n✅ Quick test PASSED!")
        print("   PPO training working on v2.0 env\n")
        
        return model, env
        
    except Exception as e:
        print(f"\n❌ Quick test FAILED:")
        print(f"   {e}\n")
        raise


def full_training():
    """Full training: 50k-100k steps
    
    Returns:
        model: Trained model
    """
    return train_ppo_baseline(
        total_timesteps=50000,
        eval_freq=5000,
        save_freq=10000,
        run_name='v2_ppo_baseline'
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Quick test mode
        model, env = quick_test_training()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # Evaluation mode
        if len(sys.argv) < 3:
            print("Usage: python train_v2_ppo.py eval <model_path>")
            sys.exit(1)
        
        model_path = sys.argv[2]
        eval_env = MHDElsasserEnv(
            grid_shape=(16, 32, 16),
            n_coils=4,
            max_episode_steps=100
        )
        results = evaluate_model(model_path, eval_env, n_episodes=10)
    else:
        # Full training mode
        model, env = full_training()
