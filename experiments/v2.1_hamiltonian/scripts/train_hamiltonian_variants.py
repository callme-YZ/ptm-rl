"""
Train Hamiltonian PPO Variants
Lambda_H ablation: 0.1, 0.5, 1.0
100k steps each, 8 parallel envs
"""

import sys
import os

# Get absolute path to v2.0
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '../src'))
sys.path.insert(0, os.path.join(script_dir, '../../v2.0'))

from sb3_policy import HamiltonianActorCriticPolicy
from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import time
import argparse

def make_env():
    """Environment factory for multiprocessing"""
    def _init():
        return MHDElsasserEnv()
    return _init

if __name__ == '__main__':
    # Parse lambda_h from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_h', type=float, required=True,
                       help='Hamiltonian guidance strength (0.1, 0.5, or 1.0)')
    args = parser.parse_args()
    
    lambda_h = args.lambda_h
    
    print("=" * 70)
    print(f"Hamiltonian PPO Training - λ_H={lambda_h}")
    print("=" * 70)

    # Create 8 parallel environments
    n_envs = 8
    print(f"\n1. Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = SubprocVecEnv([make_env() for _ in range(2)])
    print(f"✅ {n_envs} environments created")

    # Model
    print(f"\n2. Creating PPO with HamiltonianPolicy (λ_H={lambda_h})...")
    model = PPO(
        HamiltonianActorCriticPolicy,
        env,
        policy_kwargs=dict(
            lambda_h=lambda_h,
            latent_dim=8,
            h_hidden_dim=64
        ),
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"../logs/hamiltonian_lambda{lambda_h:.1f}",
        device='cpu'
    )
    print("✅ PPO model created")

    # Callbacks
    print("\n3. Setting up callbacks...")
    log_dir = f"../logs/hamiltonian_lambda{lambda_h:.1f}"
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix=f"hamiltonian_lambda{lambda_h:.1f}"
    )
    print("✅ Callbacks ready")

    # Train
    print(f"\n4. Starting training (100k steps)...")
    print(f"   Lambda_H: {lambda_h}")
    print(f"   Latent dim: 8")
    print(f"   Expected time: ~45 minutes (with {n_envs} parallel envs)")
    print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    start_time = time.time()

    model.learn(
        total_timesteps=100_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"✅ Training complete!")
    print(f"   Duration: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"   End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save final model
    final_path = f"{log_dir}/final_model"
    model.save(final_path)
    print(f"\n✅ Model saved: {final_path}")

    print("\n" + "=" * 70)
    print(f"Hamiltonian training (λ_H={lambda_h}) complete")
    print(f"Check {log_dir}/ for results")
    print("=" * 70)
