"""
Baseline PPO Training (100k steps, 8 parallel envs)
Standard PPO, no Hamiltonian guidance (λ_H=0)
"""
import sys
import os

# Get absolute path to v2.0
script_dir = os.path.dirname(os.path.abspath(__file__))
v2_path = os.path.join(script_dir, '../../v2.0')
sys.path.insert(0, os.path.abspath(v2_path))

from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import time

def make_env():
    """Environment factory for multiprocessing"""
    def _init():
        return MHDElsasserEnv()
    return _init

if __name__ == '__main__':
    print("=" * 70)
    print("Baseline PPO Training - 100k steps (8 parallel envs)")
    print("=" * 70)

    # Create 8 parallel environments
    n_envs = 8
    print(f"\n1. Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = SubprocVecEnv([make_env() for _ in range(2)])
    print(f"✅ {n_envs} environments created")

    # Model
    print("\n2. Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="../logs/baseline_100k",
        device='cpu'
    )
    print("✅ PPO model created")

    # Callbacks
    print("\n3. Setting up callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="../logs/baseline_100k",
        log_path="../logs/baseline_100k",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="../logs/baseline_100k/checkpoints",
        name_prefix="baseline_model"
    )
    print("✅ Callbacks ready")

    # Train
    print("\n4. Starting training (100k steps)...")
    print(f"   Expected time: ~1.5 hours (with {n_envs} parallel envs)")
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
    final_path = "../logs/baseline_100k/final_model"
    model.save(final_path)
    print(f"\n✅ Model saved: {final_path}")

    print("\n" + "=" * 70)
    print("Baseline training complete - check logs/baseline_100k/")
    print("=" * 70)
