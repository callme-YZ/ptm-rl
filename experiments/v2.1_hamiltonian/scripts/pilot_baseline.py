"""
Pilot: Standard PPO (λ_H=0, baseline)
1000 steps quick test
"""
import sys
import os

# Get absolute path to v2.0
script_dir = os.path.dirname(os.path.abspath(__file__))
v2_path = os.path.join(script_dir, '../../v2.0')
sys.path.insert(0, os.path.abspath(v2_path))

from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

print("=" * 60)
print("Pilot: Standard PPO Baseline (λ_H=0)")
print("=" * 60)

# Environment
env = DummyVecEnv([lambda: MHDElsasserEnv()])
eval_env = DummyVecEnv([lambda: MHDElsasserEnv()])

# Model (standard MlpPolicy, no Hamiltonian)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="../logs/pilot_baseline"
)

# Callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="../logs/pilot_baseline",
    log_path="../logs/pilot_baseline",
    eval_freq=500,
    deterministic=True,
    render=False
)

# Train 1000 steps
print("\nTraining 1000 steps (pilot)...")
model.learn(total_timesteps=1000, callback=eval_callback)

# Save
model.save("../logs/pilot_baseline/final_model")
print("\n✅ Pilot baseline complete!")
print(f"Logs: ../logs/pilot_baseline")
