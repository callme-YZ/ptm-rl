"""
Smoke test: Train Hamiltonian PPO for 10 steps
Just verify everything runs without crashes
"""

import sys
sys.path.insert(0, '../v2.0')
sys.path.insert(0, '../src')

from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("=" * 60)
print("Hamiltonian PPO Smoke Test")
print("=" * 60)

# Create environment
print("\n1. Creating environment...")
def make_env():
    return MHDElsasserEnv()

env = DummyVecEnv([make_env])
print("✅ Environment created")

# Create PPO with default policy (not Hamiltonian yet)
print("\n2. Creating PPO...")
model = PPO("MlpPolicy", env, verbose=1, n_steps=10)
print("✅ PPO model created")

# Train for 10 steps
print("\n3. Training 10 steps (smoke test)...")
model.learn(total_timesteps=10)
print("✅ Training completed")

# Test evaluation
print("\n4. Testing evaluation...")
obs = env.reset()
for i in range(5):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"  Step {i+1}: reward={rewards[0]:.4f}")

print("\n" + "=" * 60)
print("✅ Smoke test PASSED - Everything works!")
print("=" * 60)
