#!/usr/bin/env python3
"""
PPO Training Script for 3D MHD Control Environment (v1.4)

Train PPO agent to control ballooning instability using 5 coil currents.

Hyperparameters:
- Algorithm: PPO (Proximal Policy Optimization)
- n_steps: 2048 (rollout buffer)
- batch_size: 64
- learning_rate: 3e-4
- n_epochs: 10
- gamma: 0.99
- Total timesteps: 50,000 (1000 episodes × 50 steps)

Outputs:
- Best model: models/ppo_mhd_v1_4_best.zip
- TensorBoard logs: logs/ppo_mhd_v1_4/
- CSV logs: logs/ppo_mhd_v1_4_progress.csv

Usage:
    python scripts/train_mhd_ppo_v1_4.py

Author: 小A 🤖
Created: 2026-03-20
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D


class SimplifiedObsWrapper(gym.ObservationWrapper):
    """
    Wrap MHDEnv3D to provide simplified observation.
    
    Instead of full 3D fields (32×64×32 = 65k floats), extract:
    - Statistical features: energy, max_psi, max_omega
    - Spatial features: radial profiles, mode amplitudes
    
    Total observation dim: ~50 floats (manageable for MLP)
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # New observation space: Box(50,)
        # [0:5]   - Statistical: energy, max_psi, max_omega, mean_psi, mean_omega
        # [5:21]  - Radial profiles: psi(r), omega(r) (8 points each)
        # [21:29] - Mode amplitudes: |ψ_n| for n=0..7 (toroidal modes)
        # [29:37] - Mode amplitudes: |ω_n| for n=0..7
        # [37:50] - Reserved for future diagnostics
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(50,), dtype=np.float32
        )
    
    def observation(self, obs_dict):
        """Extract simplified features from full observation."""
        psi = obs_dict['psi']
        omega = obs_dict['omega']
        energy = obs_dict['energy']
        max_psi = obs_dict['max_psi']
        max_omega = obs_dict['max_omega']
        
        # Statistical features
        features = [
            energy,
            max_psi,
            max_omega,
            np.mean(np.abs(psi)),
            np.mean(np.abs(omega)),
        ]
        
        # Radial profiles (average over θ, ζ)
        psi_r = np.mean(np.abs(psi), axis=(1, 2))  # (nr,)
        omega_r = np.mean(np.abs(omega), axis=(1, 2))  # (nr,)
        
        # Sample 8 radial points
        nr = psi_r.shape[0]
        r_indices = np.linspace(0, nr-1, 8, dtype=int)
        features.extend(psi_r[r_indices])
        features.extend(omega_r[r_indices])
        
        # Toroidal mode amplitudes (FFT along ζ axis)
        psi_fft = np.fft.rfft(psi, axis=2)  # (nr, nθ, nζ/2+1)
        omega_fft = np.fft.rfft(omega, axis=2)
        
        # Average over r, θ and take first 8 modes
        psi_modes = np.mean(np.abs(psi_fft), axis=(0, 1))[:8]
        omega_modes = np.mean(np.abs(omega_fft), axis=(0, 1))[:8]
        
        # Pad if needed
        psi_modes = np.pad(psi_modes, (0, max(0, 8-len(psi_modes))))
        omega_modes = np.pad(omega_modes, (0, max(0, 8-len(omega_modes))))
        
        features.extend(psi_modes)
        features.extend(omega_modes)
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)


def make_env():
    """Create and wrap MHD environment."""
    env = MHDEnv3D(
        grid_size=(16, 32, 16),  # Smaller grid for faster training
        eta=1e-3,  # Increased resistivity for stability
        dt=0.005,  # Reduced timestep for CFL stability
        max_steps=100,  # Episode length
        I_max=0.5,  # Reduced current magnitude
        n_coils=5,
    )
    env = SimplifiedObsWrapper(env)
    return env


def main():
    print("=" * 80)
    print("PPO Training for 3D MHD Control (v1.4)")
    print("=" * 80)
    
    # Create directories
    log_dir = Path("logs/ppo_mhd_v1_4")
    model_dir = Path("models")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = make_env()
    env = Monitor(env, str(log_dir / "monitor.csv"))
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = make_env()
    eval_env = Monitor(eval_env, str(log_dir / "eval_monitor.csv"))
    eval_env = DummyVecEnv([lambda: eval_env])
    
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    
    # Create PPO model
    print("\n[2/4] Creating PPO model...")
    model = PPO(
        "MlpPolicy",  # Use MLP for Box observation
        env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    
    # Configure CSV logging
    csv_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(csv_logger)
    
    print(f"    Policy: MlpPolicy (50 features → MLP → 5 actions)")
    print(f"    n_steps: 2048")
    print(f"    batch_size: 64")
    print(f"    learning_rate: 3e-4")
    print(f"    n_epochs: 10")
    print(f"    gamma: 0.99")
    
    # Create callbacks
    print("\n[3/4] Setting up callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=5000,  # Evaluate every 5k steps (~100 episodes)
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_mhd_v1_4",
    )
    
    # Train
    print("\n[4/4] Starting training...")
    print(f"    Total timesteps: 50,000 (1000 episodes × 50 steps)")
    print(f"    Checkpoints: Every 10k steps")
    print(f"    Evaluation: Every 5k steps (10 episodes)")
    print("-" * 80)
    
    try:
        model.learn(
            total_timesteps=50_000,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
        
        # Save final model
        final_path = model_dir / "ppo_mhd_v1_4_final.zip"
        model.save(str(final_path))
        
        print("\n" + "=" * 80)
        print("✅ Training complete!")
        print("=" * 80)
        print(f"Final model: {final_path}")
        print(f"Best model: {model_dir / 'best_model.zip'}")
        print(f"Logs: {log_dir}")
        print(f"TensorBoard: tensorboard --logdir {log_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        model.save(str(model_dir / "ppo_mhd_v1_4_interrupted.zip"))
        print(f"Saved interrupted model to {model_dir / 'ppo_mhd_v1_4_interrupted.zip'}")
    
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
