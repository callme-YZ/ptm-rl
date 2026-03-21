#!/usr/bin/env python3
"""
Visualization Script for MHD Control Results (v1.4)

Generate 4 plots:
1. Training curve (mean reward vs timesteps)
2. Policy comparison (box plot of rewards)
3. Energy trajectory (E(t) for 3 policies, single episode)
4. Action heatmap (5 coil currents over time, PPO policy)

Usage:
    python scripts/visualize_control_v1_4.py

Author: 小A 🤖
Created: 2026-03-20
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
from stable_baselines3 import PPO
from pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150


class SimplifiedObsWrapper(gym.ObservationWrapper):
    """Simplified observation wrapper (matches training)."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )
    
    def observation(self, obs_dict):
        """Extract simplified features."""
        psi = obs_dict['psi']
        omega = obs_dict['omega']
        features = [
            obs_dict['energy'], obs_dict['max_psi'], obs_dict['max_omega'],
            np.mean(np.abs(psi)), np.mean(np.abs(omega)),
        ]
        psi_r = np.mean(np.abs(psi), axis=(1, 2))
        omega_r = np.mean(np.abs(omega), axis=(1, 2))
        nr = psi_r.shape[0]
        r_indices = np.linspace(0, nr-1, 8, dtype=int)
        features.extend(psi_r[r_indices])
        features.extend(omega_r[r_indices])
        psi_fft = np.fft.rfft(psi, axis=2)
        omega_fft = np.fft.rfft(omega, axis=2)
        psi_modes = np.mean(np.abs(psi_fft), axis=(0, 1))[:8]
        omega_modes = np.mean(np.abs(omega_fft), axis=(0, 1))[:8]
        psi_modes = np.pad(psi_modes, (0, max(0, 8-len(psi_modes))))
        omega_modes = np.pad(omega_modes, (0, max(0, 8-len(omega_modes))))
        features.extend(psi_modes)
        features.extend(omega_modes)
        while len(features) < 50:
            features.append(0.0)
        return np.array(features[:50], dtype=np.float32)


def plot_training_curve(log_dir: Path, output_path: Path):
    """Plot 1: Training curve from monitor.csv."""
    print("\n[1/4] Plotting training curve...")
    
    # Load monitor CSV
    monitor_csv = log_dir / "monitor.csv"
    if not monitor_csv.exists():
        print(f"  ⚠️  Monitor CSV not found: {monitor_csv}")
        return
    
    # Read CSV (skip first line which is comment)
    df = pd.read_csv(monitor_csv, skiprows=1)
    
    # Calculate rolling mean
    window = 50  # Average over 50 episodes
    df['reward_smooth'] = df['r'].rolling(window=window, min_periods=1).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['l'].cumsum(), df['r'], alpha=0.3, label='Episode reward', linewidth=0.5)
    ax.plot(df['l'].cumsum(), df['reward_smooth'], label=f'Smoothed (window={window})', linewidth=2)
    
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('PPO Training Progress (v1.4)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def plot_policy_comparison(results_csv: Path, output_path: Path):
    """Plot 2: Box plot comparing 3 policies."""
    print("\n[2/4] Plotting policy comparison...")
    
    if not results_csv.exists():
        print(f"  ⚠️  Results CSV not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 2a: Mean reward comparison
    sns.boxplot(data=df, x='policy', y='mean_reward', ax=axes[0], order=['zero', 'random', 'ppo'])
    axes[0].set_xlabel('Policy', fontsize=12)
    axes[0].set_ylabel('Mean Reward', fontsize=12)
    axes[0].set_title('Reward Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticklabels(['Zero', 'Random', 'PPO'])
    
    # Plot 2b: Energy drift comparison
    sns.boxplot(data=df, x='policy', y='final_energy_drift', ax=axes[1], order=['zero', 'random', 'ppo'])
    axes[1].set_xlabel('Policy', fontsize=12)
    axes[1].set_ylabel('Final Energy Drift |ΔE/E₀|', fontsize=12)
    axes[1].set_title('Energy Conservation', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['Zero', 'Random', 'PPO'])
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def plot_energy_trajectory(model_path: Path, output_path: Path):
    """Plot 3: Energy trajectory E(t) for 3 policies (single episode)."""
    print("\n[3/4] Plotting energy trajectory...")
    
    # Create environment (match training params)
    env = MHDEnv3D(grid_size=(16, 32, 16), eta=1e-3, dt=0.005, max_steps=100, I_max=0.5, n_coils=5)
    env = SimplifiedObsWrapper(env)
    
    # Load PPO model
    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return
    
    ppo_model = PPO.load(str(model_path))
    
    # Run 3 policies for 1 episode each
    policies = {
        'Zero': lambda obs: np.zeros(5, dtype=np.float32),
        'Random': lambda obs: np.random.uniform(-1, 1, size=5).astype(np.float32),
        'PPO': lambda obs: ppo_model.predict(obs, deterministic=True)[0],
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for policy_name, policy_fn in policies.items():
        obs, info = env.reset(seed=42)  # Fixed seed for reproducibility
        energies = [info.get('energy', 1.0)]
        
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            energies.append(info.get('energy', energies[-1]))
            done = terminated or truncated
        
        time = np.arange(len(energies)) * env.unwrapped.dt
        ax.plot(time, energies, label=policy_name, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy E(t) / E₀', fontsize=12)
    ax.set_title('Energy Evolution Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def plot_action_heatmap(model_path: Path, output_path: Path):
    """Plot 4: Action heatmap (5 coil currents over time, PPO policy)."""
    print("\n[4/4] Plotting action heatmap...")
    
    # Create environment (match training params)
    env = MHDEnv3D(grid_size=(16, 32, 16), eta=1e-3, dt=0.005, max_steps=100, I_max=0.5, n_coils=5)
    env = SimplifiedObsWrapper(env)
    
    # Load PPO model
    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return
    
    ppo_model = PPO.load(str(model_path))
    
    # Run PPO policy for 1 episode
    obs, info = env.reset(seed=42)
    actions = []
    
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Convert to array (timesteps × 5 coils)
    actions = np.array(actions)
    time = np.arange(actions.shape[0]) * env.unwrapped.dt
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(
        actions.T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        interpolation='nearest',
        extent=[0, time[-1], 0.5, 5.5],
    )
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Coil Index', fontsize=12)
    ax.set_title('PPO Control Actions (Coil Currents)', fontsize=14, fontweight='bold')
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f'Coil {i}' for i in range(1, 6)])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Current', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("MHD Control Visualization (v1.4)")
    print("=" * 80)
    
    # Paths
    log_dir = Path("logs/ppo_mhd_v1_4")
    results_dir = Path("results/phase4")
    results_csv = results_dir / "evaluation_results.csv"
    model_path = Path("models/best_model.zip")
    
    # Create output directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_training_curve(log_dir, results_dir / "training_curve.png")
    plot_policy_comparison(results_csv, results_dir / "policy_comparison.png")
    plot_energy_trajectory(model_path, results_dir / "energy_trajectory.png")
    plot_action_heatmap(model_path, results_dir / "action_heatmap.png")
    
    print("\n" + "=" * 80)
    print("✅ All visualizations complete!")
    print("=" * 80)
    print(f"Output directory: {results_dir}")
    print(f"  - training_curve.png")
    print(f"  - policy_comparison.png")
    print(f"  - energy_trajectory.png")
    print(f"  - action_heatmap.png")


if __name__ == "__main__":
    main()
