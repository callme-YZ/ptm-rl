"""
Demo: 3D MHD Gym Environment (v1.4)

Demonstrates the MHDEnv3D environment with full target grid (32×64×32).

Usage:
    python examples/demo_mhd_env_v1_4.py

Author: 小A 🤖
Created: 2026-03-20
"""

import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl')

import numpy as np
import matplotlib.pyplot as plt
from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D


def main():
    """Run demo episode with random policy."""
    print("=" * 70)
    print("3D MHD Gym Environment Demo (v1.4)")
    print("=" * 70)
    
    # Create environment with target grid size
    print("\n1. Creating environment...")
    env = MHDEnv3D(
        grid_size=(32, 64, 32),  # Target grid for RL training
        eta=1e-4,
        dt=0.01,
        max_steps=50,
        I_max=1.0,
        n_coils=5
    )
    
    print(f"   Grid: {env.grid.nr} × {env.grid.ntheta} × {env.grid.nzeta}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    
    print(f"   Initial energy E₀: {info['E0']:.6e}")
    print(f"   ψ_max: {info['psi_max']:.6e}")
    print(f"   ω_max: {info['omega_max']:.6e}")
    
    # Run episode
    print("\n3. Running episode with random policy...")
    energies = [info['E0']]
    energy_drifts = [0.0]
    rewards = []
    max_psi_vals = [info['psi_max']]
    max_omega_vals = [info['omega_max']]
    
    for step in range(50):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record diagnostics
        energies.append(info['energy'])
        energy_drifts.append(info['energy_drift'])
        rewards.append(reward)
        max_psi_vals.append(info['max_psi'])
        max_omega_vals.append(info['max_omega'])
        
        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"   Step {step+1}/50: E/E₀={info['energy']/info['E0']:.6f}, "
                  f"drift={info['energy_drift']:.4e}, reward={reward:.4e}")
    
    # Final statistics
    print("\n4. Episode complete!")
    print(f"   Final energy drift: {energy_drifts[-1]:.4e}")
    print(f"   Mean reward: {np.mean(rewards):.4e}")
    print(f"   Total reward: {np.sum(rewards):.4e}")
    print(f"   Energy dissipation: {(energies[0] - energies[-1])/energies[0]*100:.2f}%")
    
    # Plot diagnostics
    print("\n5. Plotting diagnostics...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    time = np.arange(len(energies)) * env.dt
    
    # Energy evolution
    axes[0, 0].plot(time, np.array(energies) / energies[0], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('E/E₀')
    axes[0, 0].set_title('Normalized Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy drift
    axes[0, 1].semilogy(time, np.maximum(energy_drifts, 1e-10), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('|ΔE/E₀|')
    axes[0, 1].set_title('Energy Drift (log scale)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward per step
    axes[1, 0].plot(time[1:], rewards, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Signal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max field amplitudes
    axes[1, 1].plot(time, max_psi_vals, 'b-', label='max|ψ|', linewidth=2)
    axes[1, 1].plot(time, max_omega_vals, 'r-', label='max|ω|', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Field amplitude')
    axes[1, 1].set_title('Field Maxima')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/examples/mhd_env_v1_4_demo.png', dpi=150)
    print(f"   Saved plot: examples/mhd_env_v1_4_demo.png")
    
    print("\n" + "=" * 70)
    print("Demo complete! Environment ready for RL training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
