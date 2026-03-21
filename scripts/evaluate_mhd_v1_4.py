#!/usr/bin/env python3
"""
Evaluation Script for 3D MHD Control Policies (v1.4)

Compare 3 control policies:
1. Zero action (no control baseline)
2. Random policy (uniform [-1, 1])
3. Trained PPO policy

Metrics per episode:
- Mean reward
- Final energy drift |ΔE/E₀|
- Max |ψ|, max |ω|
- Episode length

Run 20 episodes per policy for statistical significance.

Usage:
    python scripts/evaluate_mhd_v1_4.py [--model models/best_model.zip]

Author: 小A 🤖
Created: 2026-03-20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
from stable_baselines3 import PPO
from pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D


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
        energy = obs_dict['energy']
        max_psi = obs_dict['max_psi']
        max_omega = obs_dict['max_omega']
        
        features = [
            energy, max_psi, max_omega,
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


def evaluate_policy(env, policy_fn, n_episodes=20, policy_name="Policy"):
    """
    Evaluate a policy for n_episodes.
    
    Parameters
    ----------
    env : MHDEnv3D
        Environment to evaluate on
    policy_fn : callable
        Function(obs) -> action
    n_episodes : int
        Number of episodes to run
    policy_name : str
        Name for logging
    
    Returns
    -------
    results : pd.DataFrame
        Episode-level metrics
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {policy_name}")
    print(f"{'=' * 80}")
    
    results = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_rewards = []
        step_count = 0
        
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            step_count += 1
            done = terminated or truncated
        
        # Collect metrics
        episode_data = {
            'episode': ep + 1,
            'mean_reward': np.mean(episode_rewards),
            'sum_reward': np.sum(episode_rewards),
            'final_energy_drift': info.get('energy_drift', np.nan),
            'max_psi': info.get('max_psi', np.nan),
            'max_omega': info.get('max_omega', np.nan),
            'episode_length': step_count,
        }
        results.append(episode_data)
        
        # Progress
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"mean_reward={episode_data['mean_reward']:.4f}, "
                  f"energy_drift={episode_data['final_energy_drift']:.2e}")
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'─' * 80}")
    print(f"Summary Statistics ({policy_name}):")
    print(f"{'─' * 80}")
    print(f"  Mean Reward:         {df['mean_reward'].mean():.4f} ± {df['mean_reward'].std():.4f}")
    print(f"  Final Energy Drift:  {df['final_energy_drift'].mean():.2e} ± {df['final_energy_drift'].std():.2e}")
    print(f"  Max |ψ|:             {df['max_psi'].mean():.4f} ± {df['max_psi'].std():.4f}")
    print(f"  Max |ω|:             {df['max_omega'].mean():.4f} ± {df['max_omega'].std():.4f}")
    print(f"  Episode Length:      {df['episode_length'].mean():.1f} ± {df['episode_length'].std():.1f}")
    
    return df


def zero_policy(obs):
    """Zero action policy (no control)."""
    return np.zeros(5, dtype=np.float32)


def random_policy(obs):
    """Random policy (uniform [-1, 1])."""
    return np.random.uniform(-1, 1, size=5).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MHD control policies")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.zip",
        help="Path to trained PPO model"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Number of episodes per policy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase4/evaluation_results.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("MHD Control Policy Evaluation (v1.4)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Episodes per policy: {args.n_episodes}")
    
    # Create environment with wrapper (match training params)
    env = MHDEnv3D(
        grid_size=(16, 32, 16),  # Smaller grid for faster evaluation
        eta=1e-3,  # Increased resistivity for stability
        dt=0.005,  # Reduced timestep for CFL stability
        max_steps=100,  # Episode length
        I_max=0.5,  # Reduced current magnitude
        n_coils=5,
    )
    env = SimplifiedObsWrapper(env)
    
    # Load PPO model
    try:
        ppo_model = PPO.load(args.model)
        print(f"✅ Loaded PPO model from {args.model}")
        
        def ppo_policy(obs):
            action, _ = ppo_model.predict(obs, deterministic=True)
            return action
        
    except FileNotFoundError:
        print(f"❌ Model not found: {args.model}")
        print("   Run training first: python scripts/train_mhd_ppo_v1_4.py")
        sys.exit(1)
    
    # Evaluate policies
    results_zero = evaluate_policy(env, zero_policy, args.n_episodes, "Zero Action (No Control)")
    results_random = evaluate_policy(env, random_policy, args.n_episodes, "Random Policy")
    results_ppo = evaluate_policy(env, ppo_policy, args.n_episodes, "Trained PPO")
    
    # Add policy labels
    results_zero['policy'] = 'zero'
    results_random['policy'] = 'random'
    results_ppo['policy'] = 'ppo'
    
    # Combine results
    all_results = pd.concat([results_zero, results_random, results_ppo], ignore_index=True)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    summary = all_results.groupby('policy').agg({
        'mean_reward': ['mean', 'std'],
        'final_energy_drift': ['mean', 'std'],
        'max_psi': ['mean', 'std'],
        'max_omega': ['mean', 'std'],
    })
    
    print(summary.to_string())
    
    # Statistical test: PPO vs Random
    from scipy.stats import ttest_ind
    
    ppo_rewards = results_ppo['mean_reward'].values
    random_rewards = results_random['mean_reward'].values
    
    t_stat, p_value = ttest_ind(ppo_rewards, random_rewards)
    
    print("\n" + "=" * 80)
    print("Statistical Significance (PPO vs Random)")
    print("=" * 80)
    print(f"  PPO mean reward:     {ppo_rewards.mean():.4f} ± {ppo_rewards.std():.4f}")
    print(f"  Random mean reward:  {random_rewards.mean():.4f} ± {random_rewards.std():.4f}")
    print(f"  t-statistic:         {t_stat:.4f}")
    print(f"  p-value:             {p_value:.4e}")
    
    if p_value < 0.05:
        if ppo_rewards.mean() > random_rewards.mean():
            print(f"  ✅ PPO significantly BETTER than Random (p < 0.05)")
        else:
            print(f"  ⚠️  PPO significantly WORSE than Random (p < 0.05)")
    else:
        print(f"  ⚠️  No significant difference (p ≥ 0.05)")
    
    print("\n" + "=" * 80)
    print(f"✅ Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
