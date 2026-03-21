"""
Robustness Test for PPO v1.4: 48 IC Variations

Tests PPO performance across 48 different initial conditions:
- ε: [0.00005, 0.0001, 0.0002, 0.0005] (4 values)
- n: [3, 5, 7, 10] (4 toroidal modes)
- m₀: [1, 2, 3] (3 poloidal modes)

Metrics:
- Success rate (episode completion without crash)
- Energy drift |ΔE/E₀|
- Max |ψ|, max |ω|

Author: 小A 🤖
Date: 2026-03-20
Phase: 5B (Validation)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import itertools
from tqdm import tqdm
from test_env_wrapper import MHDEnv3DWithICParams, SimplifiedObsWrapper

def test_robustness(model_path, output_csv):
    """Test PPO on 48 IC configurations"""
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Parameter ranges
    epsilon_values = [0.00005, 0.0001, 0.0002, 0.0005]
    n_values = [3, 5, 7, 10]
    m0_values = [1, 2, 3]
    
    # Generate all combinations
    configs = list(itertools.product(epsilon_values, n_values, m0_values))
    print(f"Testing {len(configs)} IC configurations...")
    
    results = []
    
    for epsilon, n, m0 in tqdm(configs, desc="Robustness test"):
        try:
            # Create environment with specific IC
            base_env = MHDEnv3DWithICParams(
                grid_size=(16, 32, 16),
                eta=1e-3,
                dt=0.005,
                max_steps=100,
                mode_n=n,
                mode_m0=m0,
                epsilon_ic=epsilon,
            )
            env = SimplifiedObsWrapper(base_env)
            
            # Run one episode
            obs, info = env.reset()
            terminated = False
            truncated = False
            step = 0
            
            episode_rewards = []
            max_psi_values = []
            max_omega_values = []
            energy_drifts = []
            
            while not (terminated or truncated) and step < 100:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                max_psi_values.append(info.get('max_psi', 0.0))
                max_omega_values.append(info.get('max_omega', 0.0))
                energy_drifts.append(abs(info.get('energy_drift', 0.0)))
                
                step += 1
            
            # Episode completed successfully
            success = 1
            completion_rate = step / 100.0
            final_energy_drift = energy_drifts[-1] if energy_drifts else 0.0
            max_psi = max(max_psi_values) if max_psi_values else 0.0
            max_omega = max(max_omega_values) if max_omega_values else 0.0
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            
        except Exception as e:
            # Episode crashed
            print(f"  CRASH: ε={epsilon}, n={n}, m₀={m0}: {e}")
            success = 0
            completion_rate = 0.0
            final_energy_drift = np.nan
            max_psi = np.nan
            max_omega = np.nan
            mean_reward = np.nan
        
        # Record result
        results.append({
            'epsilon': epsilon,
            'n': n,
            'm0': m0,
            'success': success,
            'completion_rate': completion_rate,
            'final_energy_drift': final_energy_drift,
            'max_psi': max_psi,
            'max_omega': max_omega,
            'mean_reward': mean_reward,
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Statistical summary
    print("\n" + "="*60)
    print("ROBUSTNESS TEST SUMMARY")
    print("="*60)
    
    success_rate = df['success'].mean()
    print(f"Overall Success Rate: {success_rate:.1%} ({df['success'].sum()}/{len(df)})")
    
    # Success rate by parameter
    print("\nSuccess Rate by ε:")
    for eps in epsilon_values:
        rate = df[df['epsilon'] == eps]['success'].mean()
        print(f"  ε={eps:.5f}: {rate:.1%}")
    
    print("\nSuccess Rate by n:")
    for n in n_values:
        rate = df[df['n'] == n]['success'].mean()
        print(f"  n={n}: {rate:.1%}")
    
    print("\nSuccess Rate by m₀:")
    for m in m0_values:
        rate = df[df['m0'] == m]['success'].mean()
        print(f"  m₀={m}: {rate:.1%}")
    
    # Metrics for successful episodes
    successful = df[df['success'] == 1]
    if len(successful) > 0:
        print(f"\nMetrics (successful episodes only, n={len(successful)}):")
        print(f"  Energy drift: {successful['final_energy_drift'].mean():.2e} ± {successful['final_energy_drift'].std():.2e}")
        print(f"  Max |ψ|: {successful['max_psi'].mean():.2e} ± {successful['max_psi'].std():.2e}")
        print(f"  Max |ω|: {successful['max_omega'].mean():.2e} ± {successful['max_omega'].std():.2e}")
        print(f"  Mean reward: {successful['mean_reward'].mean():.4f} ± {successful['mean_reward'].std():.4f}")
    
    print(f"\nResults saved to: {output_csv}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    model_path = "models/ppo_mhd_v1_4_final.zip"
    output_csv = "results/phase5/robustness_results.csv"
    
    df = test_robustness(model_path, output_csv)
    
    # Success criterion: ≥80%
    success_rate = df['success'].mean()
    if success_rate >= 0.8:
        print(f"\n✅ PASS: Success rate {success_rate:.1%} ≥ 80%")
    else:
        print(f"\n❌ FAIL: Success rate {success_rate:.1%} < 80%")
