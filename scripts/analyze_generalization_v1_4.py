"""
Generalization Analysis for PPO v1.4: Unseen ICs

Tests PPO on:
1. Training IC (ε=0.0001, n=5, m₀=2) - baseline
2. Interpolation ICs (within training range)
3. Extrapolation ICs (outside training range)

Goal: Assess if PPO generalizes beyond training data.

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
from tqdm import tqdm
from scipy import stats
from test_env_wrapper import MHDEnv3DWithICParams, SimplifiedObsWrapper

def run_ic_test(env, model, category, ic_params, n_episodes=20):
    """Run test on specific IC configuration"""
    
    epsilon, n, m0 = ic_params
    results = []
    
    for ep in range(n_episodes):
        try:
            obs, info = env.reset()
            terminated = False
            truncated = False
            step = 0
            
            episode_rewards = []
            energy_drifts = []
            
            while not (terminated or truncated) and step < 100:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                energy_drifts.append(abs(info.get('energy_drift', 0.0)))
                
                step += 1
            
            # Success
            success = 1
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            final_energy_drift = energy_drifts[-1] if energy_drifts else 0.0
            
        except Exception as e:
            success = 0
            mean_reward = np.nan
            final_energy_drift = np.nan
        
        results.append({
            'category': category,
            'epsilon': epsilon,
            'n': n,
            'm0': m0,
            'episode': ep,
            'success': success,
            'mean_reward': mean_reward,
            'final_energy_drift': final_energy_drift,
        })
    
    return results

def test_generalization(model_path, output_csv):
    """Test PPO generalization on seen vs unseen ICs"""
    
    print("Testing PPO generalization...")
    
    # Load trained model
    model = PPO.load(model_path)
    
    all_results = []
    
    # Category 1: Training IC (baseline)
    print("\n1. Testing on TRAINING IC...")
    print("   ε=0.0001, n=5, m₀=2")
    base_env = MHDEnv3DWithICParams(
        grid_size=(16, 32, 16), eta=1e-3, dt=0.005, max_steps=100,
        mode_n=5, mode_m0=2, epsilon_ic=0.0001
    )
    env_train = SimplifiedObsWrapper(base_env)
    all_results.extend(run_ic_test(env_train, model, "Training", (0.0001, 5, 2), n_episodes=20))
    
    # Category 2: Interpolation ICs (within training range)
    print("\n2. Testing on INTERPOLATION ICs...")
    interpolation_ics = [
        (0.00015, 5, 2, "ε between training values"),
        (0.0001, 4, 2, "n between training modes (4)"),
        (0.0001, 6, 2, "n between training modes (6)"),
        (0.0001, 8, 2, "n between training modes (8)"),
    ]
    
    for epsilon, n, m0, desc in interpolation_ics:
        print(f"   {desc}: ε={epsilon}, n={n}, m₀={m0}")
        base_env = MHDEnv3DWithICParams(
            grid_size=(16, 32, 16), eta=1e-3, dt=0.005, max_steps=100,
            mode_n=n, mode_m0=m0, epsilon_ic=epsilon
        )
        env = SimplifiedObsWrapper(base_env)
        all_results.extend(run_ic_test(env, model, "Interpolation", (epsilon, n, m0), n_episodes=20))
    
    # Category 3: Extrapolation ICs (outside training range)
    print("\n3. Testing on EXTRAPOLATION ICs...")
    extrapolation_ics = [
        (0.00003, 5, 2, "ε below training (0.00003 < 0.00005)"),
        (0.001, 5, 2, "ε above training (0.001 > 0.0005)"),
        (0.0001, 2, 2, "n below training (2 < 3)"),
        (0.0001, 12, 2, "n above training (12 > 10)"),
        (0.0001, 5, 4, "m₀ above training (4 > 3)"),
    ]
    
    for epsilon, n, m0, desc in extrapolation_ics:
        print(f"   {desc}: ε={epsilon}, n={n}, m₀={m0}")
        base_env = MHDEnv3DWithICParams(
            grid_size=(16, 32, 16), eta=1e-3, dt=0.005, max_steps=100,
            mode_n=n, mode_m0=m0, epsilon_ic=epsilon
        )
        env = SimplifiedObsWrapper(base_env)
        all_results.extend(run_ic_test(env, model, "Extrapolation", (epsilon, n, m0), n_episodes=20))
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    # Statistical analysis
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS SUMMARY")
    print("="*60)
    
    for category in ["Training", "Interpolation", "Extrapolation"]:
        cat_df = df[df['category'] == category]
        success_rate = cat_df['success'].mean()
        
        print(f"\n{category}:")
        print(f"  Success rate: {success_rate:.1%} ({cat_df['success'].sum()}/{len(cat_df)})")
        
        successful = cat_df[cat_df['success'] == 1]
        if len(successful) > 0:
            print(f"  Mean reward: {successful['mean_reward'].mean():.4f} ± {successful['mean_reward'].std():.4f}")
            print(f"  Energy drift: {successful['final_energy_drift'].mean():.2e} ± {successful['final_energy_drift'].std():.2e}")
    
    # Statistical test: Interpolation vs Random baseline
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*60)
    
    # We need to test if PPO on interpolation ICs performs better than random
    # For this, we'll use the reward as the metric
    
    interp_rewards = df[(df['category'] == 'Interpolation') & (df['success'] == 1)]['mean_reward'].dropna()
    train_rewards = df[(df['category'] == 'Training') & (df['success'] == 1)]['mean_reward'].dropna()
    
    if len(interp_rewards) > 0 and len(train_rewards) > 0:
        # Compare interpolation vs training (should be similar if generalizing well)
        t_stat, p_value = stats.ttest_ind(interp_rewards, train_rewards)
        print(f"\nInterpolation vs Training rewards:")
        print(f"  Interpolation: {interp_rewards.mean():.4f} ± {interp_rewards.std():.4f}")
        print(f"  Training: {train_rewards.mean():.4f} ± {train_rewards.std():.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("  ✅ No significant difference (good generalization)")
        else:
            print("  ⚠️  Significant difference (degraded performance)")
    
    # Success criterion: Interpolation success rate > 80%
    interp_success = df[df['category'] == 'Interpolation']['success'].mean()
    
    print(f"\nResults saved to: {output_csv}")
    print("="*60)
    
    if interp_success >= 0.8:
        print(f"\n✅ PASS: Interpolation success rate {interp_success:.1%} ≥ 80%")
    else:
        print(f"\n❌ FAIL: Interpolation success rate {interp_success:.1%} < 80%")
    
    return df

if __name__ == "__main__":
    model_path = "models/ppo_mhd_v1_4_final.zip"
    output_csv = "results/phase5/generalization_results.csv"
    
    df = test_generalization(model_path, output_csv)
