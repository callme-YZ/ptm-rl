"""
Long Episode Test for PPO v1.4: 500-step Episodes

Tests stability and control over 5× longer episodes than training.

Compare:
- PPO (trained policy)
- Zero (no control)
- Random (random actions)

Metrics:
- Episode completion rate
- Energy drift over long time
- Control stability (action variance)

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
from test_env_wrapper import MHDEnv3DWithICParams, SimplifiedObsWrapper

def run_policy(env, policy_type, model=None, n_episodes=10):
    """Run a policy for multiple episodes"""
    
    results = []
    
    for ep in tqdm(range(n_episodes), desc=f"{policy_type} episodes"):
        try:
            obs, info = env.reset()
            terminated = False
            truncated = False
            step = 0
            
            episode_rewards = []
            energy_values = []
            energy_drifts = []
            actions_taken = []
            
            while not (terminated or truncated) and step < 500:
                # Select action based on policy type
                if policy_type == "PPO":
                    action, _states = model.predict(obs, deterministic=True)
                elif policy_type == "Zero":
                    action = np.zeros(env.action_space.shape)
                elif policy_type == "Random":
                    action = env.action_space.sample()
                else:
                    raise ValueError(f"Unknown policy: {policy_type}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                energy_values.append(info.get('energy', 1.0))
                energy_drifts.append(abs(info.get('energy_drift', 0.0)))
                actions_taken.append(action)
                
                step += 1
            
            # Episode completed
            success = 1
            completion_rate = step / 500.0
            final_energy = energy_values[-1] if energy_values else 1.0
            final_energy_drift = energy_drifts[-1] if energy_drifts else 0.0
            max_energy_drift = max(energy_drifts) if energy_drifts else 0.0
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            
            # Action stability (variance of actions over time)
            if len(actions_taken) > 0:
                actions_array = np.array(actions_taken)
                action_variance = np.var(actions_array, axis=0).mean()
            else:
                action_variance = np.nan
            
        except Exception as e:
            print(f"  CRASH ({policy_type}, ep {ep}): {e}")
            success = 0
            completion_rate = 0.0
            final_energy = np.nan
            final_energy_drift = np.nan
            max_energy_drift = np.nan
            mean_reward = np.nan
            action_variance = np.nan
        
        results.append({
            'policy': policy_type,
            'episode': ep,
            'success': success,
            'completion_rate': completion_rate,
            'final_energy': final_energy,
            'final_energy_drift': final_energy_drift,
            'max_energy_drift': max_energy_drift,
            'mean_reward': mean_reward,
            'action_variance': action_variance,
        })
    
    return results

def test_long_episodes(model_path, output_csv, n_episodes=10):
    """Test PPO vs baselines on 500-step episodes"""
    
    print("Testing long episodes (500 steps, 5× training length)...")
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Create environment with 500 steps
    base_env = MHDEnv3DWithICParams(
        grid_size=(16, 32, 16),
        eta=1e-3,
        dt=0.005,
        max_steps=500,
        mode_n=5,
        mode_m0=2,
        epsilon_ic=0.0001,
    )
    env = SimplifiedObsWrapper(base_env)
    
    all_results = []
    
    # Test PPO
    print("\n1. Testing PPO...")
    all_results.extend(run_policy(env, "PPO", model=model, n_episodes=n_episodes))
    
    # Test Zero
    print("\n2. Testing Zero control...")
    all_results.extend(run_policy(env, "Zero", n_episodes=n_episodes))
    
    # Test Random
    print("\n3. Testing Random control...")
    all_results.extend(run_policy(env, "Random", n_episodes=n_episodes))
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    # Statistical summary
    print("\n" + "="*60)
    print("LONG EPISODE TEST SUMMARY")
    print("="*60)
    
    for policy in ["PPO", "Zero", "Random"]:
        policy_df = df[df['policy'] == policy]
        success_rate = policy_df['success'].mean()
        
        print(f"\n{policy}:")
        print(f"  Completion rate: {success_rate:.1%} ({policy_df['success'].sum()}/{len(policy_df)})")
        
        successful = policy_df[policy_df['success'] == 1]
        if len(successful) > 0:
            print(f"  Final energy drift: {successful['final_energy_drift'].mean():.2e} ± {successful['final_energy_drift'].std():.2e}")
            print(f"  Max energy drift: {successful['max_energy_drift'].mean():.2e} ± {successful['max_energy_drift'].std():.2e}")
            print(f"  Mean reward: {successful['mean_reward'].mean():.4f} ± {successful['mean_reward'].std():.4f}")
            print(f"  Action variance: {successful['action_variance'].mean():.4f} ± {successful['action_variance'].std():.4f}")
    
    print(f"\nResults saved to: {output_csv}")
    print("="*60)
    
    # Success criterion: PPO completes 100%
    ppo_df = df[df['policy'] == 'PPO']
    ppo_success = ppo_df['success'].mean()
    
    if ppo_success == 1.0:
        print(f"\n✅ PASS: PPO completion rate {ppo_success:.1%} = 100%")
    else:
        print(f"\n⚠️  WARNING: PPO completion rate {ppo_success:.1%} < 100%")
    
    return df

if __name__ == "__main__":
    model_path = "models/ppo_mhd_v1_4_final.zip"
    output_csv = "results/phase5/long_episode_results.csv"
    
    df = test_long_episodes(model_path, output_csv, n_episodes=10)
