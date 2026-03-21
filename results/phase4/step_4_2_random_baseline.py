#!/usr/bin/env python3
"""
Phase 4 Step 4.2: Random Baseline Evaluation
Option A: w_constraint = 0.0 (remove div_B penalty)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv

def evaluate_random_baseline(n_episodes=10):
    """Evaluate random policy baseline."""
    
    # Create environment with Option A config
    env = ToroidalMHDEnv(
        nr=32,
        ntheta=64,
        max_steps=100,
        w_energy=1.0,
        w_action=0.1,
        w_constraint=0.0  # Option A: remove div_B
    )
    
    returns = []
    energy_drifts = []
    reward_components = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        done = False
        truncated = False
        step = 0
        
        # Track components
        ep_energy = 0.0
        ep_action = 0.0
        ep_constraint = 0.0
        
        while not (done or truncated):
            # Random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            step += 1
            
            # Accumulate components
            ep_energy += info.get('reward_energy', 0.0)
            ep_action += info.get('reward_action', 0.0)
            ep_constraint += info.get('reward_constraint', 0.0)
        
        returns.append(episode_return)
        energy_drifts.append(info['energy_drift'])
        reward_components.append({
            'energy': ep_energy,
            'action': ep_action,
            'constraint': ep_constraint,
            'steps': step
        })
        
        print(f"Episode {ep+1}/{n_episodes}: return={episode_return:.3f}, "
              f"energy_drift={info['energy_drift']:.4f}, steps={step}")
    
    # Statistics
    returns = np.array(returns)
    energy_drifts = np.array(energy_drifts)
    
    print("\n" + "="*60)
    print("RANDOM BASELINE STATISTICS (Option A)")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean return:   {returns.mean():.3f} ± {returns.std():.3f}")
    print(f"Range:         [{returns.min():.3f}, {returns.max():.3f}]")
    print(f"\nMean energy drift: {energy_drifts.mean():.4f} ± {energy_drifts.std():.4f}")
    print(f"Range:             [{energy_drifts.min():.4f}, {energy_drifts.max():.4f}]")
    
    # Reward component breakdown (sum over all episodes)
    total_energy = sum(rc['energy'] for rc in reward_components)
    total_action = sum(rc['action'] for rc in reward_components)
    total_constraint = sum(rc['constraint'] for rc in reward_components)
    total_reward = total_energy + total_action + total_constraint
    
    print(f"\nReward Components (cumulative, {n_episodes} episodes × 100 steps):")
    print(f"  Energy:     {total_energy:8.3f} ({100*total_energy/total_reward:5.1f}%)")
    print(f"  Action:     {total_action:8.3f} ({100*total_action/total_reward:5.1f}%)")
    print(f"  Constraint: {total_constraint:8.3f} ({100*total_constraint/total_reward:5.1f}%)")
    print(f"  Total:      {total_reward:8.3f}")
    
    # Save results
    results = {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'mean_energy_drift': float(energy_drifts.mean()),
        'std_energy_drift': float(energy_drifts.std()),
        'n_episodes': n_episodes,
        'config': 'Option A (w_constraint=0.0)',
        'reward_components': {
            'energy': float(total_energy),
            'action': float(total_action),
            'constraint': float(total_constraint),
            'energy_pct': float(100*total_energy/total_reward),
            'action_pct': float(100*total_action/total_reward),
            'constraint_pct': float(100*total_constraint/total_reward)
        }
    }
    
    import json
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    evaluate_random_baseline(n_episodes=10)
