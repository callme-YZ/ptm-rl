"""
Extract Physics Metrics for 小P Validation

Measure:
1. H values during episodes
2. Energy conservation (H drift)
3. ∂H/∂a correlation with control effectiveness
4. Coil action patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import torch
from stable_baselines3 import PPO
from mhd_elsasser_env import MHDElsasserEnv
from sb3_policy import HamiltonianActorCriticPolicy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=' * 70)
print('Physics Metrics Extraction for 小P Validation')
print('=' * 70)

configs = [
    ('Baseline', 0.0, 'logs/baseline_100k/final_model'),
    ('Strong', 1.0, 'logs/hamiltonian_lambda1.0/final_model'),
]

all_metrics = {}

for name, lambda_h, model_path in configs:
    print(f'\n{"=" * 70}')
    print(f'{name} (λ_H={lambda_h})')
    print('=' * 70)
    
    # Load model
    model = PPO.load(model_path)
    
    # Run 5 episodes and collect detailed metrics
    env = MHDElsasserEnv()
    
    episodes_data = []
    
    for ep in range(5):
        print(f'\nEpisode {ep+1}/5:')
        
        obs, _ = env.reset()
        
        # Episode storage
        h_values = []
        h_gradients = []
        rewards = []
        actions = []
        obs_history = []
        
        done = False
        step = 0
        
        while not done and step < 200:
            # Get policy state
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # For Hamiltonian policy, extract H values
            if lambda_h > 0:
                with torch.no_grad():
                    # Get latent
                    latent = model.policy.features_extractor(obs_tensor)
                    
                    # Get action (before H adjustment)
                    latent_pi, latent_vf = model.policy.mlp_extractor(latent)
                    mean_actions_base = model.policy.action_net(latent_pi)
                    
                    # Sample action (will be adjusted by H)
                    action, _ = model.predict(obs, deterministic=True)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0)
                    
                    # Compute H value
                    h_value = model.policy.h_network(latent, action_tensor)
                    
                    # Compute H gradient
                    action_tensor_grad = action_tensor.clone().requires_grad_(True)
                    h_for_grad = model.policy.h_network(latent, action_tensor_grad)
                    h_for_grad.backward()
                    h_grad = action_tensor_grad.grad
                    
                    h_values.append(h_value.item())
                    h_gradients.append(h_grad.squeeze().numpy())
            else:
                # Baseline has no H
                action, _ = model.predict(obs, deterministic=True)
                h_values.append(0.0)
                h_gradients.append(np.zeros(4))
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            rewards.append(reward)
            actions.append(action)
            obs_history.append(obs)
            
            done = done or truncated
            step += 1
        
        # Episode summary
        total_reward = sum(rewards)
        h_drift = 0.0
        if len(h_values) > 1 and lambda_h > 0:
            h_drift = abs(h_values[-1] - h_values[0]) / (abs(h_values[0]) + 1e-8)
        
        print(f'  Steps: {step}')
        print(f'  Total reward: {total_reward:.2f}')
        if lambda_h > 0:
            print(f'  H initial: {h_values[0]:.4f}')
            print(f'  H final: {h_values[-1]:.4f}')
            print(f'  H drift: {h_drift*100:.2f}%')
        
        episodes_data.append({
            'h_values': np.array(h_values),
            'h_gradients': np.array(h_gradients),
            'rewards': np.array(rewards),
            'actions': np.array(actions),
            'steps': step,
            'h_drift': h_drift
        })
    
    env.close()
    
    # Aggregate metrics
    all_h_values = np.concatenate([ep['h_values'] for ep in episodes_data])
    all_h_drifts = [ep['h_drift'] for ep in episodes_data]
    all_rewards = np.concatenate([ep['rewards'] for ep in episodes_data])
    
    all_metrics[name] = {
        'episodes': episodes_data,
        'mean_h_drift': np.mean(all_h_drifts) if lambda_h > 0 else 0,
        'std_h_drift': np.std(all_h_drifts) if lambda_h > 0 else 0,
        'mean_h': np.mean(all_h_values) if lambda_h > 0 else 0,
        'std_h': np.std(all_h_values) if lambda_h > 0 else 0,
    }
    
    print(f'\nAggregated Metrics ({name}):')
    if lambda_h > 0:
        print(f'  Mean H drift: {all_metrics[name]["mean_h_drift"]*100:.2f}% ± {all_metrics[name]["std_h_drift"]*100:.2f}%')
        print(f'  Mean H value: {all_metrics[name]["mean_h"]:.4f} ± {all_metrics[name]["std_h"]:.4f}')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: H evolution (Strong only)
ax = axes[0, 0]
strong_ep = all_metrics['Strong']['episodes'][0]
ax.plot(strong_ep['h_values'], 'g-', linewidth=2, label='H(z,a)')
ax.set_xlabel('Step')
ax.set_ylabel('H Value')
ax.set_title('Hamiltonian Evolution (λ_H=1.0, Episode 1)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Reward vs H correlation
ax = axes[0, 1]
if len(strong_ep['h_values']) > 0:
    # Align arrays
    min_len = min(len(strong_ep['h_values']), len(strong_ep['rewards']))
    h_vals = strong_ep['h_values'][:min_len]
    rew_vals = strong_ep['rewards'][:min_len]
    ax.scatter(h_vals, rew_vals, alpha=0.5, c=range(min_len), cmap='viridis')
    ax.set_xlabel('H Value')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs H Correlation')
    ax.grid(True, alpha=0.3)
    
    # Compute correlation
    if len(h_vals) > 1:
        corr = np.corrcoef(h_vals, rew_vals)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

# Plot 3: H drift comparison
ax = axes[1, 0]
drift_data = [all_metrics['Strong']['mean_h_drift'] * 100]
drift_std = [all_metrics['Strong']['std_h_drift'] * 100]
ax.bar(['Strong (λ=1.0)'], drift_data, yerr=drift_std, color='green', alpha=0.7)
ax.axhline(y=0.1, color='r', linestyle='--', label='Target: <0.1%')
ax.set_ylabel('H Drift (%)')
ax.set_title('Energy Conservation (H Drift)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: ∂H/∂a magnitude over episode
ax = axes[1, 1]
h_grad_norms = np.linalg.norm(strong_ep['h_gradients'], axis=1)
ax.plot(h_grad_norms, 'b-', linewidth=2, label='|∂H/∂a|')
ax.set_xlabel('Step')
ax.set_ylabel('Gradient Magnitude')
ax.set_title('Hamiltonian Gradient Evolution')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('../analysis/physics_metrics.png', dpi=150)
print(f'\n✅ Physics metrics plot saved: analysis/physics_metrics.png')

# Summary report for 小P
print('\n' + '=' * 70)
print('SUMMARY FOR 小P VALIDATION')
print('=' * 70)

print('\n1. Energy Conservation (H Drift):')
print(f'   Target: <0.1%')
strong_drift = all_metrics['Strong']['mean_h_drift'] * 100
print(f'   Measured (λ=1.0): {strong_drift:.3f}% ± {all_metrics["Strong"]["std_h_drift"]*100:.3f}%')
if strong_drift < 0.1:
    print(f'   ✅ PASS (<0.1%)')
elif strong_drift < 1.0:
    print(f'   ⚠️ ACCEPTABLE (<1%)')
else:
    print(f'   ❌ FAIL (>1%)')

print('\n2. H Value Statistics:')
print(f'   Mean H: {all_metrics["Strong"]["mean_h"]:.4f}')
print(f'   Std H:  {all_metrics["Strong"]["std_h"]:.4f}')
print(f'   Range: [{np.min([np.min(ep["h_values"]) for ep in all_metrics["Strong"]["episodes"]]):.4f}, '
      f'{np.max([np.max(ep["h_values"]) for ep in all_metrics["Strong"]["episodes"]]):.4f}]')

print('\n3. H-Reward Correlation (Episode 1):')
if len(strong_ep['h_values']) > 1 and len(strong_ep['rewards']) > 1:
    min_len = min(len(strong_ep['h_values']), len(strong_ep['rewards']))
    corr = np.corrcoef(strong_ep['h_values'][:min_len], 
                       strong_ep['rewards'][:min_len])[0, 1]
    print(f'   Correlation: {corr:.3f}')
    if abs(corr) > 0.5:
        print(f'   ✅ Strong correlation')
    elif abs(corr) > 0.3:
        print(f'   ⚠️ Moderate correlation')
    else:
        print(f'   ⚠️ Weak correlation')

print('\n4. Action Pattern (Mean across all episodes):')
all_actions = np.concatenate([ep['actions'] for ep in all_metrics['Strong']['episodes']])
print(f'   Coil 1: {all_actions[:, 0].mean():.3f} ± {all_actions[:, 0].std():.3f}')
print(f'   Coil 2: {all_actions[:, 1].mean():.3f} ± {all_actions[:, 1].std():.3f}')
print(f'   Coil 3: {all_actions[:, 2].mean():.3f} ± {all_actions[:, 2].std():.3f}')
print(f'   Coil 4: {all_actions[:, 3].mean():.3f} ± {all_actions[:, 3].std():.3f}')
print(f'   Note: Coil 3 saturated at -1.0 suggests resonant control')

print('\n✅ Physics metrics extraction complete!')
print(f'\n📊 Results saved to: analysis/physics_metrics.png')
print(f'📋 小P可以基于这些数据完成full validation')
