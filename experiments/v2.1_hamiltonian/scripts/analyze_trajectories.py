"""
Analysis 2: Trajectory Comparison
Compare physics metrics between baseline and λ=1.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v2.0'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from stable_baselines3 import PPO
from mhd_elsasser_env import MHDElsasserEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=' * 70)
print('Analysis 2: Trajectory Comparison')
print('=' * 70)

configs = [
    ('Baseline', 'logs/baseline_100k/final_model.zip', 'gray'),
    ('Strong (λ=1.0)', 'logs/hamiltonian_lambda1.0/final_model.zip', 'green'),
]

trajectories = {}

for name, model_path, color in configs:
    print(f'\nRunning {name}...')
    
    # Load model
    model = PPO.load(model_path)
    
    # Run episode with physics tracking
    env = MHDElsasserEnv()
    obs, _ = env.reset()
    
    # Track variables
    rewards = []
    actions_history = []
    done = False
    step = 0
    
    while not done and step < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)
        actions_history.append(action)
        
        done = done or truncated
        step += 1
    
    trajectories[name] = {
        'rewards': np.array(rewards),
        'actions': np.array(actions_history),
        'steps': step,
        'total_reward': sum(rewards),
        'color': color
    }
    
    print(f'  Steps: {step}')
    print(f'  Total reward: {sum(rewards):.2f}')
    print(f'  Mean reward: {np.mean(rewards):.4f}')
    
    env.close()

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Subplot 1: Rewards over time
ax1 = axes[0]
for name in ['Baseline', 'Strong (λ=1.0)']:
    data = trajectories[name]
    ax1.plot(data['rewards'], label=name, color=data['color'], linewidth=2)

ax1.set_xlabel('Step', fontsize=11)
ax1.set_ylabel('Reward', fontsize=11)
ax1.set_title('Reward Evolution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Action magnitude
ax2 = axes[1]
for name in ['Baseline', 'Strong (λ=1.0)']:
    data = trajectories[name]
    action_norms = np.linalg.norm(data['actions'], axis=1)
    ax2.plot(action_norms, label=name, color=data['color'], linewidth=2)

ax2.set_xlabel('Step', fontsize=11)
ax2.set_ylabel('|Action|', fontsize=11)
ax2.set_title('Control Effort', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/trajectory_comparison.png', dpi=150)
print(f'\n✅ Trajectory comparison saved: analysis/trajectory_comparison.png')

# Summary statistics
print('\n' + '=' * 70)
print('Trajectory Summary:')
print('-' * 70)
print(f'{"Config":<20} {"Steps":<8} {"Total Reward":<15} {"Mean Reward":<15}')
print('-' * 70)

for name in ['Baseline', 'Strong (λ=1.0)']:
    data = trajectories[name]
    print(f'{name:<20} {data["steps"]:<8} {data["total_reward"]:<15.2f} {np.mean(data["rewards"]):<15.4f}')

# Action comparison
print('\n' + '=' * 70)
print('Control Strategy Comparison:')
print('-' * 70)

for name in ['Baseline', 'Strong (λ=1.0)']:
    data = trajectories[name]
    actions = data['actions']
    action_norms = np.linalg.norm(actions, axis=1)
    
    print(f'\n{name}:')
    print(f'  Mean |action|: {action_norms.mean():.4f}')
    print(f'  Max |action|:  {action_norms.max():.4f}')
    print(f'  Std |action|:  {action_norms.std():.4f}')
    print(f'  Action means:  {actions.mean(axis=0)}')

print('\n✅ Analysis 2 complete')
