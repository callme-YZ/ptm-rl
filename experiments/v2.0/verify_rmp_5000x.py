"""
Verify RMP Control Effect (5000× scaling)

Author: 小A 🤖
Date: 2026-03-20

Test if 5e-3 scaling gives ~10% control effect.
"""

from mhd_elsasser_env import MHDElsasserEnv
import numpy as np

print('='*60)
print('RMP Control Verification (5000× Scaling)')
print('='*60)

env = MHDElsasserEnv(grid_shape=(16,32,16), max_episode_steps=30)

# Test 1: Baseline (no control)
print('\n1. Baseline (zero action)')
obs, info = env.reset(seed=42)
m1_baseline = [info.get('m1_amplitude', 0)]
for i in range(19):
    obs, r, done, trunc, info = env.step(np.zeros(4))
    m1_baseline.append(info.get('m1_amplitude', 0))
    if done or trunc:
        print(f'   Terminated at step {i+1}')
        break

print(f'   Initial: {m1_baseline[0]:.6f}')
print(f'   Final:   {m1_baseline[-1]:.6f}')

# Test 2: Full positive control
print('\n2. Full RMP (+40kA all coils)')
obs, info = env.reset(seed=42)
m1_pos = [info.get('m1_amplitude', 0)]
for i in range(19):
    obs, r, done, trunc, info = env.step(np.ones(4))
    m1_pos.append(info.get('m1_amplitude', 0))
    if done or trunc:
        print(f'   Terminated at step {i+1}')
        break

print(f'   Initial: {m1_pos[0]:.6f}')
print(f'   Final:   {m1_pos[-1]:.6f}')

# Test 3: Full negative control
print('\n3. Full RMP (-40kA all coils)')
obs, info = env.reset(seed=42)
m1_neg = [info.get('m1_amplitude', 0)]
for i in range(19):
    obs, r, done, trunc, info = env.step(-np.ones(4))
    m1_neg.append(info.get('m1_amplitude', 0))
    if done or trunc:
        print(f'   Terminated at step {i+1}')
        break

print(f'   Initial: {m1_neg[0]:.6f}')
print(f'   Final:   {m1_neg[-1]:.6f}')

# Analysis
print('\n' + '='*60)
print('Effect Analysis')
print('='*60)

min_len = min(len(m1_baseline), len(m1_pos), len(m1_neg))

# Intermediate checkpoints
print('\nStep-by-step comparison:')
print('Step  | Baseline | +Control | -Control | +Effect% | -Effect%')
print('------|----------|----------|----------|----------|----------')
for i in [0, 4, 9, 14, 19]:
    if i < min_len:
        base = m1_baseline[i]
        pos_diff = abs(m1_pos[i] - base)
        neg_diff = abs(m1_neg[i] - base)
        pos_pct = pos_diff/base*100 if base > 0 else 0
        neg_pct = neg_diff/base*100 if base > 0 else 0
        print(f'{i:5d} | {base:.6f} | {m1_pos[i]:.6f} | {m1_neg[i]:.6f} | '
              f'{pos_pct:6.1f}% | {neg_pct:6.1f}%')

# Final assessment
base_final = m1_baseline[min_len-1]
pos_final = m1_pos[min_len-1]
neg_final = m1_neg[min_len-1]

pos_effect = abs(pos_final - base_final) / base_final * 100
neg_effect = abs(neg_final - base_final) / base_final * 100
max_effect = max(pos_effect, neg_effect)

print('\n' + '='*60)
print('Final Assessment')
print('='*60)
print(f'Baseline final m1:   {base_final:.6f}')
print(f'+Control final m1:   {pos_final:.6f}')
print(f'-Control final m1:   {neg_final:.6f}')
print(f'\n+RMP effect: {pos_effect:.1f}%')
print(f'-RMP effect: {neg_effect:.1f}%')
print(f'Max effect:  {max_effect:.1f}%')

print('\n' + '='*60)
if max_effect >= 10:
    print('✅ EXCELLENT: RMP control effect ≥10%')
    print('   Physics realistic ✅')
    print('   RL trainable ✅')
    print('   Ready for Phase 4.5 (training)! 🚀')
elif max_effect >= 5:
    print('✅ GOOD: RMP control effect 5-10%')
    print('   Acceptable for RL training')
    print('   May tune further if needed')
elif max_effect >= 2:
    print('⚠️  WEAK: RMP control effect 2-5%')
    print('   RL may struggle')
    print('   Recommend 2× more scaling')
else:
    print('❌ TOO WEAK: RMP control effect <2%')
    print('   Not ready for RL')
    print('   Need 5-10× more scaling')

print('='*60)
