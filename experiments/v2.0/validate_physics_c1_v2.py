"""
C1: Growth Rate Verification (v2 - Velocity Measurement)

Measure ballooning mode growth rate using velocity field.

Theory: γ ~ ωA × √(β/ε)
  where ωA ~ 1 (normalized Alfvén frequency)
        β ~ 0.17 (from equilibrium)
        ε ~ 0.32 (inverse aspect ratio)

Expected: γ ~ √(0.17/0.32) ~ 0.73

Author: 小A 🤖
Date: 2026-03-21
"""

import numpy as np
import matplotlib.pyplot as plt
from mhd_elsasser_env import MHDElsasserEnv
import os

print('='*60)
print('C1: Growth Rate Verification (v2 - Velocity)')
print('='*60)

# Create environment
env = MHDElsasserEnv(grid_shape=(16,32,16), max_episode_steps=200)

# Run uncontrolled episode (zero RMP)
print('\nRunning uncontrolled episode (200 steps)...')
obs, info = env.reset(seed=42)

m2_trace = [info.get('m2_amplitude', 0)]
energy_trace = [info.get('energy', 0)]
time_steps = [0]

for i in range(200):
    action = np.zeros(4)  # No RMP control
    obs, reward, done, trunc, info = env.step(action)
    
    m2_trace.append(info.get('m2_amplitude', 0))
    energy_trace.append(info.get('energy', 0))
    time_steps.append((i+1) * 0.02)  # dt_rl = 0.02
    
    if done or trunc:
        print(f'Episode terminated at step {i+1}')
        break

print(f'Collected {len(m2_trace)} data points')

# Show first few values
print('\nFirst 10 timesteps:')
print('Time   | m2 amplitude')
print('-------|-------------')
for i in range(min(10, len(m2_trace))):
    print(f'{time_steps[i]:6.2f} | {m2_trace[i]:.6f}')

# Exponential fit for growth rate
# m2(t) = m2_0 * exp(γ*t)
# log(m2) = log(m2_0) + γ*t

# Use middle portion (skip initial transient and saturation)
start_idx = 10
end_idx = min(100, len(m2_trace)-1)

t_fit = np.array(time_steps[start_idx:end_idx])
m2_fit = np.array(m2_trace[start_idx:end_idx])

# Filter out zeros/near-zeros
valid = m2_fit > 1e-6
t_fit = t_fit[valid]
m2_fit = m2_fit[valid]

if len(t_fit) > 2:
    log_m2 = np.log(m2_fit)
    
    # Linear fit: log(m2) = a + γ*t
    coeffs = np.polyfit(t_fit, log_m2, 1)
    gamma_measured = coeffs[0]
    
    print('\n' + '='*60)
    print('Growth Rate Analysis')
    print('='*60)
    
    # Theory prediction
    beta = 0.17
    epsilon = 0.32
    gamma_theory = np.sqrt(beta / epsilon)
    
    print(f'Theory prediction:')
    print(f'  β = {beta:.3f}')
    print(f'  ε = {epsilon:.3f}')
    print(f'  γ_theory = √(β/ε) = {gamma_theory:.3f}')
    print(f'\nMeasured from simulation:')
    print(f'  γ_measured = {gamma_measured:.3f}')
    print(f'  Fit range: t={t_fit[0]:.2f} to {t_fit[-1]:.2f}')
    print(f'\nComparison:')
    print(f'  Ratio: γ_measured/γ_theory = {gamma_measured/gamma_theory:.3f}')
    
    error = abs(gamma_measured - gamma_theory) / gamma_theory * 100
    print(f'  Error: {error:.1f}%')
    
    if error < 20:
        print('\n✅ Growth rate matches theory (<20% error)')
        result = '✅ PASS'
    elif error < 50:
        print('\n⚠️  Moderate agreement (20-50% error)')
        result = '⚠️ MODERATE'
    else:
        print('\n❌ Poor agreement (>50% error)')
        result = '❌ FAIL'
    
    # Plot
    os.makedirs('./validation_results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear plot
    axes[0].plot(time_steps, m2_trace, 'b-', linewidth=2, label='Simulation')
    axes[0].axvline(t_fit[0], color='r', linestyle='--', alpha=0.5, label='Fit range')
    axes[0].axvline(t_fit[-1], color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('m=2 Amplitude (velocity)')
    axes[0].set_title('Ballooning Mode Evolution (v2 - Velocity Measurement)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Semi-log plot with fit
    axes[1].semilogy(time_steps, m2_trace, 'b-', linewidth=2, label='Simulation')
    
    # Theory line
    t_theory = np.linspace(t_fit[0], t_fit[-1], 100)
    m2_theory = m2_fit[0] * np.exp(gamma_theory * (t_theory - t_fit[0]))
    axes[1].semilogy(t_theory, m2_theory, 'r--', linewidth=2, 
                     label=f'Theory (γ={gamma_theory:.3f})')
    
    # Measured fit line
    m2_measured_fit = m2_fit[0] * np.exp(gamma_measured * (t_theory - t_fit[0]))
    axes[1].semilogy(t_theory, m2_measured_fit, 'g:', linewidth=2,
                     label=f'Fit (γ={gamma_measured:.3f})')
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('m=2 Amplitude (log scale)')
    axes[1].set_title('Exponential Growth Verification')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('./validation_results/c1_growth_rate_v2.png', dpi=150)
    print(f'\n✅ Plot saved to ./validation_results/c1_growth_rate_v2.png')
    
    # Save data
    np.savez('./validation_results/c1_growth_data_v2.npz',
             time=time_steps,
             m2=m2_trace,
             energy=energy_trace,
             gamma_measured=gamma_measured,
             gamma_theory=gamma_theory)
    print('✅ Data saved to ./validation_results/c1_growth_data_v2.npz')
    
else:
    print('\n❌ Not enough valid data points for fitting')
    result = '❌ FAIL'

print('\n' + '='*60)
print(f'C1 v2 Complete: {result}')
print('='*60)
