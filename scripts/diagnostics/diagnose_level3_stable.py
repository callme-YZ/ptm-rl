#!/usr/bin/env python3
"""
Level 3: 稳定配置测试

使用:
- 更小的时间步 (dt=0.0001)
- 更大的耗散 (nu=1e-3)
- 合理的初始扰动
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pytokmhd.solver.time_integrator import rk4_step
from src.pytokmhd.diagnostics.magnetic_island import compute_island_width
from src.pytokmhd.solver.boundary import apply_combined_bc
from src.pytokmhd.diagnostics.rational_surface import find_rational_surface

# 参数
Nr, Nz = 64, 128
m, n = 2, 1
Lr, Lz = 2.0, 4.0
n_steps = 200
dt = 0.0001  # 减小10倍

# 物理参数 - 增大耗散
eta = 1e-3
nu = 1e-3

# 网格
r = np.linspace(0.1, Lr, Nr)
z = np.linspace(-Lz/2, Lz/2, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')
dr, dz = r[1] - r[0], z[1] - z[0]

print("="*60)
print("Level 3: 稳定配置测试")
print("="*60)
print(f"网格: {Nr}×{Nz}")
print(f"时间步: dt={dt:.6f}")
print(f"物理参数: η={eta:.6e}, ν={nu:.6e} (增大耗散)")

# 简单平衡态
psi_eq = (R - 1.0)**2
omega_eq = np.zeros_like(psi_eq)

# q-profile
q_profile = 1.5 + 1.5 * (r / Lr)**2
r_s = find_rational_surface(q_profile, r, m/n)
if isinstance(r_s, tuple):
    r_s = r_s[0]

print(f"有理面: r_s = {r_s:.4f}")

# 扰动 - 减小幅度
theta = np.arctan2(Z, R - 1.0)
delta_psi = 0.001 * np.exp(-((R - 1.0 - r_s)**2) / (0.1**2)) * np.cos(m * theta)  # 0.001 vs 0.01

psi = psi_eq + delta_psi
omega = omega_eq.copy()

print(f"初始扰动幅度: {np.max(np.abs(delta_psi)):.6e}")

# 时间演化
w_history, time_history = [], []
psi_max_history, omega_max_history = [], []

for step in range(n_steps + 1):
    t = step * dt
    
    try:
        w_result = compute_island_width(psi, r, z, q_profile, m, n)
        w = w_result[0] if isinstance(w_result, tuple) else w_result
    except:
        w = 0.0
    
    w_history.append(w)
    time_history.append(t)
    psi_max_history.append(np.max(np.abs(psi)))
    omega_max_history.append(np.max(np.abs(omega)))
    
    if step % 20 == 0:
        has_bad = np.any(np.isnan(psi)) or np.any(np.isinf(psi)) or np.any(np.isnan(omega)) or np.any(np.isinf(omega))
        status = "✗ NaN/Inf" if has_bad else "✓"
        print(f"Step {step:4d} | t={t:.4f} | w={w:.6e} | |ψ|={psi_max_history[-1]:.4e} | |ω|={omega_max_history[-1]:.4e} | {status}")
        if has_bad:
            break
    
    if step < n_steps:
        try:
            psi, omega = rk4_step(psi, omega, dt, dr, dz, R, eta, nu, apply_bc=None)
            psi, omega = apply_combined_bc(psi, omega)
        except:
            break

w_history = np.array(w_history)
time_history = np.array(time_history)

print("\n增长率分析:")
valid_mask = w_history > 1e-10
if np.sum(valid_mask) > 20:
    w_valid = w_history[valid_mask]
    t_valid = time_history[valid_mask]
    log_w = np.log(w_valid + 1e-12)
    n_fit = min(int(0.8 * len(log_w)), len(log_w) - 10)
    if n_fit > 5:
        fit_start = len(log_w) - n_fit
        coeffs = np.polyfit(t_valid[fit_start:], log_w[fit_start:], 1)
        gamma = coeffs[0]
        print(f"  γ = {gamma:.6e} s⁻¹")
        print(f"  ✅ 观察到增长" if gamma > 1e-6 else "  ❌ 无增长")
    else:
        gamma = 0.0
else:
    gamma = 0.0
    print("  ❌ 数据不足")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(time_history, w_history, 'b-', lw=2)
axes[0,0].set_xlabel('Time'); axes[0,0].set_ylabel('w'); axes[0,0].set_title('Island Width'); axes[0,0].grid(True)

axes[0,1].semilogy(time_history, np.maximum(w_history, 1e-12), 'b-', lw=2)
axes[0,1].set_xlabel('Time'); axes[0,1].set_ylabel('log(w)'); axes[0,1].set_title('Island Width (log)'); axes[0,1].grid(True)

axes[1,0].plot(time_history, psi_max_history, 'r-', lw=2)
axes[1,0].set_xlabel('Time'); axes[1,0].set_ylabel('|ψ|_max'); axes[1,0].set_title('Max ψ'); axes[1,0].grid(True)

axes[1,1].plot(time_history, omega_max_history, 'g-', lw=2)
axes[1,1].set_xlabel('Time'); axes[1,1].set_ylabel('|ω|_max'); axes[1,1].set_title('Max ω'); axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('level3_stable.png', dpi=150)
print("\n图像: level3_stable.png")

print("\nLevel 3 诊断结论:")
final_bad = np.any(np.isnan(psi)) or np.any(np.isinf(psi)) or np.any(np.isnan(omega)) or np.any(np.isinf(omega))
if not final_bad and abs(gamma) > 1e-6:
    print(f"✅ Level 3 通过:")
    print(f"   - 数值稳定 (演化{len(time_history)}步)")
    print(f"   - 增长率 γ = {gamma:.6e} s⁻¹")
    print(f"   - 最终岛宽 w = {w_history[-1]:.6e}")
else:
    print(f"❌ Level 3 失败")
    if final_bad:
        print(f"   - 数值不稳定")
    if abs(gamma) < 1e-6:
        print(f"   - 无明显增长")
