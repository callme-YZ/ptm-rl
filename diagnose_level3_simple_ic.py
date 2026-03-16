#!/usr/bin/env python3
"""
Level 3: 使用简化初始条件测试自由增长

避免Solovev平衡态的大ω问题，使用简单的平衡态+扰动
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pytokmhd.solver.time_integrator import rk4_step
from src.pytokmhd.diagnostics.magnetic_island import compute_island_width
from src.pytokmhd.solver.boundary import apply_combined_bc
from src.pytokmhd.diagnostics.rational_surface import find_rational_surface

# 设置参数
Nr, Nz = 64, 128
m, n = 2, 1
Lr, Lz = 2.0, 4.0
n_steps = 100
dt = 0.0005  # 更小的时间步

# 物理参数
eta = 1e-4
nu = 1e-4

# 创建网格
r = np.linspace(0.1, Lr, Nr)
z = np.linspace(-Lz/2, Lz/2, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')
dr = r[1] - r[0]
dz = z[1] - z[0]

print("="*60)
print("Level 3: 简化初始条件测试")
print("="*60)
print(f"网格: {Nr}×{Nz}")
print(f"时间步: dt={dt:.6f}")
print(f"演化步数: {n_steps}")

# 简单平衡态：ψ_eq = (r/Lr)² 的抛物线
# 这给出 ∇²ψ = 2/Lr² + 2/(r*Lr²) ≈ 常数（小）
psi_eq = (R - 1.0)**2  # 以R=1为中心
omega_eq = np.zeros_like(psi_eq)  # 简化：ω_eq = 0

# q-profile (线性)
q_profile = 1.5 + 1.5 * (r / Lr)**2

# 找有理面
r_s = find_rational_surface(q_profile, r, m/n)
if isinstance(r_s, tuple):
    r_s = r_s[0]
print(f"\n有理面位置: r_s = {r_s:.4f}")

# 添加扰动 (tearing mode)
theta = np.arctan2(Z, R - 1.0)
delta_psi = 0.01 * np.exp(-((R - 1.0 - r_s)**2) / (0.1**2)) * np.cos(m * theta)

psi = psi_eq + delta_psi
omega = omega_eq.copy()  # ω = 0 初始

print(f"\n初始条件:")
print(f"  ψ: min={np.min(psi):.6e}, max={np.max(psi):.6e}")
print(f"  ω: min={np.min(omega):.6e}, max={np.max(omega):.6e}")
print(f"  ω幅度: {np.max(np.abs(omega)):.6e}")

# 时间演化
w_history = []
psi_max_history = []
omega_max_history = []
time_history = []

print("\n" + "="*60)
print("开始时间演化")
print("="*60)

for step in range(n_steps + 1):
    t = step * dt
    
    # 计算磁岛宽度
    try:
        w_result = compute_island_width(psi, r, z, q_profile, m, n)
        if isinstance(w_result, tuple):
            w = w_result[0]
        else:
            w = w_result
    except:
        w = 0.0
    
    w_history.append(w)
    psi_max_history.append(np.max(np.abs(psi)))
    omega_max_history.append(np.max(np.abs(omega)))
    time_history.append(t)
    
    if step % 10 == 0:
        has_nan = np.any(np.isnan(psi)) or np.any(np.isnan(omega))
        has_inf = np.any(np.isinf(psi)) or np.any(np.isinf(omega))
        
        status = "✓"
        if has_nan or has_inf:
            status = "✗ NaN/Inf"
        
        print(f"Step {step:4d} | t={t:.4f} | w={w:.6e} | "
              f"|ψ|={psi_max_history[-1]:.4e} | "
              f"|ω|={omega_max_history[-1]:.4e} | {status}")
        
        if has_nan or has_inf:
            print("\n❌ 数值崩溃")
            break
    
    # 时间演化
    if step < n_steps:
        try:
            psi, omega = rk4_step(
                psi, omega,
                dt, dr, dz,
                R,
                eta, nu,
                apply_bc=None
            )
            psi, omega = apply_combined_bc(psi, omega)
        except Exception as e:
            print(f"\n❌ Step {step}: {e}")
            break

# 转换
w_history = np.array(w_history)
time_history = np.array(time_history)

print("\n" + "="*60)
print("增长率分析")
print("="*60)

valid_mask = w_history > 1e-10
if np.sum(valid_mask) > 10:
    w_valid = w_history[valid_mask]
    t_valid = time_history[valid_mask]
    
    log_w = np.log(w_valid + 1e-12)
    n_fit = int(0.8 * len(log_w))
    if n_fit > 5:
        fit_start = len(log_w) - n_fit
        coeffs = np.polyfit(t_valid[fit_start:], log_w[fit_start:], 1)
        gamma = coeffs[0]
        
        print(f"增长率: γ = {gamma:.6e} s⁻¹")
        print(f"拟合数据点: {n_fit}")
        print(f"时间范围: [{t_valid[fit_start]:.4f}, {t_valid[-1]:.4f}]")
        
        if gamma > 1e-6:
            print(f"\n✅ 观察到增长")
        else:
            print(f"\n❌ 无明显增长")
    else:
        gamma = 0.0
        print("数据点不足")
else:
    gamma = 0.0
    print("❌ 磁岛宽度全程为0")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(time_history, w_history, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Island Width w')
ax.set_title('Island Width Evolution')
ax.grid(True)

ax = axes[0, 1]
w_plot = np.maximum(w_history, 1e-12)
ax.semilogy(time_history, w_plot, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('log(w)')
ax.set_title('Island Width (log)')
ax.grid(True)

ax = axes[1, 0]
ax.plot(time_history, psi_max_history, 'r-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('|ψ|_max')
ax.set_title('Max ψ amplitude')
ax.grid(True)

ax = axes[1, 1]
ax.plot(time_history, omega_max_history, 'g-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('|ω|_max')
ax.set_title('Max ω amplitude')
ax.grid(True)

plt.tight_layout()
plt.savefig('level3_simple_ic.png', dpi=150)
print("\n图像: level3_simple_ic.png")

print("\n" + "="*60)
print("Level 3 诊断结论")
print("="*60)

success = True
issues = []

final_nan = np.any(np.isnan(psi)) or np.any(np.isnan(omega))
final_inf = np.any(np.isinf(psi)) or np.any(np.isinf(omega))

if final_nan or final_inf:
    success = False
    issues.append("数值不稳定")

if abs(gamma) < 1e-6:
    success = False
    issues.append("无明显增长 (γ ≈ 0)")

if w_history[-1] < 1e-10:
    success = False
    issues.append("最终岛宽为0")

if success:
    print("\n✅ Level 3 通过:")
    print(f"   - 数值稳定")
    print(f"   - 增长率 γ = {gamma:.6e} s⁻¹")
    print(f"   - 最终岛宽 w = {w_history[-1]:.6e}")
else:
    print("\n❌ Level 3 失败:")
    for issue in issues:
        print(f"   - {issue}")

exit(0 if success else 1)
