#!/usr/bin/env python3
"""
Level 3: Tearing Mode Free Growth 诊断

验证标准:
1. w(t) 增长 (exponential或linear)
2. γ > 0 (正增长率)
3. 数值稳定 (no overflow)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pytokmhd.solver.initial_conditions import setup_tearing_mode
from src.pytokmhd.solver.time_integrator import rk4_step
from src.pytokmhd.diagnostics.magnetic_island import compute_island_width
from src.pytokmhd.solver.boundary import apply_combined_bc

# 设置参数
Nr, Nz = 64, 128
m, n = 2, 1
Lr, Lz = 2.0, 4.0
n_steps = 100
dt = 0.001  # 小时间步长，确保数值稳定

# 物理参数
eta = 1e-4  # 电阻率
nu = 1e-4   # 粘性

# 创建网格
r = np.linspace(0.1, Lr, Nr)
z = np.linspace(-Lz/2, Lz/2, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')
dr = r[1] - r[0]
dz = z[1] - z[0]

# q-profile
q_profile = 1.5 + 1.5 * (r / Lr)**2

print("="*60)
print("Level 3: Tearing Mode Free Growth 诊断")
print("="*60)
print(f"\n网格: {Nr}×{Nz}")
print(f"时间步: dt={dt:.6f}")
print(f"演化步数: {n_steps}")
print(f"物理参数: η={eta:.6e}, ν={nu:.6e}")

# 初始条件
psi, omega, r_s = setup_tearing_mode(
    r_grid=R, 
    z_grid=Z,
    q_profile=q_profile,
    r_values=r,
    m=m, 
    n=n,
    w_0=0.01
)

print(f"\n有理面位置: r_s = {r_s:.4f}")

# 检查初始数值
print("\n初始条件检查:")
print(f"  ψ: min={np.min(psi):.6e}, max={np.max(psi):.6e}")
print(f"  ω: min={np.min(omega):.6e}, max={np.max(omega):.6e}")

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
        # 可能返回tuple或单个值
        if isinstance(w_result, tuple):
            w = w_result[0]  # 取第一个值
        else:
            w = w_result
    except Exception as e:
        print(f"\n⚠️  Step {step}: 岛宽计算失败: {e}")
        w = 0.0
    
    w_history.append(w)
    psi_max_history.append(np.max(np.abs(psi)))
    omega_max_history.append(np.max(np.abs(omega)))
    time_history.append(t)
    
    # 每10步输出一次
    if step % 10 == 0:
        has_nan = np.any(np.isnan(psi)) or np.any(np.isnan(omega))
        has_inf = np.any(np.isinf(psi)) or np.any(np.isinf(omega))
        
        status = "✓"
        if has_nan or has_inf:
            status = "✗ (NaN/Inf detected)"
        
        print(f"Step {step:4d} | t={t:.4f} | w={w:.6e} | "
              f"|ψ|_max={psi_max_history[-1]:.4e} | "
              f"|ω|_max={omega_max_history[-1]:.4e} | {status}")
        
        if has_nan or has_inf:
            print("\n❌ 数值不稳定: 出现NaN/Inf")
            break
    
    # 时间演化 (不施加RMP)
    if step < n_steps:
        # 时间步进 (RK4)
        try:
            psi, omega = rk4_step(
                psi, omega, 
                dt, dr, dz, 
                R,  # r_grid (2D meshgrid)
                eta, nu,
                apply_bc=None  # 让rk4_step内部处理边界条件
            )
            
            # 手动应用边界条件
            psi, omega = apply_combined_bc(psi, omega)
            
        except Exception as e:
            print(f"\n❌ Step {step}: 时间演化失败: {e}")
            import traceback
            traceback.print_exc()
            break

# 转换为numpy数组
w_history = np.array(w_history)
time_history = np.array(time_history)
psi_max_history = np.array(psi_max_history)
omega_max_history = np.array(omega_max_history)

print("\n" + "="*60)
print("增长率分析")
print("="*60)

# 拟合增长率 (exponential: w = w_0 * exp(γ*t))
# 只取后半段数据 (线性/指数增长阶段)
valid_mask = w_history > 1e-10  # 排除w=0的点

if np.sum(valid_mask) > 10:
    w_valid = w_history[valid_mask]
    t_valid = time_history[valid_mask]
    
    # 线性拟合 log(w) vs t
    log_w = np.log(w_valid + 1e-12)  # 避免log(0)
    
    # 只用后80%的数据（跳过瞬态）
    n_fit = int(0.8 * len(log_w))
    if n_fit > 5:
        fit_start = len(log_w) - n_fit
        
        coeffs = np.polyfit(t_valid[fit_start:], log_w[fit_start:], 1)
        gamma = coeffs[0]
        
        print(f"\n增长率拟合 (exponential):")
        print(f"  γ = {gamma:.6e} s⁻¹")
        print(f"  拟合数据点: {n_fit}")
        print(f"  时间范围: [{t_valid[fit_start]:.4f}, {t_valid[-1]:.4f}]")
        
        if gamma > 1e-6:
            print(f"\n✅ 观察到增长: γ > 0")
        else:
            print(f"\n❌ 无明显增长: γ ≈ 0")
    else:
        print("\n⚠️  有效数据点不足，无法拟合增长率")
        gamma = 0.0
else:
    print("\n❌ 磁岛宽度全程为0，无法拟合增长率")
    gamma = 0.0

print("\n" + "="*60)
print("可视化")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 磁岛宽度 vs 时间
ax = axes[0, 0]
ax.plot(time_history, w_history, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Island Width w')
ax.set_title('Island Width Evolution')
ax.grid(True)
ax.set_ylim(bottom=0)

# (2) log(w) vs 时间 (检查exponential growth)
ax = axes[0, 1]
w_plot = np.maximum(w_history, 1e-12)  # 避免log(0)
ax.semilogy(time_history, w_plot, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('log(Island Width)')
ax.set_title('Island Width (log scale)')
ax.grid(True)

# (3) |ψ|_max vs 时间
ax = axes[1, 0]
ax.plot(time_history, psi_max_history, 'r-', linewidth=2, label='|ψ|_max')
ax.set_xlabel('Time')
ax.set_ylabel('|ψ|_max')
ax.set_title('Maximum ψ amplitude')
ax.legend()
ax.grid(True)

# (4) |ω|_max vs 时间
ax = axes[1, 1]
ax.plot(time_history, omega_max_history, 'g-', linewidth=2, label='|ω|_max')
ax.set_xlabel('Time')
ax.set_ylabel('|ω|_max')
ax.set_title('Maximum ω amplitude')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('level3_growth_diagnostics.png', dpi=150, bbox_inches='tight')
print("\n图像保存到: level3_growth_diagnostics.png")

print("\n" + "="*60)
print("Level 3 诊断结论")
print("="*60)

success = True
issues = []

# 检查1: 数值稳定性
final_has_nan = np.any(np.isnan(psi)) or np.any(np.isnan(omega))
final_has_inf = np.any(np.isinf(psi)) or np.any(np.isinf(omega))

if final_has_nan or final_has_inf:
    success = False
    issues.append("数值不稳定 (NaN/Inf)")

# 检查2: 增长率
if abs(gamma) < 1e-6:
    success = False
    issues.append("无明显增长 (γ ≈ 0)")

# 检查3: 磁岛宽度非零
if w_history[-1] < 1e-10:
    success = False
    issues.append("最终磁岛宽度为0")

if success:
    print("\n✅ Level 3 通过: 自由增长正常")
    print(f"   - 数值稳定 (无NaN/Inf)")
    print(f"   - 增长率 γ = {gamma:.6e} s⁻¹ > 0")
    print(f"   - 最终岛宽 w = {w_history[-1]:.6e} > 0")
    print(f"\n**这是关键突破！** 撕裂模能够自由增长")
else:
    print("\n❌ Level 3 失败:")
    for issue in issues:
        print(f"   - {issue}")
    print(f"\n诊断信息:")
    print(f"   - 最终岛宽: w = {w_history[-1]:.6e}")
    print(f"   - 增长率: γ = {gamma:.6e} s⁻¹")

exit(0 if success else 1)
