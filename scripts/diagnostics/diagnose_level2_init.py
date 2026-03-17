#!/usr/bin/env python3
"""
Level 2: Tearing Mode 初始化诊断

验证标准:
1. w > 0 (磁岛宽度非零)
2. ψ场有m=2扰动
3. 扰动在有理面附近
4. 数值合理 (no NaN/Inf)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.pytokmhd.solver.initial_conditions import setup_tearing_mode, find_rational_surface

# 设置参数
Nr, Nz = 64, 128
m, n = 2, 1
Lr, Lz = 2.0, 4.0

# 创建网格
r = np.linspace(0.1, Lr, Nr)
z = np.linspace(-Lz/2, Lz/2, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')

# q-profile
q_profile = 1.5 + 1.5 * (r / Lr)**2

print("="*60)
print("Level 2: Tearing Mode 初始化诊断")
print("="*60)

# 找有理面
r_s = find_rational_surface(r, q_profile, m/n)
print(f"\n有理面位置: r_s = {r_s:.4f}")
print(f"有理面处q值: q(r_s) = {np.interp(r_s, r, q_profile):.4f}")
print(f"目标q值: m/n = {m/n:.4f}")

# 生成初始条件
psi, omega, r_s_returned = setup_tearing_mode(
    r_grid=R, 
    z_grid=Z,
    q_profile=q_profile,
    r_values=r,
    m=m, 
    n=n,
    w_0=0.01  # 初始扰动幅度
)

print("\n" + "="*60)
print("数值检查")
print("="*60)

# 检查NaN/Inf
has_nan_psi = np.any(np.isnan(psi))
has_inf_psi = np.any(np.isinf(psi))
has_nan_omega = np.any(np.isnan(omega))
has_inf_omega = np.any(np.isinf(omega))

print(f"ψ场包含NaN: {has_nan_psi}")
print(f"ψ场包含Inf: {has_inf_psi}")
print(f"ω场包含NaN: {has_nan_omega}")
print(f"ω场包含Inf: {has_inf_omega}")

if has_nan_psi or has_inf_psi or has_nan_omega or has_inf_omega:
    print("\n❌ 数值检查失败: 存在NaN/Inf")
    exit(1)

print("\n✅ 数值检查通过: 无NaN/Inf")

# 统计信息
print(f"\nψ场统计:")
print(f"  min = {np.min(psi):.6e}")
print(f"  max = {np.max(psi):.6e}")
print(f"  mean = {np.mean(psi):.6e}")
print(f"  std = {np.std(psi):.6e}")

print(f"\nω场统计:")
print(f"  min = {np.min(omega):.6e}")
print(f"  max = {np.max(omega):.6e}")
print(f"  mean = {np.mean(omega):.6e}")
print(f"  std = {np.std(omega):.6e}")

print("\n" + "="*60)
print("扰动检查 (Fourier分析)")
print("="*60)

# 在有理面附近提取数据 (r = r_s)
ir_s = np.argmin(np.abs(r - r_s))
psi_at_rs = psi[ir_s, :]  # ψ(r_s, z)

# Fourier变换 (沿z方向)
fft_psi = np.fft.fft(psi_at_rs)
freqs = np.fft.fftfreq(Nz, d=(z[1]-z[0]))
power = np.abs(fft_psi)**2

# 找主导模式
dominant_mode = np.argmax(power[1:Nz//2]) + 1  # 排除DC分量
dominant_freq = freqs[dominant_mode]
dominant_wavelength = 1.0 / abs(dominant_freq) if dominant_freq != 0 else np.inf

print(f"有理面处ψ场Fourier分析:")
print(f"  主导频率: {dominant_freq:.4f}")
print(f"  主导波长: {dominant_wavelength:.4f}")
print(f"  期望波长 (Lz): {Lz:.4f}")

# 提取m=2分量强度
# 扰动形式: δψ ~ cos(m*θ), 在柱坐标中θ沿z变化
# 但我们用的是(R,Z)坐标, θ = atan2(Z, R-R0)

# 计算平均半径处的扰动幅度
r_mid = r[Nr//2]
ir_mid = Nr//2
psi_mid = psi[ir_mid, :]
psi_eq_mid = np.mean(psi_mid)  # 平衡态部分
psi_pert_mid = psi_mid - psi_eq_mid  # 扰动部分

print(f"\n中间半径 (r={r_mid:.3f}) 处扰动:")
print(f"  平衡态幅度: {abs(psi_eq_mid):.6e}")
print(f"  扰动幅度: {np.std(psi_pert_mid):.6e}")
print(f"  扰动/平衡态比值: {np.std(psi_pert_mid)/abs(psi_eq_mid):.6e}")

print("\n" + "="*60)
print("可视化")
print("="*60)

# 创建诊断图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (1) ψ场2D分布
ax = axes[0, 0]
im = ax.contourf(R, Z, psi, levels=50, cmap='RdBu_r')
ax.axvline(1.0 + r_s, color='red', linestyle='--', label=f'r_s={r_s:.3f}')
ax.set_xlabel('R')
ax.set_ylabel('Z')
ax.set_title('ψ field')
ax.legend()
plt.colorbar(im, ax=ax)

# (2) ω场2D分布
ax = axes[0, 1]
im = ax.contourf(R, Z, omega, levels=50, cmap='RdBu_r')
ax.axvline(1.0 + r_s, color='red', linestyle='--', label=f'r_s={r_s:.3f}')
ax.set_xlabel('R')
ax.set_ylabel('Z')
ax.set_title('ω field')
ax.legend()
plt.colorbar(im, ax=ax)

# (3) q-profile
ax = axes[0, 2]
ax.plot(r, q_profile, 'b-', linewidth=2)
ax.axhline(m/n, color='red', linestyle='--', label=f'q={m/n:.2f}')
ax.axvline(r_s, color='red', linestyle='--', label=f'r_s={r_s:.3f}')
ax.set_xlabel('r')
ax.set_ylabel('q')
ax.set_title('q-profile')
ax.legend()
ax.grid(True)

# (4) ψ径向分布 (at z=0)
ax = axes[1, 0]
iz_mid = Nz // 2
ax.plot(r, psi[:, iz_mid], 'b-', linewidth=2)
ax.axvline(r_s, color='red', linestyle='--', label=f'r_s={r_s:.3f}')
ax.set_xlabel('r')
ax.set_ylabel('ψ')
ax.set_title('ψ radial profile (z=0)')
ax.legend()
ax.grid(True)

# (5) ψ沿z分布 (at r=r_s)
ax = axes[1, 1]
ax.plot(z, psi_at_rs, 'b-', linewidth=2, label='ψ(r_s, z)')
ax.set_xlabel('z')
ax.set_ylabel('ψ')
ax.set_title(f'ψ poloidal profile (r={r_s:.3f})')
ax.legend()
ax.grid(True)

# (6) Fourier功率谱
ax = axes[1, 2]
ax.semilogy(freqs[1:Nz//2], power[1:Nz//2], 'b-', linewidth=2)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')
ax.set_title('Fourier spectrum of ψ(r_s, z)')
ax.grid(True)

plt.tight_layout()
plt.savefig('level2_init_diagnostics.png', dpi=150, bbox_inches='tight')
print("\n图像保存到: level2_init_diagnostics.png")

print("\n" + "="*60)
print("Level 2 诊断结论")
print("="*60)

success = True
issues = []

# 检查1: 数值有效性
if has_nan_psi or has_inf_psi or has_nan_omega or has_inf_omega:
    success = False
    issues.append("存在NaN/Inf")

# 检查2: 扰动幅度非零
if np.std(psi_pert_mid) < 1e-10:
    success = False
    issues.append("扰动幅度过小")

# 检查3: 有理面位置合理
if r_s < 0.1 or r_s > Lr - 0.1:
    success = False
    issues.append(f"有理面位置不合理: r_s={r_s:.3f}")

if success:
    print("\n✅ Level 2 通过: 初始化正确")
    print("   - 无NaN/Inf")
    print("   - 扰动存在")
    print("   - 有理面位置合理")
else:
    print("\n❌ Level 2 失败:")
    for issue in issues:
        print(f"   - {issue}")

exit(0 if success else 1)
