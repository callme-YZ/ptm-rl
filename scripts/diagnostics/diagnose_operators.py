#!/usr/bin/env python3
"""
诊断MHD算子的数值精度和稳定性

测试:
1. Laplacian算子: ∇²f
2. Poisson bracket: [f, g]
3. Gradient算子
"""

import numpy as np
from src.pytokmhd.solver.mhd_equations import (
    laplacian_cylindrical,
    poisson_bracket,
    gradient_r,
    gradient_z
)

# 网格
Nr, Nz = 64, 128
r = np.linspace(0.1, 2.0, Nr)
z = np.linspace(-2.0, 2.0, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')
dr = r[1] - r[0]
dz = z[1] - z[0]

print("="*60)
print("算子诊断")
print("="*60)

# Test 1: Laplacian of a simple function
print("\n[Test 1] Laplacian算子")
print("-"*60)

# f = r² → ∇²f = 2/r² + 2 + 2/r ≈ 2 + 2/r (在柱坐标)
# 实际: ∇²(r²) = ∂²(r²)/∂r² + (1/r)∂(r²)/∂r = 2 + 2r/r = 4
f_test = R**2
lap_f_numerical = laplacian_cylindrical(f_test, dr, dz, R)
lap_f_exact = 4.0 * np.ones_like(R)  # ∇²(r²) = 4

error = np.abs(lap_f_numerical - lap_f_exact)
print(f"测试函数: f = r²")
print(f"理论: ∇²f = 4")
print(f"数值: ∇²f range = [{np.min(lap_f_numerical):.4f}, {np.max(lap_f_numerical):.4f}]")
print(f"误差: max|error| = {np.max(error):.6e}")
print(f"相对误差: {np.max(error)/4.0:.2%}")

if np.max(error) > 0.5:
    print("⚠️  Laplacian精度较低")
else:
    print("✓ Laplacian精度合理")

# Test 2: Poisson bracket基本性质
print("\n[Test 2] Poisson bracket算子")
print("-"*60)

# 测试反对称性: [f, g] = -[g, f]
f = np.sin(2*np.pi*Z/4.0) * (1 - R**2)
g = np.cos(2*np.pi*Z/4.0) * R

pb_fg = poisson_bracket(f, g, dr, dz)
pb_gf = poisson_bracket(g, f, dr, dz)

antisymmetry_error = np.abs(pb_fg + pb_gf)
print(f"反对称性测试: [f,g] + [g,f] = 0")
print(f"误差: max|[f,g]+[g,f]| = {np.max(antisymmetry_error):.6e}")

if np.max(antisymmetry_error) < 1e-10:
    print("✓ 反对称性成立")
else:
    print("⚠️  反对称性有误差")

# Test 3: 检查是否产生爆炸性增长
print("\n[Test 3] Poisson bracket数值稳定性")
print("-"*60)

# 简单状态
psi_test = (R - 1.0)**2
omega_test = np.zeros_like(psi_test)

# 计算一步RHS (不含耗散)
from src.pytokmhd.solver.poisson_solver import solve_poisson

phi_test = solve_poisson(omega_test, dr, dz, R, rhs_sign=-1.0)
J_test = laplacian_cylindrical(psi_test, dr, dz, R)

pb_phi_psi = poisson_bracket(phi_test, psi_test, dr, dz)
pb_phi_omega = poisson_bracket(phi_test, omega_test, dr, dz)
pb_psi_J = poisson_bracket(psi_test, J_test, dr, dz)

print(f"∂ψ/∂t = -[φ,ψ] + η∇²ψ")
print(f"  [φ,ψ] range: [{np.min(pb_phi_psi):.4e}, {np.max(pb_phi_psi):.4e}]")
print(f"  |[φ,ψ]|_max = {np.max(np.abs(pb_phi_psi)):.4e}")

print(f"\n∂ω/∂t = -[φ,ω] + [ψ,J]")
print(f"  [φ,ω] range: [{np.min(pb_phi_omega):.4e}, {np.max(pb_phi_omega):.4e}]")
print(f"  [ψ,J] range: [{np.min(pb_psi_J):.4e}, {np.max(pb_psi_J):.4e}]")
print(f"  |[ψ,J]|_max = {np.max(np.abs(pb_psi_J)):.4e}")

# 估计时间步长约束
if np.max(np.abs(pb_psi_J)) > 0:
    dt_est = 0.1 / np.max(np.abs(pb_psi_J))
    print(f"\n时间步长估计 (CFL-like):")
    print(f"  dt < {dt_est:.6e} 秒")
    
    if dt_est < 1e-6:
        print(f"⚠️  需要极小的时间步长 → 可能数值不稳定")
    else:
        print(f"✓ 时间步长约束合理")

# Test 4: 检查NaN/Inf来源
print("\n[Test 4] 检查NaN/Inf")
print("-"*60)

has_nan = {
    'psi': np.any(np.isnan(psi_test)),
    'omega': np.any(np.isnan(omega_test)),
    'phi': np.any(np.isnan(phi_test)),
    'J': np.any(np.isnan(J_test)),
    '[φ,ψ]': np.any(np.isnan(pb_phi_psi)),
    '[ψ,J]': np.any(np.isnan(pb_psi_J))
}

has_inf = {
    'psi': np.any(np.isinf(psi_test)),
    'omega': np.any(np.isinf(omega_test)),
    'phi': np.any(np.isinf(phi_test)),
    'J': np.any(np.isinf(J_test)),
    '[φ,ψ]': np.any(np.isinf(pb_phi_psi)),
    '[ψ,J]': np.any(np.isinf(pb_psi_J))
}

print("NaN检查:")
for key, val in has_nan.items():
    print(f"  {key}: {'✗ 有NaN' if val else '✓'}")

print("\nInf检查:")
for key, val in has_inf.items():
    print(f"  {key}: {'✗ 有Inf' if val else '✓'}")

if any(has_nan.values()) or any(has_inf.values()):
    print("\n❌ 检测到NaN/Inf → 算子实现有问题")
else:
    print("\n✓ 算子无NaN/Inf")

print("\n" + "="*60)
print("诊断总结")
print("="*60)

issues = []
if np.max(error) > 0.5:
    issues.append("Laplacian精度低")
if np.max(antisymmetry_error) > 1e-10:
    issues.append("Poisson bracket不满足反对称性")
if 'dt_est' in locals() and dt_est < 1e-6:
    issues.append("需要极小时间步长")
if any(has_nan.values()) or any(has_inf.values()):
    issues.append("算子产生NaN/Inf")

if len(issues) == 0:
    print("\n✅ 算子实现正常")
else:
    print("\n❌ 发现问题:")
    for issue in issues:
        print(f"   - {issue}")
