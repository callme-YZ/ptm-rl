#!/usr/bin/env python3
"""
Phase 4 Validation Tests - Large Grid Verification
验证"Validation tests失败是因为小网格分辨率不足"的假设
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.control.validation import test_rmp_suppression_open_loop

print("=" * 80)
print("Phase 4 Validation Tests - Large Grid Verification")
print("=" * 80)
print()

# Test 1: 小网格 (预期失败)
print("=" * 60)
print("Test 1: Small Grid (32×64, 50 steps)")
print("=" * 60)
small_grid_success = False
small_grid_diag = {}

try:
    success, diag = test_rmp_suppression_open_loop(
        Nr=32, Nz=64, m=2, n=1, n_steps=50
    )
    small_grid_success = success
    small_grid_diag = diag
    
    print(f"Result: {'PASS ✅' if success else 'FAIL ❌'}")
    print(f"Diagnostics: {diag}")
    
    if 'gamma_free' in diag and 'gamma_rmp' in diag:
        reduction = (diag['gamma_free'] - diag['gamma_rmp']) / diag['gamma_free'] * 100
        print(f"\nAnalysis:")
        print(f"  γ_free = {diag['gamma_free']:.6f}")
        print(f"  γ_rmp  = {diag['gamma_rmp']:.6f}")
        print(f"  Reduction = {reduction:.1f}%")
        print(f"  Target: >50%")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# Test 2: 大网格 (验证)
print("=" * 60)
print("Test 2: Large Grid (64×128, 100 steps)")
print("=" * 60)
large_grid_success = False
large_grid_diag = {}

try:
    success, diag = test_rmp_suppression_open_loop(
        Nr=64, Nz=128, m=2, n=1, n_steps=100
    )
    large_grid_success = success
    large_grid_diag = diag
    
    print(f"Result: {'PASS ✅' if success else 'FAIL ❌'}")
    print(f"Diagnostics: {diag}")
    
    # 详细分析
    if 'gamma_free' in diag and 'gamma_rmp' in diag:
        reduction = (diag['gamma_free'] - diag['gamma_rmp']) / diag['gamma_free'] * 100
        print(f"\nAnalysis:")
        print(f"  γ_free = {diag['gamma_free']:.6f}")
        print(f"  γ_rmp  = {diag['gamma_rmp']:.6f}")
        print(f"  Reduction = {reduction:.1f}%")
        print(f"  Target: >50%")
        print(f"  Status: {'✅ PASS' if reduction > 50 else '❌ FAIL'}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# 总结对比
print("=" * 80)
print("Verification Summary")
print("=" * 80)
print()
print(f"Small Grid (32×64, 50 steps): {'PASS ✅' if small_grid_success else 'FAIL ❌'}")
print(f"Large Grid (64×128, 100 steps): {'PASS ✅' if large_grid_success else 'FAIL ❌'}")
print()

if 'gamma_free' in small_grid_diag and 'gamma_rmp' in small_grid_diag:
    small_reduction = (small_grid_diag['gamma_free'] - small_grid_diag['gamma_rmp']) / small_grid_diag['gamma_free'] * 100
    print(f"Small Grid Reduction: {small_reduction:.1f}%")
    
if 'gamma_free' in large_grid_diag and 'gamma_rmp' in large_grid_diag:
    large_reduction = (large_grid_diag['gamma_free'] - large_grid_diag['gamma_rmp']) / large_grid_diag['gamma_free'] * 100
    print(f"Large Grid Reduction: {large_reduction:.1f}%")

print()

# 结论
print("Conclusion:")
if not small_grid_success and large_grid_success:
    print("✅ 假设验证成功: Validation tests失败是因为小网格分辨率不足")
    print("   建议: 采用大网格配置 (64×128, 100 steps) 进行后续测试")
elif not small_grid_success and not large_grid_success:
    print("⚠️  假设被推翻: 大网格仍然失败,问题不是简单的网格分辨率问题")
    print("   建议: 需要深入调试物理模型和控制逻辑")
else:
    print("ℹ️  意外结果: 小网格通过或其他情况")

print("=" * 80)
