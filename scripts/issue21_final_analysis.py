#!/usr/bin/env python3
"""
Issue #21: Final Deep Analysis (修正版)

小P要求的补充分析:
1. 分辨率scaling
2. JAX backend验证  
3. JIT使用检查

Author: 小A 🤖
Date: 2026-03-25 08:00
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import jax.numpy as jnp
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 80)
print("Issue #21: Final Deep Analysis")
print("=" * 80)
print()

# ==============================================================================
# Analysis 1: Resolution Scaling
# ==============================================================================

print("=" * 80)
print("Analysis 1: Resolution Scaling")
print("=" * 80)
print()

resolutions = [
    (16, 32, 4),
    (32, 64, 8),
    (48, 96, 12),
]

print("Testing different grid resolutions (30 steps each)...\n")
print("Resolution      | Grid Points | Step Time | Frequency | vs Baseline")
print("-" * 78)

results = []
baseline_time = None

for nr, ntheta, nz in resolutions:
    env_res = make_hamiltonian_mhd_env(
        nr=nr, ntheta=ntheta, nz=nz,
        dt=1e-4, max_steps=100,
        eta=0.05, nu=1e-4,
        normalize_obs=False
    )
    psi_res, phi_res = create_tearing_ic(nr=nr, ntheta=ntheta)
    env_res.mhd_solver.initialize(jnp.array(psi_res, dtype=jnp.float32), jnp.array(phi_res, dtype=jnp.float32))
    
    # Warm up (JIT compilation)
    for _ in range(10):
        env_res.step(np.array([1.0, 1.0]), compute_obs=False)
    
    # Measure
    times = []
    for _ in range(30):
        start = time.perf_counter()
        obs, r, term, trunc, info = env_res.step(np.array([1.0, 1.0]), compute_obs=False)
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    freq = 1000.0 / mean_time
    grid_points = nr * ntheta * nz
    
    if nr == 32 and ntheta == 64:
        baseline_time = mean_time
        scaling_str = "1.00× (baseline)"
    else:
        if baseline_time:
            scaling = mean_time / baseline_time
            scaling_str = f"{scaling:.2f}×"
        else:
            scaling_str = "—"
    
    results.append((nr, ntheta, nz, grid_points, mean_time, freq))
    print(f"{nr:2d}×{ntheta:2d}×{nz:2d}       | {grid_points:11d} | {mean_time:7.2f}±{std_time:.2f} ms | {freq:7.1f} Hz | {scaling_str}")

print()

# Fit scaling law
if len(results) >= 2:
    grid_points = np.array([r[3] for r in results])
    times = np.array([r[4] for r in results])
    
    log_N = np.log(grid_points)
    log_t = np.log(times)
    b, log_a = np.polyfit(log_N, log_t, 1)
    
    print(f"📊 Scaling law: time ∝ N^{b:.2f}")
    if abs(b - 1.0) < 0.15:
        print(f"   → Near-linear O(N) scaling ✅ (理想)")
        interpretation = "Good! 算法效率高"
    elif b < 1.5:
        print(f"   → Slightly super-linear O(N^{b:.2f}) ⚠️")
        interpretation = "可接受,有优化空间"
    elif abs(b - 2.0) < 0.15:
        print(f"   → Quadratic O(N²) ⚠️⚠️")
        interpretation = "差!需要优化算法"
    else:
        print(f"   → Non-standard O(N^{b:.2f})")
        interpretation = "需要深入分析"
    
    print(f"   {interpretation}")
    print()

# ==============================================================================
# Analysis 2: JAX Backend Verification
# ==============================================================================

print("=" * 80)
print("Analysis 2: JAX Backend & JIT Status")
print("=" * 80)
print()

# Check source code for JIT usage
print("Checking complete_solver_v2.py for @jax.jit...")
try:
    with open('src/pim_rl/physics/v2/complete_solver_v2.py', 'r') as f:
        solver_code = f.read()
    
    has_jax_import = 'import jax' in solver_code
    has_jit_decorator = '@jax.jit' in solver_code or '@jit' in solver_code
    has_partial_jit = '@partial(jax.jit' in solver_code
    
    print(f"✅ JAX imported: {'Yes' if has_jax_import else 'No'}")
    print(f"{'✅' if has_jit_decorator or has_partial_jit else '❌'} @jax.jit decorators: {'Yes' if has_jit_decorator or has_partial_jit else 'No'}")
    
    if not has_jit_decorator and not has_partial_jit:
        print()
        print("⚠️ CRITICAL FINDING: No @jax.jit decorators found!")
        print("   → RHS computation NOT JIT-compiled")
        print("   → This is a major performance bottleneck")
        print("   → Expected 2-5× speedup from adding JIT")
except Exception as e:
    print(f"Error reading source: {e}")
    has_jit_decorator = None

print()

# Check bracket implementation
print("Checking toroidal_bracket.py...")
try:
    with open('src/pim_rl/physics/v2/toroidal_bracket.py', 'r') as f:
        bracket_code = f.read()
    
    bracket_has_jit = '@jax.jit' in bracket_code or '@jit' in bracket_code
    print(f"{'✅' if bracket_has_jit else '❌'} Bracket JIT: {'Yes' if bracket_has_jit else 'No'}")
except:
    bracket_has_jit = None

print()

# ==============================================================================
# Summary & Recommendations
# ==============================================================================

print("=" * 80)
print("SUMMARY: Deep Analysis Results")
print("=" * 80)
print()

print("📐 Resolution Scaling:")
for r in results:
    nr, ntheta, nz, grid_points, mean_time, freq = r
    print(f"   {nr:2d}×{ntheta:2d}×{nz:2d} ({grid_points:6d} points): {mean_time:6.2f} ms → {freq:5.1f} Hz")

if len(results) >= 2:
    print(f"\n   Scaling: O(N^{b:.2f})")
    
    # Extrapolate to lower resolution
    N_16x32 = 16 * 32 * 4
    N_32x64 = 32 * 64 * 8
    if baseline_time:
        predicted_16x32 = baseline_time * (N_16x32 / N_32x64) ** b
        actual_16x32 = results[0][4]
        print(f"   Prediction accuracy: {abs(predicted_16x32 - actual_16x32)/actual_16x32*100:.1f}% error")
print()

print("🔧 JAX & JIT Status:")
if has_jit_decorator is not None:
    if not has_jit_decorator:
        print("   ❌ JIT compilation: NOT ENABLED")
        print("   ⚠️ This is the PRIMARY optimization opportunity!")
        print()
        print("   Expected impact from adding JIT:")
        print("      - Speedup: 2-5×")
        print("      - Current: 60 Hz → Target: 120-300 Hz")
        print("      - Effort: LOW (add @jax.jit decorators)")
    else:
        print("   ✅ JIT compilation: ENABLED")
        print("   → Already optimized, focus on other bottlenecks")
else:
    print("   ⚠️ Could not determine JIT status")
print()

print("🎯 Recommendations for Issue #15 (this afternoon):")
print()

if has_jit_decorator is False:
    print("   1. ⭐ HIGH PRIORITY: Add @jax.jit to complete_solver_v2.py")
    print("      - Functions to JIT: rhs(), hamiltonian()")
    print("      - Expected: 2-5× speedup")
    print("      - Effort: ~30 minutes")
    print()
    print("   2. MEDIUM: Test GPU backend (if available)")
    print("      - Additional 2-3× speedup")
    print("      - Effort: ~1 hour")
    print()
else:
    print("   1. Test GPU acceleration")
    print("   2. Optimize Poisson solver (if not already done)")
    print()

if len(results) >= 2 and b > 1.5:
    print("   3. INVESTIGATE: Algorithm complexity O(N^{:.2f})".format(b))
    print("      - Should be closer to O(N)")
    print("      - May indicate inefficient operations")
    print()

print("=" * 80)
print("Analysis Complete - Ready for Issue #15")
print("=" * 80)
