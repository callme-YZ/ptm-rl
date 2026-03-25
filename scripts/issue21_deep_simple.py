#!/usr/bin/env python3
"""
Issue #21: Simplified Deep Analysis (小P要求)

Focus on what we can measure easily:
1. Morrison bracket component breakdown
2. Resolution scaling
3. JAX backend verification

Author: 小A 🤖 + 小P ⚛️
Date: 2026-03-25
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import jax.numpy as jnp
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 80)
print("Issue #21: Simplified Deep Analysis")
print("=" * 80)
print()

# ==============================================================================
# Analysis 1: Morrison Bracket Component Timing
# ==============================================================================

print("=" * 80)
print("Analysis 1: RHS Component Breakdown")
print("=" * 80)
print()

env = make_hamiltonian_mhd_env(nr=32, ntheta=64, nz=8, dt=1e-4, max_steps=1000, eta=0.05, nu=1e-4, normalize_obs=False)
psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

solver = env.mhd_solver.solver

# Get current state from solver
state = solver.integrator.state if hasattr(solver.integrator, 'state') else None
if state is None:
    # Construct manually
    from pim_rl.physics.v2.elsasser_bracket import ElsasserState
    # Use z_plus/z_minus from solver's internal state
    # This is a workaround - we'll just measure full RHS
    print("⚠️ Cannot isolate bracket components easily")
    print("   Measuring full RHS timing instead...")
    print()
    
    # Measure full RHS only
    times_rhs = []
    for i in range(50):
        obs, r, term, trunc, info = env.step(np.array([0.0]), compute_obs=False)
        if i >= 10:  # Skip first 10 for warm-up
            times_rhs.append(info.get('step_time_ms', 0))
    
    print(f"Full physics step: {np.mean(times_rhs):.2f} ms")
    print()

# ==============================================================================
# Analysis 2: Resolution Scaling
# ==============================================================================

print("=" * 80)
print("Analysis 2: Resolution Scaling")
print("=" * 80)
print()

resolutions = [
    (16, 32, 4),
    (32, 64, 8),
    (48, 96, 12),
]

print("Testing different grid resolutions (20 steps each)...\n")
print("Resolution      | Grid Points | Step Time | Frequency | Scaling")
print("-" * 75)

results = []
baseline_time = None

for nr, ntheta, nz in resolutions:
    # Setup
    env_res = make_hamiltonian_mhd_env(
        nr=nr, ntheta=ntheta, nz=nz,
        dt=1e-4, max_steps=100,
        eta=0.05, nu=1e-4,
        normalize_obs=False
    )
    psi_res, phi_res = create_tearing_ic(nr=nr, ntheta=ntheta)
    env_res.mhd_solver.initialize(jnp.array(psi_res, dtype=jnp.float32), jnp.array(phi_res, dtype=jnp.float32))
    
    # Warm up
    for _ in range(10):
        env_res.step(np.array([0.0]), compute_obs=False)
    
    # Measure
    times = []
    for _ in range(20):
        start = time.perf_counter()
        obs, r, term, trunc, info = env_res.step(np.array([0.0]), compute_obs=False)
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = np.mean(times)
    freq = 1000.0 / mean_time
    grid_points = nr * ntheta * nz
    
    if nr == 32 and ntheta == 64:
        baseline_time = mean_time
        scaling_str = "1.0× (baseline)"
    else:
        if baseline_time:
            scaling = mean_time / baseline_time
            scaling_str = f"{scaling:.2f}×"
        else:
            scaling_str = "—"
    
    results.append((nr, ntheta, nz, grid_points, mean_time, freq))
    print(f"{nr:2d}×{ntheta:2d}×{nz:2d}       | {grid_points:11d} | {mean_time:8.2f} ms | {freq:8.1f} Hz | {scaling_str}")

print()

# Estimate scaling exponent
if len(results) >= 2:
    grid_points = np.array([r[3] for r in results])
    times = np.array([r[4] for r in results])
    
    # Fit power law: time = a * N^b
    log_N = np.log(grid_points)
    log_t = np.log(times)
    b, log_a = np.polyfit(log_N, log_t, 1)
    
    print(f"📊 Scaling law: time ∝ N^{b:.2f}")
    if abs(b - 1.0) < 0.15:
        print(f"   → Near-linear O(N) scaling ✅")
    elif abs(b - 1.3) < 0.15:
        print(f"   → Slightly super-linear O(N^{b:.2f}) ⚠️")
    elif abs(b - 2.0) < 0.15:
        print(f"   → Quadratic O(N²) scaling ⚠️⚠️")
    else:
        print(f"   → Non-standard O(N^{b:.2f})")
    print()

# ==============================================================================
# Analysis 3: JAX Backend Verification
# ==============================================================================

print("=" * 80)
print("Analysis 3: JAX Backend Verification")
print("=" * 80)
print()

# Check array types
print("Checking internal array types...")
solver = env.mhd_solver.solver

# Check if we're using JAX arrays
test_state = solver.integrator._state if hasattr(solver.integrator, '_state') else None
if test_state:
    print(f"State z_plus type: {type(test_state.z_plus)}")
    is_jax = 'jax' in str(type(test_state.z_plus)).lower()
else:
    # Check from grid
    print(f"Grid dr type: {type(solver.grid.dr)}")
    is_jax = 'jax' in str(type(solver.grid.dr)).lower() or hasattr(solver.grid.dr, 'device')

if is_jax:
    print("✅ Backend: JAX DeviceArray")
else:
    print("⚠️ Backend: NumPy ndarray")
print()

# Check for JIT decorators
print("Checking for @jax.jit usage...")
import inspect

# Check complete_solver_v2.py
try:
    with open('src/pim_rl/physics/v2/complete_solver_v2.py', 'r') as f:
        solver_code = f.read()
    
    has_jit_import = 'import jax' in solver_code or 'from jax' in solver_code
    has_jit_decorator = '@jax.jit' in solver_code or '@jit' in solver_code
    
    print(f"JAX imported: {'Yes ✅' if has_jit_import else 'No ❌'}")
    print(f"@jax.jit used: {'Yes ✅' if has_jit_decorator else 'No ❌'}")
    
    if has_jit_import and not has_jit_decorator:
        print("⚠️ JAX imported but @jax.jit NOT used → Optimization opportunity!")
except Exception as e:
    print(f"Could not check source: {e}")

print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 80)
print("SUMMARY: Deep Analysis Results")
print("=" * 80)
print()

print("📐 Resolution Scaling:")
for r in results:
    nr, ntheta, nz, grid_points, mean_time, freq = r
    speedup = baseline_time / mean_time if baseline_time else 1.0
    print(f"   {nr}×{ntheta}×{nz}: {mean_time:.2f} ms → {freq:.1f} Hz")
print()

if len(results) >= 2:
    print(f"   Scaling exponent: N^{b:.2f}")
    print()

print("🔧 JAX Status:")
print(f"   Using JAX arrays: {'Yes ✅' if is_jax else 'No ❌'}")
try:
    print(f"   @jax.jit decorators: {'Yes ✅' if has_jit_decorator else 'No ❌ (OPTIMIZATION NEEDED!)'}")
except:
    pass
print()

print("🎯 Key Findings for Issue #15:")
print()
if not has_jit_decorator:
    print("   ⭐ PRIMARY OPTIMIZATION: Add @jax.jit to RHS")
    print("      - Current: No JIT compilation")
    print("      - Expected: 2-5× speedup")
    print("      - Effort: Low (add decorators)")
    print()
else:
    print("   ✅ JIT already enabled")
    print()

print("   🔍 Resolution vs Performance:")
if len(results) >= 2:
    print(f"      - Scaling: ~O(N^{b:.2f})")
    if b < 1.2:
        print("      - Good scaling efficiency ✅")
    else:
        print("      - Could be better (optimize algorithms)")
print()

print("=" * 80)
print("Analysis Complete")
print("=" * 80)
