#!/usr/bin/env python3
"""
Issue #21: Deep Performance Analysis (补充小P要求的物理分析)

Missing from comprehensive profiling:
1. Morrison bracket内部breakdown
2. 不同分辨率性能
3. Physics correctness baseline
4. JAX vs NumPy usage verification

Author: 小A 🤖
Date: 2026-03-25 07:48
Issue: #21 (补充分析)
Requested by: 小P ⚛️
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import jax
import jax.numpy as jnp
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pim_rl.physics.v2.tearing_ic import create_tearing_ic
from pim_rl.physics.v2.elsasser_bracket import ElsasserState

print("=" * 80)
print("Issue #21: Deep Performance Analysis (小P要求的补充)")
print("=" * 80)
print()

# ==============================================================================
# Analysis 1: Morrison Bracket Internal Breakdown
# ==============================================================================

print("=" * 80)
print("Analysis 1: Morrison Bracket Internal Breakdown")
print("=" * 80)
print()

print("Setting up test case...")
env = make_hamiltonian_mhd_env(nr=32, ntheta=64, nz=8, dt=1e-4, max_steps=1000, eta=0.05, nu=1e-4, normalize_obs=False)
psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

# Get current state
solver = env.mhd_solver.solver
state = solver.grid.to_elsasser(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

# Warm up
for _ in range(5):
    _ = solver.rhs(state)

print("Profiling Morrison bracket components (50 calls each)...\n")

# 1. Full RHS
times_rhs_full = []
for _ in range(50):
    start = time.perf_counter()
    _ = solver.rhs(state)
    times_rhs_full.append((time.perf_counter() - start) * 1000)
times_rhs_full = np.array(times_rhs_full)
print(f"1. Full RHS:                    {times_rhs_full.mean():.2f} ms")

# 2. Hamiltonian computation only
from pim_rl.physics.v2.toroidal_hamiltonian import toroidal_hamiltonian
times_hamiltonian = []
for _ in range(50):
    start = time.perf_counter()
    _ = toroidal_hamiltonian(state, solver.grid, solver.epsilon)
    times_hamiltonian.append((time.perf_counter() - start) * 1000)
times_hamiltonian = np.array(times_hamiltonian)
print(f"2. Hamiltonian (energy):        {times_hamiltonian.mean():.2f} ms")

# 3. Functional derivative (∂H/∂z±)
from pim_rl.physics.v2.elsasser_bracket import functional_derivative
times_deriv = []
for _ in range(50):
    def H(s, g):
        return toroidal_hamiltonian(s, g, solver.epsilon)
    start = time.perf_counter()
    _ = functional_derivative(H, state, solver.grid)
    times_deriv.append((time.perf_counter() - start) * 1000)
times_deriv = np.array(times_deriv)
print(f"3. Functional derivative:       {times_deriv.mean():.2f} ms")

# 4. Bracket operation only
times_bracket = []
def H(s, g):
    return toroidal_hamiltonian(s, g, solver.epsilon)
dH = functional_derivative(H, state, solver.grid)
for _ in range(50):
    start = time.perf_counter()
    _ = solver.grid.bracket(state, dH)
    times_bracket.append((time.perf_counter() - start) * 1000)
times_bracket = np.array(times_bracket)
print(f"4. Morrison bracket {{z,H}}:     {times_bracket.mean():.2f} ms")

# 5. Resistive term
from pim_rl.physics.v2.resistive_dynamics import resistive_mhd_rhs
ideal_bracket = solver.grid.bracket(state, dH)
times_resistive = []
for _ in range(50):
    start = time.perf_counter()
    _ = resistive_mhd_rhs(state, solver.grid, ideal_bracket, solver.eta, solver.pressure_scale)
    times_resistive.append((time.perf_counter() - start) * 1000)
times_resistive = np.array(times_resistive)
print(f"5. Resistive + pressure term:   {times_resistive.mean():.2f} ms")

print()
print("📊 Breakdown (% of RHS time):")
total = times_rhs_full.mean()
print(f"   Hamiltonian:       {times_hamiltonian.mean():.2f} ms ({times_hamiltonian.mean()/total*100:.1f}%)")
print(f"   Functional deriv:  {times_deriv.mean():.2f} ms ({times_deriv.mean()/total*100:.1f}%)")
print(f"   Bracket:           {times_bracket.mean():.2f} ms ({times_bracket.mean()/total*100:.1f}%)")
print(f"   Resistive:         {times_resistive.mean():.2f} ms ({times_resistive.mean()/total*100:.1f}%)")
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
    (24, 48, 6),
    (32, 64, 8),
    (48, 96, 12),
]

print("Testing different grid resolutions...\n")
print("Resolution (Nr×Nθ×Nz) | Grid Points | RHS Time | Frequency | Scaling")
print("-" * 80)

results = []
for nr, ntheta, nz in resolutions:
    # Setup
    env_res = make_hamiltonian_mhd_env(nr=nr, ntheta=ntheta, nz=nz, dt=1e-4, max_steps=1000, eta=0.05, nu=1e-4, normalize_obs=False)
    psi_res, phi_res = create_tearing_ic(nr=nr, ntheta=ntheta)
    env_res.mhd_solver.initialize(jnp.array(psi_res, dtype=jnp.float32), jnp.array(phi_res, dtype=jnp.float32))
    
    solver_res = env_res.mhd_solver.solver
    state_res = solver_res.grid.to_elsasser(jnp.array(psi_res, dtype=jnp.float32), jnp.array(phi_res, dtype=jnp.float32))
    
    # Warm up
    for _ in range(5):
        _ = solver_res.rhs(state_res)
    
    # Measure
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = solver_res.rhs(state_res)
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = np.mean(times)
    freq = 1000.0 / mean_time
    grid_points = nr * ntheta * nz
    
    results.append((nr, ntheta, nz, grid_points, mean_time, freq))
    
    # Scaling relative to 32×64×8
    if nr == 32 and ntheta == 64:
        scaling = "1.0× (baseline)"
    else:
        baseline_time = [r[4] for r in results if r[0]==32 and r[1]==64][0] if any(r[0]==32 and r[1]==64 for r in results) else mean_time
        scaling = f"{mean_time/baseline_time:.2f}×"
    
    print(f"{nr:2d} × {ntheta:2d} × {nz:2d}        | {grid_points:6d}      | {mean_time:6.2f} ms | {freq:6.1f} Hz | {scaling}")

print()

# Estimate scaling law
if len(results) >= 3:
    grid_points = np.array([r[3] for r in results])
    times = np.array([r[4] for r in results])
    
    # Fit power law: time = a * N^b
    log_N = np.log(grid_points)
    log_t = np.log(times)
    b, log_a = np.polyfit(log_N, log_t, 1)
    
    print(f"📊 Scaling law: time ∝ N^{b:.2f}")
    if abs(b - 1.0) < 0.1:
        print("   → Near-linear scaling (O(N)) ✅")
    elif abs(b - 2.0) < 0.1:
        print("   → Quadratic scaling (O(N²)) ⚠️")
    else:
        print(f"   → Non-standard scaling")
print()

# ==============================================================================
# Analysis 3: Physics Correctness Baseline
# ==============================================================================

print("=" * 80)
print("Analysis 3: Physics Correctness Baseline")
print("=" * 80)
print()

print("Verifying physics conservation laws before optimization...\n")

# Setup
env = make_hamiltonian_mhd_env(nr=32, ntheta=64, nz=8, dt=1e-4, max_steps=100, eta=0.05, nu=1e-4, normalize_obs=False)
psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

solver = env.mhd_solver.solver
state0 = solver.grid.to_elsasser(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

# Initial energy
H0 = solver.hamiltonian(state0)

# Evolve
state = state0
energies = [H0]
for i in range(100):
    state = solver.step(state, 1e-4)
    H = solver.hamiltonian(state)
    energies.append(H)

energies = np.array(energies)
energy_change = (energies[-1] - energies[0]) / abs(energies[0]) * 100

print(f"Initial energy:  {energies[0]:.6e}")
print(f"Final energy:    {energies[-1]:.6e}")
print(f"Relative change: {energy_change:+.2f}%")
print()

if abs(energy_change) < 5.0:
    print("✅ Energy conservation: GOOD (|ΔH/H| < 5%)")
elif abs(energy_change) < 20.0:
    print("⚠️ Energy conservation: ACCEPTABLE (5% < |ΔH/H| < 20%)")
else:
    print("❌ Energy conservation: POOR (|ΔH/H| > 20%)")
print()

# Energy drift rate
print("Energy time series:")
for i in [0, 25, 50, 75, 100]:
    print(f"  Step {i:3d}: H = {energies[i]:.6e} ({(energies[i]-H0)/abs(H0)*100:+.2f}%)")
print()

# ==============================================================================
# Analysis 4: JAX vs NumPy Usage
# ==============================================================================

print("=" * 80)
print("Analysis 4: JAX vs NumPy Backend Verification")
print("=" * 80)
print()

print("Checking which backend is actually used...\n")

# Check array types
state_test = solver.grid.to_elsasser(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))

print(f"State z_plus type:  {type(state_test.z_plus)}")
print(f"State z_minus type: {type(state_test.z_minus)}")
print(f"State P type:       {type(state_test.P)}")
print()

is_jax = 'jax' in str(type(state_test.z_plus)).lower()
if is_jax:
    print("✅ Backend: JAX (DeviceArray)")
else:
    print("❌ Backend: NumPy (ndarray) - NOT using JAX!")
print()

# Check if JIT is being used
print("Checking JAX JIT usage...")
import inspect

# Check RHS function
rhs_source = inspect.getsource(solver.rhs)
has_jit_decorator = '@jax.jit' in rhs_source or '@jit' in rhs_source

if has_jit_decorator:
    print("✅ RHS function: @jax.jit decorator found")
else:
    print("❌ RHS function: No @jax.jit decorator (NOT JIT-compiled!)")
print()

# Check bracket function
bracket_source = inspect.getsource(solver.grid.bracket)
bracket_has_jit = '@jax.jit' in bracket_source or '@jit' in bracket_source

if bracket_has_jit:
    print("✅ Bracket function: @jax.jit decorator found")
else:
    print("❌ Bracket function: No @jax.jit decorator")
print()

# ==============================================================================
# Final Summary
# ==============================================================================

print("=" * 80)
print("SUMMARY: Deep Analysis Results")
print("=" * 80)
print()

print("🔍 Morrison Bracket Breakdown:")
print(f"   1. Hamiltonian:       {times_hamiltonian.mean():.2f} ms ({times_hamiltonian.mean()/total*100:.1f}%)")
print(f"   2. Func derivative:   {times_deriv.mean():.2f} ms ({times_deriv.mean()/total*100:.1f}%)")
print(f"   3. Bracket:           {times_bracket.mean():.2f} ms ({times_bracket.mean()/total*100:.1f}%)")
print(f"   4. Resistive:         {times_resistive.mean():.2f} ms ({times_resistive.mean()/total*100:.1f}%)")
print()

print("📐 Resolution Scaling:")
for r in results:
    nr, ntheta, nz, grid_points, mean_time, freq = r
    print(f"   {nr}×{ntheta}×{nz}: {mean_time:.2f} ms ({freq:.1f} Hz)")
print()

print("⚖️ Physics Correctness:")
print(f"   Energy drift: {energy_change:+.2f}% over 100 steps")
if abs(energy_change) < 5.0:
    print("   Status: ✅ GOOD")
elif abs(energy_change) < 20.0:
    print("   Status: ⚠️ ACCEPTABLE")
else:
    print("   Status: ❌ POOR")
print()

print("🔧 Backend Status:")
print(f"   Arrays: {'JAX ✅' if is_jax else 'NumPy ❌'}")
print(f"   RHS JIT: {'Yes ✅' if has_jit_decorator else 'No ❌'}")
print(f"   Bracket JIT: {'Yes ✅' if bracket_has_jit else 'No ❌'}")
print()

print("=" * 80)
print("Deep Analysis Complete")
print("=" * 80)
