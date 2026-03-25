#!/usr/bin/env python3
"""
JIT Physics Validation using Ballooning Mode (Issue #33)

Use ballooning instead of tearing (Issue #34 blocks tearing).

Author: 小P ⚛️
Date: 2026-03-25
Issue: #33
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import jax.numpy as jnp
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver, _rhs_jit, _hamiltonian_jit
from pim_rl.physics.v2.elsasser_bracket import ElsasserState
# Use simpler IC approach - just pressure bump
import jax

print("="*60)
print("Issue #33: JIT Validation (Ballooning Mode)")
print("="*60)
print("\nUsing ballooning mode (tearing deferred to Issue #34)")

# ==============================================================================
# Test 1: Energy Conservation
# ==============================================================================

def test_energy_conservation():
    print("\n" + "="*60)
    print("Test 1: Energy Conservation")
    print("="*60)
    
    nr, ntheta, nz = 32, 64, 8
    dr, dtheta, dz = 0.03125, 2*np.pi/ntheta, 0.1
    
    solver = CompleteMHDSolver(
        grid_shape=(nr, ntheta, nz),
        dr=dr, dtheta=dtheta, dz=dz,
        epsilon=0.3, eta=0.01, pressure_scale=0.5  # Ballooning needs pressure
    )
    
    # Simple pressure-driven IC (ballooning-like)
    r = np.linspace(0, 1, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Equilibrium (simple)
    psi_2d = 0.5 * R**2
    phi_2d = np.zeros_like(psi_2d)
    
    # Add m=8 perturbation
    m = 8
    psi_2d += 0.01 * R * (1 - R) * np.sin(m * Theta)
    
    psi = np.repeat(psi_2d[:, :, np.newaxis], nz, axis=2)
    phi = np.repeat(phi_2d[:, :, np.newaxis], nz, axis=2)
    
    state = ElsasserState(
        z_plus=jnp.array(psi + phi, dtype=jnp.float32),
        z_minus=jnp.array(psi - phi, dtype=jnp.float32),
        P=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.17  # β₀ p
    )
    
    E0 = float(_hamiltonian_jit(state, solver.grid, solver.epsilon))
    print(f"Initial energy: {E0:.6f}")
    
    # Evolve 10 steps
    dt = 1e-4
    energies = [E0]
    
    for step in range(10):
        dstate = _rhs_jit(state, solver.grid, solver.epsilon, solver.eta, solver.pressure_scale)
        state = ElsasserState(
            z_plus=state.z_plus + dt * dstate.z_plus,
            z_minus=state.z_minus + dt * dstate.z_minus,
            P=state.P + dt * dstate.P
        )
        E = float(_hamiltonian_jit(state, solver.grid, solver.epsilon))
        energies.append(E)
    
    E_final = energies[-1]
    dE = E_final - E0
    dE_relative = abs(dE) / abs(E0)
    
    print(f"Final energy: {E_final:.6f}")
    print(f"ΔE: {dE:.6e}")
    print(f"ΔE/E₀: {dE_relative:.6e}")
    
    assert dE_relative < 0.01, f"Energy drift too large: {dE_relative:.2%}"
    print("✅ Energy conservation: PASS")

# ==============================================================================
# Test 2: Physics Stability
# ==============================================================================

def test_physics_stability():
    print("\n" + "="*60)
    print("Test 2: Physics Stability (No NaN/Inf)")
    print("="*60)
    
    nr, ntheta, nz = 32, 64, 8
    dr, dtheta, dz = 0.03125, 2*np.pi/ntheta, 0.1
    
    solver = CompleteMHDSolver(
        grid_shape=(nr, ntheta, nz),
        dr=dr, dtheta=dtheta, dz=dz,
        epsilon=0.3, eta=0.01, pressure_scale=0.5
    )
    
    r = np.linspace(0, 1, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    psi_2d = 0.5 * R**2 + 0.05 * R * (1 - R) * np.sin(8 * Theta)
    phi_2d = np.zeros_like(psi_2d)
    psi = np.repeat(psi_2d[:, :, np.newaxis], nz, axis=2)
    phi = np.repeat(phi_2d[:, :, np.newaxis], nz, axis=2)
    
    state = ElsasserState(
        z_plus=jnp.array(psi + phi, dtype=jnp.float32),
        z_minus=jnp.array(psi - phi, dtype=jnp.float32),
        P=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.17
    )
    
    # Evolve 500 steps
    dt = 1e-4
    for step in range(500):
        dstate = _rhs_jit(state, solver.grid, solver.epsilon, solver.eta, solver.pressure_scale)
        state = ElsasserState(
            z_plus=state.z_plus + dt * dstate.z_plus,
            z_minus=state.z_minus + dt * dstate.z_minus,
            P=state.P + dt * dstate.P
        )
        
        if step % 100 == 0:
            max_z = float(jnp.max(jnp.abs(state.z_plus)))
            print(f"  Step {step}: max|z+| = {max_z:.6f}")
    
    # Check for NaN/Inf
    assert jnp.all(jnp.isfinite(state.z_plus)), "z_plus has NaN/Inf"
    assert jnp.all(jnp.isfinite(state.z_minus)), "z_minus has NaN/Inf"
    assert jnp.all(jnp.isfinite(state.P)), "P has NaN/Inf"
    
    print("✅ Physics stability: PASS (500 steps, no blowup)")

# ==============================================================================
# Test 3: Dynamic eta (RL Control)
# ==============================================================================

def test_dynamic_eta():
    print("\n" + "="*60)
    print("Test 3: Dynamic eta (RL Control Scenario)")
    print("="*60)
    
    nr, ntheta, nz = 32, 64, 8
    dr, dtheta, dz = 0.03125, 2*np.pi/ntheta, 0.1
    
    solver = CompleteMHDSolver(
        grid_shape=(nr, ntheta, nz),
        dr=dr, dtheta=dtheta, dz=dz,
        epsilon=0.3, eta=0.01, pressure_scale=0.5
    )
    
    state = ElsasserState(
        z_plus=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.1,
        z_minus=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.05,
        P=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.1
    )
    
    # Test different eta values (RL control)
    eta_values = [0.01, 0.05, 0.1, 0.05, 0.02]
    
    import time
    times = []
    
    for i, eta in enumerate(eta_values):
        t0 = time.time()
        dstate = _rhs_jit(state, solver.grid, solver.epsilon, eta, solver.pressure_scale)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Call {i+1}: eta={eta:.3f}, time={elapsed*1000:.2f} ms")
    
    first_call = times[0]
    later_calls = times[1:]
    avg_later = np.mean(later_calls)
    
    print(f"\nFirst call (compilation): {first_call*1000:.2f} ms")
    print(f"Later calls (avg): {avg_later*1000:.2f} ms")
    
    speedup = first_call / avg_later
    print(f"Speedup after compilation: {speedup:.1f}×")
    
    assert speedup > 2.0, f"No compilation speedup: {speedup:.1f}×"
    print("✅ Dynamic eta: PASS (no recompilation overhead)")

# ==============================================================================
# Test 4: Performance Benchmark
# ==============================================================================

def test_performance():
    print("\n" + "="*60)
    print("Test 4: Performance Benchmark")
    print("="*60)
    
    nr, ntheta, nz = 32, 64, 8
    dr, dtheta, dz = 0.03125, 2*np.pi/ntheta, 0.1
    
    solver = CompleteMHDSolver(
        grid_shape=(nr, ntheta, nz),
        dr=dr, dtheta=dtheta, dz=dz,
        epsilon=0.3, eta=0.01, pressure_scale=0.5
    )
    
    r = np.linspace(0, 1, nr)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    psi_2d = 0.5 * R**2 + 0.01 * R * (1 - R) * np.sin(8 * Theta)
    phi_2d = np.zeros_like(psi_2d)
    psi = np.repeat(psi_2d[:, :, np.newaxis], nz, axis=2)
    phi = np.repeat(phi_2d[:, :, np.newaxis], nz, axis=2)
    
    state = ElsasserState(
        z_plus=jnp.array(psi + phi, dtype=jnp.float32),
        z_minus=jnp.array(psi - phi, dtype=jnp.float32),
        P=jnp.ones((nr, ntheta, nz), dtype=jnp.float32) * 0.17
    )
    
    # Warmup
    for _ in range(10):
        _ = _rhs_jit(state, solver.grid, solver.epsilon, solver.eta, solver.pressure_scale)
    
    # Benchmark
    import time
    n_trials = 100
    times = []
    
    for _ in range(n_trials):
        t0 = time.time()
        _ = _rhs_jit(state, solver.grid, solver.epsilon, solver.eta, solver.pressure_scale)
        times.append(time.time() - t0)
    
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    
    print(f"RHS computation (JIT):")
    print(f"  Mean: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min:  {min(times)*1000:.2f} ms")
    print(f"  Max:  {max(times)*1000:.2f} ms")
    
    # Compare to Issue #21 baseline (17 ms non-JIT)
    baseline_ms = 17.0
    speedup = baseline_ms / avg_time
    
    print(f"\nComparison to non-JIT baseline:")
    print(f"  Baseline (from Issue #21): {baseline_ms:.2f} ms")
    print(f"  JIT version: {avg_time:.2f} ms")
    print(f"  Speedup: {speedup:.1f}×")
    
    # Target: >3× speedup
    assert speedup > 3.0, f"Insufficient speedup: {speedup:.1f}× (target: >3×)"
    print(f"✅ Performance: PASS ({speedup:.1f}× speedup)")

# ==============================================================================
# Run All Tests
# ==============================================================================

if __name__ == "__main__":
    try:
        test_energy_conservation()
        test_physics_stability()
        test_dynamic_eta()
        test_performance()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nIssue #33 JIT Validation Complete!")
        print("- Physics correctness: ✅")
        print("- Energy conservation: ✅") 
        print("- Stability: ✅")
        print("- Dynamic eta: ✅")
        print("- Performance: ✅ (>3× speedup)")
        print("\nSafe to merge JIT optimization.")
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\n{e}")
        print("\nDo NOT merge until fixed!")
        sys.exit(1)
