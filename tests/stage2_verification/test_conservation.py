"""
Stage 2 Verification: Task 2.1 - Long-term Energy Conservation Tests

Tests ideal MHD (η=0, ν=0) energy conservation over 1000+ steps.

Expected: |dH/dt| < 1e-12 (machine precision)

Author: 小P ⚛️
For: 小A 🤖 to execute
Date: 2026-03-23
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from pytokmhd.grid import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd import HamiltonianMHD
from pytokmhd.physics.initial_conditions import ballooning_ic


def compute_energy(solver, psi, omega):
    """
    Compute total energy H = ∫[½|∇φ|² + ½|∇ψ|²] dV
    
    From Stage 1 theory:
    - Kinetic: E_kin = ∫ ½|∇φ|² dV (φ from ∇²φ = ω)
    - Magnetic: E_mag = ∫ ½|∇ψ|² dV
    """
    grid = solver.grid
    
    # Solve for φ from ω
    phi = solver.poisson_solver.solve(omega)
    
    # Compute gradients
    grad_phi_r, grad_phi_theta = solver.gradient(phi)
    grad_psi_r, grad_psi_theta = solver.gradient(psi)
    
    # |∇f|² in toroidal geometry
    grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.R[:, None])**2
    grad_psi_sq = grad_psi_r**2 + (grad_psi_theta / grid.R[:, None])**2
    
    # Integrate with volume element R dr dθ
    dV = grid.dR * grid.dtheta * grid.R[:, None]
    
    E_kin = 0.5 * np.sum(grad_phi_sq * dV)
    E_mag = 0.5 * np.sum(grad_psi_sq * dV)
    
    return E_kin + E_mag


def test_ideal_conservation_short():
    """
    Test 1: Short run (100 steps) for quick validation
    
    Expected: |dH/dt| < 1e-10
    """
    print("\n" + "="*60)
    print("TEST 1: Ideal MHD Conservation (100 steps)")
    print("="*60)
    
    # Setup
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    # CRITICAL: η=0, ν=0 for ideal MHD
    solver = HamiltonianMHD(
        grid=grid,
        dt=1e-3,
        eta=0.0,    # ← No resistivity
        nu=0.0      # ← No viscosity
    )
    
    # Initial condition: ballooning mode
    psi, omega = ballooning_ic(grid, beta=0.17, q_axis=1.2, shear=0.5)
    
    # Run simulation
    n_steps = 100
    energies = []
    
    print(f"Running {n_steps} steps...")
    for step in range(n_steps):
        H = compute_energy(solver, psi, omega)
        energies.append(H)
        
        # Evolve
        psi, omega = solver.step(psi, omega)
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: H = {H:.12e}")
    
    energies = np.array(energies)
    
    # Analysis
    H_init = energies[0]
    H_final = energies[-1]
    dH = H_final - H_init
    rel_drift = np.abs(dH / H_init)
    
    max_drift = np.max(np.abs(energies - H_init)) / H_init
    
    print("\nResults:")
    print(f"  H_initial: {H_init:.12e}")
    print(f"  H_final:   {H_final:.12e}")
    print(f"  ΔH:        {dH:.12e}")
    print(f"  Rel drift: {rel_drift:.12e}")
    print(f"  Max drift: {max_drift:.12e}")
    
    # Verdict
    threshold = 1e-10
    if max_drift < threshold:
        print(f"\n✅ PASS: Energy drift < {threshold:.0e}")
        return True
    else:
        print(f"\n❌ FAIL: Energy drift {max_drift:.2e} > {threshold:.0e}")
        return False


def test_ideal_conservation_long():
    """
    Test 2: Long run (1000 steps) for Stage 2 requirement
    
    Expected: |dH/dt| < 1e-12
    Secular drift: No long-term trend
    """
    print("\n" + "="*60)
    print("TEST 2: Ideal MHD Conservation (1000 steps)")
    print("="*60)
    
    # Setup (same as Test 1)
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    solver = HamiltonianMHD(
        grid=grid,
        dt=1e-3,
        eta=0.0,
        nu=0.0
    )
    
    psi, omega = ballooning_ic(grid, beta=0.17, q_axis=1.2, shear=0.5)
    
    # Run simulation
    n_steps = 1000
    energies = []
    times = []
    
    print(f"Running {n_steps} steps (this may take a minute)...")
    for step in range(n_steps):
        H = compute_energy(solver, psi, omega)
        energies.append(H)
        times.append(step * solver.dt)
        
        psi, omega = solver.step(psi, omega)
        
        if step % 100 == 0:
            print(f"  Step {step:4d}: H = {H:.12e}")
    
    energies = np.array(energies)
    times = np.array(times)
    
    # Analysis
    H_init = energies[0]
    H_final = energies[-1]
    dH = H_final - H_init
    rel_drift = np.abs(dH / H_init)
    max_drift = np.max(np.abs(energies - H_init)) / H_init
    
    # Check for secular drift (linear fit)
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(times, energies - H_init, deg=1)
    secular_slope = p.coef[1]
    
    print("\nResults:")
    print(f"  H_initial:     {H_init:.12e}")
    print(f"  H_final:       {H_final:.12e}")
    print(f"  ΔH:            {dH:.12e}")
    print(f"  Rel drift:     {rel_drift:.12e}")
    print(f"  Max drift:     {max_drift:.12e}")
    print(f"  Secular slope: {secular_slope:.2e} (should be ~0)")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Absolute energy
    ax1.plot(times, energies, 'b-', linewidth=1)
    ax1.axhline(H_init, color='r', linestyle='--', label='Initial')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy H')
    ax1.set_title('Energy Conservation (Ideal MHD, 1000 steps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative drift
    ax2.plot(times, (energies - H_init) / H_init, 'b-', linewidth=1)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('(H - H₀) / H₀')
    ax2.set_title('Relative Energy Drift')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-max_drift*2, max_drift*2])
    
    plt.tight_layout()
    plt.savefig('stage2_energy_conservation.png', dpi=150)
    print("\n📊 Plot saved: stage2_energy_conservation.png")
    
    # Verdict
    threshold = 1e-12
    secular_threshold = 1e-15
    
    pass_drift = max_drift < threshold
    pass_secular = np.abs(secular_slope) < secular_threshold
    
    if pass_drift and pass_secular:
        print(f"\n✅ PASS: Energy conservation verified")
        print(f"  - Max drift {max_drift:.2e} < {threshold:.0e}")
        print(f"  - No secular trend (slope {secular_slope:.2e})")
        return True
    else:
        print(f"\n❌ FAIL:")
        if not pass_drift:
            print(f"  - Drift {max_drift:.2e} > {threshold:.0e}")
        if not pass_secular:
            print(f"  - Secular drift detected (slope {secular_slope:.2e})")
        return False


def test_resistive_dissipation():
    """
    Test 3: Resistive case (η > 0) should show energy decrease
    
    Expected: dH/dt < 0 (matches Stage 1 theory)
    Rate: dH/dt ≈ -∫ η|∇J|² dV
    """
    print("\n" + "="*60)
    print("TEST 3: Resistive MHD Dissipation (η=1e-4)")
    print("="*60)
    
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    # Resistive case
    solver = HamiltonianMHD(
        grid=grid,
        dt=1e-3,
        eta=1e-4,    # ← Resistivity ON
        nu=0.0       # ← No viscosity (isolate resistive effect)
    )
    
    psi, omega = ballooning_ic(grid, beta=0.17, q_axis=1.2, shear=0.5)
    
    n_steps = 500
    energies = []
    
    print(f"Running {n_steps} steps...")
    for step in range(n_steps):
        H = compute_energy(solver, psi, omega)
        energies.append(H)
        
        psi, omega = solver.step(psi, omega)
        
        if step % 100 == 0:
            print(f"  Step {step:3d}: H = {H:.12e}")
    
    energies = np.array(energies)
    
    # Analysis
    H_init = energies[0]
    H_final = energies[-1]
    dH = H_final - H_init
    
    # Estimate dH/dt
    dH_dt = np.gradient(energies, solver.dt)
    avg_dH_dt = np.mean(dH_dt[10:])  # Skip initial transient
    
    print("\nResults:")
    print(f"  H_initial: {H_init:.12e}")
    print(f"  H_final:   {H_final:.12e}")
    print(f"  ΔH:        {dH:.12e}")
    print(f"  dH/dt:     {avg_dH_dt:.2e} (should be < 0)")
    
    # Verdict
    if dH < 0 and avg_dH_dt < 0:
        print(f"\n✅ PASS: Energy dissipation verified (dH/dt < 0)")
        return True
    else:
        print(f"\n❌ FAIL: Energy should decrease for resistive MHD")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Stage 2 Verification: Energy Conservation Tests")
    print(" Issue #23 - Task 2.1")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['short'] = test_ideal_conservation_short()
    results['long'] = test_ideal_conservation_long()
    results['resistive'] = test_resistive_dissipation()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Ideal, 100 steps):  {'✅ PASS' if results['short'] else '❌ FAIL'}")
    print(f"Test 2 (Ideal, 1000 steps): {'✅ PASS' if results['long'] else '❌ FAIL'}")
    print(f"Test 3 (Resistive):         {'✅ PASS' if results['resistive'] else '❌ FAIL'}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n🎉 All tests PASSED - Energy conservation verified! ✅")
        print("\nNext: Task 2.2 - Poisson bracket properties")
    else:
        print("\n⚠️  Some tests FAILED - investigate before proceeding")
    
    print("="*70)
