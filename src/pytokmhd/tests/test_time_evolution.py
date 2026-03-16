"""
Unit Tests for Time Evolution

Tests:
1. RK4 stability over 100 timesteps
2. Energy conservation < 1%
3. No NaN/Inf divergence

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

from pytokmhd.solver import time_integrator, boundary, mhd_equations
from pytokmhd.solver.poisson_solver import solve_poisson


def compute_energy(psi, omega, dr, dz, r_grid):
    """
    Compute total energy: E = ∫(B² + v²) dr dz.
    
    Simplified: E ≈ ∫(|∇ψ|² + ω²) dr dz
    """
    # Gradient energy
    dpsi_dr = mhd_equations.gradient_r(psi, dr)
    dpsi_dz = mhd_equations.gradient_z(psi, dz)
    
    E_mag = 0.5 * np.sum((dpsi_dr**2 + dpsi_dz**2) * r_grid) * dr * dz
    E_kin = 0.5 * np.sum(omega**2 * r_grid) * dr * dz
    
    return E_mag + E_kin


def test_rk4_stability():
    """Test: RK4 remains stable for 100 steps."""
    print("\n=== Test: RK4 Stability ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Initial condition: small perturbation
    psi0 = 0.1 * np.sin(2*np.pi*Z/Lz) * (1 - R**2)
    omega0 = np.zeros_like(psi0)
    
    # Physics parameters
    eta = 1e-3
    dt = 0.001
    n_steps = 100
    
    # Evolve
    psi = psi0.copy()
    omega = omega0.copy()
    
    energies = []
    
    for step in range(n_steps):
        # RK4 step
        psi, omega = time_integrator.rk4_step(
            psi, omega, dt, dr, dz, R, eta, nu=0.0,
            apply_bc=boundary.apply_combined_bc
        )
        
        # Check for NaN/Inf
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            raise AssertionError(f"NaN/Inf detected at step {step}")
        
        # Compute energy
        E = compute_energy(psi, omega, dr, dz, R)
        energies.append(E)
    
    # Check stability: energy should not blow up
    E_final = energies[-1]
    E_initial = energies[0]
    
    print(f"Initial energy: {E_initial:.6e}")
    print(f"Final energy:   {E_final:.6e}")
    print(f"Relative change: {abs(E_final - E_initial)/E_initial * 100:.2f}%")
    
    # Stability criterion: energy change < 100% (should not explode)
    assert abs(E_final - E_initial)/E_initial < 1.0, "RK4 unstable: energy exploded"
    
    print("✅ PASSED: RK4 stable over 100 steps")
    
    # Return removed for pytest compliance


def test_energy_conservation():
    """Test: Energy conservation < 1% (for conservative case)."""
    print("\n=== Test: Energy Conservation ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Initial condition: equilibrium (should be conserved)
    psi0 = (1 - R**2)  # Parabolic profile (equilibrium)
    omega0 = np.zeros_like(psi0)
    
    # Physics parameters (very small resistivity)
    eta = 1e-6  # Minimal dissipation
    dt = 0.001
    n_steps = 100
    
    # Evolve
    psi = psi0.copy()
    omega = omega0.copy()
    
    E_initial = compute_energy(psi, omega, dr, dz, R)
    
    for step in range(n_steps):
        psi, omega = time_integrator.rk4_step(
            psi, omega, dt, dr, dz, R, eta, nu=0.0,
            apply_bc=boundary.apply_combined_bc
        )
    
    E_final = compute_energy(psi, omega, dr, dz, R)
    
    # Energy drift
    drift = abs(E_final - E_initial) / E_initial * 100
    
    print(f"Initial energy: {E_initial:.6e}")
    print(f"Final energy:   {E_final:.6e}")
    print(f"Drift: {drift:.2f}%")
    
    # Conservation criterion: drift < 1%
    assert drift < 1.0, f"Energy conservation violated: drift {drift:.2f}% > 1%"
    
    print("✅ PASSED: Energy conserved within 1%")
    
    # Return removed for pytest compliance


def test_no_divergence():
    """Test: No NaN/Inf over 100 steps."""
    print("\n=== Test: No Divergence ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Initial condition: random perturbation (stress test)
    np.random.seed(42)
    psi0 = 0.01 * np.random.randn(Nr, Nz) * (1 - R**2)
    omega0 = 0.01 * np.random.randn(Nr, Nz)
    
    # Physics parameters
    eta = 1e-3
    dt = 0.001
    n_steps = 100
    
    # Evolve
    psi = psi0.copy()
    omega = omega0.copy()
    
    for step in range(n_steps):
        psi, omega = time_integrator.rk4_step(
            psi, omega, dt, dr, dz, R, eta, nu=0.0,
            apply_bc=boundary.apply_combined_bc
        )
        
        # Check for NaN/Inf
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            raise AssertionError(f"Divergence at step {step}: NaN/Inf in psi")
        if np.any(np.isnan(omega)) or np.any(np.isinf(omega)):
            raise AssertionError(f"Divergence at step {step}: NaN/Inf in omega")
        
        # Check magnitude (should not explode)
        if np.max(np.abs(psi)) > 1e6:
            raise AssertionError(f"Divergence at step {step}: psi too large")
        if np.max(np.abs(omega)) > 1e6:
            raise AssertionError(f"Divergence at step {step}: omega too large")
    
    print(f"Final max(|psi|):   {np.max(np.abs(psi)):.2e}")
    print(f"Final max(|omega|): {np.max(np.abs(omega)):.2e}")
    print("✅ PASSED: No divergence over 100 steps")
    
    pass  # Test passed


def run_all_tests():
    """Run all time evolution tests."""
    print("="*60)
    print("PyTokMHD Time Evolution Tests")
    print("="*60)
    
    results = {}
    
    try:
        results['stability'] = test_rk4_stability()
        results['conservation'] = test_energy_conservation()
        results['no_divergence'] = test_no_divergence()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print("\nSummary:")
        print(f"  RK4 stable over 100 steps")
        print(f"  Energy drift: {results['conservation']:.2f}%")
        print(f"  No NaN/Inf divergence")
        
        pass  # Test passed
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        assert False, "No divergence check failed"


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
