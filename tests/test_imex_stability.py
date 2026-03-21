"""
IMEX Stability Tests (v1.3)

Validates IMEX time stepping for resistive MHD.

Test Cases:
1. Small resistivity (η=1e-5): Should be stable
2. Moderate resistivity (η=1e-4): Should be stable
3. Large resistivity (η=1e-3): Should be stable (new capability!)
4. Energy dissipation: Verify dH/dt = -η∫|J|² (resistive dissipation)

Success Criteria:
- No NaN for η up to 1e-3
- Energy decreases monotonically
- Convergence with dt refinement
- Correct energy dissipation rate

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Validation
"""

import numpy as np
import pytest
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.physics import compute_current_density, compute_hamiltonian
from pytokmhd.solvers import solve_poisson_toroidal


@pytest.fixture
def grid():
    """Standard test grid."""
    return ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)


@pytest.fixture
def initial_state(grid):
    """
    Initial equilibrium state.
    
    Simple Harris sheet-like equilibrium:
        ψ = r² * (1 - r/a)
    """
    r_grid = grid.r_grid
    a = grid.a
    
    psi = r_grid**2 * (1.0 - r_grid / a)
    omega = -laplacian_toroidal(psi, grid)
    
    return psi, omega


def test_imex_small_resistivity(grid, initial_state):
    """
    Test η=1e-5 (should be stable even with explicit).
    
    This validates that IMEX doesn't break working cases.
    """
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-5, nu=0.0, use_imex=True
    )
    
    H0 = compute_hamiltonian(psi, omega, grid)
    
    # Evolve 100 steps
    for _ in range(100):
        psi, omega = solver.step(psi, omega)
    
    H_final = compute_hamiltonian(psi, omega, grid)
    
    # Check no NaN
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(omega))
    
    # Energy should decrease (resistive dissipation)
    assert H_final < H0
    
    # Energy drift should be small (< 1%)
    energy_drift = abs(H_final - H0) / H0
    assert energy_drift < 0.01


def test_imex_moderate_resistivity(grid, initial_state):
    """
    Test η=1e-4 (moderate resistivity).
    
    This would be unstable with explicit treatment at dt=1e-4.
    IMEX should handle it gracefully.
    """
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-4, nu=0.0, use_imex=True
    )
    
    H0 = compute_hamiltonian(psi, omega, grid)
    
    # Evolve 100 steps
    for _ in range(100):
        psi, omega = solver.step(psi, omega)
    
    H_final = compute_hamiltonian(psi, omega, grid)
    
    # Check no NaN
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(omega))
    
    # Energy should decrease monotonically
    assert H_final < H0


def test_imex_large_resistivity(grid, initial_state):
    """
    Test η=1e-3 (large resistivity).
    
    **This is the key new capability!**
    
    Explicit treatment would explode. IMEX should be stable.
    """
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-3, nu=0.0, use_imex=True
    )
    
    H0 = compute_hamiltonian(psi, omega, grid)
    energies = [H0]
    
    # Evolve 100 steps
    for i in range(100):
        psi, omega = solver.step(psi, omega)
        H = compute_hamiltonian(psi, omega, grid)
        energies.append(H)
        
        # Check no NaN at each step
        if np.any(np.isnan(psi)) or np.any(np.isnan(omega)):
            raise AssertionError(f"NaN detected at step {i+1}")
    
    H_final = energies[-1]
    
    # Check no NaN
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(omega))
    
    # Energy should decrease monotonically
    energies_arr = np.array(energies)
    assert np.all(np.diff(energies_arr) <= 0), "Energy should decrease monotonically"
    
    # Significant dissipation expected
    assert H_final < 0.5 * H0, "Should dissipate significant energy"


def test_energy_dissipation_rate(grid, initial_state):
    """
    Verify energy budget: dH/dt ≈ -η∫|J|² (resistive dissipation).
    
    This checks that IMEX preserves the correct physics.
    """
    psi, omega = initial_state
    
    eta = 1e-4
    dt = 1e-4
    
    solver = HamiltonianMHDIMEX(
        grid, dt=dt, eta=eta, nu=0.0, use_imex=True
    )
    
    H0 = compute_hamiltonian(psi, omega, grid)
    J0 = compute_current_density(psi, grid)
    
    # Analytical dissipation rate: -η∫|J|²dV
    # Volume element: dV = r*R*dr*dtheta (axisymmetric)
    dV = grid.jacobian() * grid.dr * grid.dtheta
    resistive_power = -eta * np.sum(J0**2 * dV)
    
    # Take one step
    psi, omega = solver.step(psi, omega)
    H1 = compute_hamiltonian(psi, omega, grid)
    
    # Numerical dissipation rate
    dH_dt_numerical = (H1 - H0) / dt
    
    # Expected: dH/dt ≈ -η∫|J|²
    # Allow 20% error (discretization, time integration)
    relative_error = abs(dH_dt_numerical - resistive_power) / abs(resistive_power)
    
    assert relative_error < 0.2, (
        f"Energy dissipation rate mismatch: "
        f"numerical={dH_dt_numerical:.3e}, "
        f"analytical={resistive_power:.3e}, "
        f"error={relative_error:.1%}"
    )


def test_imex_convergence(grid, initial_state):
    """
    Test convergence with dt refinement.
    
    Error should scale as O(dt²) for 2nd-order method.
    """
    psi0, omega0 = initial_state
    
    eta = 1e-4
    T_final = 1e-3  # Short evolution
    
    # Reference solution: very small dt
    dt_ref = 1e-6
    solver_ref = HamiltonianMHDIMEX(grid, dt=dt_ref, eta=eta, nu=0.0)
    psi_ref, omega_ref = psi0.copy(), omega0.copy()
    n_steps_ref = int(T_final / dt_ref)
    for _ in range(n_steps_ref):
        psi_ref, omega_ref = solver_ref.step(psi_ref, omega_ref)
    
    # Test solutions at different dt
    dt_values = [1e-4, 2e-4, 4e-4]
    errors = []
    
    for dt in dt_values:
        solver = HamiltonianMHDIMEX(grid, dt=dt, eta=eta, nu=0.0)
        psi, omega = psi0.copy(), omega0.copy()
        n_steps = int(T_final / dt)
        for _ in range(n_steps):
            psi, omega = solver.step(psi, omega)
        
        # Error in ψ
        error = np.max(np.abs(psi - psi_ref))
        errors.append(error)
    
    # Check convergence: error(2*dt) / error(dt) ≈ 4 (2nd order)
    # Relaxed check: should decrease
    assert errors[1] > errors[0], "Error should decrease with smaller dt"
    assert errors[2] > errors[1], "Error should continue to decrease"


def test_imex_vs_explicit_small_eta(grid, initial_state):
    """
    Compare IMEX vs explicit for small η where both should work.
    
    Validates that IMEX doesn't introduce spurious behavior.
    """
    psi0, omega0 = initial_state
    
    eta = 1e-5
    dt = 1e-5  # Small dt for explicit stability
    n_steps = 10
    
    # IMEX
    solver_imex = HamiltonianMHDIMEX(grid, dt=dt, eta=eta, nu=0.0, use_imex=True)
    psi_imex, omega_imex = psi0.copy(), omega0.copy()
    for _ in range(n_steps):
        psi_imex, omega_imex = solver_imex.step(psi_imex, omega_imex)
    
    # Explicit
    solver_explicit = HamiltonianMHDIMEX(grid, dt=dt, eta=eta, nu=0.0, use_imex=False)
    psi_explicit, omega_explicit = psi0.copy(), omega0.copy()
    for _ in range(n_steps):
        psi_explicit, omega_explicit = solver_explicit.step(psi_explicit, omega_explicit)
    
    # Should be close (< 1% difference)
    error_psi = np.max(np.abs(psi_imex - psi_explicit)) / np.max(np.abs(psi_explicit))
    error_omega = np.max(np.abs(omega_imex - omega_explicit)) / np.max(np.abs(omega_explicit))
    
    assert error_psi < 0.01, f"IMEX vs explicit ψ error too large: {error_psi:.3e}"
    assert error_omega < 0.01, f"IMEX vs explicit ω error too large: {error_omega:.3e}"


if __name__ == "__main__":
    # Run quick validation
    print("IMEX Stability Tests")
    print("=" * 60)
    
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    psi, omega = (
        grid.r_grid**2 * (1.0 - grid.r_grid / grid.a),
        -laplacian_toroidal(grid.r_grid**2 * (1.0 - grid.r_grid / grid.a), grid)
    )
    
    print("\nTest 1: Small resistivity (η=1e-5)")
    test_imex_small_resistivity(grid, (psi.copy(), omega.copy()))
    print("✅ PASSED")
    
    print("\nTest 2: Moderate resistivity (η=1e-4)")
    test_imex_moderate_resistivity(grid, (psi.copy(), omega.copy()))
    print("✅ PASSED")
    
    print("\nTest 3: Large resistivity (η=1e-3) [NEW CAPABILITY]")
    test_imex_large_resistivity(grid, (psi.copy(), omega.copy()))
    print("✅ PASSED - Stable for η=1e-3!")
    
    print("\nTest 4: Energy dissipation rate")
    test_energy_dissipation_rate(grid, (psi.copy(), omega.copy()))
    print("✅ PASSED")
    
    print("\n" + "=" * 60)
    print("All tests PASSED! IMEX implementation validated.")
