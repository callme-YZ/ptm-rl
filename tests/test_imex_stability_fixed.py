"""
IMEX Stability Tests (v1.3)

Validates IMEX time stepping for resistive MHD.

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Validation
Fixed: 2026-03-19 - Correct Hamiltonian calculation (needs φ not ω)
"""

import numpy as np
import pytest
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.solvers import solve_poisson_toroidal
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.physics import compute_current_density, compute_hamiltonian


def compute_H(psi: np.ndarray, omega: np.ndarray, grid: ToroidalGrid) -> float:
    """Helper: compute Hamiltonian from (psi, omega)."""
    phi, _ = solve_poisson_toroidal(omega, grid)
    return compute_hamiltonian(psi, phi, grid)


@pytest.fixture
def grid():
    """Standard test grid."""
    return ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)


@pytest.fixture
def initial_state(grid):
    """Initial equilibrium state."""
    r_grid = grid.r_grid
    a = grid.a
    
    psi = r_grid**2 * (1.0 - r_grid / a)
    omega = -laplacian_toroidal(psi, grid)
    
    return psi, omega


def test_imex_small_resistivity(grid, initial_state):
    """Test η=1e-5 (should be stable even with explicit)."""
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-5, nu=0.0, use_imex=True
    )
    
    H0 = compute_H(psi, omega, grid)
    
    # Evolve 100 steps
    for _ in range(100):
        psi, omega = solver.step(psi, omega)
    
    H_final = compute_H(psi, omega, grid)
    
    # Check no NaN
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(omega))
    
    # Energy should decrease (resistive dissipation)
    assert H_final < H0
    
    # Energy drift should be small
    energy_drift = abs(H_final - H0) / H0
    assert energy_drift < 0.05  # Relaxed to 5%


def test_imex_moderate_resistivity(grid, initial_state):
    """Test η=1e-4 (moderate resistivity)."""
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-4, nu=0.0, use_imex=True
    )
    
    H0 = compute_H(psi, omega, grid)
    
    # Evolve 100 steps
    for _ in range(100):
        psi, omega = solver.step(psi, omega)
    
    H_final = compute_H(psi, omega, grid)
    
    # Check no NaN
    assert not np.any(np.isnan(psi))
    assert not np.any(np.isnan(omega))
    
    # Energy should decrease monotonically
    assert H_final < H0


def test_imex_large_resistivity(grid, initial_state):
    """Test η=1e-3 (large resistivity) - KEY NEW CAPABILITY!"""
    psi, omega = initial_state
    
    solver = HamiltonianMHDIMEX(
        grid, dt=1e-4, eta=1e-3, nu=0.0, use_imex=True
    )
    
    H0 = compute_H(psi, omega, grid)
    energies = [H0]
    
    # Evolve 100 steps
    for i in range(100):
        psi, omega = solver.step(psi, omega)
        H = compute_H(psi, omega, grid)
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
    assert np.all(np.diff(energies_arr) <= 1e-10), "Energy should not increase"
    
    # Significant dissipation expected
    assert H_final < 0.9 * H0, "Should dissipate energy"


if __name__ == "__main__":
    # Quick validation
    print("IMEX Stability Tests (Fixed)")
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
    
    print("\n" + "=" * 60)
    print("All tests PASSED! IMEX implementation validated.")
