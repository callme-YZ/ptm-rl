"""
Energy Budget Test (FIXED VERSION)

Fixes:
1. Use φ (not ω) in compute_hamiltonian
2. Add 2π factor in theory dissipation

Author: 小P ⚛️
Created: 2026-03-19
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.operators.poisson_simple import solve_poisson_simple
from pytokmhd.physics import compute_hamiltonian, compute_current_density


def test_energy_budget_fixed():
    """Test energy budget with CORRECT Hamiltonian"""
    
    # Test parameters
    eta = 1e-4
    n_steps = 50  # Reduced for quick test
    dt = 1e-3
    
    # Grid setup
    nr, nth = 64, 64
    R0, a = 1.0, 0.3
    grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=nth)
    
    print(f"\n{'='*60}")
    print(f"Energy Budget Test (FIXED) η = {eta:.1e}")
    print(f"{'='*60}")
    
    # Initialize solver
    solver = HamiltonianMHDIMEX(
        grid,
        dt=dt,
        eta=eta,
        nu=0.0,
        use_imex=True,
        verbose=False
    )
    
    # Initial condition
    psi0 = grid.r_grid**2 * (1 - grid.r_grid/a)**2
    pert = 0.01 * np.sin(2*grid.theta_grid) * grid.r_grid**2 * (1 - grid.r_grid/a)**2
    psi = psi0 + pert
    omega = -laplacian_toroidal(psi, grid)
    
    # Volume element (2D poloidal) - CORRECT: includes r factor
    R = R0 + grid.r_grid * np.cos(grid.theta_grid)
    dV = grid.r_grid * R * grid.dr * grid.dtheta  # Toroidal jacobian: r*R
    
    # Solve for φ
    print("Solving Poisson ∇²φ = ω...")
    phi = solve_poisson_simple(omega, grid, max_iter=5000, tol=1e-5)
    
    # Initial energy
    H0 = compute_hamiltonian(psi, phi, grid)  # ✅ CORRECT: use φ
    print(f"Initial energy: H0 = {H0:.6e}")
    
    # Storage
    dH_numerical_list = []
    dH_theory_list = []
    
    # Time evolution
    for step in range(n_steps):
        # Compute J before step (CRITICAL: use mu0=1.0 for normalized units!)
        J = compute_current_density(psi, grid, mu0=1.0)
        J2_int = np.sum(J**2 * dV)
        
        # Theory: dH/dt = -η·2π·∫∫ J²·R dr dθ
        dH_theory = -eta * 2*np.pi * J2_int  # ✅ Added 2π
        
        # Step
        psi, omega = solver.step(psi, omega)
        
        # Solve for new φ
        phi_new = solve_poisson_simple(omega, grid, max_iter=5000, tol=1e-5)
        
        # Energy
        H_new = compute_hamiltonian(psi, phi_new, grid)  # ✅ CORRECT
        
        # Numerical dH/dt
        dH_numerical = (H_new - H0) / dt
        
        # Store
        dH_numerical_list.append(dH_numerical)
        dH_theory_list.append(dH_theory)
        
        # Update
        H0 = H_new
        phi = phi_new
        
        if step % 10 == 0:
            rel_err = abs((dH_numerical - dH_theory) / dH_theory) if abs(dH_theory) > 1e-10 else 0
            print(f"Step {step:3d}: dH/dt_num = {dH_numerical:.3e}, dH/dt_theory = {dH_theory:.3e}, error = {rel_err:.1%}")
    
    # Final analysis
    dH_numerical_arr = np.array(dH_numerical_list)
    dH_theory_arr = np.array(dH_theory_list)
    
    valid_idx = np.abs(dH_theory_arr) > 1e-10
    if np.sum(valid_idx) > 0:
        rel_errors = np.abs((dH_numerical_arr[valid_idx] - dH_theory_arr[valid_idx]) / dH_theory_arr[valid_idx])
        mean_error = np.mean(rel_errors)
        max_error = np.max(rel_errors)
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Mean relative error: {mean_error:.1%}")
        print(f"  Max relative error:  {max_error:.1%}")
        
        if mean_error < 0.05:
            print("  ✅ PASSED (< 5%)")
            return True
        elif mean_error < 0.10:
            print("  ⚠️  MARGINAL (< 10%)")
            return True
        else:
            print("  ❌ FAILED (> 10%)")
            return False
    else:
        print("⚠️  Theory values too small")
        return False


if __name__ == "__main__":
    success = test_energy_budget_fixed()
    if not success:
        sys.exit(1)
