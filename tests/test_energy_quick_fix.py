"""
Quick Fix Test: Just fix μ₀ units

Test only Fix #1 to see if that's the main issue.
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.physics import compute_current_density, compute_hamiltonian

# Test parameters
eta = 1e-4
n_steps = 20  # Reduced for speed
dt = 1e-3

# Grid
nr, nth = 64, 64
R0, a = 1.0, 0.3
grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=nth)

print("Testing Fix #1: μ₀ = 1 (normalized units)")

# Solver
solver = HamiltonianMHDIMEX(grid, dt=dt, eta=eta, nu=0.0, use_imex=True, verbose=False)

# Initial condition
psi0 = grid.r_grid**2 * (1 - grid.r_grid/a)**2
pert = 0.01 * np.sin(2*grid.theta_grid) * grid.r_grid**2 * (1 - grid.r_grid/a)**2
psi = psi0 + pert
omega = -laplacian_toroidal(psi, grid)

# Volume element
dV = (R0 + grid.r_grid * np.cos(grid.theta_grid)) * grid.dr * grid.dtheta

# Initial energy (still using omega, but that's OK for now)
H0 = compute_hamiltonian(psi, omega, grid)

print(f"Initial H = {H0:.6e}")

# Evolution
errors = []

for step in range(n_steps):
    # FIX #1: Use μ₀ = 1 (normalized units)
    J = compute_current_density(psi, grid, mu0=1.0)  # ✅ FIXED
    J2_int = np.sum(J**2 * dV)
    
    # FIX #3: Add 2π
    dH_theory = -eta * 2*np.pi * J2_int  # ✅ FIXED
    
    # Step
    psi, omega = solver.step(psi, omega)
    
    # Energy (still using omega - Fix #2 pending)
    H = compute_hamiltonian(psi, omega, grid)
    dH_numeric = (H - H0) / dt
    H0 = H
    
    if abs(dH_theory) > 1e-10:
        rel_err = abs((dH_numeric - dH_theory) / dH_theory)
        errors.append(rel_err)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: dH_num={dH_numeric:.3e}, dH_theory={dH_theory:.3e}, error={rel_err:.1%}")

# Summary
if errors:
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"\nResults (Fix #1 + #3 only):")
    print(f"Mean error: {mean_err:.1%}")
    print(f"Max error: {max_err:.1%}")
    
    if mean_err < 0.05:
        print("✅ PASSED (< 5%)")
    elif mean_err < 0.10:
        print("⚠️  MARGINAL (< 10%)")
    else:
        print("❌ FAILED")
