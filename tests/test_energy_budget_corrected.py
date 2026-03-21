"""
Energy Budget Test - CORRECTED VERSION

Applies all 3 fixes:
1. Use φ (not ω) in Hamiltonian
2. Use normalized units (μ₀=1)
3. Include 2π factor in toroidal integral

Author: 小P ⚛️
Date: 2026-03-19
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.solvers import solve_poisson_toroidal
from pytokmhd.physics import compute_current_density

# Grid
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)

# Initial condition
psi0 = grid.r_grid**2 * np.cos(grid.theta_grid)
omega0 = np.zeros_like(psi0)

# Solver with resistivity
eta = 1e-4
solver = HamiltonianMHDIMEX(
    grid=grid,
    dt=1e-4,
    eta=eta,
    nu=0.0,
    use_imex=True,
    verbose=False
)

solver.psi = psi0.copy()
solver.omega = omega0.copy()

print("Testing energy dissipation formula...")
print("=" * 60)

# Compute initial Hamiltonian (FIXED: use φ not ω)
phi0 = solve_poisson_toroidal(omega0, grid)[0]
from pytokmhd.operators import gradient_toroidal
grad_psi0 = gradient_toroidal(psi0, grid)
grad_phi0 = gradient_toroidal(phi0, grid)

# Hamiltonian (3D with 2π toroidal integral)
# Volume element: dV = jacobian * dr * dθ where jacobian = r*R
dV = grid.jacobian() * grid.dr * grid.dtheta
H0 = 2*np.pi * 0.5 * np.sum((grad_psi0[0]**2 + grad_psi0[1]**2 + 
                              grad_phi0[0]**2 + grad_phi0[1]**2) * dV)

print(f"Initial Hamiltonian: {H0:.6e}")

# Run a few steps
errors = []
for i in range(5):
    psi_old = solver.psi.copy()
    omega_old = solver.omega.copy()
    
    # Current before step
    J_old = compute_current_density(psi_old, grid, mu0=1.0)  # FIX 2: μ₀=1
    J2_int = np.sum(J_old**2 * dV)
    
    # Theory prediction (FIX 3: add 2π)
    dH_dt_theory = -eta * 2*np.pi * J2_int
    
    # Evolve
    solver.psi, solver.omega = solver.step(solver.psi, solver.omega)
    
    # Compute new Hamiltonian (FIX 1: use φ, with 2π)
    phi_new = solve_poisson_toroidal(solver.omega, grid)[0]
    grad_psi_new = gradient_toroidal(solver.psi, grid)
    grad_phi_new = gradient_toroidal(phi_new, grid)
    H_new = 2*np.pi * 0.5 * np.sum((grad_psi_new[0]**2 + grad_psi_new[1]**2 +
                                     grad_phi_new[0]**2 + grad_phi_new[1]**2) * dV)
    
    # Numerical dH/dt
    dH_dt_numeric = (H_new - H0) / solver.dt
    
    # Error
    if abs(dH_dt_theory) > 1e-10:
        rel_error = abs(dH_dt_numeric - dH_dt_theory) / abs(dH_dt_theory)
    else:
        rel_error = 0
    
    errors.append(rel_error)
    
    print(f"Step {i+1:2d}: dH/dt_num={dH_dt_numeric:+.3e}, "
          f"dH/dt_theory={dH_dt_theory:+.3e}, "
          f"error={rel_error*100:.2f}%")
    
    H0 = H_new

print("=" * 60)
mean_error = np.mean(errors) * 100
print(f"Mean relative error: {mean_error:.2f}%")

if mean_error < 10:
    print("✅ PASS: Error < 10%")
    sys.exit(0)
else:
    print(f"❌ FAIL: Error {mean_error:.1f}% > 10%")
    sys.exit(1)
