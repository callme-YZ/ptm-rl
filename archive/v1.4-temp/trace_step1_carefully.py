"""Carefully trace Step 1"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.solvers import solve_poisson_toroidal
from pytokmhd.physics import compute_hamiltonian

grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)

psi0 = grid.r_grid**2 * np.cos(grid.theta_grid)
omega0 = np.zeros_like(psi0)

print("Initial condition:")
print(f"psi: min={psi0.min():.6e}, max={psi0.max():.6e}")
print(f"omega: min={omega0.min():.6e}, max={omega0.max():.6e}")

phi0 = solve_poisson_toroidal(omega0, grid)[0]
print(f"phi0: min={phi0.min():.6e}, max={phi0.max():.6e}")

H0 = compute_hamiltonian(psi0, phi0, grid)
print(f"H0 = {H0:.6e}")

# Create solver
eta = 1e-4
solver = HamiltonianMHDIMEX(grid=grid, dt=1e-4, eta=eta, nu=0.0, use_imex=True)
solver.psi = psi0.copy()
solver.omega = omega0.copy()

print("\n=== Evolving one step ===")

# Manually step through to see what happens
from pytokmhd.operators import poisson_bracket

# Step 1: Compute φ^n
phi_n = solve_poisson_toroidal(omega0, grid)[0]
print(f"\nphi_n: min={phi_n.min():.6e}, max={phi_n.max():.6e}")
print(f"  (Should be ~0 since omega=0)")

# Step 2: Explicit half-step
psi_phi_bracket = poisson_bracket(psi0, phi_n, grid)
print(f"\n{psi,phi} bracket: min={psi_phi_bracket.min():.6e}, max={psi_phi_bracket.max():.6e}")
print(f"  (Should be ~0 since phi=0)")

psi_star = psi0 + 0.5 * solver.dt * psi_phi_bracket
print(f"\npsi_star: min={psi_star.min():.6e}, max={psi_star.max():.6e}")
print(f"  |psi_star - psi0|: {np.abs(psi_star - psi0).max():.6e}")
print(f"  (Should be ~0 since bracket=0)")

# Step 3: Implicit resistive
from pytokmhd.solvers.implicit_resistive import solve_implicit_resistive
psi_half, niter = solve_implicit_resistive(
    psi_star, 0.5*solver.dt, eta, grid, psi_boundary=None, verbose=False
)

print(f"\npsi_half (after implicit): min={psi_half.min():.6e}, max={psi_half.max():.6e}")
print(f"  |psi_half - psi_star|: {np.abs(psi_half - psi_star).max():.6e}")
print(f"  Expected ~ dt*eta*J ~ 1e-8")

# Step 4: Enforce BC
# (done inside solve_implicit_resistive)

# Continue with ω step...
phi_half = solve_poisson_toroidal(omega0, grid)[0]  # Still ~0
omega_phi_bracket = poisson_bracket(omega0, phi_half, grid)
omega_new = omega0 + solver.dt * omega_phi_bracket

print(f"\nomega_new: min={omega_new.min():.6e}, max={omega_new.max():.6e}")
print(f"  (Should still be ~0)")

# Second half-step for psi
phi_new = solve_poisson_toroidal(omega_new, grid)[0]
psi_phi_bracket_new = poisson_bracket(psi_half, phi_new, grid)
psi_star_star = psi_half + 0.5 * solver.dt * psi_phi_bracket_new

psi_new, niter2 = solve_implicit_resistive(
    psi_star_star, 0.5*solver.dt, eta, grid, psi_boundary=None, verbose=False
)

psi_new[0, :] = np.mean(psi_new[0, :])
psi_new[-1, :] = 0.0

print(f"\npsi_new (final): min={psi_new.min():.6e}, max={psi_new.max():.6e}")
print(f"  |psi_new - psi0|: {np.abs(psi_new - psi0).max():.6e}")

# Compute H_new
phi_final = solve_poisson_toroidal(omega_new, grid)[0]
H_new = compute_hamiltonian(psi_new, phi_final, grid)

print(f"\nH_new = {H_new:.6e}")
print(f"ΔH = {H_new - H0:.6e}")
print(f"ΔH/H0 = {(H_new - H0)/H0*100:.1f}%")
