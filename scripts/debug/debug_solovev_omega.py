import numpy as np
from src.pytokmhd.solver.initial_conditions import solovev_equilibrium

r = np.linspace(0.1, 2.0, 64)
z = np.linspace(-2.0, 2.0, 128)

psi_eq, omega_eq = solovev_equilibrium(r, z)

print(f"Solovev equilibrium:")
print(f"  ψ_eq: min={np.min(psi_eq):.6e}, max={np.max(psi_eq):.6e}")
print(f"  ω_eq: min={np.min(omega_eq):.6e}, max={np.max(omega_eq):.6e}")
print(f"  ω_eq ratio: max/min = {np.max(np.abs(omega_eq))/np.min(np.abs(omega_eq[omega_eq != 0])):.2e}")

# Check typical reduced MHD equilibrium: ω_eq should be ~ O(1) or smaller
if np.max(np.abs(omega_eq)) > 10:
    print(f"\n⚠️  WARNING: |ω_eq|_max = {np.max(np.abs(omega_eq)):.2e} >> 1")
    print("  This is likely too large for numerical stability!")
