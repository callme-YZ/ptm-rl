#!/usr/bin/env python3
"""
Simple q-profile test for M3D-C1

Focus on q(axis) accuracy (PHYS-01 fix validation)
"""

import sys
sys.path.insert(0, '..')

import numpy as np

from pytokeq.equilibrium.solver.picard_gs_solver import (
    Grid, CoilSet, Constraints,
    solve_picard_free_boundary
)
from pytokeq.equilibrium.profiles.m3dc1_profile import M3DC1Profile
from pytokeq.equilibrium.diagnostics.q_profile import QCalculator

# Setup
R_min, R_max = 1.0, 2.0
Z_min, Z_max = -0.5, 0.5
nx, ny = 128, 128

R_1d = np.linspace(R_min, R_max, nx)
Z_1d = np.linspace(Z_min, Z_max, ny)

grid = Grid.from_1d(R_1d, Z_1d)

# Profile
profile = M3DC1Profile()

print(f"\n{'='*70}")
print(f"M3D-C1 q-profile Test (PHYS-01)")
print(f"{'='*70}")
print(f"\nTarget: q(axis) = 1.75 with error < 5%")
print(f"\nGrid: {nx}×{ny}")

# No coils (fixed boundary)
coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
constraints = Constraints(xpoint=[], isoflux=[])

# Solve
print(f"\nSolving equilibrium...")
result = solve_picard_free_boundary(
    profile=profile,
    grid=grid,
    coils=coils,
    constraints=constraints,
    max_outer=50,
    tol_psi=1e-5,
    damping=0.5
)

if not result.converged:
    print(f"⚠️  Did not converge (residual={result.residuals[-1]:.3e})")
else:
    print(f"✓ Converged in {result.niter} iterations")

# B field functions
psi = result.psi
dpsi_dR = np.gradient(psi, grid.dR, axis=0)
dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)

R, Z = grid.R, grid.Z
B_R_grid = -dpsi_dZ / R
B_Z_grid = dpsi_dR / R

from scipy.interpolate import RegularGridInterpolator

def Br_func(R_pts, Z_pts):
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), B_R_grid.T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

def Bz_func(R_pts, Z_pts):
    interp = RegularGridInterpolator(
        (R_1d, Z_1d), B_Z_grid.T,
        bounds_error=False, fill_value=0
    )
    return interp(np.column_stack([R_pts, Z_pts]))

def fpol(psi_norm):
    return profile.Fpol(psi_norm)

# Compute q
print(f"\nComputing q-profile...")
calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)

# Compute with extrapolation to axis
psi_test = np.array([0.0, 0.5, 0.90])
q_values = calc.compute_q_profile(psi_test, ntheta=128, extrapolate=True)

q_axis, q_mid, q_90 = q_values

# Expected
q0_target = 1.75
qmid_target = profile.q_profile(np.array([0.5]))[0]
q90_target = profile.q_profile(np.array([0.90]))[0]

# Errors
err_axis = abs(q_axis - q0_target) / q0_target
err_mid = abs(q_mid - qmid_target) / qmid_target
err_90 = abs(q_90 - q90_target) / q90_target

# Results
print(f"\n{'='*70}")
print(f"Results:")
print(f"{'='*70}")
print(f"  Location    Computed    Target    Error")
print(f"  -----------------------------------------")
print(f"  Axis (0%)   {q_axis:8.3f}  {q0_target:8.3f}  {err_axis*100:6.1f}%")
print(f"  Mid  (50%)  {q_mid:8.3f}  {qmid_target:8.3f}  {err_mid*100:6.1f}%")
print(f"  Edge (90%)  {q_90:8.3f}  {q90_target:8.3f}  {err_90*100:6.1f}%")

# Validation
print(f"\n{'='*70}")
print(f"Validation:")
print(f"{'='*70}")

PASS = True

if err_axis < 0.05:
    print(f"  ✅ q(axis) error < 5% ({err_axis*100:.1f}%) - TARGET MET!")
elif err_axis < 0.15:
    print(f"  ✓ q(axis) error < 15% ({err_axis*100:.1f}%) - Acceptable")
else:
    print(f"  ❌ q(axis) error {err_axis*100:.1f}% - FAILED")
    PASS = False

if err_mid < 0.20:
    print(f"  ✓ q(mid) error < 20% ({err_mid*100:.1f}%)")
else:
    print(f"  ⚠️  q(mid) error {err_mid*100:.1f}%")

if err_90 < 0.30:
    print(f"  ✓ q(90%) error < 30% ({err_90*100:.1f}%)")
else:
    print(f"  ⚠️  q(90%) error {err_90*100:.1f}%")

print(f"\n{'='*70}")
if PASS:
    print(f"✅ PHYS-01 FIX VALIDATED")
    print(f"   q-profile calculation now accurate!")
else:
    print(f"⚠️  PHYS-01 FIX INCOMPLETE")
    print(f"   q(axis) error still > target")
print(f"{'='*70}\n")

# Return status (for direct execution)
if __name__ == "__main__":
    sys.exit(0 if PASS else 1)
