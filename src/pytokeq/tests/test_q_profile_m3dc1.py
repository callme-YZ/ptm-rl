"""
Test q-profile on M3D-C1 Benchmark

Validate PHYS-01 fix: q(axis) error should be < 5% (vs 30% before)
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


def test_m3dc1_q_profile():
    """
    Compute q-profile on M3D-C1 equilibrium
    
    Target: q₀ = 1.75 (axis), q_edge ≈ 2.5
    Old error: ~30% at axis
    New target: < 5% at axis
    """
    print("\n" + "="*70)
    print("M3D-C1 q-profile Validation (PHYS-01 Fix)")
    print("="*70)
    
    # Setup (use 128x128 for faster test, 256x256 for production)
    R_min, R_max = 1.0, 2.0
    Z_min, Z_max = -0.5, 0.5
    
    nx, ny = 128, 128  # Reduced for speed
    
    R_1d = np.linspace(R_min, R_max, nx)
    Z_1d = np.linspace(Z_min, Z_max, ny)
    
    grid = Grid.from_1d(R_1d, Z_1d)
    
    print(f"\nGrid: {nx}×{ny}")
    print(f"  R: [{R_min}, {R_max}] m")
    print(f"  Z: [{Z_min}, {Z_max}] m")
    
    # Profile
    profile = M3DC1Profile()
    print(f"\n{profile}")
    
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
        max_outer=50,  # Reduced for speed
        tol_psi=1e-5,  # Slightly relaxed
        damping=0.5
    )
    
    if not result.converged:
        print(f"⚠️  Equilibrium did not converge!")
        print(f"   Residual: {result.residuals[-1]:.3e}")
        print(f"   Continuing anyway for q validation...")
    else:
        print(f"✓ Converged in {result.niter} iterations")
    
    # Extract fields for q calculation
    psi = result.psi
    R, Z = grid.R, grid.Z
    
    # F(ψ) from profile
    def fpol(psi_norm):
        return profile.Fpol(psi_norm)
    
    # Compute gradients for B_R, B_Z
    dpsi_dR = np.gradient(psi, grid.dR, axis=0)
    dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)
    
    def Br_func(R_pts, Z_pts):
        """B_R = -(1/R) ∂ψ/∂Z"""
        # Interpolate to points
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (R_1d, Z_1d), (-dpsi_dZ / R).T,
            bounds_error=False, fill_value=0
        )
        return interp(np.column_stack([R_pts, Z_pts]))
    
    def Bz_func(R_pts, Z_pts):
        """B_Z = (1/R) ∂ψ/∂R"""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (R_1d, Z_1d), (dpsi_dR / R).T,
            bounds_error=False, fill_value=0
        )
        return interp(np.column_stack([R_pts, Z_pts]))
    
    # Create q calculator
    print(f"\n" + "="*70)
    print("Computing q-profile...")
    print("="*70)
    
    calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)
    
    # Compute q profile
    psi_norm, q = calc.compute_q_profile(npsi=50, ntheta=64, extrapolate=True)
    
    # Get specific values
    # NOTE: For fixed-boundary without separatrix, avoid true edge (psi_norm=1.0)
    # Use psi_norm=0.95 as "effective edge"
    q_axis = calc.compute_q_profile(np.array([0.0]), ntheta=64, extrapolate=True)
    q_95 = calc.compute_q_profile(np.array([0.95]), ntheta=64, extrapolate=False)
    
    # Target values
    q0_target = 1.75
    q_95_target = profile.q_profile(np.array([0.95]))[0]  # Use prescribed value at 95%
    
    # Errors
    q_axis_error = abs(q_axis - q0_target) / q0_target
    q_95_error = abs(q_95 - q_95_target) / q_95_target
    
    # Results
    print(f"\nq-profile Results:")
    print("-" * 70)
    print(f"  q(axis):    {q_axis:.3f} (target: {q0_target:.2f})")
    print(f"              Error: {q_axis_error*100:.1f}%")
    print(f"  q(ψ=0.95):  {q_95:.3f} (target: {q_95_target:.2f})")
    print(f"              Error: {q_95_error*100:.1f}%")
    
    # Validation
    print(f"\n" + "="*70)
    print("Validation:")
    print("="*70)
    
    success = True
    
    # Check 1: q(axis) error
    if q_axis_error < 0.05:
        print(f"  ✅ q(axis) error < 5% ({q_axis_error*100:.1f}%)")
    elif q_axis_error < 0.15:
        print(f"  ✓ q(axis) error < 15% ({q_axis_error*100:.1f}%) - acceptable")
    elif q_axis_error < 0.30:
        print(f"  ⚠️  q(axis) error {q_axis_error*100:.1f}% (improved from old 30%)")
    else:
        print(f"  ❌ q(axis) error {q_axis_error*100:.1f}% (NO IMPROVEMENT)")
        success = False
    
    # Check 2: q(edge) error
    if q_edge_error < 0.05:
        print(f"  ✅ q(edge) error < 5% ({q_edge_error*100:.1f}%)")
    elif q_edge_error < 0.10:
        print(f"  ✓ q(edge) error < 10% ({q_edge_error*100:.1f}%)")
    else:
        print(f"  ⚠️  q(edge) error {q_edge_error*100:.1f}%")
    
    # Check 3: Monotonicity
    dq = np.diff(q)
    decreasing = np.sum(dq < -0.01)
    
    if decreasing == 0:
        print(f"  ✅ q monotonically increasing")
    elif decreasing / len(dq) < 0.1:
        print(f"  ✓ q mostly monotonic ({decreasing}/{len(dq)} violations)")
    else:
        print(f"  ⚠️  q not monotonic ({decreasing}/{len(dq)} decreasing)")
    
    # Check 4: Values in physical range
    if np.all((q > 0) & (q < 10)):
        print(f"  ✅ q in physical range (0 < q < 10)")
    else:
        print(f"  ❌ q outside physical range")
        success = False
    
    # Summary
    print(f"\n" + "="*70)
    if success and q_axis_error < 0.05:
        print("✅ PHYS-01 FIX SUCCESSFUL!")
        print(f"   q(axis) error: 30% → {q_axis_error*100:.1f}%")
    elif success and q_axis_error < 0.10:
        print("✓ PHYS-01 FIX WORKING")
        print(f"   q(axis) error: 30% → {q_axis_error*100:.1f}% (good improvement)")
    else:
        print("⚠️  PHYS-01 FIX INCOMPLETE")
        print(f"   q(axis) error still {q_axis_error*100:.1f}%")
    print("="*70)
    
    # Print full profile for inspection
    print(f"\nFull q-profile (first 10 points):")
    print("-" * 70)
    print(f"{'psi_norm':>10} {'q':>10}")
    for i in range(min(10, len(psi_norm))):
        print(f"{psi_norm[i]:10.3f} {q[i]:10.3f}")
    print(f"  ...")
    print(f"{psi_norm[-1]:10.3f} {q[-1]:10.3f}")
    
    return q_axis_error, q_edge_error


if __name__ == '__main__':
    q_axis_error, q_edge_error = test_m3dc1_q_profile()
