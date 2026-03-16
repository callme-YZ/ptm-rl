"""
Step 6: M3D-C1 Benchmark Equilibrium Test

Test Picard solver with M3D-C1 profile on 256×256 grid
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import warnings
from pytokeq.equilibrium.solver.picard_gs_solver import (
    Grid, CoilSet, Constraints,
    solve_picard_free_boundary
)
from pytokeq.equilibrium.profiles.m3dc1_profile import M3DC1Profile


def test_m3dc1_equilibrium():
    """
    Generate M3D-C1 benchmark equilibrium
    
    Parameters:
      - Grid: 256×256 (high resolution for tearing validation)
      - Profile: M3D-C1 (q₀=1.75, q_edge=2.5)
      - Boundary: Fixed (no coils for initial test)
    
    Success criteria:
      - Converges (residual < 1e-6)
      - Axis in interior
      - Force balance reasonable
    """
    print("\n" + "="*70)
    print("Step 6: M3D-C1 Equilibrium Generation")
    print("="*70)
    
    # Setup grid (256×256 from Phase 1)
    R_min, R_max = 1.0, 2.0  # [m]
    Z_min, Z_max = -0.5, 0.5  # [m]
    
    R_1d = np.linspace(R_min, R_max, 256)
    Z_1d = np.linspace(Z_min, Z_max, 256)
    
    grid = Grid.from_1d(R_1d, Z_1d)
    
    print(f"\nGrid Setup:")
    print(f"  Size: {grid.nr}×{grid.nz}")
    print(f"  R: [{R_min}, {R_max}] m")
    print(f"  Z: [{Z_min}, {Z_max}] m")
    print(f"  ΔR = {grid.dR:.6f} m")
    print(f"  ΔZ = {grid.dZ:.6f} m")
    
    # Check M3D-C1 grid resolution requirement
    delta_tearing = 0.025  # M3D-C1 tearing layer width [m]
    resolution_ratio = grid.dR / delta_tearing
    
    if resolution_ratio > 0.2:
        warnings.warn(
            f"\n⚠️ Grid too coarse for M3D-C1!\n"
            f"  Δr/δ = {resolution_ratio:.2f} > 0.2 (required)\n"
            f"  Current: {grid.nr}×{grid.nz} → ΔR = {grid.dR:.4f} m\n"
            f"  Minimum: 128×128 → ΔR < 0.005 m\n"
            f"  Recommended: 256×256 → ΔR < 0.004 m\n"
            f"  This may cause numerical instability or divergence!"
        )
    else:
        print(f"  Δr/δ = {resolution_ratio:.3f} ✓ (< 0.2, suitable for M3D-C1)")
    
    # Tearing layer width estimate
    delta_tearing = 0.025  # [m] rough estimate
    points_per_layer = delta_tearing / grid.dR
    print(f"  Points per tearing layer: {points_per_layer:.1f}")
    
    if grid.dR > 0.2 * delta_tearing:
        print(f"  ⚠️  Resolution marginal (Δr/δ = {grid.dR/delta_tearing:.2f})")
    else:
        print(f"  ✓  Resolution adequate (Δr/δ = {grid.dR/delta_tearing:.2f})")
    
    # Profile
    profile = M3DC1Profile()
    print(f"\n{profile}")
    
    # No coils (fixed-boundary)
    coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
    constraints = Constraints(xpoint=[], isoflux=[])
    
    print(f"\nBoundary Conditions:")
    print(f"  Type: Fixed (ψ=0 at computational boundary)")
    print(f"  Coils: None")
    print(f"  Constraints: None")
    
    # Solve
    print(f"\nRunning Picard Solver...")
    print(f"  Max iterations: 100")
    print(f"  Tolerance: 1e-6")
    print(f"  Damping: 0.5")
    
    result = solve_picard_free_boundary(
        profile=profile,
        grid=grid,
        coils=coils,
        constraints=constraints,
        max_outer=100,
        tol_psi=1e-6,
        damping=0.5
    )
    
    # Results
    print(f"\n" + "="*70)
    print("Convergence Results:")
    print("="*70)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.niter}")
    print(f"  Final residual: {result.residuals[-1]:.3e}")
    
    if len(result.residuals) > 0:
        print(f"\n  Residual history (first 10):")
        for i in range(min(10, len(result.residuals))):
            print(f"    {i}: {result.residuals[i]:.3e}")
    
    print(f"\n  Magnetic Axis:")
    print(f"    R = {result.R_axis:.3f} m")
    print(f"    Z = {result.Z_axis:.3f} m")
    print(f"    ψ_axis = {result.psi_axis:.3e} Wb")
    
    print(f"\n  Boundary:")
    print(f"    ψ_boundary = {result.psi_boundary:.3e} Wb")
    
    # Validation
    print(f"\n" + "="*70)
    print("Validation:")
    print("="*70)
    
    # Check 1: Convergence
    if result.residuals[-1] < 1e-6:
        print(f"  ✓ Converged (residual < 1e-6)")
    elif result.residuals[-1] < 1e-5:
        print(f"  ✓ Acceptable (residual < 1e-5)")
    else:
        print(f"  ⚠️  Poor convergence (residual = {result.residuals[-1]:.3e})")
    
    # Check 2: Axis location
    R_center = (R_min + R_max) / 2
    Z_center = (Z_min + Z_max) / 2
    axis_offset = np.sqrt((result.R_axis - R_center)**2 + (result.Z_axis - Z_center)**2)
    
    if axis_offset < 0.1:
        print(f"  ✓ Axis near center (offset = {axis_offset:.3f} m)")
    else:
        print(f"  ⚠️  Axis displaced (offset = {axis_offset:.3f} m)")
    
    # Check 3: Flux direction
    if result.psi_axis > result.psi_boundary:
        print(f"  ✓ Correct flux direction (ψ_axis > ψ_boundary)")
    else:
        print(f"  ⚠️  Wrong flux direction!")
    
    # Check 4: Current density sign
    from pytokeq.equilibrium.solver.picard_gs_solver import compute_current_density
    Jtor_final = compute_current_density(result.psi, grid, profile, result.psi_axis)
    J_axis = Jtor_final[result.i_axis, result.j_axis]
    
    print(f"  Current at axis: J_φ = {J_axis:.3e} A/m²")
    if J_axis > 0:
        print(f"  ✓ J > 0 (normal tokamak current)")
    else:
        print(f"  ❌ J < 0 (REVERSED - unstable!)")
    
    # Check 5: Monotonic convergence
    if len(result.residuals) > 5:
        ratio = result.residuals[-1] / result.residuals[0]
        print(f"  Convergence rate: {ratio:.2e}")
        
        if ratio < 0.01:
            print(f"  ✓ Excellent convergence (>100× reduction)")
        elif ratio < 0.1:
            print(f"  ✓ Good convergence (>10× reduction)")
        else:
            print(f"  ⚠️  Slow convergence")
    
    # Summary
    print(f"\n" + "="*70)
    if (result.residuals[-1] < 1e-5 and 
        axis_offset < 0.1 and 
        result.psi_axis > result.psi_boundary):
        print("✅ STEP 6.3 PASSED: Picard solver working")
    else:
        print("⚠️  STEP 6.3 INCOMPLETE: Check issues above")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = test_m3dc1_equilibrium()
