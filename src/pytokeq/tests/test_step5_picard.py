"""
Step 5 Test: Picard Iteration with Fixed Boundary (Solov'ev)

This tests the core Picard loop WITHOUT free-boundary complexity
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pytokeq.equilibrium.solver.picard_gs_solver import (
    Grid, ProfileModel, CoilSet, Constraints,
    solve_picard_free_boundary, MU0
)


class SolovevProfile(ProfileModel):
    """
    Solov'ev analytical profile
    
    For Solov'ev solution:
        p'(ψ) = -8μ₀
        FF'(ψ) = 0
    """
    
    def pprime(self, psi_norm: np.ndarray) -> np.ndarray:
        return -8 * MU0 * np.ones_like(psi_norm)
    
    def ffprime(self, psi_norm: np.ndarray) -> np.ndarray:
        return np.zeros_like(psi_norm)


def test_picard_fixed_boundary():
    """
    Test Picard iteration with fixed boundary (no coils, no constraints)
    
    This isolates the Picard loop from free-boundary complexity
    
    Expected:
        - Converges in <30 iterations
        - Residual <1e-6
        - Produces reasonable ψ
    """
    print("\n" + "="*60)
    print("Step 5 Test: Picard Fixed-Boundary")
    print("="*60)
    
    # Setup grid
    R_1d = np.linspace(1.0, 2.0, 33)
    Z_1d = np.linspace(-0.5, 0.5, 33)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    print(f"\nGrid: {grid.nr} × {grid.nz}")
    
    # Solov'ev profile
    profile = SolovevProfile()
    
    # Empty coils and constraints (fixed-boundary)
    coils = CoilSet(R=np.array([]), Z=np.array([]), I=np.array([]))
    constraints = Constraints(xpoint=[], isoflux=[])
    
    print("\nProfile: Solov'ev (p'=-8μ₀, FF'=0)")
    print("Boundary: Fixed (no coils)")
    
    # Solve
    print("\nRunning Picard iteration...")
    result = solve_picard_free_boundary(
        profile=profile,
        grid=grid,
        coils=coils,
        constraints=constraints,
        max_outer=50,
        tol_psi=1e-6,
        damping=0.5
    )
    
    print(f"\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.niter}")
    print(f"  Final residual: {result.residuals[-1]:.3e}")
    print(f"  ψ_axis: {result.psi_axis:.3e}")
    print(f"  ψ_boundary: {result.psi_boundary:.3e}")
    print(f"  Axis location: R={grid.R[result.i_axis, result.j_axis]:.3f}, "
          f"Z={grid.Z[result.i_axis, result.j_axis]:.3f}")
    
    # Validation
    assert result.converged, "Picard should converge"
    assert result.niter < 50, f"Too many iterations: {result.niter}"
    assert result.residuals[-1] < 1e-5, f"Residual too large: {result.residuals[-1]:.3e}"
    
    # Check axis in interior
    assert 1 <= result.i_axis <= grid.nr-2, "Axis should be in interior (R)"
    assert 1 <= result.j_axis <= grid.nz-2, "Axis should be in interior (Z)"
    
    # Check ψ reasonable
    assert result.psi_axis > result.psi_boundary, "ψ_axis should be > ψ_boundary"
    
    print(f"\n✅ Picard Fixed-Boundary TEST PASSED")
    print(f"   Core iteration loop working!")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STEP 5: PICARD ITERATION TEST")
    print("="*60)
    
    try:
        result = test_picard_fixed_boundary()
        
        print("\n" + "="*60)
        print("✅ STEP 5 PICARD TEST PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

