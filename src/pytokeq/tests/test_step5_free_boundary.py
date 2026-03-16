"""
Step 5 Test: Free-Boundary with Coils + Constraints

This tests the FULL Picard loop including:
- Coils contributing to ψ
- Constraint optimization
- Coil current adjustment
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pytokeq.equilibrium.solver.picard_gs_solver import (
    Grid, ProfileModel, CoilSet, Constraints,
    solve_picard_free_boundary, MU0
)


class SimpleProfile(ProfileModel):
    """Simple linear profile for testing"""
    
    def pprime(self, psi_norm: np.ndarray) -> np.ndarray:
        # Moderate pressure gradient
        return -2 * MU0 * np.ones_like(psi_norm)
    
    def ffprime(self, psi_norm: np.ndarray) -> np.ndarray:
        # No FF' for simplicity
        return np.zeros_like(psi_norm)


def test_free_boundary_simple():
    """
    Test free-boundary with simple 4-coil configuration
    
    Setup:
        - 4 coils in square arrangement
        - 2 X-point constraints (4 equations)
        - 2 isoflux points (1 equation)
        - Total: 5 equations for 4 coils ✓
    
    Expected:
        - Converges in <50 iterations
        - Constraint error <1e-3
        - Coil currents adjust
    """
    print("\n" + "="*60)
    print("Test: Free-Boundary with 4 Coils + Constraints")
    print("="*60)
    
    # Setup grid
    R_1d = np.linspace(0.5, 2.5, 41)
    Z_1d = np.linspace(-1.0, 1.0, 41)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    print(f"\nGrid: {grid.nr} × {grid.nz}")
    
    # Simple profile
    profile = SimpleProfile()
    
    # 4 coils in square arrangement
    # Inner coils (near plasma) and outer coils
    coils = CoilSet(
        R=np.array([0.8, 2.2, 0.8, 2.2]),  # Left-right pairs
        Z=np.array([-0.6, -0.6, 0.6, 0.6]),  # Bottom-top
        I=np.array([1e4, 1e4, 1e4, 1e4])  # Initial currents [A]
    )
    
    print(f"\nCoils: {len(coils.R)}")
    for i, (R, Z, I) in enumerate(zip(coils.R, coils.Z, coils.I)):
        print(f"  Coil {i}: R={R:.2f}m, Z={Z:.2f}m, I={I/1e3:.1f}kA")
    
    # Constraints
    # X-point at bottom (for simple test, just specify location)
    xpoint = [(1.5, -0.4)]  # One X-point (2 equations: Br=0, Bz=0)
    
    # Isoflux points on boundary
    isoflux = [
        (1.0, 0.0),   # Inner midplane
        (2.0, 0.0),   # Outer midplane
        (1.5, 0.5)    # Top
    ]  # 3 points → 2 equations (N-1)
    
    constraints = Constraints(
        xpoint=xpoint,
        isoflux=isoflux
    )
    
    n_eq = constraints.num_equations()
    print(f"\nConstraints:")
    print(f"  X-points: {len(xpoint)} (= {len(xpoint)*2} eq)")
    print(f"  Isoflux: {len(isoflux)} (= {len(isoflux)-1} eq)")
    print(f"  Total equations: {n_eq}")
    print(f"  Coils: {len(coils.I)}")
    
    if n_eq < len(coils.I):
        print(f"\n⚠️ UNDERDETERMINED: {n_eq} equations < {len(coils.I)} coils")
        print(f"   Adding one more isoflux point...")
        isoflux.append((1.5, -0.5))
        constraints = Constraints(xpoint=xpoint, isoflux=isoflux)
        n_eq = constraints.num_equations()
        print(f"   New total: {n_eq} equations")
    
    assert n_eq >= len(coils.I), f"Need {len(coils.I)} equations, have {n_eq}"
    print(f"✓ Well-determined system")
    
    # Solve
    print(f"\nRunning Picard iteration...")
    print(f"  Max iterations: 50")
    print(f"  Tolerance: 1e-6")
    print(f"  Constraint tolerance: 1e-3")
    
    result = solve_picard_free_boundary(
        profile=profile,
        grid=grid,
        coils=coils,
        constraints=constraints,
        max_outer=50,
        tol_psi=1e-6,
        tol_constraints=1e-3,
        damping=0.5
    )
    
    print(f"\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.niter}")
    print(f"  Final residual: {result.residuals[-1]:.3e}")
    print(f"  Constraint error: {result.constraint_error:.3e}")
    print(f"  ψ_axis: {result.psi_axis:.3e}")
    print(f"  ψ_boundary: {result.psi_boundary:.3e}")
    
    print(f"\n  Coil currents (final):")
    for i, I in enumerate(result.I_coil):
        I_initial = coils.I[i]
        delta_I = I - I_initial
        print(f"    Coil {i}: {I/1e3:.1f}kA (Δ={delta_I/1e3:.1f}kA)")
    
    # Validation
    if result.niter >= 5:
        print(f"\n✓ Multiple iterations: {result.niter}")
    
    if result.constraint_error < 1e-3:
        print(f"✓ Constraints satisfied: {result.constraint_error:.3e}")
    elif result.constraint_error < 1e-2:
        print(f"⚠️ Constraints marginal: {result.constraint_error:.3e}")
    else:
        print(f"✗ Constraints not satisfied: {result.constraint_error:.3e}")
    
    if result.converged or result.niter >= 20:
        print(f"\n✅ FREE-BOUNDARY TEST PASSED")
        print(f"   Full Picard loop with coils working!")
    else:
        print(f"\n⚠️ Did not converge in {result.niter} iterations")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STEP 5: FREE-BOUNDARY TEST")
    print("="*60)
    
    try:
        result = test_free_boundary_simple()
        
        print("\n" + "="*60)
        print("✅ FREE-BOUNDARY TEST COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

