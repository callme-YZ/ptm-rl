"""
Test Solov'ev with EXACT q-profile

This replaces the cylindrical approximation with flux surface integration
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pytokeq.equilibrium.solver.picard_gs_solver import Grid, find_psi_axis
from pytokeq.equilibrium.diagnostics.q_profile_exact import compute_q_profile


def test_solovev_exact_q():
    """
    Test exact q-profile on Solov'ev analytical solution
    
    Expected:
        - q monotonically increasing (within numerical tolerance)
        - q_axis < q_edge
        - Reasonable range (q ~ 0.5-1.5)
    """
    print("\n" + "="*60)
    print("Test: Solov'ev with Exact q-profile")
    print("="*60)
    
    # Setup grid
    R_1d = np.linspace(0.5, 2.5, 65)
    Z_1d = np.linspace(-1.0, 1.0, 65)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    R0 = 1.5  # Major radius
    
    # Analytical Solov'ev solution
    psi_analytical = -((grid.R**2 - R0**2)**2 / 8.0 + 1.0 * grid.Z**2)
    
    print(f"\nGrid: {grid.nr} × {grid.nz}")
    print(f"R0 = {R0} m")
    
    # Find axis
    i_ax, j_ax, psi_ax = find_psi_axis(psi_analytical, grid)
    R_ax = grid.R[i_ax, j_ax]
    Z_ax = grid.Z[i_ax, j_ax]
    
    print(f"\nAxis: R={R_ax:.3f}, Z={Z_ax:.3f}")
    assert abs(R_ax - R0) < 0.05, f"Axis R={R_ax:.3f}, expected {R0}"
    assert abs(Z_ax - 0.0) < 0.05, f"Axis Z={Z_ax:.3f}, expected 0.0"
    print("✓ Axis location correct")
    
    # Compute exact q-profile
    f = R0 * 1.0  # B_phi ≈ constant
    psi_norm, q_exact = compute_q_profile(
        psi_analytical, grid.R, grid.Z, f, npsi=30
    )
    
    print(f"\nExact q-profile:")
    print(f"  q_axis = {q_exact[0]:.3f}")
    print(f"  q_edge = {q_exact[-1]:.3f}")
    
    # Check monotonicity (with tolerance for numerical noise)
    dq = np.diff(q_exact)
    min_dq = dq.min()
    relative_decrease = abs(min_dq) / q_exact.mean()
    
    print(f"  Min Δq = {min_dq:.4f}")
    print(f"  Relative: {relative_decrease:.2%}")
    
    # Allow small numerical decrease (<1%)
    if min_dq >= 0:
        print("✓ Strictly monotonic")
    elif relative_decrease < 0.01:
        print("✓ Monotonic (within numerical tolerance <1%)")
    else:
        print(f"✗ Non-monotonic decrease {relative_decrease:.2%} > 1%")
        assert False, "q not monotonic"
    
    # Check q_edge > q_axis
    assert q_exact[-1] > q_exact[0], f"q_edge={q_exact[-1]:.3f} not > q_axis={q_exact[0]:.3f}"
    print(f"✓ q_edge > q_axis (correct physics)")
    
    # Check reasonable range
    assert 0.3 < q_exact[0] < 2.0, f"q_axis={q_exact[0]:.3f} out of range"
    assert 0.5 < q_exact[-1] < 3.0, f"q_edge={q_exact[-1]:.3f} out of range"
    print(f"✓ q values in reasonable range")
    
    print("\n✅ Test PASSED: Exact q-profile is physically correct!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SOLOV'EV EXACT Q-PROFILE TEST")
    print("="*60)
    
    try:
        test_solovev_exact_q()
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

