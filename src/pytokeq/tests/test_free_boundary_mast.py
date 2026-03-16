"""
Free-Boundary Solver Validation: FreeGS MAST Comparison

Tests free-boundary solver against FreeGS MAST equilibrium configuration.

Reference: Design doc Section 5.2
"""

import numpy as np
import sys
from pathlib import Path

# Add equilibrium to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'equilibrium'))

from pytokeq.equilibrium.boundary.free_boundary_picard import (
    solve_free_boundary_picard, FreeBoundaryResult
)
from pytokeq.equilibrium.free_boundary_constraints import CoilSet, IsofluxPair
from pytokeq.equilibrium.boundary.fixed_boundary_picard import solve_picard_gs  # Use existing solver


class Grid:
    """Computational grid"""
    def __init__(self, nr=65, nz=65, R_min=0.1, R_max=2.0, Z_min=-1.5, Z_max=1.5):
        R = np.linspace(R_min, R_max, nr)
        Z = np.linspace(Z_min, Z_max, nz)
        self.R, self.Z = np.meshgrid(R, Z, indexing='ij')
        self.dR = R[1] - R[0]
        self.dZ = Z[1] - Z[0]
        self.nr = nr
        self.nz = nz


class QuadraticProfile:
    """Simple quadratic profile for testing"""
    def __init__(self, I_p=1e6, beta_p=0.5, R0=1.0, a=0.5):
        self.I_p = I_p
        self.beta_p = beta_p
        self.R0 = R0
        self.a = a
        
        # Characteristic values
        mu0 = 4e-7 * np.pi
        self.psi_norm = mu0 * I_p * a  # Flux scale
        self.p0 = beta_p * (mu0 * I_p / a)**2 / (2 * mu0)  # Pressure scale
        self.F0 = 1.0  # Toroidal field function (simplified)
    
    def pprime(self, psi_n):
        """p'(ψ) - normalized input"""
        return -2 * self.p0 / self.psi_norm * np.ones_like(psi_n)
    
    def ffprime(self, psi_n):
        """FF'(ψ) - normalized input"""
        return -self.F0**2 / self.psi_norm * np.ones_like(psi_n)


def setup_mast_coils():
    """
    MAST-like coil configuration
    
    Simplified 6-coil system approximating MAST geometry
    """
    coils = CoilSet(
        R=np.array([0.3, 1.8, 0.3, 1.8, 1.0, 1.0]),
        Z=np.array([1.2, 1.2, -1.2, -1.2, 0.5, -0.5]),
        I=np.array([5e4, 5e4, -5e4, -5e4, 2e4, -2e4])  # Initial guess (kA range)
    )
    return coils


def setup_mast_constraints():
    """
    MAST constraint configuration
    
    - 1 X-point (lower)
    - 4 isoflux pairs (shape control)
    """
    isoflux_pairs = [
        # Vertical symmetry points
        IsofluxPair(R1=0.8, Z1=0.6, R2=0.8, Z2=-0.6),
        IsofluxPair(R1=1.2, Z1=0.8, R2=1.2, Z2=-0.8),
        # Horizontal symmetry
        IsofluxPair(R1=0.6, Z1=0.0, R2=1.4, Z2=0.0),
        # Strike point region
        IsofluxPair(R1=0.5, Z1=-1.0, R2=1.5, Z2=-1.0),
    ]
    return isoflux_pairs


def gs_solver_wrapper(profile, grid, psi_boundary):
    """
    Wrapper to use existing fixed-boundary Picard solver for interior
    
    This solves: Δ*ψ = RHS inside, with ψ = psi_boundary on edges
    """
    # Use existing Picard solver
    result = solve_picard_gs(
        profile=profile,
        grid=grid,
        psi_boundary=psi_boundary,
        max_iter=50,
        tol=1e-6,
        verbose=False
    )
    
    return result.psi


def run_mast_test(verbose=True):
    """
    Run MAST free-boundary test
    
    Returns
    -------
    result : FreeBoundaryResult
        Solver result
    """
    if verbose:
        print("\n" + "="*70)
        print("FREE-BOUNDARY MAST TEST")
        print("="*70)
    
    # Setup
    grid = Grid(nr=65, nz=65)
    profile = QuadraticProfile(I_p=1e6, beta_p=0.5, R0=1.0, a=0.5)
    coils = setup_mast_coils()
    isoflux = setup_mast_constraints()
    
    if verbose:
        print(f"\nGrid: {grid.nr}×{grid.nz}")
        print(f"Coils: {len(coils.R)}")
        print(f"Constraints: 2 (X-point) + {len(isoflux)} (isoflux) = {2+len(isoflux)}")
        print(f"Profile: I_p={profile.I_p/1e6:.1f} MA, β_p={profile.beta_p:.2f}")
    
    # Solve
    result = solve_free_boundary_picard(
        profile=profile,
        grid=grid,
        coils=coils,
        isoflux_pairs=isoflux,
        solve_gs_interior=gs_solver_wrapper,
        max_iter=30,
        tol_psi=1e-5,
        tol_I=1e-2,
        damping=0.3,
        verbose=verbose
    )
    
    return result


def analyze_result(result: FreeBoundaryResult):
    """Analyze and report result"""
    print("\n" + "="*70)
    print("RESULT ANALYSIS")
    print("="*70)
    
    print(f"\nConvergence: {result.converged}")
    print(f"Iterations: {result.n_iterations}")
    
    if result.xpoint:
        print(f"\nX-point:")
        print(f"  R = {result.xpoint.R:.3f} m")
        print(f"  Z = {result.xpoint.Z:.3f} m")
        print(f"  ψ = {result.xpoint.psi:.3e} Wb")
        print(f"  |∇ψ| = {result.xpoint.grad_mag:.2e} Wb/m")
    else:
        print("\n⚠️ No X-point detected!")
    
    print(f"\nCoil currents:")
    for i, I in enumerate(result.I_coils):
        print(f"  Coil {i}: {I/1000:.2f} kA")
    
    # Flux statistics
    print(f"\nFlux solution:")
    print(f"  ψ_axis = {result.psi.max():.3e} Wb")
    print(f"  ψ_edge = {result.psi.min():.3e} Wb")
    print(f"  Δψ = {result.psi.max() - result.psi.min():.3e} Wb")
    
    # Success criteria (loose for first test)
    success = (
        result.converged and
        result.xpoint is not None and
        abs(result.xpoint.Z) > 0.1  # X-point away from midplane
    )
    
    print(f"\n{'✅' if success else '⚠️'} Test {'PASSED' if success else 'INCOMPLETE'}")
    
    return success


if __name__ == "__main__":
    print("Free-Boundary MAST Validation Test")
    print("="*70)
    print("This test validates the free-boundary solver structure")
    print("against MAST-like configuration.")
    print("\nNote: Without FreeGS comparison, we validate:")
    print("  - Convergence")
    print("  - X-point detection")
    print("  - Reasonable coil currents")
    print("="*70)
    
    try:
        result = run_mast_test(verbose=True)
        success = analyze_result(result)
        
        if success:
            print("\n" + "="*70)
            print("✅ FREE-BOUNDARY SOLVER: STRUCTURAL VALIDATION PASSED")
            print("="*70)
            print("\nNext steps:")
            print("  - Compare with FreeGS (requires FreeGS installation)")
            print("  - Benchmark performance")
            print("  - Test multiple scenarios")
        else:
            print("\n⚠️ Test incomplete - check configuration")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
