"""
Free Boundary Condition using Green's Function

Implements self-consistent boundary condition for free-boundary G-S solver:
    ψ_boundary = ∫∫ G(boundary; R',Z') · Jtor(R',Z') dR' dZ'

Adapted from FreeGS boundary.py (LGPL license)
"""

import numpy as np
from scipy.integrate import romb
from greens_function import greens_psi


def free_boundary_greens(R, Z, Jtor, psi):
    """
    Apply free boundary condition using Green's function integration
    
    Computes boundary ψ from plasma current distribution via:
        ψ(R_b, Z_b) = ∫∫_domain G(R_b,Z_b; R',Z') · Jtor(R',Z') dR' dZ'
    
    This is the CORRECT free-boundary formulation:
        - Boundary ψ includes plasma contribution
        - Self-consistent with interior solution
        - Converges to physical equilibrium
        
    Parameters
    ----------
    R : ndarray (nx, ny)
        Major radius grid
    Z : ndarray (nx, ny)
        Vertical coordinate grid
    Jtor : ndarray (nx, ny)
        Toroidal current density [A/m²]
    psi : ndarray (nx, ny)
        Poloidal flux array (modified in-place at boundary)
        
    Returns
    -------
    None
        Modifies psi[boundary] in-place
        
    Notes
    -----
    Complexity: O(N_boundary × N_domain)
        For 65×65 grid: ~260 boundary points × 4225 interior = ~1M operations
        With Romberg integration: manageable (<1 second)
        
    This method is less efficient than von Hagenow but simpler to implement.
    Future optimization: implement freeBoundaryHagenow (line integral).
        
    Mathematical basis:
        G-S equation: Δ*ψ = S(ψ)
        Integral form: ψ(r) = ∫∫ G(r;r')·S(r') dA + boundary_terms
        
        For free-boundary: solve for both ψ_interior AND ψ_boundary
        Boundary equation: same integral, evaluated at boundary points
        
    Physical interpretation:
        Magnetic flux at boundary = superposition of contributions
        from all plasma current elements (Green's function = response)
        
    Reference:
        FreeGS boundary.py: freeBoundary function
        Our analysis: greens-function-boundary-derivation.md
    """
    
    nx, ny = psi.shape
    
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 1]
    
    # List of boundary indices
    # Order: bottom, top, left, right (avoiding corner duplicates)
    boundary_indices = []
    
    # Bottom edge (Z=Zmin)
    boundary_indices.extend([(i, 0) for i in range(nx)])
    
    # Top edge (Z=Zmax)
    boundary_indices.extend([(i, ny-1) for i in range(nx)])
    
    # Left edge (R=Rmin), skip corners
    boundary_indices.extend([(0, j) for j in range(1, ny-1)])
    
    # Right edge (R=Rmax), skip corners
    boundary_indices.extend([(nx-1, j) for j in range(1, ny-1)])
    
    print(f"Computing free boundary condition...")
    print(f"  Grid: {nx}×{ny}")
    print(f"  Boundary points: {len(boundary_indices)}")
    print(f"  Integration method: Romberg")
    
    # Loop over boundary points
    for idx, (i, j) in enumerate(boundary_indices):
        # Boundary point location
        R_boundary = R[i, j]
        Z_boundary = Z[i, j]
        
        # Calculate Green's function from all domain points to this boundary point
        greenfunc = greens_psi(R, Z, R_boundary, Z_boundary)
        
        # Prevent infinity/NaN at the boundary point itself
        # (Green's function singular at r'=r, but measure zero in integration)
        greenfunc[i, j] = 0.0
        
        # Integrate Green's function × Jtor over domain
        # Use Romberg integration (requires 2^n+1 points, which we have!)
        psi[i, j] = romb(romb(greenfunc * Jtor)) * dR * dZ
        
        # Progress indicator (every 10%)
        if (idx + 1) % (len(boundary_indices) // 10) == 0:
            progress = 100 * (idx + 1) / len(boundary_indices)
            print(f"  Progress: {progress:.0f}%")
    
    print(f"  ✅ Boundary condition computed")


def free_boundary_simple(R, Z, Jtor, psi):
    """
    Simplified free boundary: set ψ=0 on boundary
    
    This is for testing/debugging. NOT physically correct for free-boundary!
    Use free_boundary_greens() for actual solver.
    
    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates (not used)
    Jtor : ndarray
        Toroidal current (not used)
    psi : ndarray
        Flux array, boundary set to zero
    """
    
    psi[0, :] = 0.0
    psi[:, 0] = 0.0
    psi[-1, :] = 0.0
    psi[:, -1] = 0.0


def test_free_boundary():
    """
    Test free boundary condition implementation
    """
    
    print("=" * 70)
    print("Testing Free Boundary Condition")
    print("=" * 70)
    print()
    
    # Setup test grid
    R_1D = np.linspace(0.5, 2.0, 33)  # 2^5+1 points (Romberg requirement)
    Z_1D = np.linspace(-1.0, 1.0, 33)
    R, Z = np.meshgrid(R_1D, Z_1D, indexing='ij')
    
    # Test current distribution: Gaussian centered at (1.2, 0.0)
    R0 = 1.2
    Z0 = 0.0
    sigma = 0.3
    
    Jtor = 1e6 * np.exp(-((R - R0)**2 + (Z - Z0)**2) / (2 * sigma**2))
    
    print(f"Test setup:")
    print(f"  Grid: {R.shape[0]}×{R.shape[1]}")
    print(f"  R range: [{R.min():.2f}, {R.max():.2f}] m")
    print(f"  Z range: [{Z.min():.2f}, {Z.max():.2f}] m")
    print(f"  Jtor peak: {Jtor.max():.2e} A/m²")
    print(f"  Jtor center: R={R0}m, Z={Z0}m")
    print()
    
    # Initialize psi
    psi = np.zeros_like(R)
    
    # Apply free boundary condition
    import time
    t0 = time.time()
    
    free_boundary_greens(R, Z, Jtor, psi)
    
    t1 = time.time()
    
    print()
    print(f"Results:")
    print(f"  Computation time: {t1-t0:.2f} seconds")
    print(f"  Boundary ψ range: [{psi[psi!=0].min():.6e}, {psi[psi!=0].max():.6e}] Wb")
    
    # Check that interior is still zero (only boundary modified)
    interior_sum = np.sum(np.abs(psi[1:-1, 1:-1]))
    print(f"  Interior ψ sum: {interior_sum:.6e} (should be ~0)")
    
    # Check boundary values are non-zero
    boundary_nonzero = np.sum(psi[0,:] != 0) + np.sum(psi[-1,:] != 0) + \
                       np.sum(psi[:,0] != 0) + np.sum(psi[:,-1] != 0)
    
    print(f"  Non-zero boundary points: {boundary_nonzero}")
    
    print()
    print("=" * 70)
    print("Test complete!")
    

if __name__ == "__main__":
    test_free_boundary()
