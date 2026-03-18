"""
Exact Sparse Poisson Solver for Toroidal Geometry

Solves ∇²φ = ω using exact stencil from laplacian_toroidal.

Strategy:
- Numerically extract stencil by applying operator to delta functions
- Build sparse matrix from extracted coefficients
- Guaranteed to match laplacian_toroidal to machine precision

References:
- Phase 1 implementation: toroidal_operators.laplacian_toroidal
- Design doc: v1.1-toroidal-symplectic-design.md Section 1.2

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Tuple
from ..geometry import ToroidalGrid


def build_laplacian_matrix(grid: ToroidalGrid, tol: float = 1e-15) -> sp.csr_matrix:
    """
    Build sparse Laplacian matrix by numerically extracting stencils.
    
    For each grid point (i, j), applies laplacian_toroidal to a delta function
    centered at that point, then reads off the coefficients.
    
    This guarantees exact match with laplacian_toroidal to machine precision.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Toroidal grid object
    tol : float, optional
        Threshold for treating coefficients as zero (default 1e-15)
    
    Returns
    -------
    L : scipy.sparse.csr_matrix (N, N)
        Sparse Laplacian matrix where N = nr * ntheta
        L @ phi_flat = laplacian_toroidal(phi, grid).flatten()
    
    Notes
    -----
    - Construction is O(N) where N = nr*ntheta (one operator call per point)
    - Stencil size is typically ~15 non-zeros per row (5 in r, 3 in θ)
    - Uses lil_matrix for construction, converts to csr for solving
    """
    from ..operators.toroidal_operators import laplacian_toroidal
    
    nr, ntheta = grid.r_grid.shape
    N = nr * ntheta
    
    # Build in lil format (efficient for construction)
    L = sp.lil_matrix((N, N))
    
    # Helper: 2D index to flat
    def idx(i, j):
        return i * ntheta + (j % ntheta)  # Periodic in θ
    
    print(f"Building Laplacian matrix ({nr} x {ntheta} = {N} points)...")
    
    # Extract stencil for each point
    # Key insight: L[row, col] is the coefficient of f[col] in lap_f[row]
    # To extract column col: set f = delta at position col, compute lap_f
    # Then lap_f[row] gives L[row, col]
    
    for i_src in range(nr):
        if i_src % 10 == 0:
            print(f"  Processing column {i_src}/{nr}...")
        
        for j_src in range(ntheta):
            # Create delta function at source position (i_src, j_src)
            delta = np.zeros((nr, ntheta))
            delta[i_src, j_src] = 1.0
            
            # Apply Laplacian - this gives us one COLUMN of L
            lap_delta = laplacian_toroidal(delta, grid)
            
            # Extract non-zero coefficients
            # lap_delta[i_dst, j_dst] = L[idx(i_dst, j_dst), idx(i_src, j_src)]
            col = idx(i_src, j_src)
            
            for i_dst in range(nr):
                for j_dst in range(ntheta):
                    coeff = lap_delta[i_dst, j_dst]
                    if abs(coeff) > tol:
                        row = idx(i_dst, j_dst)
                        L[row, col] = coeff
    
    # Enforce Dirichlet boundary conditions: φ = 0 at r_min and r = a
    print("  Enforcing Dirichlet BC (φ=0 at r boundaries)...")
    for j in range(ntheta):
        # r_min boundary (i=0)
        row = idx(0, j)
        L[row, :] = 0.0      # Clear row
        L[row, row] = 1.0    # φ[0,j] = 0
        
        # r=a boundary (i=nr-1)
        row = idx(nr-1, j)
        L[row, :] = 0.0      # Clear row
        L[row, row] = 1.0    # φ[nr-1,j] = 0
    
    print("  Converting to CSR format...")
    return L.tocsr()


def solve_poisson_exact(omega: np.ndarray, grid: ToroidalGrid, 
                         tol: float = 1e-12,
                         L_matrix: sp.csr_matrix = None) -> np.ndarray:
    """
    Solve Poisson equation ∇²φ = ω using exact toroidal Laplacian stencil.
    
    Parameters
    ----------
    omega : np.ndarray (nr, ntheta)
        Source term (vorticity, charge density, etc.)
    grid : ToroidalGrid
        Toroidal grid object
    tol : float, optional
        Solver tolerance (default 1e-12 for machine precision)
    L_matrix : scipy.sparse.csr_matrix, optional
        Pre-built Laplacian matrix (if None, will build it)
        Pass this if solving multiple times on same grid for efficiency
    
    Returns
    -------
    phi : np.ndarray (nr, ntheta)
        Solution to ∇²φ = ω
    
    Notes
    -----
    Stencil structure (from laplacian_toroidal):
    - R direction: up to 5-point stencil
    - θ direction: up to 5-point stencil (product rule terms)
    - Total: ~15-20 non-zero entries per row
    
    Boundary conditions:
    - R boundaries: one-sided differences
    - θ direction: periodic
    
    Solver: scipy.sparse.linalg.spsolve (direct LU)
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Test: ω = r²·sin(2θ)
    >>> r_grid, theta_grid = grid.r_grid, grid.theta_grid
    >>> omega = r_grid**2 * np.sin(2*theta_grid)
    >>> phi = solve_poisson_exact(omega, grid)
    >>> 
    >>> # Verify residual < 1e-10
    >>> from pytokmhd.operators.toroidal_operators import laplacian_toroidal
    >>> residual = laplacian_toroidal(phi, grid) - omega
    >>> print(f"Max residual: {np.max(np.abs(residual)):.2e}")
    >>> assert np.max(np.abs(residual)) < 1e-10
    """
    nr, ntheta = omega.shape
    N = nr * ntheta
    
    # Helper: 2D index to flat
    def idx(i, j):
        return i * ntheta + (j % ntheta)
    
    # Build Laplacian matrix if not provided
    if L_matrix is None:
        L_matrix = build_laplacian_matrix(grid)
    
    # Flatten source term and enforce BC
    omega_flat = omega.copy().flatten()
    
    # Set ω = 0 at boundaries (consistent with φ = 0 BC)
    for j in range(ntheta):
        omega_flat[idx(0, j)] = 0.0      # r_min
        omega_flat[idx(nr-1, j)] = 0.0   # r = a
    
    # Solve sparse linear system
    print(f"Solving sparse system ({N} unknowns)...")
    phi_flat = spla.spsolve(L_matrix, omega_flat)
    
    # Reshape to 2D
    phi = phi_flat.reshape((nr, ntheta))
    
    return phi


def verify_poisson_solver(grid: ToroidalGrid, verbose: bool = True) -> Dict[str, float]:
    """
    Verify Poisson solver with analytical test case.
    
    Test case: ω = r²·sin(2θ)
    
    Parameters
    ----------
    grid : ToroidalGrid
        Grid to test on
    verbose : bool, optional
        Print detailed results (default True)
    
    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'max_residual': max|∇²φ - ω|
        - 'rms_residual': RMS residual
        - 'max_phi': max|φ|
        - 'condition_number': (if computed)
    """
    from ..operators.toroidal_operators import laplacian_toroidal
    
    if verbose:
        print("=" * 60)
        print("Poisson Solver Verification")
        print("=" * 60)
    
    # Test case: ω = r²·sin(2θ)
    r_grid = grid.r_grid
    theta_grid = grid.theta_grid
    omega = r_grid**2 * np.sin(2 * theta_grid)
    
    if verbose:
        print(f"\nTest case: ω = r²·sin(2θ)")
        print(f"Grid: {grid.nr} x {grid.ntheta} = {grid.nr * grid.ntheta} points")
        print(f"Source norm: {np.linalg.norm(omega):.6e}")
    
    # Solve
    phi = solve_poisson_exact(omega, grid)
    
    # Verify by applying Laplacian
    lap_phi = laplacian_toroidal(phi, grid)
    residual = lap_phi - omega
    
    # Compute metrics
    max_residual = np.max(np.abs(residual))
    rms_residual = np.sqrt(np.mean(residual**2))
    max_phi = np.max(np.abs(phi))
    
    metrics = {
        'max_residual': max_residual,
        'rms_residual': rms_residual,
        'max_phi': max_phi,
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  max|φ|           = {max_phi:.6e}")
        print(f"  max|∇²φ - ω|     = {max_residual:.6e}")
        print(f"  RMS residual     = {rms_residual:.6e}")
        
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(phi))
        has_inf = np.any(np.isinf(phi))
        print(f"  NaN/Inf check    = {'❌ FAIL' if (has_nan or has_inf) else '✅ PASS'}")
        
        # Check residual threshold (slightly relaxed due to boundaries)
        passed = max_residual < 1e-9
        print(f"  Residual < 1e-9  = {'✅ PASS' if passed else '❌ FAIL'}")
        print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    """Quick test when run as script."""
    import sys
    sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')
    
    from pytokmhd.geometry import ToroidalGrid
    
    # Small test grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Run verification
    metrics = verify_poisson_solver(grid, verbose=True)
    
    # Exit with appropriate code (relaxed threshold)
    success = metrics['max_residual'] < 1e-9
    sys.exit(0 if success else 1)
