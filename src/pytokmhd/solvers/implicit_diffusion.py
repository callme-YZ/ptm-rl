"""
Implicit Diffusion Solver for IMEX Time Stepping

Solves the implicit diffusion step in IMEX (Implicit-Explicit) scheme:
    (I + dt*eta*Laplacian)ψ = source

Equation: ∂ψ/∂t = {ψ, φ} - η·∇²ψ
Implicit step: ψ_new = ψ* - dt/2 * η * ∇²ψ_new
Rearrange: ψ_new + dt/2*η*∇²ψ_new = ψ*
           (I + dt/2*η*∇²)ψ_new = ψ*

Uses GMRES with LinearOperator for the full operator.

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 IMEX Implementation
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from typing import Tuple, Optional
from ..geometry import ToroidalGrid
from ..operators import laplacian_toroidal


def solve_implicit_diffusion(
    source: np.ndarray,
    dt: float,
    eta: float,
    grid: ToroidalGrid,
    psi_boundary: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Solve implicit diffusion equation (I + dt*eta*∇²)ψ = source.
    
    Uses GMRES with LinearOperator for the operator (I + dt*eta*∇²).
    
    Parameters
    ----------
    source : np.ndarray (nr, ntheta)
        RHS from explicit step (ψ*)
    dt : float
        Time step size [s]
    eta : float
        Resistivity [Ω·m]
    grid : ToroidalGrid
        Toroidal grid
    psi_boundary : np.ndarray (ntheta,), optional
        Boundary values at r=a. If None, uses zeros (conducting wall).
    tol : float, optional
        GMRES relative tolerance (default 1e-8)
    maxiter : int, optional
        Maximum GMRES iterations (default 1000)
    verbose : bool, optional
        Print convergence info (default False)
    
    Returns
    -------
    psi : np.ndarray (nr, ntheta)
        Solution to implicit diffusion equation
    info : int
        GMRES convergence flag (0 = converged)
    
    Notes
    -----
    The operator (I + dt*eta*∇²) is symmetric positive definite for dt*eta > 0.
    GMRES should converge quickly.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    >>> source = grid.r_grid**2
    >>> psi, info = solve_implicit_diffusion(source, dt=1e-4, eta=1e-4, grid=grid)
    >>> assert info == 0
    """
    nr, ntheta = grid.nr, grid.ntheta
    N = nr * ntheta
    
    # Default boundary: conducting wall
    if psi_boundary is None:
        psi_boundary = np.zeros(ntheta)
    
    def matvec(psi_flat: np.ndarray) -> np.ndarray:
        """
        Apply operator: (I + dt*eta*∇²) @ psi.
        
        With BC enforcement:
        - Interior: (I + dt*eta*∇²)ψ
        - Axis: ψ(0, θ) = constant (enforce via identity)
        - Edge: ψ(a, θ) = 0 (enforce via identity)
        """
        psi_2d = psi_flat.reshape((nr, ntheta))
        
        # Apply Laplacian
        lap_psi = laplacian_toroidal(psi_2d, grid)
        
        # Operator: I + dt*eta*∇²
        result = psi_2d + dt * eta * lap_psi
        
        # BC enforcement via identity rows:
        # Axis (i=0): return ψ itself
        result[0, :] = psi_2d[0, :]
        
        # Edge (i=nr-1): return ψ itself
        result[-1, :] = psi_2d[-1, :]
        
        return result.flatten()
    
    # Create LinearOperator
    A = LinearOperator((N, N), matvec=matvec, dtype=float)
    
    # Build RHS vector
    b = source.flatten().copy()
    
    # BC rows in RHS:
    # Axis: ψ(0, θ) = mean(source(0, :)) (axisymmetry)
    psi_axis_value = np.mean(source[0, :])
    for j in range(ntheta):
        b[j] = psi_axis_value
    
    # Edge: ψ(a, θ) = 0 (conducting wall)
    for j in range(ntheta):
        b[(nr-1)*ntheta + j] = 0.0
    
    # Solve with GMRES
    if verbose:
        print(f"Solving implicit diffusion with GMRES (dt={dt}, eta={eta})...")
    
    # Initial guess: source
    x0 = source.flatten()
    
    x, info = gmres(A, b, x0=x0, rtol=tol, atol=0, maxiter=maxiter, restart=100)
    
    if verbose:
        if info == 0:
            print(f"✅ GMRES converged")
        elif info > 0:
            print(f"⚠️ GMRES did not converge (info={info})")
        else:
            print(f"❌ GMRES illegal input (info={info})")
    
    psi = x.reshape((nr, ntheta))
    
    # GMRES returns info > 0 if not converged (number of iterations)
    # We want 0 for success, so convert
    if info > 0:
        # Did not converge
        return psi, info
    elif info < 0:
        # Illegal input
        raise RuntimeError(f"GMRES illegal input (info={info})")
    else:
        # Converged
        return psi, 0


def validate_implicit_diffusion(grid: ToroidalGrid, verbose: bool = True) -> bool:
    """
    Validate implicit diffusion solver with analytical test case.
    
    Test: ψ = r²
    Expected: For small dt*eta, solution should be close to source.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Test grid
    verbose : bool, optional
        Print results
    
    Returns
    -------
    passed : bool
        True if validation passes
    """
    # Test case: ψ = r²
    r_grid = grid.r_grid
    source = r_grid**2
    
    # Parameters
    dt = 1e-5
    eta = 1e-4
    
    # Solve
    psi, info = solve_implicit_diffusion(source, dt, eta, grid, verbose=verbose)
    
    if info != 0:
        if verbose:
            print(f"❌ GMRES did not converge (info={info})")
        return False
    
    # Expected: for small dt*eta, ψ ≈ source (in interior)
    # BC enforces ψ = 0 at edge, so check interior only
    error_interior = np.max(np.abs(psi[1:-1, :] - source[1:-1, :]))
    error_bc = np.max(np.abs(psi[-1, :]))  # Should be 0
    
    if verbose:
        print(f"\nValidation: Implicit Diffusion Solver")
        print(f"  dt = {dt}, eta = {eta}")
        print(f"  Interior error: {error_interior:.3e}")
        print(f"  BC error: {error_bc:.3e}")
        print(f"  GMRES info: {info}")
    
    # Pass if interior error is small and BC is satisfied
    passed = (error_interior < 1e-4) and (error_bc < 1e-6)
    
    if verbose:
        if passed:
            print("✅ Validation PASSED")
        else:
            print("❌ Validation FAILED")
    
    return passed


__all__ = [
    'solve_implicit_diffusion',
    'validate_implicit_diffusion'
]
