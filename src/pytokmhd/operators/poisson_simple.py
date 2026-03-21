"""
Simple Poisson Solver (Jacobi Iteration)

Quick implementation for testing energy budget fix.
Uses iterative Jacobi method instead of FFT/banded solver.

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from ..geometry import ToroidalGrid


def solve_poisson_simple(
    omega: np.ndarray,
    grid: ToroidalGrid,
    max_iter: int = 10000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Solve ∇²φ = ω using Jacobi iteration.
    
    Parameters
    ----------
    omega : np.ndarray (nr, ntheta)
        Source term
    grid : ToroidalGrid
        Grid object
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    phi : np.ndarray (nr, ntheta)
        Solution
    """
    nr, ntheta = omega.shape
    dr = grid.dr
    dtheta = grid.dtheta
    r_grid = grid.r_grid
    
    # Initialize φ = 0
    phi = np.zeros_like(omega)
    phi_new = np.zeros_like(omega)
    
    # Jacobi iteration
    for iteration in range(max_iter):
        # Interior points
        for i in range(1, nr-1):
            for j in range(ntheta):
                r = r_grid[i, j]
                
                # Neighbors (periodic in θ)
                j_prev = (j - 1) % ntheta
                j_next = (j + 1) % ntheta
                
                # Discretization of ∇²φ = ω:
                # (φ[i+1] - 2φ[i] + φ[i-1])/dr² + (1/r)(φ[i+1] - φ[i-1])/(2dr)
                # + (φ[j+1] - 2φ[j] + φ[j-1])/(r²dθ²) = ω[i,j]
                #
                # Solve for φ[i,j]:
                coeff_main = -2/dr**2 - 2/(r**2 * dtheta**2)
                
                rhs = omega[i, j]
                rhs -= (1/dr**2 + 1/(2*r*dr)) * phi[i+1, j]
                rhs -= (1/dr**2 - 1/(2*r*dr)) * phi[i-1, j]
                rhs -= phi[i, j_next] / (r**2 * dtheta**2)
                rhs -= phi[i, j_prev] / (r**2 * dtheta**2)
                
                phi_new[i, j] = -rhs / coeff_main
        
        # Boundary conditions
        # r = 0 (axis): φ = 0 or dφ/dr = 0
        phi_new[0, :] = 0  # Simplified: φ(r=0) = 0
        
        # r = a (edge): φ = 0
        phi_new[-1, :] = 0
        
        # Check convergence
        residual = np.max(np.abs(phi_new - phi))
        phi[:] = phi_new
        
        if residual < tol:
            print(f"Poisson converged in {iteration+1} iterations (residual={residual:.3e})")
            break
    else:
        print(f"⚠️  Poisson did not converge (residual={residual:.3e})")
    
    return phi
