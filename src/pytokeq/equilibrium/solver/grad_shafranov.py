"""
Grad-Shafranov equilibrium solver

Solves: Δ*ψ = -μ0 R² dp/dψ - F dF/dψ

Methods:
- Picard iteration (simple, robust)
- Newton iteration (fast, for future)

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.operators import build_grad_shafranov_operator
from equilibrium.profiles import EquilibriumProfile


class ConvergenceError(Exception):
    """Raised when G-S solver fails to converge"""
    pass


def solve_grad_shafranov_picard(
    R: np.ndarray,
    Z: np.ndarray,
    profile: EquilibriumProfile,
    psi_init: Optional[np.ndarray] = None,
    psi_boundary: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Solve Grad-Shafranov equation using Picard iteration.
    
    Δ*ψ = S(ψ)
    where S(ψ) = -μ0 R² dp/dψ - F dF/dψ
    
    Algorithm:
    1. Start with initial guess ψ^(0)
    2. Compute S(ψ^(n))
    3. Solve linear: Δ*ψ^(n+1) = S(ψ^(n))
    4. Apply boundary conditions
    5. Check convergence: ||ψ^(n+1) - ψ^(n)|| < tol
    6. Repeat until converged
    
    Parameters
    ----------
    R : np.ndarray, shape (Nr,)
        Major radius grid
    Z : np.ndarray, shape (Nz,)
        Vertical grid
    profile : EquilibriumProfile
        Pressure and F profiles
    psi_init : np.ndarray, shape (Nr, Nz), optional
        Initial guess (default: zero)
    psi_boundary : float
        Boundary value for ψ (default: 0)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance (relative L2 norm)
    verbose : bool
        Print iteration info
    
    Returns
    -------
    psi : np.ndarray, shape (Nr, Nz)
        Equilibrium flux
    info : dict
        Convergence info (iterations, residuals, etc.)
    
    Raises
    ------
    ConvergenceError
        If not converged within max_iter
    
    Notes
    -----
    Picard is first-order convergence but very robust.
    Good for getting initial solution or difficult cases.
    """
    Nr, Nz = len(R), len(Z)
    N = Nr * Nz
    
    # Build Δ* operator matrix (once)
    L = build_grad_shafranov_operator(R, Z)
    
    # Initial guess
    if psi_init is None:
        psi = np.zeros((Nr, Nz))
    else:
        psi = psi_init.copy()
    
    # Create R grid (2D)
    R_2d = R[:, None] * np.ones((Nr, Nz))
    
    # Boundary mask
    boundary_mask = np.zeros((Nr, Nz), dtype=bool)
    boundary_mask[0, :] = True   # R_min
    boundary_mask[-1, :] = True  # R_max
    boundary_mask[:, 0] = True   # Z_min
    boundary_mask[:, -1] = True  # Z_max
    
    # Store convergence history
    residuals = []
    
    if verbose:
        print("=" * 60)
        print("Grad-Shafranov Picard Iteration")
        print("=" * 60)
        print(f"Grid: {Nr}×{Nz}, tol={tol:.1e}, max_iter={max_iter}")
        print("-" * 60)
    
    for n in range(max_iter):
        # Compute source term S(ψ^n) at current ψ
        S = profile.source_term(psi, R_2d)
        S_flat = S.flatten()
        
        # Solve linear system: Δ*ψ^(n+1) = S(ψ^n)
        psi_new_flat = spsolve(L, S_flat)
        psi_new = psi_new_flat.reshape(Nr, Nz)
        
        # Apply boundary conditions
        psi_new[boundary_mask] = psi_boundary
        
        # Compute residual (relative L2 norm)
        diff = psi_new - psi
        residual = np.linalg.norm(diff) / (np.linalg.norm(psi) + 1e-10)
        residuals.append(residual)
        
        if verbose:
            print(f"Iter {n+1:3d}: residual = {residual:.3e}")
        
        # Check convergence
        if residual < tol:
            if verbose:
                print("-" * 60)
                print(f"✓ Converged in {n+1} iterations")
                print("=" * 60)
            
            info = {
                'converged': True,
                'iterations': n + 1,
                'residual': residual,
                'residuals': residuals
            }
            return psi_new, info
        
        # Update for next iteration
        psi = psi_new
    
    # Did not converge
    if verbose:
        print("-" * 60)
        print(f"✗ Not converged after {max_iter} iterations")
        print(f"  Final residual: {residuals[-1]:.3e}")
        print("=" * 60)
    
    raise ConvergenceError(
        f"Picard iteration did not converge after {max_iter} iterations. "
        f"Final residual: {residuals[-1]:.3e}"
    )


def compute_force_balance_error(psi: np.ndarray, 
                                 R: np.ndarray, 
                                 Z: np.ndarray,
                                 profile: EquilibriumProfile) -> float:
    """
    Compute force balance error for equilibrium.
    
    Checks: Δ*ψ - S(ψ) ≈ 0
    
    Parameters
    ----------
    psi : np.ndarray, shape (Nr, Nz)
        Flux to check
    R, Z : np.ndarray
        Grid
    profile : EquilibriumProfile
        Profile used for S(ψ)
    
    Returns
    -------
    error : float
        Max absolute error in interior
    """
    from core.operators import apply_grad_shafranov_operator
    
    # Compute Δ*ψ
    Delta_psi = apply_grad_shafranov_operator(psi, R, Z)
    
    # Compute S(ψ)
    R_2d = R[:, None] * np.ones_like(psi)
    S = profile.source_term(psi, R_2d)
    
    # Error in interior (exclude boundaries)
    error_field = np.abs(Delta_psi - S)
    error_interior = error_field[1:-1, 1:-1]
    
    return error_interior.max()


def create_initial_guess_soloviev(R: np.ndarray, 
                                   Z: np.ndarray,
                                   R0: float = None,
                                   Z0: float = None,
                                   amplitude: float = 0.1) -> np.ndarray:
    """
    Create Soloviev-like initial guess.
    
    ψ = amplitude * [(R-R0)² + (Z-Z0)²]
    
    Simple, smooth, satisfies boundary conditions if amplitude chosen right.
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid
    R0, Z0 : float, optional
        Magnetic axis (default: grid center)
    amplitude : float
        Overall scale
    
    Returns
    -------
    psi_init : np.ndarray, shape (Nr, Nz)
        Initial guess
    """
    if R0 is None:
        R0 = (R[0] + R[-1]) / 2
    if Z0 is None:
        Z0 = (Z[0] + Z[-1]) / 2
    
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    psi_init = amplitude * ((R_2d - R0)**2 + (Z_2d - Z0)**2)
    
    # Ensure boundary = 0
    psi_init[0, :] = 0
    psi_init[-1, :] = 0
    psi_init[:, 0] = 0
    psi_init[:, -1] = 0
    
    return psi_init
