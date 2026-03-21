"""
Implicit Resistive Diffusion Solver for IMEX Time Stepping

Solves the implicit resistive step in IMEX scheme:
    (I - dt*eta*[J operator])ψ = source

where J operator corresponds to compute_current_density:
    J = Δ*ψ/(μ₀R)
    
Physical equation: ∂ψ/∂t = {ψ, φ} - η·J

Implicit step: ψ_new = ψ* - dt/2 * η * J(ψ_new)
Rearrange: ψ_new - dt/2*η*J(ψ_new) = ψ*
           (I - dt/2*η*[J operator])ψ_new = ψ*

Author: 小P ⚛️
Created: 2026-03-19 (Fixed)
Phase: v1.3 IMEX Implementation - Corrected Physics
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from typing import Tuple, Optional
from ..geometry import ToroidalGrid
from ..physics import compute_current_density


def solve_implicit_resistive(
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
    Solve implicit resistive diffusion: (I + dt*eta*[J op])ψ = source.
    
    Uses GMRES with LinearOperator for the operator (I + dt*eta*J_op)
    where J_op corresponds to compute_current_density.
    
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
        Solution to implicit resistive equation
    info : int
        GMRES convergence flag (0 = converged)
    
    Notes
    -----
    The resistive term uses compute_current_density which includes:
    - Grad-Shafranov operator Δ*ψ
    - Division by μ₀R
    - Proper axis handling (L'Hôpital for r=0)
    
    This is the CORRECT physics for toroidal resistive MHD.
    NOT simple Laplacian!
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    >>> source = grid.r_grid**2
    >>> psi, info = solve_implicit_resistive(source, dt=1e-4, eta=1e-4, grid=grid)
    >>> assert info == 0
    """
    nr, ntheta = grid.nr, grid.ntheta
    N = nr * ntheta
    
    # Default boundary: conducting wall
    if psi_boundary is None:
        psi_boundary = np.zeros(ntheta)
    
    def matvec(psi_flat: np.ndarray) -> np.ndarray:
        """
        Apply operator: (I - dt*eta*[J op]) @ psi.
        
        J operator = compute_current_density = Δ*ψ/(μ₀R)
        NOTE: Minus sign for resistive damping!
        
        With BC enforcement:
        - Interior: (I - dt*eta*J_op)ψ
        - Axis: ψ(0, θ) = constant (enforce via identity)
        - Edge: ψ(a, θ) = psi_boundary (enforce via identity)
        """
        psi_2d = psi_flat.reshape((nr, ntheta))
        
        # Apply J operator (compute_current_density)
        # Use normalized units (μ₀=1.0) for consistency
        J_psi = compute_current_density(psi_2d, grid, mu0=1.0)
        
        # Operator: I - dt*eta*J  (CORRECTED: minus sign for resistive damping!)
        result = psi_2d - dt * eta * J_psi
        
        # Enforce BCs by identity rows
        # Axis: average over theta
        psi_axis_avg = np.mean(psi_2d[0, :])
        result[0, :] = psi_axis_avg
        
        # Edge: conducting wall
        result[-1, :] = psi_boundary
        
        return result.ravel()
    
    # Create LinearOperator
    A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    
    # Enforce BCs on source
    source_bc = source.copy()
    source_bc[0, :] = np.mean(source[0, :])  # Axis
    source_bc[-1, :] = psi_boundary  # Edge
    
    # Initial guess: source itself (good guess!)
    x0 = source_bc.ravel()
    
    # Solve with GMRES
    psi_flat, info = gmres(
        A, source_bc.ravel(),
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        atol=0  # Use relative tolerance only
    )
    
    if verbose:
        if info == 0:
            print(f"GMRES converged in {maxiter} iterations (or less)")
        else:
            print(f"GMRES did not converge: info={info}")
    
    psi = psi_flat.reshape((nr, ntheta))
    
    # Final BC enforcement (safety)
    psi[0, :] = np.mean(psi[0, :])
    psi[-1, :] = psi_boundary
    
    return psi, info
