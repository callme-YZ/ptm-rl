"""
Utility Functions for MHD Operators

Helper functions for common MHD operations.

Author: 小P ⚛️
Created: 2026-03-17 (M3.2)
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid
from .toroidal_operators import gradient_toroidal


def B_poloidal_from_psi(
    psi: np.ndarray,
    grid: ToroidalGrid
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute poloidal magnetic field from flux function ψ.
    
    In toroidal geometry with axisymmetry:
        B_pol = ∇ψ × ∇φ
    
    where φ is the toroidal angle.
    
    This gives:
        B_r = (1/r) ∂ψ/∂θ
        B_θ = -∂ψ/∂r
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux function
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    B_r : np.ndarray (nr, ntheta)
        Radial component of poloidal field
    B_theta : np.ndarray (nr, ntheta)
        Poloidal component of poloidal field
    
    Notes
    -----
    gradient_toroidal returns (∂ψ/∂r, (1/r²)∂ψ/∂θ), so we need to
    recover ∂ψ/∂θ by multiplying by r².
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = simple_equilibrium(grid)
    >>> B_r, B_theta = B_poloidal_from_psi(psi, grid)
    >>> # Verify ∇·B ≈ 0
    >>> from pytokmhd.operators import divergence_toroidal
    >>> div_B = divergence_toroidal(B_r, B_theta, grid)
    >>> assert np.max(np.abs(div_B)) < 1e-6
    """
    # Get gradient components
    # gradient_toroidal returns: (∂ψ/∂r, (1/r²)∂ψ/∂θ)
    grad_r, grad_theta = gradient_toroidal(psi, grid)
    
    # Recover ∂ψ/∂θ from (1/r²)∂ψ/∂θ
    dpsi_dtheta = grad_theta * grid.r_grid**2
    
    # Get R(r, θ) = R₀ + r cos(θ)
    R_grid = grid.R_grid
    
    # Correct poloidal field components (from cylindrical derivation)
    # B_r = (1/(r·R)) ∂ψ/∂θ
    B_r = dpsi_dtheta / (grid.r_grid * R_grid)
    
    # B_θ = -(1/R) ∂ψ/∂r
    B_theta = -grad_r / R_grid
    
    return B_r, B_theta
