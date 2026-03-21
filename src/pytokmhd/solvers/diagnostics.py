"""
MHD Diagnostics

Energy and other physics diagnostics.

Author: 小P ⚛️
Created: 2026-03-17 (M3.3)
"""

import numpy as np
from ..geometry import ToroidalGrid
from ..operators import gradient_toroidal


def compute_energy(
    psi: np.ndarray,
    omega: np.ndarray,
    grid: ToroidalGrid
) -> float:
    """
    Compute total MHD energy.
    
    E = E_magnetic + E_kinetic
    
    E_mag = (1/2) ∫ |∇ψ|² √g dV
    E_kin = (1/2) ∫ |ω|² √g dV
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux
    omega : np.ndarray (nr, ntheta)
        Vorticity
    grid : ToroidalGrid
    
    Returns
    -------
    E : float
        Total energy
    
    Notes
    -----
    The Jacobian √g = r*R in toroidal coordinates (r, θ, φ).
    
    For magnetic energy:
        |∇ψ|² = (∂ψ/∂r)² + (1/r² ∂ψ/∂θ)²
    
    The gradient_toroidal returns:
        grad_r = ∂ψ/∂r
        grad_theta = (1/r²)∂ψ/∂θ
    
    So |∇ψ|² = grad_r² + grad_theta²
    
    Volume element:
        dV = √g dr dθ dφ = r*R dr dθ dφ
    
    Integrating over φ gives factor 2π, which we include implicitly.
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> from pytokmhd.solvers.equilibrium import circular_equilibrium
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    >>> psi = circular_equilibrium(grid)
    >>> omega = np.zeros_like(psi)
    >>> E = compute_energy(psi, omega, grid)
    >>> assert E > 0  # Should have positive magnetic energy
    >>> assert np.isfinite(E)
    """
    J = grid.jacobian()  # √g = r*R
    
    # Magnetic energy: (1/2) ∫ |∇ψ|² √g dV
    grad_r, grad_theta = gradient_toroidal(psi, grid)
    
    # |∇ψ|² = (∂ψ/∂r)² + (1/r² ∂ψ/∂θ)²
    # grad_theta already includes the 1/r² factor
    grad_psi_sq = grad_r**2 + grad_theta**2
    
    E_mag = 0.5 * np.sum(grad_psi_sq * J * grid.dr * grid.dtheta)
    
    # Kinetic energy: (1/2) ∫ ω² √g dV
    E_kin = 0.5 * np.sum(omega**2 * J * grid.dr * grid.dtheta)
    
    return E_mag + E_kin
