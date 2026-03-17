"""
Simple Equilibrium Solutions

Provides simple analytical equilibria for testing.

Author: 小P ⚛️
Created: 2026-03-17 (M3.2)
"""

import numpy as np
from ..geometry import ToroidalGrid


def circular_equilibrium(grid: ToroidalGrid, psi0: float = 1.0, 
                         epsilon: float = 0.1, m: int = 1) -> np.ndarray:
    """
    Simple equilibrium with poloidal variation.
    
    ψ = psi0 * r² * (1 + ε*cos(m*θ))
    
    This ensures:
        ∂ψ/∂θ ≠ 0 → B_r ≠ 0
        ∂ψ/∂r ≠ 0 → B_θ ≠ 0
    
    Parameters
    ----------
    grid : ToroidalGrid
        Toroidal grid
    psi0 : float, optional
        Central flux value (default: 1.0)
    epsilon : float, optional
        Poloidal modulation amplitude (default: 0.1)
    m : int, optional
        Poloidal mode number (default: 1)
    
    Returns
    -------
    psi : np.ndarray (nr, ntheta)
        Equilibrium poloidal flux
    
    Notes
    -----
    The poloidal variation breaks perfect axisymmetry, ensuring:
    - B_r is non-zero (from ∂ψ/∂θ term)
    - ∇·B = 0 can be tested (not trivially zero)
    
    Physical interpretation:
    - epsilon controls the up-down asymmetry
    - m controls the number of poloidal lobes
    
    For testing purposes, epsilon should be small (<<1) to maintain
    approximate equilibrium properties while having non-trivial topology.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = circular_equilibrium(grid)
    >>> # Should be smooth and well-behaved
    >>> assert np.all(np.isfinite(psi))
    >>> assert np.min(psi) >= 0
    """
    r = grid.r_grid
    theta = grid.theta_grid
    
    # With poloidal variation
    psi = psi0 * r**2 * (1.0 + epsilon * np.cos(m * theta))
    
    # Normalize to [0, 1]
    psi = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)
    
    return psi
