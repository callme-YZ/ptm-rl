"""
Poisson Bracket Operator for Toroidal Geometry

Implements the canonical Poisson bracket [f, g] in toroidal coordinates
for reduced MHD Hamiltonian formulation.

Mathematical Foundation
-----------------------
For axisymmetric toroidal coordinates (r, θ, φ), the canonical Poisson bracket:

    [f, g] = (1/R²) (∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)

where R = R₀ + r*cos(θ) is the major radius.

The factor 1/R² comes from the toroidal metric g^φφ = 1/R² in the 
canonical formulation.

Properties
----------
1. **Anti-symmetry**: [f, g] = -[g, f]
2. **Linearity**: [af + bg, h] = a[f, h] + b[g, h]
3. **Jacobi identity**: [f, [g, h]] + [g, [h, f]] + [h, [f, g]] = 0
4. **Leibniz rule**: [fg, h] = f[g, h] + g[f, h]

Physical Interpretation
-----------------------
The Poisson bracket generates canonical transformations and governs
the time evolution of observables in Hamiltonian mechanics.

For reduced MHD:
    ∂A/∂t = [A, H]

where H is the Hamiltonian and A is any field variable.

References
----------
- Morrison (1998): Hamiltonian description of the ideal fluid
- Brizard & Hahm (2007): Foundations of nonlinear gyrokinetic theory
- Hazeltine & Meiss (2003): Plasma Confinement, Chapter 3

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid


def poisson_bracket(f: np.ndarray, g: np.ndarray, grid: ToroidalGrid) -> np.ndarray:
    """
    Compute canonical Poisson bracket [f, g] in toroidal geometry.
    
    Formula:
        [f, g] = (1/R²) (∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)
    
    Uses 2nd-order centered finite differences for all derivatives.
    
    Parameters
    ----------
    f : np.ndarray (nr, ntheta)
        First scalar field
    g : np.ndarray (nr, ntheta)
        Second scalar field
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    bracket : np.ndarray (nr, ntheta)
        Poisson bracket [f, g]
    
    Notes
    -----
    - Anti-symmetric: [f, g] = -[g, f] to machine precision
    - Satisfies Jacobi identity within discretization error O(dr² + dθ²)
    - Periodic boundary in θ direction
    - One-sided differences at radial boundaries
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> f = grid.r_grid**2
    >>> g = np.sin(grid.theta_grid)
    >>> bracket_fg = poisson_bracket(f, g, grid)
    >>> bracket_gf = poisson_bracket(g, f, grid)
    >>> # Verify anti-symmetry
    >>> assert np.allclose(bracket_fg, -bracket_gf, atol=1e-12)
    """
    nr, ntheta = f.shape
    dr = grid.dr
    dtheta = grid.dtheta
    R_grid = grid.R_grid  # R = R₀ + r*cos(θ)
    
    # Compute derivatives
    df_dr, df_dtheta = _compute_derivatives(f, grid)
    dg_dr, dg_dtheta = _compute_derivatives(g, grid)
    
    # Poisson bracket: (1/R²)(∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)
    bracket = (df_dr * dg_dtheta - df_dtheta * dg_dr) / R_grid**2
    
    return bracket


def _compute_derivatives(f: np.ndarray, grid: ToroidalGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂f/∂r and ∂f/∂θ using centered finite differences.
    
    Parameters
    ----------
    f : np.ndarray (nr, ntheta)
        Scalar field
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    df_dr : np.ndarray (nr, ntheta)
        Radial derivative ∂f/∂r
    df_dtheta : np.ndarray (nr, ntheta)
        Poloidal derivative ∂f/∂θ
    
    Notes
    -----
    - 2nd-order centered differences in interior
    - One-sided 2nd-order at radial boundaries
    - Periodic in θ direction
    """
    nr, ntheta = f.shape
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Radial derivative: ∂f/∂r
    df_dr = np.zeros_like(f)
    
    # Interior: centered difference
    df_dr[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dr)
    
    # Boundaries: one-sided difference (2nd-order)
    df_dr[0, :] = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dr)
    df_dr[-1, :] = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dr)
    
    # Poloidal derivative: ∂f/∂θ
    df_dtheta = np.zeros_like(f)
    
    # Interior: centered difference
    df_dtheta[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dtheta)
    
    # Periodic boundary in θ
    df_dtheta[:, 0] = (f[:, 1] - f[:, -1]) / (2*dtheta)
    df_dtheta[:, -1] = (f[:, 0] - f[:, -2]) / (2*dtheta)
    
    return df_dr, df_dtheta


def jacobi_identity_residual(f: np.ndarray, g: np.ndarray, h: np.ndarray, 
                              grid: ToroidalGrid) -> float:
    """
    Compute residual of Jacobi identity: [f, [g, h]] + [g, [h, f]] + [h, [f, g]].
    
    The Jacobi identity must be satisfied for the Poisson bracket to define
    a valid Lie algebra structure. Residual should be O(dr² + dθ²) due to
    discretization error.
    
    Parameters
    ----------
    f, g, h : np.ndarray (nr, ntheta)
        Three scalar fields
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    residual : float
        Maximum absolute value of Jacobi identity residual
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> f = grid.r_grid**2
    >>> g = np.sin(grid.theta_grid)
    >>> h = grid.r_grid * np.cos(grid.theta_grid)
    >>> residual = jacobi_identity_residual(f, g, h, grid)
    >>> # Should be small (discretization error)
    >>> assert residual < 1e-6
    """
    # Compute nested brackets
    gh = poisson_bracket(g, h, grid)
    hf = poisson_bracket(h, f, grid)
    fg = poisson_bracket(f, g, grid)
    
    term1 = poisson_bracket(f, gh, grid)
    term2 = poisson_bracket(g, hf, grid)
    term3 = poisson_bracket(h, fg, grid)
    
    # Jacobi identity: sum should be zero
    jacobi_sum = term1 + term2 + term3
    
    residual = np.max(np.abs(jacobi_sum))
    
    return residual


def advection_bracket(psi: np.ndarray, omega: np.ndarray, 
                      grid: ToroidalGrid) -> np.ndarray:
    """
    Compute advection term [ψ, ω] for reduced MHD.
    
    This is the nonlinear advection of vorticity ω by the stream function ψ:
        [ψ, ω] = (1/R²)(∂ψ/∂r ∂ω/∂θ - ∂ψ/∂θ ∂ω/∂r)
    
    Physically represents E×B advection in the poloidal plane.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Stream function (or flux function)
    omega : np.ndarray (nr, ntheta)
        Vorticity (or any scalar field)
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    advection : np.ndarray (nr, ntheta)
        Advection term [ψ, ω]
    
    Notes
    -----
    Special case of poisson_bracket(psi, omega, grid).
    Provided as named function for clarity in MHD context.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> from pytokmhd.solvers.equilibrium import circular_equilibrium
    >>> psi = circular_equilibrium(grid)
    >>> omega = grid.laplacian(psi)
    >>> adv = advection_bracket(psi, omega, grid)
    """
    return poisson_bracket(psi, omega, grid)
