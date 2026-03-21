"""
Pressure Profile and Gradient Calculations

Implements equilibrium pressure profiles P(ψ) and gradients ∇P
for force balance in toroidal MHD.

Physical Background
-------------------
In MHD equilibrium, the force balance equation is:
    J×B = ∇P

where:
    - J: current density
    - B: magnetic field
    - P: plasma pressure

For axisymmetric toroidal equilibrium, the pressure is a flux function:
    P = P(ψ)
    
where ψ is the poloidal magnetic flux.

The pressure gradient is:
    ∇P = (dP/dψ)·∇ψ

Standard Profiles
-----------------
Common choices for P(ψ):

1. **Power law** (this implementation):
   P(ψ) = P₀(1 - ψ/ψ_edge)^α
   
   - P₀: central pressure
   - ψ_edge: edge flux value
   - α: peaking parameter (α > 0)

2. **Polynomial**:
   P(ψ) = Σᵢ aᵢ(ψ/ψ_edge)^i

3. **Exponential**:
   P(ψ) = P₀ exp(-βψ/ψ_edge)

References
----------
- Grad & Rubin (1958): "Hydromagnetic Equilibria and Force-Free Fields"
- Shafranov (1966): "Plasma Equilibrium in a Magnetic Field"
- Solov'ev (1968): "The Theory of Hydromagnetic Stability of Toroidal Plasma Configurations"
- Freidberg (2014): "Ideal MHD", Chapter 6

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid


def pressure_profile(
    psi: np.ndarray,
    P0: float,
    psi_edge: float,
    alpha: float = 2.0
) -> np.ndarray:
    """
    Compute equilibrium pressure profile P(ψ).
    
    Uses power-law profile:
        P(ψ) = P₀(1 - ψ/ψ_edge)^α   for ψ < ψ_edge
        P(ψ) = 0                     for ψ ≥ ψ_edge
    
    This is the standard profile used in Grad-Shafranov solvers.
    
    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux ψ (any shape)
    P0 : float
        Central pressure [Pa] (at ψ=0), must be > 0
    psi_edge : float
        Edge flux value [Wb], defines ψ=ψ_edge as separatrix
    alpha : float, optional
        Peaking exponent (default: 2.0)
        - α=1: linear profile
        - α=2: parabolic (typical for tokamaks)
        - α>2: more peaked
    
    Returns
    -------
    P : np.ndarray
        Pressure [Pa], same shape as psi
    
    Notes
    -----
    - Assumes ψ=0 at magnetic axis, ψ=ψ_edge at plasma edge
    - Normalized flux: ψ_n = ψ/ψ_edge ∈ [0,1]
    - P decreases monotonically from axis to edge
    - P=0 outside separatrix (scrape-off layer)
    
    Examples
    --------
    >>> psi = np.linspace(0, 1.0, 100)
    >>> P = pressure_profile(psi, P0=1e5, psi_edge=1.0, alpha=2.0)
    >>> assert np.isclose(P[0], 1e5)  # Central pressure
    >>> assert np.isclose(P[-1], 0.0)  # Edge pressure
    
    >>> # Verify monotonic decrease
    >>> assert np.all(np.diff(P) <= 0)
    """
    # Validation
    if P0 <= 0:
        raise ValueError(f"Central pressure P0 must be positive, got {P0}")
    if alpha <= 0:
        raise ValueError(f"Peaking exponent alpha must be positive, got {alpha}")
    
    # Normalized flux
    psi_n = psi / psi_edge
    
    # Pressure profile
    P = np.zeros_like(psi)
    mask = psi_n < 1.0  # Inside separatrix
    P[mask] = P0 * (1.0 - psi_n[mask])**alpha
    
    return P


def pressure_gradient_psi(
    psi: np.ndarray,
    P0: float,
    psi_edge: float,
    alpha: float = 2.0
) -> np.ndarray:
    """
    Compute dP/dψ analytically.
    
    For power-law profile P(ψ) = P₀(1 - ψ/ψ_edge)^α:
        dP/dψ = -α·P₀/ψ_edge · (1 - ψ/ψ_edge)^(α-1)
    
    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux ψ (any shape)
    P0 : float
        Central pressure [Pa], must be > 0
    psi_edge : float
        Edge flux value [Wb]
    alpha : float, optional
        Peaking exponent (default: 2.0)
    
    Returns
    -------
    dP_dpsi : np.ndarray
        Pressure gradient dP/dψ [Pa/Wb], same shape as psi
    
    Notes
    -----
    - dP/dψ < 0 (pressure decreases outward)
    - Vanishes at edge (ψ=ψ_edge)
    - Used in force balance: ∇P = (dP/dψ)·∇ψ
    
    Examples
    --------
    >>> psi = np.linspace(0, 1.0, 100)
    >>> dP = pressure_gradient_psi(psi, P0=1e5, psi_edge=1.0, alpha=2.0)
    >>> # Gradient should be negative everywhere
    >>> assert np.all(dP[psi < 1.0] < 0)
    >>> # Gradient should vanish at edge
    >>> assert np.isclose(dP[-1], 0.0, atol=1e-10)
    """
    # Validation
    if P0 <= 0:
        raise ValueError(f"Central pressure P0 must be positive, got {P0}")
    if alpha <= 0:
        raise ValueError(f"Peaking exponent alpha must be positive, got {alpha}")
    
    # Normalized flux
    psi_n = psi / psi_edge
    
    # dP/dψ
    dP_dpsi = np.zeros_like(psi)
    mask = psi_n < 1.0  # Inside separatrix
    dP_dpsi[mask] = -alpha * P0 / psi_edge * (1.0 - psi_n[mask])**(alpha - 1.0)
    
    return dP_dpsi


def pressure_gradient(
    psi: np.ndarray,
    P0: float,
    psi_edge: float,
    grid: ToroidalGrid,
    alpha: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pressure gradient ∇P = (dP/dψ)·∇ψ in toroidal geometry.
    
    The pressure gradient in (r, θ) coordinates:
        ∇P = (dP/dψ)·∇ψ = (dP/dψ)(∂ψ/∂r, (1/r²)∂ψ/∂θ)
    
    Components:
        ∇P_r = (dP/dψ)·(∂ψ/∂r)
        ∇P_θ = (dP/dψ)·(1/r²)·(∂ψ/∂θ)
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux ψ on toroidal grid
    P0 : float
        Central pressure [Pa]
    psi_edge : float
        Edge flux value [Wb]
    grid : ToroidalGrid
        Toroidal grid object
    alpha : float, optional
        Peaking exponent (default: 2.0)
    
    Returns
    -------
    grad_P_r : np.ndarray (nr, ntheta)
        Radial component of ∇P [Pa/m]
    grad_P_theta : np.ndarray (nr, ntheta)
        Poloidal component of ∇P [Pa/m]
    
    Notes
    -----
    - Uses centered finite differences for ∂ψ/∂r and ∂ψ/∂θ
    - Metric factor 1/r² included in poloidal component
    - Force balance: J×B = ∇P requires this to be accurate
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Simple test: constant ψ field
    >>> psi = grid.r_grid**2
    >>> grad_P_r, grad_P_theta = pressure_gradient(psi, P0=1e5, psi_edge=0.09, grid=grid)
    """
    # Compute dP/dψ
    dP_dpsi = pressure_gradient_psi(psi, P0, psi_edge, alpha)
    
    # Compute ∇ψ using centered differences
    dpsi_dr = np.zeros_like(psi)
    dpsi_dtheta = np.zeros_like(psi)
    
    dr = grid.dr
    dtheta = grid.dtheta
    nr, ntheta = psi.shape
    
    # ∂ψ/∂r (radial derivative)
    # Interior: centered difference
    dpsi_dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dr)
    # Boundaries: one-sided difference (2nd-order)
    dpsi_dr[0, :] = (-3*psi[0, :] + 4*psi[1, :] - psi[2, :]) / (2*dr)
    dpsi_dr[-1, :] = (3*psi[-1, :] - 4*psi[-2, :] + psi[-3, :]) / (2*dr)
    
    # ∂ψ/∂θ (poloidal derivative)
    # Interior: centered difference
    dpsi_dtheta[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2*dtheta)
    # Periodic boundary in θ
    dpsi_dtheta[:, 0] = (psi[:, 1] - psi[:, -1]) / (2*dtheta)
    dpsi_dtheta[:, -1] = (psi[:, 0] - psi[:, -2]) / (2*dtheta)
    
    # ∇P = (dP/dψ)·∇ψ
    grad_P_r = dP_dpsi * dpsi_dr
    grad_P_theta = dP_dpsi * dpsi_dtheta / grid.r_grid**2  # Metric factor
    
    return grad_P_r, grad_P_theta


def beta_poloidal(
    P0: float,
    B_pol: float,
    mu0: float = 4*np.pi*1e-7
) -> float:
    """
    Compute poloidal beta βₚ = 2μ₀P/B_pol².
    
    Poloidal beta measures ratio of plasma pressure to poloidal magnetic pressure.
    Typical tokamak values: βₚ ~ 0.1 - 1.0
    
    Parameters
    ----------
    P0 : float
        Central pressure [Pa]
    B_pol : float
        Poloidal magnetic field [T]
    mu0 : float, optional
        Permeability of free space [H/m] (default: 4π×10⁻⁷)
    
    Returns
    -------
    beta_p : float
        Poloidal beta (dimensionless)
    
    Examples
    --------
    >>> P0 = 1e5  # 1 bar
    >>> B_pol = 0.5  # 0.5 T
    >>> beta = beta_poloidal(P0, B_pol)
    >>> print(f"βₚ = {beta:.3f}")
    βₚ = 0.003
    """
    if B_pol <= 0:
        raise ValueError(f"Poloidal field B_pol must be positive, got {B_pol}")
    
    beta_p = 2 * mu0 * P0 / B_pol**2
    
    return beta_p
