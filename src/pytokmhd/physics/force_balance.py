"""
Force Balance Verification

Implements J×B = ∇P force balance calculations for equilibrium verification.

Physical Background
-------------------
In MHD equilibrium, plasma is in force balance:
    J×B = ∇P

where:
    - J: current density [A/m²]
    - B: magnetic field [T]
    - P: plasma pressure [Pa]

For axisymmetric toroidal equilibrium:
    - Toroidal current: Jφ = (1/μ₀R)Δ*ψ
    - Poloidal current: Jₚ = (1/μ₀)dF/dψ ∇ψ
    - Force balance becomes Grad-Shafranov equation

Grad-Shafranov Operator
-----------------------
In (r, θ) coordinates, the Grad-Shafranov operator is:
    Δ*ψ = R²∇·(∇ψ/R²) = ∂²ψ/∂r² + (1/r²)∂²ψ/∂θ² + (∂R/∂r)(∂ψ/∂r)/R
    
For circular cross-section:
    R = R₀ + r·cos(θ)
    ∂R/∂r = cos(θ)
    
    Δ*ψ = ∂²ψ/∂r² + (1/r²)∂²ψ/∂θ² + cos(θ)/(R₀+r·cos(θ))·∂ψ/∂r

Solovev Equilibrium
-------------------
The Solovev solution is an exact analytical equilibrium:
    ψ(r,θ) = ψ₀ + c₁r² + c₂r⁴ + c₃r⁴cos(2θ)
    
Force balance J×B = ∇P is satisfied exactly (to machine precision).
We use it as a benchmark.

References
----------
- Grad & Rubin (1958): Original force balance formulation
- Shafranov (1966): Toroidal equilibrium equation
- Solov'ev (1968): Analytical equilibrium solutions
- Freidberg (2014): "Ideal MHD", Chapter 6
- Wesson & Campbell (2011): "Tokamaks", Chapter 3

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid
from ..equilibrium import pressure_gradient


def compute_current_density(
    psi: np.ndarray,
    grid: ToroidalGrid,
    mu0: float = 4*np.pi*1e-7
) -> np.ndarray:
    """
    Compute toroidal current density Jφ from poloidal flux ψ.
    
    In axisymmetric equilibrium:
        Jφ = (1/μ₀R)Δ*ψ
    
    where Δ* is the Grad-Shafranov operator:
        Δ*ψ = R²∇·(∇ψ/R²)
    
    For circular cross-section in (r, θ) coordinates:
        Δ*ψ = ∂²ψ/∂r² + (1/r²)∂²ψ/∂θ² + cos(θ)/(R₀+r·cos(θ))·∂ψ/∂r
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux ψ [Wb]
    grid : ToroidalGrid
        Toroidal grid object
    mu0 : float, optional
        Permeability of free space [H/m]
    
    Returns
    -------
    J_phi : np.ndarray (nr, ntheta)
        Toroidal current density [A/m²]
    
    Notes
    -----
    - Uses 2nd-order centered finite differences
    - Periodic boundary in θ direction
    - One-sided differences at radial boundaries
    - Δ*ψ includes metric terms for toroidal geometry
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2  # Simple test case
    >>> J_phi = compute_current_density(psi, grid)
    """
    nr, ntheta = psi.shape
    dr = grid.dr
    dtheta = grid.dtheta
    r_grid = grid.r_grid
    R_grid = grid.R_grid
    theta_grid = grid.theta_grid
    
    # Compute ∂²ψ/∂r²
    d2psi_dr2 = np.zeros_like(psi)
    # Interior: centered difference
    d2psi_dr2[1:-1, :] = (psi[2:, :] - 2*psi[1:-1, :] + psi[:-2, :]) / dr**2
    # Boundaries: one-sided (2nd-order)
    d2psi_dr2[0, :] = (2*psi[0, :] - 5*psi[1, :] + 4*psi[2, :] - psi[3, :]) / dr**2
    d2psi_dr2[-1, :] = (2*psi[-1, :] - 5*psi[-2, :] + 4*psi[-3, :] - psi[-4, :]) / dr**2
    
    # Compute ∂²ψ/∂θ²
    d2psi_dtheta2 = np.zeros_like(psi)
    # Interior: centered difference
    d2psi_dtheta2[:, 1:-1] = (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2]) / dtheta**2
    # Periodic boundary in θ
    d2psi_dtheta2[:, 0] = (psi[:, 1] - 2*psi[:, 0] + psi[:, -1]) / dtheta**2
    d2psi_dtheta2[:, -1] = (psi[:, 0] - 2*psi[:, -1] + psi[:, -2]) / dtheta**2
    
    # Compute ∂ψ/∂r for metric term
    dpsi_dr = np.zeros_like(psi)
    # Interior: centered difference
    dpsi_dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dr)
    # Boundaries: one-sided (2nd-order)
    dpsi_dr[0, :] = (-3*psi[0, :] + 4*psi[1, :] - psi[2, :]) / (2*dr)
    dpsi_dr[-1, :] = (3*psi[-1, :] - 4*psi[-2, :] + psi[-3, :]) / (2*dr)
    
    # Grad-Shafranov operator: Δ*ψ
    Delta_star_psi = (d2psi_dr2 + d2psi_dtheta2 / r_grid**2 + 
                      np.cos(theta_grid) / R_grid * dpsi_dr)
    
    # Current density: Jφ = (1/μ₀R)Δ*ψ
    J_phi = Delta_star_psi / (mu0 * R_grid)
    
    return J_phi


def compute_lorentz_force(
    psi: np.ndarray,
    grid: ToroidalGrid,
    mu0: float = 4*np.pi*1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lorentz force J×B in (r, θ) components.
    
    For axisymmetric equilibrium with:
        - Jφ: toroidal current
        - Bᵣ = -(1/r)∂ψ/∂θ: radial poloidal field
        - Bθ = ∂ψ/∂r: poloidal field (θ component)
    
    The Lorentz force components are:
        (J×B)ᵣ = Jφ·Bθ = Jφ·(∂ψ/∂r)
        (J×B)θ = -Jφ·Bᵣ = Jφ·(1/r)·(∂ψ/∂θ)
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux ψ [Wb]
    grid : ToroidalGrid
        Toroidal grid object
    mu0 : float, optional
        Permeability of free space [H/m]
    
    Returns
    -------
    JxB_r : np.ndarray (nr, ntheta)
        Radial component of J×B [N/m³]
    JxB_theta : np.ndarray (nr, ntheta)
        Poloidal component of J×B [N/m³]
    
    Notes
    -----
    - In equilibrium: J×B = ∇P
    - Used for force balance verification
    - Poloidal field: Bₚ = ∇ψ × ∇φ / R
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2  # Test case
    >>> JxB_r, JxB_theta = compute_lorentz_force(psi, grid)
    """
    # Compute toroidal current density
    J_phi = compute_current_density(psi, grid, mu0)
    
    # Compute ∇ψ
    dr = grid.dr
    dtheta = grid.dtheta
    nr, ntheta = psi.shape
    r_grid = grid.r_grid
    
    # ∂ψ/∂r
    dpsi_dr = np.zeros_like(psi)
    # Interior: centered difference
    dpsi_dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dr)
    # Boundaries: one-sided (2nd-order)
    dpsi_dr[0, :] = (-3*psi[0, :] + 4*psi[1, :] - psi[2, :]) / (2*dr)
    dpsi_dr[-1, :] = (3*psi[-1, :] - 4*psi[-2, :] + psi[-3, :]) / (2*dr)
    
    # ∂ψ/∂θ
    dpsi_dtheta = np.zeros_like(psi)
    # Interior: centered difference
    dpsi_dtheta[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2*dtheta)
    # Periodic boundary in θ
    dpsi_dtheta[:, 0] = (psi[:, 1] - psi[:, -1]) / (2*dtheta)
    dpsi_dtheta[:, -1] = (psi[:, 0] - psi[:, -2]) / (2*dtheta)
    
    # Poloidal field components
    # Bᵣ = -(1/r)∂ψ/∂θ
    # Bθ = ∂ψ/∂r
    B_r = -dpsi_dtheta / r_grid
    B_theta = dpsi_dr
    
    # J×B components
    # (J×B)ᵣ = Jφ·Bθ
    # (J×B)θ = -Jφ·Bᵣ
    JxB_r = J_phi * B_theta
    JxB_theta = -J_phi * B_r
    
    return JxB_r, JxB_theta


def force_balance_residual(
    psi: np.ndarray,
    P0: float,
    psi_edge: float,
    grid: ToroidalGrid,
    alpha: float = 2.0,
    mu0: float = 4*np.pi*1e-7
) -> dict:
    """
    Verify force balance J×B = ∇P and return residual.
    
    Computes both sides of equilibrium equation:
        J×B = ∇P
    
    Returns componentwise residuals and maximum error.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux ψ [Wb]
    P0 : float
        Central pressure [Pa]
    psi_edge : float
        Edge flux value [Wb]
    grid : ToroidalGrid
        Toroidal grid object
    alpha : float, optional
        Pressure peaking exponent (default: 2.0)
    mu0 : float, optional
        Permeability of free space [H/m]
    
    Returns
    -------
    result : dict
        {
            'residual_r': np.ndarray,     # (J×B)ᵣ - (∇P)ᵣ
            'residual_theta': np.ndarray, # (J×B)θ - (∇P)θ
            'max_residual': float,        # max(|residual_r|, |residual_theta|)
            'rms_residual': float,        # sqrt(mean(residual²))
            'relative_error': float,      # max_residual / max(|∇P|)
            'JxB_r': np.ndarray,          # Lorentz force (r)
            'JxB_theta': np.ndarray,      # Lorentz force (θ)
            'gradP_r': np.ndarray,        # Pressure gradient (r)
            'gradP_theta': np.ndarray,    # Pressure gradient (θ)
        }
    
    Notes
    -----
    - For Solovev equilibrium: max_residual should be O(1e-6) or smaller
    - relative_error gives normalized measure of force balance quality
    - Used for equilibrium verification and benchmarking
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Load Solovev equilibrium (example)
    >>> psi = load_solovev_equilibrium(grid)  # Not implemented here
    >>> result = force_balance_residual(psi, P0=1e5, psi_edge=1.0, grid=grid)
    >>> print(f"Max residual: {result['max_residual']:.2e}")
    >>> print(f"Relative error: {result['relative_error']:.2e}")
    """
    # Compute J×B
    JxB_r, JxB_theta = compute_lorentz_force(psi, grid, mu0)
    
    # Compute ∇P
    gradP_r, gradP_theta = pressure_gradient(psi, P0, psi_edge, grid, alpha)
    
    # Residual: J×B - ∇P
    residual_r = JxB_r - gradP_r
    residual_theta = JxB_theta - gradP_theta
    
    # Error metrics
    max_residual = max(np.max(np.abs(residual_r)), np.max(np.abs(residual_theta)))
    rms_residual = np.sqrt(np.mean(residual_r**2 + residual_theta**2))
    
    # Relative error (normalized by ∇P magnitude)
    gradP_mag = np.sqrt(gradP_r**2 + gradP_theta**2)
    max_gradP = np.max(gradP_mag)
    relative_error = max_residual / max_gradP if max_gradP > 0 else np.inf
    
    return {
        'residual_r': residual_r,
        'residual_theta': residual_theta,
        'max_residual': max_residual,
        'rms_residual': rms_residual,
        'relative_error': relative_error,
        'JxB_r': JxB_r,
        'JxB_theta': JxB_theta,
        'gradP_r': gradP_r,
        'gradP_theta': gradP_theta,
    }


def pressure_force_term(
    psi: np.ndarray,
    P0: float,
    psi_edge: float,
    grid: ToroidalGrid,
    alpha: float = 2.0
) -> np.ndarray:
    """
    Compute pressure force term for vorticity equation.
    
    This term represents the pressure-driven vorticity source in the
    generalized vorticity equation:
        ∂ω/∂t = [ω, H] + S_P + dissipation
    
    where S_P is the pressure force term:
        S_P = (1/R²)(dP/dψ) ∇²ψ
    
    or equivalently in Poisson bracket form:
        S_P = (1/R²)(dP/dψ)[∇ψ, ∇·(∇ψ/R²)]
    
    Physical Interpretation
    -----------------------
    This term couples pressure gradient ∇P to vorticity evolution,
    ensuring force balance J×B = ∇P is maintained in equilibrium.
    
    In equilibrium (∂ω/∂t = 0), this term balances the magnetic stress.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux ψ [Wb]
    P0 : float
        Central pressure [Pa]
    psi_edge : float
        Edge flux value [Wb]
    grid : ToroidalGrid
        Toroidal grid object
    alpha : float, optional
        Pressure peaking exponent (default: 2.0)
    
    Returns
    -------
    S_P : np.ndarray (nr, ntheta)
        Pressure force term [s⁻¹]
    
    Notes
    -----
    - Simplified form: S_P = (1/R²)(dP/dψ)·Δ*ψ
    - Where Δ*ψ = ∇²ψ + geometric terms (Grad-Shafranov operator)
    - This is consistent with J×B = ∇P force balance
    
    References
    ----------
    - Freidberg (2014): "Ideal MHD", Chapter 6
    - Hazeltine & Meiss (2003): "Plasma Confinement"
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2  # Test flux
    >>> S_P = pressure_force_term(psi, P0=1e5, psi_edge=0.09, grid=grid)
    >>> assert S_P.shape == psi.shape
    """
    # Compute dP/dψ
    from ..equilibrium import pressure_gradient_psi
    dP_dpsi = pressure_gradient_psi(psi, P0, psi_edge, alpha)
    
    # Compute Δ*ψ (Grad-Shafranov operator)
    # Using J_phi = (1/μ₀R)Δ*ψ from compute_current_density
    mu0 = 4*np.pi*1e-7
    J_phi = compute_current_density(psi, grid, mu0)
    Delta_star_psi = mu0 * grid.R_grid * J_phi
    
    # Pressure force term: S_P = (1/R²)(dP/dψ)·Δ*ψ
    S_P = (dP_dpsi * Delta_star_psi) / grid.R_grid**2
    
    return S_P
