"""
Hamiltonian Formulation for Reduced MHD

Implements the Hamiltonian energy functional for reduced MHD in toroidal geometry.

Mathematical Foundation
-----------------------
The reduced MHD Hamiltonian in axisymmetric toroidal geometry:

    H[ψ, φ] = ∫ d³x [ (1/2)|∇φ|² + (1/2)|∇ψ|² ]

where:
    ψ: poloidal magnetic flux
    φ: electrostatic potential (stream function)
    ∇φ: velocity field
    ∇ψ: magnetic field

Volume element in toroidal coordinates:
    d³x = √g dr dθ dφ = r*R dr dθ dφ

For axisymmetric case (∂/∂φ = 0), integrate over toroidal angle:
    ∫₀²ᵖ dφ = 2π

Physical Interpretation
-----------------------
H = K + U where:

K = ∫ (1/2)|∇φ|² dV  : Kinetic energy (E×B flow)
U = ∫ (1/2)|∇ψ|² dV  : Magnetic energy (poloidal field)

Conservation Laws
-----------------
1. **Energy Conservation**: dH/dt = 0 (for ideal MHD)
2. **Hamiltonian Evolution**:
   ∂ψ/∂t = [ψ, H]
   ∂ω/∂t = [ω, H]  where ω = ∇²φ (vorticity)

Numerical Implementation
------------------------
Gradient in toroidal coordinates:
    |∇f|² = (∂f/∂r)² + (1/r²)(∂f/∂θ)²

Volume integral:
    ∫ d³x = ∫∫ r*R dr dθ * 2π

References
----------
- Morrison (1998): Hamiltonian description of the ideal fluid
- Strauss (1976): Numerical studies of nonlinear evolution of kink modes
- Hazeltine & Meiss (2003): Plasma Confinement

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from ..geometry import ToroidalGrid
from ..operators.toroidal_operators import gradient_toroidal


def hamiltonian_density(psi: np.ndarray, phi: np.ndarray, 
                        grid: ToroidalGrid) -> np.ndarray:
    """
    Compute Hamiltonian energy density h = (1/2)|∇φ|² + (1/2)|∇ψ|².
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal magnetic flux
    phi : np.ndarray (nr, ntheta)
        Electrostatic potential (stream function)
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    h : np.ndarray (nr, ntheta)
        Hamiltonian density at each grid point
    
    Notes
    -----
    Energy density formula:
        h = (1/2)[(∂φ/∂r)² + (1/r²)(∂φ/∂θ)²] 
          + (1/2)[(∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)²]
    
    This is the integrand before volume element √g = r*R.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2  # Simple test field
    >>> phi = np.zeros_like(psi)
    >>> h = hamiltonian_density(psi, phi, grid)
    >>> assert h.shape == psi.shape
    >>> assert np.all(h >= 0)  # Energy density is non-negative
    """
    # Compute gradients
    grad_phi_r, grad_phi_theta = gradient_toroidal(phi, grid)
    grad_psi_r, grad_psi_theta = gradient_toroidal(psi, grid)
    
    # |∇φ|² = (∂φ/∂r)² + (1/r)(∂φ/∂θ)²
    # Note: gradient_toroidal returns (∂f/∂r, (1/r)∂f/∂θ)
    # So we need to be careful with the metric
    
    # Actually, for |∇f|² in orthogonal coords:
    # |∇f|² = g^rr (∂f/∂r)² + g^θθ (∂f/∂θ)²
    #       = (∂f/∂r)² + (1/r²)(∂f/∂θ)²
    
    # But gradient_toroidal returns physical components:
    # (∂f/∂r, (1/r)∂f/∂θ)
    
    # So |∇f|² = (∂f/∂r)² + ((1/r)∂f/∂θ)²
    
    # Let me recalculate from scratch using raw derivatives
    r_grid = grid.r_grid
    
    # Get raw derivatives (not physical components)
    from ..operators.poisson_bracket import _compute_derivatives
    
    dphi_dr, dphi_dtheta = _compute_derivatives(phi, grid)
    dpsi_dr, dpsi_dtheta = _compute_derivatives(psi, grid)
    
    # |∇φ|² = (∂φ/∂r)² + (1/r²)(∂φ/∂θ)²
    grad_phi_squared = dphi_dr**2 + (dphi_dtheta / r_grid)**2
    
    # |∇ψ|² = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)²
    grad_psi_squared = dpsi_dr**2 + (dpsi_dtheta / r_grid)**2
    
    # Hamiltonian density
    h = 0.5 * (grad_phi_squared + grad_psi_squared)
    
    return h


def compute_hamiltonian(psi: np.ndarray, phi: np.ndarray, 
                        grid: ToroidalGrid) -> float:
    """
    Compute total Hamiltonian energy H[ψ, φ].
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal magnetic flux
    phi : np.ndarray (nr, ntheta)
        Electrostatic potential
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    H : float
        Total Hamiltonian energy
    
    Notes
    -----
    Volume integral:
        H = ∫ h dV = ∫∫ h * r*R dr dθ * 2π
    
    Uses trapezoidal rule for numerical integration.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2
    >>> phi = np.zeros_like(psi)
    >>> H = compute_hamiltonian(psi, phi, grid)
    >>> assert H > 0  # Energy is positive
    """
    # Compute energy density
    h = hamiltonian_density(psi, phi, grid)
    
    # Volume element: dV = r*R dr dθ * 2π
    jacobian = grid.jacobian()  # √g = r*R
    
    # Integrate over (r, θ) using trapezoidal rule
    # For 2D: ∫∫ f(r,θ) dr dθ ≈ Σᵢⱼ f[i,j] * dr * dtheta
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Energy in poloidal plane
    energy_2d = np.sum(h * jacobian) * dr * dtheta
    
    # Multiply by 2π for toroidal direction
    H = 2 * np.pi * energy_2d
    
    return H


def kinetic_energy(phi: np.ndarray, grid: ToroidalGrid) -> float:
    """
    Compute kinetic energy K = ∫ (1/2)|∇φ|² dV.
    
    Parameters
    ----------
    phi : np.ndarray (nr, ntheta)
        Electrostatic potential (stream function)
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    K : float
        Kinetic energy
    
    Notes
    -----
    This is the E×B flow energy.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> phi = grid.r_grid * np.sin(grid.theta_grid)
    >>> K = kinetic_energy(phi, grid)
    >>> assert K >= 0
    """
    psi_zero = np.zeros_like(phi)
    H_total = compute_hamiltonian(psi_zero, phi, grid)
    return H_total


def magnetic_energy(psi: np.ndarray, grid: ToroidalGrid) -> float:
    """
    Compute magnetic energy U = ∫ (1/2)|∇ψ|² dV.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal magnetic flux
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    U : float
        Magnetic energy
    
    Notes
    -----
    This is the poloidal magnetic field energy.
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> from pytokmhd.solvers.equilibrium import circular_equilibrium
    >>> psi = circular_equilibrium(grid)
    >>> U = magnetic_energy(psi, grid)
    >>> assert U > 0
    """
    phi_zero = np.zeros_like(psi)
    H_total = compute_hamiltonian(psi, phi_zero, grid)
    return H_total


def energy_partition(psi: np.ndarray, phi: np.ndarray, 
                     grid: ToroidalGrid) -> dict:
    """
    Compute energy partition: total, kinetic, and magnetic.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux
    phi : np.ndarray (nr, ntheta)
        Electrostatic potential
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    energy : dict
        Dictionary with keys:
        - 'total': Total Hamiltonian H
        - 'kinetic': Kinetic energy K
        - 'magnetic': Magnetic energy U
        - 'fraction_kinetic': K/H
        - 'fraction_magnetic': U/H
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> psi = grid.r_grid**2
    >>> phi = grid.r_grid * np.sin(grid.theta_grid)
    >>> energy = energy_partition(psi, phi, grid)
    >>> print(f"Total: {energy['total']:.3e}")
    >>> print(f"Kinetic fraction: {energy['fraction_kinetic']:.2%}")
    """
    H_total = compute_hamiltonian(psi, phi, grid)
    K = kinetic_energy(phi, grid)
    U = magnetic_energy(psi, grid)
    
    energy = {
        'total': H_total,
        'kinetic': K,
        'magnetic': U,
        'fraction_kinetic': K / H_total if H_total > 0 else 0,
        'fraction_magnetic': U / H_total if H_total > 0 else 0,
    }
    
    return energy
