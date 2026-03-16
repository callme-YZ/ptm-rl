"""
Initial Conditions for PyTokMHD

Provides equilibrium initialization methods:
- Phase 1: Simplified Harris current sheet
- Phase 2: Real tokamak equilibrium from PyTokEq

Author: 小P ⚛️
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .equilibrium_cache import EquilibriumCache


def harris_sheet_initial(
    r: np.ndarray,
    z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified Harris current sheet equilibrium (Phase 1)
    
    Simple analytical equilibrium for testing and validation.
    
    Args:
        r: Radial grid (Nr,)
        z: Vertical grid (Nz,)
        
    Returns:
        psi: Magnetic flux (Nr, Nz)
        omega: Vorticity (Nr, Nz)
    """
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Harris sheet profile
    psi = np.tanh(Z) * (1 - R**2)
    
    # Consistent vorticity (∇²ψ ≈ constant for simplified case)
    omega = np.zeros_like(psi)
    
    return psi, omega


def pytokeq_initial(
    r: np.ndarray,
    z: np.ndarray,
    eq_cache: EquilibriumCache,
    perturbation_amplitude: float = 0.01,
    mode_number: int = 2,
    target_q: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real tokamak equilibrium from PyTokEq with tearing mode perturbation (Phase 2)
    
    Loads equilibrium from cache and adds m=2 tearing mode perturbation
    at the q=2 rational surface.
    
    Args:
        r: Radial grid (Nr,)
        z: Vertical grid (Nz,)
        eq_cache: EquilibriumCache instance
        perturbation_amplitude: Tearing mode amplitude (default 0.01)
        mode_number: Poloidal mode number m (default 2)
        target_q: Target q-value for rational surface (default 2.0)
        
    Returns:
        psi: Equilibrium flux + tearing perturbation (Nr, Nz)
        omega: Equilibrium vorticity (Nr, Nz)
    """
    # Get equilibrium from cache
    eq = eq_cache.get_equilibrium(perturb=False)
    
    psi_eq = eq['psi_eq']
    j_eq = eq['j_eq']
    p_eq = eq['p_eq']
    q_profile = eq['q_profile']
    
    # Find rational surface q = target_q
    r_s = find_rational_surface(r, q_profile, target_q)
    
    # Add tearing mode perturbation
    delta_psi = tearing_mode_perturbation(
        r, z, r_s, 
        mode_number=mode_number, 
        amplitude=perturbation_amplitude
    )
    
    psi = psi_eq + delta_psi
    
    # Compute equilibrium vorticity
    omega = compute_equilibrium_vorticity(r, z, psi_eq, j_eq, p_eq)
    
    return psi, omega


def find_rational_surface(
    r: np.ndarray,
    q_profile: np.ndarray,
    target_q: float
) -> float:
    """
    Find radial location of rational surface q = target_q
    
    Args:
        r: Radial grid (Nr,)
        q_profile: Safety factor profile (Nr,)
        target_q: Target q-value
        
    Returns:
        r_s: Radial location of rational surface
    """
    # Ensure q_profile is monotonic (typical for tokamaks)
    if len(q_profile) != len(r):
        # If q_profile is shorter, interpolate to r grid
        r_q = np.linspace(r[0], r[-1], len(q_profile))
        q_interp = np.interp(r, r_q, q_profile)
    else:
        q_interp = q_profile
    
    # Find where q crosses target_q
    # Linear interpolation
    if target_q < q_interp.min() or target_q > q_interp.max():
        # If target_q outside range, use midpoint
        return r[len(r) // 2]
    
    idx = np.searchsorted(q_interp, target_q)
    if idx == 0:
        return r[0]
    if idx >= len(r):
        return r[-1]
    
    # Linear interpolation between idx-1 and idx
    r1, r2 = r[idx-1], r[idx]
    q1, q2 = q_interp[idx-1], q_interp[idx]
    
    r_s = r1 + (target_q - q1) * (r2 - r1) / (q2 - q1)
    
    return r_s


def tearing_mode_perturbation(
    r: np.ndarray,
    z: np.ndarray,
    r_s: float,
    mode_number: int = 2,
    amplitude: float = 0.01,
    width: float = 0.1
) -> np.ndarray:
    """
    Tearing mode perturbation at rational surface
    
    Form: δψ = A * exp(-(r-r_s)²/w²) * cos(m*θ)
    where θ = arctan(z/r) is poloidal angle
    
    Args:
        r: Radial grid (Nr,)
        z: Vertical grid (Nz,)
        r_s: Rational surface radius
        mode_number: Poloidal mode number m
        amplitude: Perturbation amplitude
        width: Radial width of perturbation
        
    Returns:
        delta_psi: Tearing mode perturbation (Nr, Nz)
    """
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Poloidal angle
    theta = np.arctan2(Z, R - r_s)
    
    # Radial envelope (Gaussian centered at r_s)
    radial_envelope = np.exp(-((R - r_s) / width)**2)
    
    # Mode structure
    delta_psi = amplitude * radial_envelope * np.cos(mode_number * theta)
    
    return delta_psi


def compute_equilibrium_vorticity(
    r: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    j_tor: np.ndarray,
    pressure: np.ndarray
) -> np.ndarray:
    """
    Compute equilibrium vorticity from equilibrium fields
    
    From reduced MHD: ω = ∇²ψ (in equilibrium)
    Use finite difference Laplacian.
    
    Args:
        r: Radial grid (Nr,)
        z: Vertical grid (Nz,)
        psi: Equilibrium flux (Nr, Nz)
        j_tor: Toroidal current (Nr, Nz) [not used but kept for API consistency]
        pressure: Pressure (Nr, Nz) [not used but kept for API consistency]
        
    Returns:
        omega: Equilibrium vorticity (Nr, Nz)
    """
    Nr, Nz = len(r), len(z)
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    omega = np.zeros_like(psi)
    
    # Interior points: 5-point Laplacian
    for i in range(1, Nr - 1):
        for j in range(1, Nz - 1):
            d2psi_dr2 = (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j]) / dr**2
            d2psi_dz2 = (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1]) / dz**2
            
            # Cylindrical Laplacian: ∇² = ∂²/∂r² + (1/r)∂/∂r + ∂²/∂z²
            dpsi_dr = (psi[i+1, j] - psi[i-1, j]) / (2 * dr)
            
            omega[i, j] = d2psi_dr2 + dpsi_dr / r[i] + d2psi_dz2
    
    # Boundaries: simple copy from interior
    omega[0, :] = omega[1, :]
    omega[-1, :] = omega[-2, :]
    omega[:, 0] = omega[:, 1]
    omega[:, -1] = omega[:, -2]
    
    return omega


def solovev_equilibrium(
    r: np.ndarray,
    z: np.ndarray,
    R0: float = 1.0,
    a: float = 0.5,
    epsilon: float = 0.32,
    kappa: float = 1.7,
    delta: float = 0.33
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical Solovev equilibrium for testing
    
    Provides exact analytical equilibrium for validation.
    
    Args:
        r: Radial grid (Nr,)
        z: Vertical grid (Nz,)
        R0: Major radius
        a: Minor radius
        epsilon: Inverse aspect ratio
        kappa: Elongation
        delta: Triangularity
        
    Returns:
        psi: Magnetic flux (Nr, Nz)
        omega: Vorticity (Nr, Nz)
    """
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Solovev solution (simplified form)
    psi = ((R - R0)**2 + (Z / kappa)**2 - a**2)**2 / (4 * R0**2)
    
    # Vorticity from Laplacian
    omega = compute_equilibrium_vorticity(r, z, psi, 
                                          np.zeros_like(psi), 
                                          np.zeros_like(psi))
    
    return psi, omega
