"""
Interchange Mode Initial Conditions (Issue #27)

Physics:
- Pressure-driven instability
- Rayleigh-Taylor type (heavy on light)
- Medium m-number (m=2-4)

References:
- Freidberg (1987) - Ideal MHD Ch 9
- Wesson (2011) - Tokamaks Ch 6.3 (Mercier criterion)
- Goedbloed & Poedts (2004) - Principles of MHD

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def pressure_interchange_equilibrium(
    r: jnp.ndarray,
    p0: float = 1.0,
    r_peak: float = 0.6,
    width: float = 0.15
) -> jnp.ndarray:
    """
    Pressure profile with localized bump (interchange unstable).
    
    p(r) = p0 * exp(-((r - r_peak) / width)²)
    
    Creates steep gradient at r = r_peak → interchange instability.
    
    Parameters
    ----------
    r : array, shape (nr,)
        Radial grid
    p0 : float
        Peak pressure
    r_peak : float
        Location of pressure peak
    width : float
        Pressure gradient scale length
        
    Returns
    -------
    p : array, shape (nr,)
        Pressure profile
        
    Notes
    -----
    Mercier criterion: D_I = (r/p)(dp/dr) - (2/q²)
    For steep gradient: D_I > 0 → unstable
    """
    p = p0 * jnp.exp(-((r - r_peak) / width)**2)
    return p


def psi_interchange_equilibrium(
    r: jnp.ndarray,
    p0: float = 1.0,
    r_peak: float = 0.6,
    width: float = 0.15,
    B0: float = 1.0
) -> jnp.ndarray:
    """
    Flux function for pressure-driven equilibrium.
    
    Simplified Grad-Shafranov:
    ∇²ψ ≈ -μ₀ r² dp/dψ
    
    For weak pressure, use approximate solution:
    ψ(r) ≈ ∫ p(r') r' dr' (normalized)
    
    Parameters
    ----------
    r : array
        Radial grid
    p0 : float
        Peak pressure
    r_peak : float
        Pressure peak location
    width : float
        Pressure width
    B0 : float
        Field strength normalization
        
    Returns
    -------
    psi : array
        Flux function
        
    Notes
    -----
    This is approximate - full GS solution would require iteration.
    For IC purposes, we just need reasonable equilibrium structure.
    """
    # Get pressure
    p = pressure_interchange_equilibrium(r, p0, r_peak, width)
    
    # Approximate: ψ ~ integral of pressure weighted by r
    # Use cumulative trapezoidal integration
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    integrand = p * r
    
    # Cumulative integral
    psi = jnp.cumsum(integrand) * dr
    
    # Normalize and scale
    psi = psi / jnp.max(jnp.abs(psi) + 1e-10) * B0
    
    return psi


def psi_interchange_perturbation(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 0.01,
    r_unstable: float = 0.6,
    width: float = 0.15,
    m: int = 2
) -> jnp.ndarray:
    """
    Interchange mode perturbation.
    
    δψ = ε * f(r) * cos(m*θ)
    
    where f(r) is Gaussian envelope at unstable radius.
    
    Parameters
    ----------
    r : array, shape (nr,)
        Radial grid
    theta : array, shape (ntheta,)
        Poloidal angle
    eps : float
        Perturbation amplitude
    r_unstable : float
        Location of instability (pressure gradient)
    width : float
        Radial width of mode
    m : int
        Poloidal mode number (2-4 typical for interchange)
        
    Returns
    -------
    delta_psi : array, shape (nr, ntheta)
        Flux perturbation
        
    Notes
    -----
    Interchange typically has m=2-4 (vs m=1 for kink).
    """
    R, Theta = jnp.meshgrid(r, theta, indexing='ij')
    
    # Radial envelope (Gaussian at unstable radius)
    f_r = jnp.exp(-((R - r_unstable) / width)**2)
    
    # Poloidal structure (cos for ψ)
    delta_psi = eps * f_r * jnp.cos(m * Theta)
    
    return delta_psi


def phi_interchange_perturbation(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 0.01,
    r_unstable: float = 0.6,
    width: float = 0.15,
    m: int = 2
) -> jnp.ndarray:
    """
    Stream function perturbation for interchange.
    
    δφ = ε * f(r) * sin(m*θ)
    
    Similar phase structure to ψ (characteristic of pressure modes).
    
    Parameters
    ----------
    r, theta : arrays
        Grid
    eps : float
        Perturbation amplitude
    r_unstable : float
        Unstable radius
    width : float
        Radial width
    m : int
        Mode number
        
    Returns
    -------
    delta_phi : array, shape (nr, ntheta)
        Stream function perturbation
    """
    R, Theta = jnp.meshgrid(r, theta, indexing='ij')
    
    # Same radial envelope
    f_r = jnp.exp(-((R - r_unstable) / width)**2)
    
    # sin for φ (90° shift from ψ)
    delta_phi = eps * f_r * jnp.sin(m * Theta)
    
    return delta_phi


def create_interchange_ic(
    nr: int = 32,
    ntheta: int = 64,
    p0: float = 1.0,
    r_peak: float = 0.6,
    width: float = 0.15,
    eps: float = 0.01,
    m: int = 2,
    B0: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create complete interchange mode initial condition.
    
    ψ = ψ_eq(r) + δψ(r,θ)
    φ = δφ(r,θ)
    
    Parameters
    ----------
    nr, ntheta : int
        Grid resolution
    p0 : float
        Peak pressure
    r_peak : float
        Pressure peak location (instability location)
    width : float
        Pressure gradient scale length
    eps : float
        Perturbation amplitude
    m : int
        Poloidal mode number (2-4 typical)
    B0 : float
        Magnetic field strength
        
    Returns
    -------
    psi : array, shape (nr, ntheta)
        Total flux function
    phi : array, shape (nr, ntheta)
        Stream function
        
    Notes
    -----
    Growth rate (Freidberg Ch 9):
    γ ≈ √[(p₀/ρ) / L_p²]
    
    For p0=1.0, ρ=1.0, L_p=0.15:
    γ ≈ √(1 / 0.0225) ≈ 6.7 s⁻¹
    
    Observable growth in 0.1s: e^(0.67) ≈ 1.95 (95% increase)
    """
    # Create grid
    r = jnp.linspace(0, 1, nr)
    theta = jnp.linspace(0, 2*jnp.pi, ntheta, endpoint=False)
    
    # Equilibrium (axisymmetric)
    psi_eq_1d = psi_interchange_equilibrium(r, p0=p0, r_peak=r_peak, 
                                            width=width, B0=B0)
    psi_eq = jnp.tile(psi_eq_1d[:, None], (1, ntheta))
    
    # Perturbation (m-mode)
    delta_psi = psi_interchange_perturbation(r, theta, eps=eps, 
                                             r_unstable=r_peak, 
                                             width=width, m=m)
    delta_phi = phi_interchange_perturbation(r, theta, eps=eps,
                                             r_unstable=r_peak,
                                             width=width, m=m)
    
    # Total
    psi = psi_eq + delta_psi
    phi = delta_phi
    
    return psi, phi


def get_expected_growth_rate(
    p0: float = 1.0,
    rho: float = 1.0,
    L_p: float = 0.15
) -> float:
    """
    Theoretical interchange growth rate (Freidberg Ch 9).
    
    For pressure-driven interchange:
    γ ≈ √[(p₀/ρ) / L_p²]
    
    where L_p is pressure gradient scale length.
    
    Parameters
    ----------
    p0 : float
        Peak pressure
    rho : float
        Mass density
    L_p : float
        Pressure gradient scale length
        
    Returns
    -------
    gamma : float
        Growth rate (s⁻¹)
        
    Notes
    -----
    Typical values:
    - p0 ~ 1 (normalized)
    - ρ ~ 1 (normalized)
    - L_p ~ 0.1-0.2 (steep gradient)
    - γ ~ 5-10 s⁻¹ (observable in 0.1s)
    
    Compare to:
    - Kink: γ ~ V_A/R₀ ~ 0.1-1 s⁻¹ (slower with scaling)
    - Tearing: γ ~ S^(-3/5) ω_A ~ 1-10 s⁻¹
    - Interchange: γ ~ √(β) ω_A ~ 5-10 s⁻¹ (medium)
    """
    gamma = jnp.sqrt((p0 / rho) / L_p**2)
    return gamma


def compute_mode_amplitude(psi: np.ndarray, m: int = 2) -> float:
    """
    Extract mode amplitude from flux function.
    
    Parameters
    ----------
    psi : array, shape (nr, ntheta)
        Flux function
    m : int
        Mode number to extract
        
    Returns
    -------
    mode_amp : float
        RMS amplitude of mode-m Fourier component
    """
    # FFT in poloidal direction
    psi_fft = np.fft.fft(psi, axis=1) / psi.shape[1]
    
    # Extract mode-m component
    mode_complex = psi_fft[:, m]
    
    # RMS amplitude
    mode_amp = np.sqrt(np.mean(np.abs(mode_complex)**2))
    
    return mode_amp


# Parameter sets for different scenarios

FAST_INTERCHANGE = {
    'p0': 2.0,        # Strong pressure
    'r_peak': 0.6,
    'width': 0.1,     # Steep gradient → fast growth
    'eps': 0.02,
    'm': 2,
    'B0': 1.0,
}

MODERATE_INTERCHANGE = {
    'p0': 1.0,
    'r_peak': 0.6,
    'width': 0.15,
    'eps': 0.01,
    'm': 2,
    'B0': 1.0,
}

SLOW_INTERCHANGE = {
    'p0': 0.5,        # Weaker pressure
    'r_peak': 0.7,
    'width': 0.2,     # Gentler gradient → slow growth
    'eps': 0.005,
    'm': 3,           # Higher m (more stable)
    'B0': 0.8,
}

M3_INTERCHANGE = {
    'p0': 1.0,
    'r_peak': 0.6,
    'width': 0.15,
    'eps': 0.01,
    'm': 3,           # m=3 mode (vs m=2 default)
    'B0': 1.0,
}
