"""
Kink Mode Initial Conditions (Issue #27)

Physics:
- m=1 helical instability
- Current-driven (q ≈ 1 resonance)
- Ideal MHD (no resistivity needed)

References:
- Kadomtsev (1975) - Internal kink theory
- Freidberg (1987) - Ideal MHD textbook
- Wesson (2011) - Tokamaks Ch 7

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def current_kink_equilibrium(
    r: jnp.ndarray,
    j0: float = 2.0,
    a: float = 0.8
) -> jnp.ndarray:
    """
    Current profile for kink equilibrium (q ≈ 1).
    
    Parabolic current profile:
    J_z(r) = j0 * (1 - (r/a)²)
    
    This gives q(r) ≈ constant ≈ 1 when properly normalized.
    
    Parameters
    ----------
    r : array, shape (nr,)
        Radial grid
    j0 : float
        Peak current density (at r=0)
    a : float
        Current profile width
        
    Returns
    -------
    J_z : array, shape (nr,)
        Current density profile
    """
    r_norm = jnp.clip(r / a, 0, 1)
    J_z = j0 * (1 - r_norm**2)
    return J_z


def psi_kink_equilibrium(
    r: jnp.ndarray,
    j0: float = 2.0,
    a: float = 0.8,
    B0: float = 1.0
) -> jnp.ndarray:
    """
    Flux function for kink equilibrium.
    
    Solve: ∇²ψ = -μ₀ J_z
    
    For cylindrical with J_z = j0(1-(r/a)²):
    ψ(r) = -(j0 a²/4) [(r/a)² - (r/a)⁴/2] + C
    
    Choose C such that ψ(r=0) = 0.
    
    Parameters
    ----------
    r : array
        Radial grid
    j0 : float
        Peak current
    a : float
        Profile width
    B0 : float
        Normalization (for matching units)
        
    Returns
    -------
    psi : array
        Flux function
    """
    r_norm = jnp.clip(r / a, 0, 1)
    
    # Analytical solution to ∇²ψ = -J_z
    # (assuming cylindrical, μ₀ = 1 for normalization)
    psi = -(j0 * a**2 / 4) * (r_norm**2 - r_norm**4 / 2)
    
    # Scale by B0 for proper units
    psi = psi * B0
    
    return psi


def psi_kink_perturbation(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 0.01,
    r_res: float = 0.5,
    width: float = 0.2,
    m: int = 1
) -> jnp.ndarray:
    """
    Kink mode perturbation (m=1 helical displacement).
    
    δψ = ε * f(r) * sin(m*θ)
    
    where f(r) is Gaussian envelope peaked at q=1 surface.
    
    Parameters
    ----------
    r : array, shape (nr,)
        Radial grid
    theta : array, shape (ntheta,)
        Poloidal angle
    eps : float
        Perturbation amplitude
    r_res : float
        Resonant surface radius (q=1 location)
    width : float
        Radial width of mode
    m : int
        Poloidal mode number (default: 1 for kink)
        
    Returns
    -------
    delta_psi : array, shape (nr, ntheta)
        Flux perturbation
    """
    R, Theta = jnp.meshgrid(r, theta, indexing='ij')
    
    # Radial envelope (Gaussian at resonance)
    f_r = jnp.exp(-((R - r_res) / width)**2)
    
    # Helical structure
    delta_psi = eps * f_r * jnp.sin(m * Theta)
    
    return delta_psi


def phi_kink_perturbation(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 0.01,
    r_res: float = 0.5,
    width: float = 0.2,
    m: int = 1
) -> jnp.ndarray:
    """
    Stream function perturbation for kink.
    
    δφ = ε * f(r) * cos(m*θ)
    
    Phase-shifted 90° from δψ (characteristic of kink).
    
    Parameters
    ----------
    r, theta : arrays
        Grid
    eps : float
        Perturbation amplitude
    r_res : float
        Resonant surface
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
    f_r = jnp.exp(-((R - r_res) / width)**2)
    
    # 90° phase shift (cos vs sin)
    delta_phi = eps * f_r * jnp.cos(m * Theta)
    
    return delta_phi


def create_kink_ic(
    nr: int = 32,
    ntheta: int = 64,
    r_res: float = 0.5,
    j0: float = 2.0,
    a: float = 0.8,
    eps: float = 0.01,
    B0: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create complete kink mode initial condition.
    
    ψ = ψ_eq(r) + δψ(r,θ)
    φ = δφ(r,θ)
    
    Parameters
    ----------
    nr, ntheta : int
        Grid resolution
    r_res : float
        Resonant surface (q=1 location, default 0.5)
    j0 : float
        Peak current (controls q profile)
    a : float
        Current profile width
    eps : float
        Perturbation amplitude
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
    Growth rate (Freidberg 1987):
    γ ≈ 0.3 V_A / R₀  (internal kink, q₀ ≈ 0.9)
    
    For observable growth in 0.1s:
    - Need γ ~ 10 s⁻¹
    - Requires scaled parameters (reduced B0 or increased mass)
    """
    # Create grid
    r = jnp.linspace(0, 1, nr)
    theta = jnp.linspace(0, 2*jnp.pi, ntheta, endpoint=False)
    
    # Equilibrium (axisymmetric)
    psi_eq_1d = psi_kink_equilibrium(r, j0=j0, a=a, B0=B0)
    psi_eq = jnp.tile(psi_eq_1d[:, None], (1, ntheta))
    
    # Perturbation (m=1 helical)
    delta_psi = psi_kink_perturbation(r, theta, eps=eps, r_res=r_res)
    delta_phi = phi_kink_perturbation(r, theta, eps=eps, r_res=r_res)
    
    # Total
    psi = psi_eq + delta_psi
    phi = delta_phi
    
    return psi, phi


def get_expected_growth_rate(
    B0: float = 1.0,
    rho: float = 1.0,
    R0: float = 1.0,
    q0: float = 0.9
) -> float:
    """
    Theoretical kink growth rate (Freidberg 1987).
    
    For internal kink with q₀ < 1:
    γ ≈ 0.3 V_A / R₀
    
    where V_A = B₀ / √(μ₀ ρ) is Alfvén velocity.
    
    Parameters
    ----------
    B0 : float
        Magnetic field strength
    rho : float
        Mass density
    R0 : float
        Major radius
    q0 : float
        On-axis safety factor (should be <1)
        
    Returns
    -------
    gamma : float
        Growth rate (s⁻¹)
        
    Notes
    -----
    For realistic tokamak:
    - B0 ~ 1-5 T
    - ρ ~ 1e-6 kg/m³
    - V_A ~ 1e6 m/s
    - γ ~ 1e5 s⁻¹ (µs timescale!)
    
    For simulation (observable in 0.1s):
    - Reduce B0 or increase ρ
    - Target γ ~ 1-10 s⁻¹
    """
    mu0 = 1.0  # Normalized units
    V_A = B0 / jnp.sqrt(mu0 * rho)
    
    # Freidberg formula (0.3 is numerical coefficient from theory)
    gamma = 0.3 * V_A / R0
    
    return gamma


def compute_m1_amplitude(psi: np.ndarray) -> float:
    """
    Extract m=1 mode amplitude from flux function.
    
    Same as tearing_ic.compute_m1_amplitude.
    
    Parameters
    ----------
    psi : array, shape (nr, ntheta)
        Flux function
        
    Returns
    -------
    m1_amp : float
        RMS amplitude of m=1 Fourier component
    """
    # FFT in poloidal direction
    psi_fft = np.fft.fft(psi, axis=1) / psi.shape[1]
    
    # Extract m=1 component
    m1_complex = psi_fft[:, 1]
    
    # RMS amplitude
    m1_amp = np.sqrt(np.mean(np.abs(m1_complex)**2))
    
    return m1_amp


# Parameter sets for different scenarios

FAST_KINK = {
    'r_res': 0.5,
    'j0': 3.0,     # Stronger current → faster growth
    'a': 0.8,
    'eps': 0.02,   # Larger perturbation
    'B0': 1.5,     # Stronger field
}

MODERATE_KINK = {
    'r_res': 0.5,
    'j0': 2.0,
    'a': 0.8,
    'eps': 0.01,
    'B0': 1.0,
}

SLOW_KINK = {
    'r_res': 0.6,
    'j0': 1.5,     # Weaker current → marginal stability
    'a': 0.8,
    'eps': 0.005,
    'B0': 0.8,
}
