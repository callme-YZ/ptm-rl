"""
Magnetic Island Diagnostics

Detect and quantify magnetic island structures from MHD simulations.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from .rational_surface import find_rational_surface


def extract_flux_surface(psi, r, z, r_target, n_theta=256, z_center=0.0):
    """
    Extract flux values along a circle at radius r_target
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux on 2D grid
    r : ndarray (Nr,)
        Radial coordinates
    z : ndarray (Nz,)
        Vertical coordinates
    r_target : float
        Target radius
    n_theta : int, optional
        Number of poloidal angle points (default: 256)
    z_center : float, optional
        Vertical center of the circle (default: 0.0)
    
    Returns
    -------
    psi_theta : ndarray (n_theta,)
        Flux values along circle
    theta : ndarray (n_theta,)
        Poloidal angles [0, 2π)
    
    Notes
    -----
    For cylindrical geometry (R,Z), we parameterize a circle:
    R = r_target
    Z = z_center + R_minor * sin(θ)
    
    For simplicity, take R_minor = r_target (circular cross-section)
    """
    # Create interpolator
    interp = RectBivariateSpline(r, z, psi)
    
    # Generate poloidal angles
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Parametric circle in (R, Z) plane
    # For magnetic island detection, scan around rational surface
    # Use minor radius proportional to r_target
    R_minor = min(r_target, 0.3)  # Smaller minor radius to stay in domain
    
    R_points = r_target + R_minor * np.cos(theta)
    Z_points = z_center + R_minor * np.sin(theta)
    
    # Ensure points are within grid bounds
    R_points = np.clip(R_points, r.min(), r.max())
    Z_points = np.clip(Z_points, z.min(), z.max())
    
    psi_theta = interp(R_points, Z_points, grid=False)
    
    return psi_theta, theta


def find_extrema(f_theta, theta=None):
    """
    Find O-points and X-points from periodic function
    
    Parameters
    ----------
    f_theta : ndarray
        Periodic function values (e.g., flux along theta)
    theta : ndarray, optional
        Angular coordinates (default: uniform [0, 2π))
    
    Returns
    -------
    extrema : dict
        - 'o_points': list of (index, value) for local maxima
        - 'x_points': list of (index, value) for local minima
        - 'phase': dominant phase (radians)
    """
    n = len(f_theta)
    if theta is None:
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Extend periodically for edge handling
    f_ext = np.concatenate([f_theta[-1:], f_theta, f_theta[:1]])
    
    # Find local maxima (O-points)
    o_points = []
    for i in range(1, n+1):
        if f_ext[i] > f_ext[i-1] and f_ext[i] > f_ext[i+1]:
            o_points.append((i-1, f_ext[i]))
    
    # Find local minima (X-points)
    x_points = []
    for i in range(1, n+1):
        if f_ext[i] < f_ext[i-1] and f_ext[i] < f_ext[i+1]:
            x_points.append((i-1, f_ext[i]))
    
    # Estimate phase from first O-point
    if len(o_points) > 0:
        phase = theta[o_points[0][0]]
    else:
        phase = 0.0
    
    return {
        'o_points': o_points,
        'x_points': x_points,
        'phase': phase
    }


def compute_separatrix_width(extrema, dr, r_s=1.0):
    """
    Compute island width from extrema positions
    
    Parameters
    ----------
    extrema : dict
        Output from find_extrema()
    dr : float
        Radial grid spacing
    r_s : float, optional
        Rational surface radius (default: 1.0)
    
    Returns
    -------
    width : float
        Island width (distance between O-point and X-point)
    """
    if len(extrema['o_points']) == 0 or len(extrema['x_points']) == 0:
        return 0.0
    
    # Get flux values
    o_vals = [v for _, v in extrema['o_points']]
    x_vals = [v for _, v in extrema['x_points']]
    
    # Width estimate from flux difference
    delta_psi = max(o_vals) - min(x_vals)
    
    if delta_psi <= 0:
        return 0.0
    
    # Improved estimate using island theory
    # For tearing modes: w ≈ 4√(δψ / |ψ'|)
    # Approximate ψ' ~ r_s (for typical equilibria)
    psi_prime_approx = r_s if r_s > 0 else 1.0
    
    width = 4.0 * np.sqrt(abs(delta_psi) / psi_prime_approx)
    
    return width


def compute_island_width(psi, r, z, q_profile, m=2, n=1):
    """
    Compute magnetic island width from Poincaré section
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux on 2D grid
    r : ndarray (Nr,)
        Radial coordinates
    z : ndarray (Nz,)
        Vertical coordinates
    q_profile : ndarray (Nr,)
        Safety factor profile q(r)
    m : int, optional
        Poloidal mode number (default: 2)
    n : int, optional
        Toroidal mode number (default: 1)
    
    Returns
    -------
    w : float
        Island width
    r_s : float
        Rational surface radius
    phase : float
        Island phase (radians)
    
    Notes
    -----
    Method:
    1. Find rational surface q(r_s) = m/n
    2. Extract psi along θ at r_s
    3. Identify O-points and X-points
    4. Compute width from separatrix
    """
    # Find rational surface
    q_target = m / n
    r_s, acc = find_rational_surface(q_profile, r, q_target, method='spline')
    
    if np.isnan(r_s):
        return 0.0, np.nan, 0.0
    
    # Extract flux at rational surface
    psi_theta, theta = extract_flux_surface(psi, r, z, r_s)
    
    # Find extrema
    extrema = find_extrema(psi_theta, theta)
    
    # Compute width
    dr = r[1] - r[0]
    w = compute_separatrix_width(extrema, dr, r_s)
    
    return w, r_s, extrema['phase']


def compute_helical_flux(psi, r, z, m=2, n=1):
    """
    Compute helical flux δψ_mn using Fourier decomposition
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    r : ndarray (Nr,)
        Radial coordinates
    z : ndarray (Nz,)
        Vertical coordinates
    m : int, optional
        Poloidal mode number (default: 2)
    n : int, optional
        Toroidal mode number (default: 1)
    
    Returns
    -------
    delta_psi : complex
        Helical flux amplitude δψ_mn
    
    Notes
    -----
    δψ_mn = ∫ ψ(r,θ) * exp(-i m θ) dθ
    
    Island width: w ≈ 4√(|δψ_mn|/|ψ'(r_s)|)
    """
    Nr, Nz = psi.shape
    n_theta = 256
    
    delta_psi_profile = np.zeros(Nr, dtype=complex)
    
    for i, r_val in enumerate(r):
        # Extract flux at this radius
        psi_theta, theta = extract_flux_surface(psi, r, z, r_val, n_theta)
        
        # Fourier coefficient
        integrand = psi_theta * np.exp(-1j * m * theta)
        delta_psi_profile[i] = np.trapz(integrand, theta) / (2*np.pi)
    
    # Return maximum amplitude
    delta_psi = delta_psi_profile[np.argmax(np.abs(delta_psi_profile))]
    
    return delta_psi
