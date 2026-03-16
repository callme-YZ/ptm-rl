"""
Growth Rate Diagnostics

Measure tearing mode instability growth rates.
"""

import numpy as np


def compute_growth_rate(w_history, t_history, transient_fraction=0.2):
    """
    Compute growth rate from island width time history
    
    Parameters
    ----------
    w_history : array_like
        Island width time series
    t_history : array_like
        Time stamps
    transient_fraction : float, optional
        Fraction of initial data to skip (default: 0.2)
    
    Returns
    -------
    gamma : float
        Growth rate (1/time)
    sigma_gamma : float
        Uncertainty estimate (standard error)
    
    Notes
    -----
    Fits exponential growth: w(t) = w_0 * exp(γ t)
    Uses log-linear regression: log(w) = log(w_0) + γ t
    """
    w_history = np.asarray(w_history)
    t_history = np.asarray(t_history)
    
    # Remove initial transient
    n_skip = int(transient_fraction * len(w_history))
    t_fit = t_history[n_skip:]
    w_fit = w_history[n_skip:]
    
    # Filter out non-positive values
    valid = w_fit > 0
    t_fit = t_fit[valid]
    w_fit = w_fit[valid]
    
    if len(w_fit) < 2:
        return np.nan, np.nan
    
    # Log-linear fit
    log_w = np.log(w_fit)
    
    # Linear regression
    gamma, log_w0 = np.polyfit(t_fit, log_w, 1)
    
    # Uncertainty estimate
    log_w_pred = log_w0 + gamma * t_fit
    residuals = log_w - log_w_pred
    sigma_gamma = np.std(residuals) / np.sqrt(len(t_fit))
    
    return gamma, sigma_gamma


def compute_energy(psi, dr, dz, r_grid):
    """
    Compute total magnetic energy
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    dr, dz : float
        Grid spacing
    r_grid : ndarray (Nr,)
        Radial coordinates
    
    Returns
    -------
    E : float
        Magnetic energy ∫ B² dV / (2μ₀)
    
    Notes
    -----
    B² ∝ |∇ψ|² in cylindrical geometry
    """
    Nr, Nz = psi.shape
    
    # Compute gradients
    dpsi_dr = np.gradient(psi, dr, axis=0)
    dpsi_dz = np.gradient(psi, dz, axis=1)
    
    # |∇ψ|² in cylindrical coordinates
    grad_psi_sq = dpsi_dr**2 + dpsi_dz**2
    
    # Volume element: dV = 2π r dr dz
    # Create 2D r array
    r_2d = r_grid[:, np.newaxis]
    
    # Energy density (ignore constants)
    energy_density = grad_psi_sq * r_2d
    
    # Integrate
    E = np.sum(energy_density) * dr * dz * 2 * np.pi
    
    return E


def compute_energy_time_derivative(psi, omega, dr, dz, r_grid):
    """
    Compute dE/dt from Ohm's law
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    omega : ndarray (Nr, Nz)
        Vorticity
    dr, dz : float
        Grid spacing
    r_grid : ndarray (Nr,)
        Radial coordinates
    
    Returns
    -------
    dE_dt : float
        Energy time derivative
    
    Notes
    -----
    From Ohm's law: dE/dt ∝ ∫ J·E dV
    """
    # Current density: J ∝ ∇²ψ (in 2D approximation)
    d2psi_dr2 = np.gradient(np.gradient(psi, dr, axis=0), dr, axis=0)
    d2psi_dz2 = np.gradient(np.gradient(psi, dz, axis=1), dz, axis=1)
    
    laplacian_psi = d2psi_dr2 + d2psi_dz2
    
    # Electric field: E ∝ ∂ψ/∂t
    # In steady state with resistivity: ∂ψ/∂t ∝ -∇²ψ
    # Use omega as proxy for time derivative
    
    r_2d = r_grid[:, np.newaxis]
    
    # J·E integrand
    integrand = laplacian_psi * omega * r_2d
    
    dE_dt = np.sum(integrand) * dr * dz * 2 * np.pi
    
    return dE_dt


def energy_growth_rate(psi, omega, dr, dz, r_grid):
    """
    Compute growth rate from energy evolution
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    omega : ndarray (Nr, Nz)
        Vorticity
    dr, dz : float
        Grid spacing
    r_grid : ndarray (Nr,)
        Radial coordinates
    
    Returns
    -------
    gamma : float
        Growth rate γ = (1/2E) dE/dt
    
    Notes
    -----
    For exponential growth: dE/dt = 2γE
    """
    E = compute_energy(psi, dr, dz, r_grid)
    dE_dt = compute_energy_time_derivative(psi, omega, dr, dz, r_grid)
    
    if E == 0:
        return 0.0
    
    gamma = 0.5 * dE_dt / E
    
    return gamma


def compute_growth_rate_sliding_window(w_history, t_history, window_size=50):
    """
    Compute time-varying growth rate using sliding window
    
    Parameters
    ----------
    w_history : array_like
        Island width time series
    t_history : array_like
        Time stamps
    window_size : int, optional
        Window size for local fit (default: 50)
    
    Returns
    -------
    gamma_history : ndarray
        Growth rate time series
    t_gamma : ndarray
        Time stamps for gamma (window centers)
    sigma_history : ndarray
        Uncertainty estimates
    """
    w_history = np.asarray(w_history)
    t_history = np.asarray(t_history)
    
    n = len(w_history)
    if n < window_size:
        return np.array([]), np.array([]), np.array([])
    
    gamma_history = []
    sigma_history = []
    t_gamma = []
    
    for i in range(window_size, n):
        w_window = w_history[i-window_size:i]
        t_window = t_history[i-window_size:i]
        
        gamma, sigma = compute_growth_rate(w_window, t_window, 
                                           transient_fraction=0.0)
        
        if not np.isnan(gamma):
            gamma_history.append(gamma)
            sigma_history.append(sigma)
            t_gamma.append(t_window[-1])
    
    return np.array(gamma_history), np.array(t_gamma), np.array(sigma_history)
