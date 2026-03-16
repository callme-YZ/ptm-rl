"""
Diagnostics Visualization Tools

Plot tearing mode diagnostics results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_island_evolution(monitor, save_path=None, figsize=(10, 8)):
    """
    Plot island width and growth rate evolution
    
    Parameters
    ----------
    monitor : TearingModeMonitor
        Monitor instance with history data
    save_path : str, optional
        Path to save figure (default: None, display only)
    figsize : tuple, optional
        Figure size (default: (10, 8))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Island width evolution
    ax1.plot(monitor.t_history, monitor.w_history, 'b-', 
             linewidth=2, label='Island width')
    
    # Exponential fit if enough data
    if len(monitor.gamma_history) > 0:
        gamma_avg, sigma = monitor.get_latest_gamma()
        
        if not np.isnan(gamma_avg) and len(monitor.t_history) > 0:
            t_fit = np.array(monitor.t_history)
            w0 = monitor.w_history[0] if monitor.w_history[0] > 0 else 1e-6
            w_fit = w0 * np.exp(gamma_avg * (t_fit - t_fit[0]))
            
            ax1.plot(t_fit, w_fit, 'r--', linewidth=2, alpha=0.7,
                    label=f'γ = {gamma_avg:.4f} ± {sigma:.4f}')
    
    ax1.set_ylabel('Island width', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Tearing Mode Evolution (m={monitor.m}, n={monitor.n})', 
                  fontsize=14, fontweight='bold')
    
    # Growth rate evolution
    if len(monitor.gamma_history) > 0:
        # Get time stamps for gamma (offset by window size)
        t_gamma = monitor.t_history[monitor.gamma_window:]
        
        ax2.plot(t_gamma, monitor.gamma_history, 'g-', 
                linewidth=2, label='Growth rate')
        
        # Add uncertainty band
        gamma_arr = np.array(monitor.gamma_history)
        sigma_arr = np.array(monitor.sigma_history)
        
        ax2.fill_between(t_gamma, 
                         gamma_arr - 2*sigma_arr,
                         gamma_arr + 2*sigma_arr,
                         alpha=0.3, color='g', label='2σ band')
        
        # Zero line
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.set_ylabel('Growth rate γ', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for growth rate',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color='gray')
        ax2.set_xlabel('Time', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_poincare_section(psi, r, z, r_s=None, levels=20, figsize=(8, 6)):
    """
    Plot Poincaré section (flux contours)
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    r : ndarray (Nr,)
        Radial coordinates
    z : ndarray (Nz,)
        Vertical coordinates
    r_s : float, optional
        Rational surface radius to highlight (default: None)
    levels : int, optional
        Number of contour levels (default: 20)
    figsize : tuple, optional
        Figure size (default: (8, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D grid
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Contour plot
    contours = ax.contour(R, Z, psi, levels=levels, colors='b', linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8)
    
    # Highlight rational surface
    if r_s is not None:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_s * np.cos(theta), r_s * np.sin(theta), 
               'r--', linewidth=2, label=f'r_s = {r_s:.3f}')
        ax.legend(loc='best')
    
    ax.set_xlabel('R', fontsize=12)
    ax.set_ylabel('Z', fontsize=12)
    ax.set_title('Poincaré Section (Flux Contours)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_flux_surface(psi_theta, theta, extrema=None, figsize=(10, 5)):
    """
    Plot flux along a surface and mark extrema
    
    Parameters
    ----------
    psi_theta : ndarray
        Flux values along poloidal angle
    theta : ndarray
        Poloidal angles
    extrema : dict, optional
        Extrema from find_extrema() (default: None)
    figsize : tuple, optional
        Figure size (default: (10, 5))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(theta, psi_theta, 'b-', linewidth=2, label='Flux')
    
    if extrema is not None:
        # O-points
        if len(extrema['o_points']) > 0:
            o_idx = [i for i, _ in extrema['o_points']]
            o_vals = [v for _, v in extrema['o_points']]
            ax.plot(theta[o_idx], o_vals, 'ro', markersize=10, 
                   label='O-points (maxima)')
        
        # X-points
        if len(extrema['x_points']) > 0:
            x_idx = [i for i, _ in extrema['x_points']]
            x_vals = [v for _, v in extrema['x_points']]
            ax.plot(theta[x_idx], x_vals, 'gx', markersize=12, 
                   markeredgewidth=3, label='X-points (minima)')
    
    ax.set_xlabel('Poloidal angle θ', fontsize=12)
    ax.set_ylabel('Flux ψ', fontsize=12)
    ax.set_title('Flux Along Rational Surface', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2*np.pi])
    
    return fig


def plot_diagnostics_summary(monitor, psi=None, r=None, z=None, 
                             save_path=None, figsize=(14, 10)):
    """
    Comprehensive diagnostics summary plot
    
    Parameters
    ----------
    monitor : TearingModeMonitor
        Monitor instance
    psi : ndarray (Nr, Nz), optional
        Current poloidal flux for Poincaré section
    r, z : ndarray, optional
        Grid coordinates (required if psi provided)
    save_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (14, 10))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    """
    if psi is not None and (r is None or z is None):
        raise ValueError("Must provide r and z if psi is given")
    
    # Create layout
    if psi is not None:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Island width
        ax1 = fig.add_subplot(gs[0, :])
        
        # Growth rate
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Poincaré section
        ax3 = fig.add_subplot(gs[1, 1])
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        gs = None
        ax3 = None
    
    # Plot island width
    ax1.plot(monitor.t_history, monitor.w_history, 'b-', linewidth=2)
    
    if len(monitor.gamma_history) > 0:
        gamma_avg, sigma = monitor.get_latest_gamma()
        if not np.isnan(gamma_avg):
            t_fit = np.array(monitor.t_history)
            w0 = monitor.w_history[0] if monitor.w_history[0] > 0 else 1e-6
            w_fit = w0 * np.exp(gamma_avg * (t_fit - t_fit[0]))
            ax1.plot(t_fit, w_fit, 'r--', linewidth=2, alpha=0.7,
                    label=f'γ = {gamma_avg:.4f}')
    
    ax1.set_ylabel('Island width', fontsize=12)
    ax1.set_title(f'Tearing Mode m={monitor.m}, n={monitor.n}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot growth rate
    if len(monitor.gamma_history) > 0:
        t_gamma = monitor.t_history[monitor.gamma_window:]
        ax2.plot(t_gamma, monitor.gamma_history, 'g-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Growth rate γ', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Plot Poincaré section if provided
    if ax3 is not None and psi is not None:
        R, Z = np.meshgrid(r, z, indexing='ij')
        contours = ax3.contour(R, Z, psi, levels=15, colors='b', linewidths=1)
        
        # Highlight rational surface
        if len(monitor.r_s_history) > 0:
            r_s = monitor.r_s_history[-1]
            if not np.isnan(r_s):
                theta_circ = np.linspace(0, 2*np.pi, 100)
                ax3.plot(r_s * np.cos(theta_circ), r_s * np.sin(theta_circ),
                        'r--', linewidth=2, label=f'r_s={r_s:.3f}')
                ax3.legend()
        
        ax3.set_xlabel('R', fontsize=12)
        ax3.set_ylabel('Z', fontsize=12)
        ax3.set_title('Current Poincaré Section', fontsize=12)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
