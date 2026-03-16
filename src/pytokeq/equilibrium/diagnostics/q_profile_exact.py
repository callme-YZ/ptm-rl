"""
Exact q-profile calculation using flux surface integration

q(ψ) = (1/2π) ∮ f/(R²·B_θ) dl

Reference: FreeGS method (learned 2026-03-12 10:29-10:36)
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def trace_flux_surface_simple(psi, psi_target, R, Z, ntheta=100):
    """
    Trace flux surface using simple angular sampling
    
    Assumes roughly circular/elliptical flux surfaces
    Works well for Solov'ev and similar equilibria
    
    Args:
        psi: (nr, nz) Poloidal flux
        psi_target: Target ψ value
        R, Z: (nr, nz) Grid coordinates
        ntheta: Number of angular points
        
    Returns:
        R_surface, Z_surface: (ntheta,) Points on flux surface
    """
    # Find approximate center (axis)
    interior = psi[1:-1, 1:-1]
    i_max, j_max = np.unravel_index(interior.argmax(), interior.shape)
    i_axis, j_axis = i_max + 1, j_max + 1
    R0 = R[i_axis, j_axis]
    Z0 = Z[i_axis, j_axis]
    
    # Create interpolator for ψ(R,Z)
    nr, nz = psi.shape
    R_1d = R[:, 0]
    Z_1d = Z[0, :]
    psi_interp = RectBivariateSpline(R_1d, Z_1d, psi, kx=3, ky=3)
    
    # Angular sampling
    theta = np.linspace(0, 2*np.pi, ntheta)
    
    # Estimate radius where ψ ≈ psi_target
    # Start from axis, expand radially
    psi_axis = psi[i_axis, j_axis]
    psi_edge = psi[0, :].mean()
    
    # Fraction
    frac = (psi_target - psi_axis) / (psi_edge - psi_axis)
    r_guess = frac * (R_1d.max() - R0)
    
    R_surface = np.zeros(ntheta)
    Z_surface = np.zeros(ntheta)
    
    for i, th in enumerate(theta):
        # Search radially from axis at angle th
        r_min, r_max = 0.01, R_1d.max() - R0
        
        # Binary search for r where ψ(R0+r*cos(th), Z0+r*sin(th)) = psi_target
        for _ in range(20):  # Max 20 iterations
            r_mid = 0.5 * (r_min + r_max)
            R_test = R0 + r_mid * np.cos(th)
            Z_test = Z0 + r_mid * np.sin(th)
            
            # Check bounds
            if R_test < R_1d.min() or R_test > R_1d.max():
                break
            if Z_test < Z_1d.min() or Z_test > Z_1d.max():
                break
            
            psi_test = psi_interp(R_test, Z_test)[0,0]
            
            if abs(psi_test - psi_target) < 1e-6 * abs(psi_edge - psi_axis):
                break
            
            if psi_test > psi_target:
                r_min = r_mid
            else:
                r_max = r_mid
        
        R_surface[i] = R0 + r_mid * np.cos(th)
        Z_surface[i] = Z0 + r_mid * np.sin(th)
    
    return R_surface, Z_surface


def compute_q_at_surface(R_surface, Z_surface, psi, R_grid, Z_grid, f):
    """
    Compute q on a single flux surface
    
    q = (1/2π) ∮ f/(R²·B_θ) dl
    
    Args:
        R_surface, Z_surface: (n,) Points on flux surface
        psi: (nr, nz) Poloidal flux
        R_grid, Z_grid: (nr, nz) Grid coordinates
        f: Toroidal flux function f = R*B_φ [T·m]
        
    Returns:
        q: Safety factor (dimensionless)
    """
    # Compute B_θ = |∇ψ| / R at each surface point
    
    nr, nz = psi.shape
    R_1d = R_grid[:, 0]
    Z_1d = Z_grid[0, :]
    dR = R_1d[1] - R_1d[0]
    dZ = Z_1d[1] - Z_1d[0]
    
    # Compute gradients on grid
    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)
    
    # Create interpolators
    dpsi_dR_interp = RectBivariateSpline(R_1d, Z_1d, dpsi_dR, kx=1, ky=1)
    dpsi_dZ_interp = RectBivariateSpline(R_1d, Z_1d, dpsi_dZ, kx=1, ky=1)
    
    # Evaluate at surface points
    dpsi_dR_surf = np.array([dpsi_dR_interp(R, Z)[0,0] 
                             for R, Z in zip(R_surface, Z_surface)])
    dpsi_dZ_surf = np.array([dpsi_dZ_interp(R, Z)[0,0] 
                             for R, Z in zip(R_surface, Z_surface)])
    
    # |∇ψ|
    grad_psi_mag = np.sqrt(dpsi_dR_surf**2 + dpsi_dZ_surf**2)
    
    # B_θ = |∇ψ| / R
    B_theta = grad_psi_mag / R_surface
    B_theta = np.maximum(B_theta, 1e-10)  # Avoid division by zero
    
    # Integrand: f / (R² B_θ)
    integrand = f / (R_surface**2 * B_theta)
    
    # Line integral: ∮ integrand dl
    dR = np.diff(R_surface)
    dZ = np.diff(Z_surface)
    dl = np.sqrt(dR**2 + dZ**2)
    
    # Close the loop
    dR_close = R_surface[0] - R_surface[-1]
    dZ_close = Z_surface[0] - Z_surface[-1]
    dl = np.append(dl, np.sqrt(dR_close**2 + dZ_close**2))
    
    # Trapezoidal rule
    integrand_periodic = np.append(integrand, integrand[0])
    integrand_avg = 0.5 * (integrand_periodic[:-1] + integrand_periodic[1:])
    integral = np.sum(integrand_avg * dl)
    
    # q = (1/2π) × integral
    q = integral / (2 * np.pi)
    
    return q


def compute_q_profile(psi, R, Z, f, npsi=50):
    """
    Compute q-profile using exact flux surface integration
    
    Args:
        psi: (nr, nz) Poloidal flux
        R, Z: (nr, nz) Grid coordinates
        f: Toroidal flux function [T·m]
        npsi: Number of flux surfaces
        
    Returns:
        psi_norm: (npsi,) Normalized ψ [0,1]
        q: (npsi,) Safety factor
    """
    # Find axis and edge
    interior = psi[1:-1, 1:-1]
    i_max, j_max = np.unravel_index(interior.argmax(), interior.shape)
    i_axis, j_axis = i_max + 1, j_max + 1
    psi_axis = psi[i_axis, j_axis]
    psi_edge = psi[0, :].mean()
    
    # ψ values (skip very close to axis where B_θ→0)
    psi_values = np.linspace(psi_axis, psi_edge, npsi)[1:]  # Skip axis
    psi_norm_values = (psi_values - psi_axis) / (psi_edge - psi_axis)
    
    q_profile = np.zeros(len(psi_values))
    
    for i, psi_val in enumerate(psi_values):
        try:
            # Trace flux surface
            R_surf, Z_surf = trace_flux_surface_simple(
                psi, psi_val, R, Z, ntheta=100
            )
            
            # Compute q
            q_profile[i] = compute_q_at_surface(R_surf, Z_surf, psi, R, Z, f)
            
        except Exception as e:
            # Fallback
            if i > 0:
                q_profile[i] = q_profile[i-1]
            else:
                q_profile[i] = 1.0
    
    # Add axis value (extrapolate from first two)
    if len(q_profile) >= 2:
        q_axis = 2*q_profile[0] - q_profile[1]
    else:
        q_axis = q_profile[0]
    
    q_full = np.concatenate([[q_axis], q_profile])
    psi_norm_full = np.concatenate([[0.0], psi_norm_values])
    
    return psi_norm_full, q_full


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, '..')
    from picard_gs_solver import Grid
    
    print("Testing exact q-profile calculation...")
    
    # Simple circular test
    R_1d = np.linspace(1.0, 2.0, 33)
    Z_1d = np.linspace(-0.5, 0.5, 33)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    R0, Z0 = 1.5, 0.0
    r = np.sqrt((grid.R - R0)**2 + (grid.Z - Z0)**2)
    psi = -(r**2)
    
    f = R0 * 1.0
    
    psi_norm, q = compute_q_profile(psi, grid.R, grid.Z, f, npsi=20)
    
    print(f"\nCircular test:")
    print(f"  q[0] (axis): {q[0]:.3f}")
    print(f"  q[-1] (edge): {q[-1]:.3f}")
    print(f"  Monotonic? {np.all(np.diff(q) >= 0)}")
    
    if q[-1] > q[0] and np.all(np.diff(q) >= 0):
        print("  ✓ PASS: q increasing")
    else:
        print("  ✗ FAIL")

