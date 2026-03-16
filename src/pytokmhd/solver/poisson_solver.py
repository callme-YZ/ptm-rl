"""
Poisson Solver for MHD

FFT-based solver for ∇²φ = f in cylindrical coordinates.
Adapted from PyTearRL implementation.

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np
from scipy.fft import fft2, ifft2


def solve_poisson(rhs, dr, dz, r_grid, rhs_sign=1.0):
    """
    Solve ∇²φ = rhs_sign * rhs using FFT.
    
    In cylindrical coordinates:
    ∇²φ = ∂²φ/∂r² + (1/r)∂φ/∂r + ∂²φ/∂z²
    
    This is solved in Fourier space (z-direction) and finite difference (r-direction).
    
    Parameters
    ----------
    rhs : np.ndarray (Nr, Nz)
        Right-hand side of Poisson equation
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    rhs_sign : float, optional
        Sign of RHS (default 1.0, use -1.0 for ∇²φ = -ω)
    
    Returns
    -------
    phi : np.ndarray (Nr, Nz)
        Solution to Poisson equation
    
    Notes
    -----
    Boundary conditions:
    - φ(r=0) = finite (regularity)
    - φ(r=Lr) = 0 (Dirichlet)
    - Periodic in z
    
    Algorithm:
    1. FFT in z-direction
    2. Solve tridiagonal system in r for each k_z mode
    3. IFFT back to real space
    """
    Nr, Nz = rhs.shape
    Lz = dz * Nz
    
    # FFT in z-direction
    rhs_k = fft2(rhs_sign * rhs, axes=(1,))
    phi_k = np.zeros_like(rhs_k, dtype=complex)
    
    # Wave numbers in z
    kz = 2*np.pi * np.fft.fftfreq(Nz, d=dz)
    
    # Solve for each mode
    for j in range(Nz):
        kz_j = kz[j]
        
        # Build tridiagonal matrix for radial equation:
        # ∂²φ/∂r² + (1/r)∂φ/∂r - kz²φ = rhs_k[j]
        
        # Coefficients for finite difference
        # φ_{i-1} * a_i + φ_i * b_i + φ_{i+1} * c_i = rhs_k[i, j]
        
        a = np.zeros(Nr)
        b = np.zeros(Nr)
        c = np.zeros(Nr)
        rhs_vec = rhs_k[:, j].copy()
        
        # Interior points (i = 1, ..., Nr-2)
        for i in range(1, Nr-1):
            r_i = r_grid[i, 0]  # r value at grid point i
            
            # Centered difference for ∂²/∂r²
            a[i] = 1.0 / dr**2 - 1.0 / (2*r_i*dr)  # Coefficient of φ_{i-1}
            c[i] = 1.0 / dr**2 + 1.0 / (2*r_i*dr)  # Coefficient of φ_{i+1}
            b[i] = -2.0 / dr**2 - kz_j**2          # Coefficient of φ_i
        
        # Boundary: r=0 (axis) - regularity condition
        # Use L'Hôpital: ∂φ/∂r = 0 at r=0
        b[0] = -2.0 / dr**2 - kz_j**2
        c[0] = 2.0 / dr**2  # φ[1] - φ[-1] = 0  =>  c[0] = 2/dr²
        
        # Boundary: r=Lr (wall) - Dirichlet φ=0
        b[-1] = 1.0
        a[-1] = 0.0
        c[-1] = 0.0
        rhs_vec[-1] = 0.0
        
        # Solve tridiagonal system
        phi_k[:, j] = solve_tridiagonal(a, b, c, rhs_vec)
    
    # IFFT back to real space
    phi = np.real(ifft2(phi_k, axes=(1,)))
    
    return phi


def solve_tridiagonal(a, b, c, d):
    """
    Solve tridiagonal system: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i].
    
    Uses Thomas algorithm (O(N) complexity).
    
    Parameters
    ----------
    a : np.ndarray (N,)
        Lower diagonal (a[0] is ignored)
    b : np.ndarray (N,)
        Main diagonal
    c : np.ndarray (N,)
        Upper diagonal (c[-1] is ignored)
    d : np.ndarray (N,)
        Right-hand side
    
    Returns
    -------
    x : np.ndarray (N,)
        Solution
    """
    N = len(d)
    c_prime = np.zeros(N, dtype=complex)
    d_prime = np.zeros(N, dtype=complex)
    x = np.zeros(N, dtype=complex)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, N):
        denom = b[i] - a[i] * c_prime[i-1]
        if i < N-1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
    
    # Back substitution
    x[-1] = d_prime[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x
