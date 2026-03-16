"""
Boundary Conditions for MHD

Handles boundary conditions for cylindrical MHD solver.

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np


def apply_dirichlet_boundary(psi, omega, r_idx_boundary=None):
    """
    Apply Dirichlet boundary conditions: ψ = 0 at boundaries.
    
    Typically used at conducting wall (r = Lr).
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    r_idx_boundary : int or list, optional
        Radial indices to enforce BC (default: last index)
    
    Returns
    -------
    psi : np.ndarray (Nr, Nz)
        Modified flux
    omega : np.ndarray (Nr, Nz)
        Modified vorticity (unchanged in this BC)
    """
    if r_idx_boundary is None:
        r_idx_boundary = [-1]  # Outer boundary
    
    if not isinstance(r_idx_boundary, (list, tuple)):
        r_idx_boundary = [r_idx_boundary]
    
    for idx in r_idx_boundary:
        psi[idx, :] = 0.0
    
    return psi, omega


def apply_axis_boundary(psi, omega):
    """
    Apply regularity condition at axis (r = 0).
    
    For cylindrical symmetry:
    - ∂ψ/∂r|_{r=0} = 0 (regularity)
    - ω can be non-zero
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    
    Returns
    -------
    psi : np.ndarray (Nr, Nz)
        Modified flux (with regularity enforced)
    omega : np.ndarray (Nr, Nz)
        Modified vorticity
    """
    # Enforce ∂ψ/∂r = 0 at r=0 using one-sided difference
    # psi[1] - psi[0] ≈ 0  =>  psi[0] = psi[1]
    psi[0, :] = psi[1, :]
    
    # Similarly for omega (if needed)
    omega[0, :] = omega[1, :]
    
    return psi, omega


def apply_periodic_z(psi, omega):
    """
    Apply periodic boundary conditions in z-direction.
    
    This is typically automatic in operators that assume periodicity,
    but can be enforced explicitly here.
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    
    Returns
    -------
    psi : np.ndarray (Nr, Nz)
        Modified flux
    omega : np.ndarray (Nr, Nz)
        Modified vorticity
    
    Notes
    -----
    No modification needed if operators already handle periodicity.
    """
    # Periodic BC: f(z=0) = f(z=Lz)
    # This is typically handled in gradient operators
    # No action needed here
    return psi, omega


def apply_combined_bc(psi, omega):
    """
    Apply standard tokamak boundary conditions:
    - Axis regularity (r=0)
    - Conducting wall (r=Lr): ψ=0
    - Periodic in z
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    
    Returns
    -------
    psi : np.ndarray (Nr, Nz)
        Modified flux
    omega : np.ndarray (Nr, Nz)
        Modified vorticity
    """
    # Axis
    psi, omega = apply_axis_boundary(psi, omega)
    
    # Wall
    psi, omega = apply_dirichlet_boundary(psi, omega, r_idx_boundary=[-1])
    
    # Periodic z (no action needed)
    psi, omega = apply_periodic_z(psi, omega)
    
    return psi, omega
