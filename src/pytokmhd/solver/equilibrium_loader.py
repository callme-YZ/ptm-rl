"""
PyTokEq Equilibrium Loader for PyTokMHD

Loads PyTokEq equilibrium data and interpolates to MHD grid.

Phase 2 Component - PyTokEq Integration
Author: 小P ⚛️
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple, Optional
import pickle


def load_pytokeq_equilibrium(
    eq_file_path: str,
    target_grid: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Load PyTokEq equilibrium and interpolate to MHD grid
    
    Args:
        eq_file_path: Path to PyTokEq output (npz or pickle)
        target_grid: MHD grid (r_mhd, z_mhd) arrays
        
    Returns:
        Dictionary containing:
            - psi_eq: Equilibrium flux on MHD grid (Nr, Nz)
            - j_eq: Equilibrium current on MHD grid (Nr, Nz)
            - p_eq: Equilibrium pressure on MHD grid (Nr, Nz)
            - q_profile: Safety factor profile (Nr,)
            - R_axis: Magnetic axis R coordinate
            - Z_axis: Magnetic axis Z coordinate
    """
    r_mhd, z_mhd = target_grid
    
    # Load equilibrium data
    if eq_file_path.endswith('.npz'):
        eq_data = np.load(eq_file_path)
        equilibrium = {
            'psi': eq_data['psi'],
            'j_tor': eq_data['j_tor'],
            'pressure': eq_data['pressure'],
            'q_profile': eq_data['q_profile'],
            'r': eq_data['r'],
            'z': eq_data['z'],
            'R_axis': float(eq_data['R_axis']),
            'Z_axis': float(eq_data['Z_axis'])
        }
    elif eq_file_path.endswith('.pkl') or eq_file_path.endswith('.pickle'):
        with open(eq_file_path, 'rb') as f:
            equilibrium = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {eq_file_path}")
    
    # Extract PyTokEq grid
    r_eq = equilibrium['r']
    z_eq = equilibrium['z']
    
    # Interpolate psi
    psi_eq_mhd = interpolate_equilibrium(
        equilibrium['psi'], r_eq, z_eq, r_mhd, z_mhd
    )
    
    # Interpolate j_tor
    j_eq_mhd = interpolate_equilibrium(
        equilibrium['j_tor'], r_eq, z_eq, r_mhd, z_mhd
    )
    
    # Interpolate pressure
    p_eq_mhd = interpolate_equilibrium(
        equilibrium['pressure'], r_eq, z_eq, r_mhd, z_mhd
    )
    
    return {
        'psi_eq': psi_eq_mhd,
        'j_eq': j_eq_mhd,
        'p_eq': p_eq_mhd,
        'q_profile': equilibrium['q_profile'],
        'R_axis': equilibrium['R_axis'],
        'Z_axis': equilibrium['Z_axis']
    }


def interpolate_equilibrium(
    field_eq: np.ndarray,
    r_eq: np.ndarray,
    z_eq: np.ndarray,
    r_mhd: np.ndarray,
    z_mhd: np.ndarray
) -> np.ndarray:
    """
    2D interpolation from PyTokEq grid to MHD grid
    
    Uses scipy RegularGridInterpolator with cubic method
    for smooth, accurate interpolation.
    
    Args:
        field_eq: Field on PyTokEq grid (Nr_eq, Nz_eq)
        r_eq: PyTokEq radial grid (Nr_eq,)
        z_eq: PyTokEq vertical grid (Nz_eq,)
        r_mhd: MHD radial grid (Nr,)
        z_mhd: MHD vertical grid (Nz,)
        
    Returns:
        field_mhd: Field interpolated to MHD grid (Nr, Nz)
    """
    # Create interpolator
    interp = RegularGridInterpolator(
        (r_eq, z_eq),
        field_eq,
        method='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    
    # Create MHD grid meshgrid
    R_mhd, Z_mhd = np.meshgrid(r_mhd, z_mhd, indexing='ij')
    
    # Interpolate
    points = np.stack([R_mhd.ravel(), Z_mhd.ravel()], axis=-1)
    field_mhd = interp(points).reshape(R_mhd.shape)
    
    return field_mhd


def compute_interpolation_error(
    field_original: np.ndarray,
    field_interp: np.ndarray,
    r_eq: np.ndarray,
    z_eq: np.ndarray,
    r_mhd: np.ndarray,
    z_mhd: np.ndarray
) -> float:
    """
    Compute relative interpolation error
    
    Re-samples interpolated field back to original grid
    and compares with original.
    
    Args:
        field_original: Original field on PyTokEq grid
        field_interp: Interpolated field on MHD grid
        r_eq, z_eq: PyTokEq grid
        r_mhd, z_mhd: MHD grid
        
    Returns:
        max_relative_error: Maximum relative error
    """
    # Interpolate back to original grid
    interp_back = RegularGridInterpolator(
        (r_mhd, z_mhd),
        field_interp,
        method='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    
    R_eq, Z_eq = np.meshgrid(r_eq, z_eq, indexing='ij')
    points = np.stack([R_eq.ravel(), Z_eq.ravel()], axis=-1)
    field_back = interp_back(points).reshape(R_eq.shape)
    
    # Compute relative error in significant regions
    field_max = np.max(np.abs(field_original))
    mask = np.abs(field_original) > 0.01 * field_max
    
    abs_error = np.abs(field_back - field_original)
    
    if np.sum(mask) > 0:
        rel_error = abs_error[mask] / (np.abs(field_original[mask]) + 1e-10)
        return np.nanmax(rel_error)
    else:
        # If no significant regions, return normalized absolute error
        return np.max(abs_error) / (field_max + 1e-10)
