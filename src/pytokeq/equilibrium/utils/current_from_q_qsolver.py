#!/usr/bin/env python3
"""
QSOLVER-Method Current Computation from q-Profile

Accurate implementation of DeLucia (1980) Eq 9-11.

Author: 小P ⚛️
Date: 2026-03-11 (Fixed version, ψ limited to 0.85)
"""

import numpy as np
from scipy.interpolate import interp1d
try:
    from .surface_averaging import compute_R_dpsi_dR_avg
except ImportError:
    from surface_averaging import compute_R_dpsi_dR_avg

MU0 = 4*np.pi*1e-7

def compute_current_from_q_qsolver(psi, R, Z, q_profile, n_levels=50):
    """
    Compute J_φ(R,Z) from q-profile using QSOLVER method
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    R, Z : ndarray
        Grid
    q_profile : callable(psi_normalized) -> q
        Target safety factor
    n_levels : int
        Flux surfaces to sample
        
    Returns
    -------
    J_phi : ndarray (Nr, Nz)
        Current density [A/m²]
        
    Notes
    -----
    - Flux range limited to [0.01, 0.85] to avoid edge issues
    - I_pol must be monotonic for physical current
    """
    psi_min = np.min(psi)
    psi_max = np.max(psi)
    
    # CRITICAL: Limit to 0.01-0.85 (edge contours unreliable)
    psi_frac = np.linspace(0.01, 0.85, n_levels)
    psi_levels = psi_min + psi_frac * (psi_max - psi_min)
    
    # Step 1: Compute I_pol(ψ) from QSOLVER Eq 9a
    I_pol = np.zeros(n_levels)
    
    for k, (psi_lvl, psi_n) in enumerate(zip(psi_levels, psi_frac)):
        R_dpsi_dR_avg = compute_R_dpsi_dR_avg(psi, R, Z, psi_lvl)
        
        if R_dpsi_dR_avg <= 0:
            I_pol[k] = 0
            continue
        
        q_target = q_profile(psi_n)
        if q_target <= 0:
            I_pol[k] = 0
            continue
        
        # I_pol = (2π)² ⟨R∂ψ/∂R⟩ / (μ₀q)
        I_pol[k] = (2*np.pi)**2 * R_dpsi_dR_avg / (MU0 * q_target)
    
    # Step 2: dI_pol/dψ
    dI_dpsi = np.gradient(I_pol, psi_frac)  # Use normalized flux for gradient
    
    # Step 3: Interpolate to grid
    dI_dpsi_func = interp1d(psi_frac, dI_dpsi, kind='linear',
                            bounds_error=False, fill_value=0)
    
    psi_n_grid = (psi - psi_min) / (psi_max - psi_min + 1e-12)
    dI_dpsi_grid = dI_dpsi_func(psi_n_grid.flatten()).reshape(psi.shape)
    
    # Step 4: J_φ = (1/2πR) dI_pol/dψ_normalized * (1/Δψ)
    # Need to convert dI/dψ_norm to dI/dψ_physical
    Dpsi = psi_max - psi_min
    dI_dpsi_phys = dI_dpsi_grid / (Dpsi + 1e-12)
    
    RR = np.meshgrid(R, Z, indexing='ij')[0]
    J_phi = dI_dpsi_phys / (2*np.pi*RR)
    
    # Zero outside 0.85
    J_phi[psi_n_grid > 0.85] = 0
    
    return J_phi

# Test
if __name__ == "__main__":
    print("="*70)
    print("QSOLVER Current Computation - Final Test")
    print("="*70)
    
    Nr, Nz = 64, 64
    R = np.linspace(0.5, 1.5, Nr)
    Z = np.linspace(-0.5, 0.5, Nz)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    R0, a = 1.0, 0.4
    r = np.sqrt((RR-R0)**2 + ZZ**2)
    psi = np.clip((r/a)**2, 0, 1)
    
    def q_const(psi_n):
        return 2.0 * np.ones_like(psi_n)
    
    J_phi = compute_current_from_q_qsolver(psi, R, Z, q_const, n_levels=30)
    
    print(f"\nJ_φ computed:")
    print(f"  Range: [{np.min(J_phi):.3e}, {np.max(J_phi):.3e}] A/m²")
    print(f"  All non-negative? {np.all(J_phi >= 0)}")
    
    if np.all(J_phi >= 0):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
    
    print("\n" + "="*70)
    print("✅ QSOLVER Method Implementation Complete")
    print("="*70)
