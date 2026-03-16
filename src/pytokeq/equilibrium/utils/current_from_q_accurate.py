#!/usr/bin/env python3
"""
Accurate Current Computation from q-Profile

Implements QSOLVER method (Eq 9-11) to compute J_φ(ψ) from prescribed q(ψ)
using flux surface averaging.

This is the ACCURATE implementation, not the simplified J~q₀/q approximation.

Reference: DeLucia, Jardin, Todd (1980) Sections II-III
Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Callable
from surface_averaging import compute_R_dpsi_dR_avg

# Physical constants
MU0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]


def compute_I_pol_from_q(psi: np.ndarray,
                         R: np.ndarray,
                         Z: np.ndarray,
                         q_profile: Callable,
                         n_levels: int = 50) -> tuple:
    """
    Compute poloidal current I_pol(ψ) from target q-profile
    
    Uses QSOLVER Eq 9a:
    I_pol(ψ) = (2π)² ⟨R·∂ψ/∂R⟩_ψ / [μ₀ q(ψ)]
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux (not necessarily normalized)
    R, Z : ndarray
        Grid coordinates
    q_profile : callable(psi_normalized) -> q
        Target safety factor as function of normalized flux
    n_levels : int
        Number of flux surfaces to sample
        
    Returns
    -------
    psi_levels : ndarray (n_levels,)
        Flux values where I_pol computed
    I_pol : ndarray (n_levels,)
        Poloidal current enclosed by each surface [A]
    """
    # Flux range
    psi_min = np.min(psi)
    psi_max = np.max(psi)
    
    # Sample flux surfaces (skip psi_min to avoid axis singularity)
    psi_levels = np.linspace(psi_min, psi_max, n_levels)
    psi_levels[0] = psi_min + (psi_max - psi_min) * 0.01  # Slight offset from axis
    
    # Normalize for q-profile input
    psi_n_levels = (psi_levels - psi_min) / (psi_max - psi_min + 1e-12)
    
    # Compute I_pol at each level
    I_pol = np.zeros(n_levels)
    
    for k, (psi_lvl, psi_n) in enumerate(zip(psi_levels, psi_n_levels)):
        # Surface average ⟨R·∂ψ/∂R⟩
        R_dpsi_dR_avg = compute_R_dpsi_dR_avg(psi, R, Z, psi_lvl)
        
        if R_dpsi_dR_avg <= 0:
            # Invalid surface or outside domain
            I_pol[k] = 0
            continue
        
        # Target q at this flux
        q_target = q_profile(psi_n)
        
        if q_target <= 0:
            # Unphysical q
            I_pol[k] = 0
            continue
        
        # QSOLVER Eq 9a
        I_pol[k] = (2 * np.pi)**2 * R_dpsi_dR_avg / (MU0 * q_target)
    
    return psi_levels, I_pol


def compute_current_from_q_accurate(psi: np.ndarray,
                                    R: np.ndarray,
                                    Z: np.ndarray,
                                    q_profile: Callable,
                                    n_levels: int = 50) -> np.ndarray:
    """
    Compute current density J_φ(R,Z) from target q-profile
    
    Uses QSOLVER method:
    1. Compute I_pol(ψ) from q(ψ) via surface averaging
    2. Differentiate: dI_pol/dψ
    3. Map to grid: J_φ = (1/2πR) dI_pol/dψ
    
    Parameters
    ----------
    psi : ndarray (Nr, Nz)
        Poloidal flux
    R, Z : ndarray
        Grid coordinates
    q_profile : callable(psi_normalized) -> q
        Target safety factor
    n_levels : int
        Number of flux surfaces (accuracy vs speed tradeoff)
        
    Returns
    -------
    J_phi : ndarray (Nr, Nz)
        Toroidal current density [A/m²]
    """
    # Step 1: Compute I_pol(ψ)
    psi_levels, I_pol = compute_I_pol_from_q(psi, R, Z, q_profile, n_levels)
    
    # Step 2: Compute dI_pol/dψ
    dI_dpsi = np.gradient(I_pol, psi_levels)
    
    # Step 3: Create interpolator for dI_pol/dψ
    dI_dpsi_func = interp1d(
        psi_levels, dI_dpsi,
        kind='cubic',
        bounds_error=False,
        fill_value=0  # Zero outside plasma
    )
    
    # Step 4: Evaluate on grid
    psi_flat = psi.flatten()
    dI_dpsi_grid = dI_dpsi_func(psi_flat).reshape(psi.shape)
    
    # Step 5: Convert to current density
    # J_φ = (1/2πR) dI_pol/dψ
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    J_phi = dI_dpsi_grid / (2 * np.pi * RR)
    
    # Zero outside plasma (ψ_normalized > 0.99)
    psi_min = np.min(psi)
    psi_max = np.max(psi)
    psi_normalized = (psi - psi_min) / (psi_max - psi_min + 1e-12)
    J_phi[psi_normalized > 0.99] = 0
    
    return J_phi


def verify_q_from_current(psi: np.ndarray,
                          R: np.ndarray,
                          Z: np.ndarray,
                          J_phi: np.ndarray,
                          q_profile_target: Callable,
                          n_check: int = 10) -> dict:
    """
    Verify that computed J_φ reproduces target q-profile
    
    Back-computes q from J_φ and compares with target
    
    Parameters
    ----------
    psi : ndarray
        Poloidal flux
    R, Z : ndarray
        Grid
    J_phi : ndarray
        Current density (from compute_current_from_q_accurate)
    q_profile_target : callable
        Target q-profile function
    n_check : int
        Number of flux surfaces to check
        
    Returns
    -------
    diagnostics : dict
        q_error_mean, q_error_max, q_computed, q_target at sample points
    """
    # Sample flux surfaces
    psi_min = np.min(psi)
    psi_max = np.max(psi)
    psi_check = np.linspace(psi_min + 0.05*(psi_max-psi_min), 
                            psi_max - 0.05*(psi_max-psi_min), 
                            n_check)
    psi_n_check = (psi_check - psi_min) / (psi_max - psi_min)
    
    # Target q
    q_target = np.array([q_profile_target(pn) for pn in psi_n_check])
    
    # Back-compute q from J_phi
    # q = (2π)² ⟨R∂ψ/∂R⟩ / [μ₀ I_pol]
    # where I_pol = ∫∫ J_φ dA
    
    q_computed = np.zeros(n_check)
    
    for k, psi_lvl in enumerate(psi_check):
        # Surface average
        from surface_averaging import compute_R_dpsi_dR_avg
        R_dpsi_dR_avg = compute_R_dpsi_dR_avg(psi, R, Z, psi_lvl)
        
        # Compute I_pol by integrating J_φ inside this surface
        # Simple approximation: sum J_φ where ψ < psi_lvl
        mask = (psi < psi_lvl)
        
        # Grid cell area
        dR = R[1] - R[0]
        dZ = Z[1] - Z[0]
        dA = dR * dZ
        
        # I_pol ≈ ∫∫ J_φ dA
        I_pol_computed = np.sum(J_phi[mask]) * dA * 2 * np.pi  # Factor 2π for toroidal integration
        
        if I_pol_computed > 1e-10 and R_dpsi_dR_avg > 0:
            q_computed[k] = (2*np.pi)**2 * R_dpsi_dR_avg / (MU0 * I_pol_computed)
        else:
            q_computed[k] = 0
    
    # Error
    valid = (q_computed > 0) & (q_target > 0)
    if np.any(valid):
        q_error = np.abs(q_computed[valid] - q_target[valid]) / q_target[valid]
        q_error_mean = np.mean(q_error)
        q_error_max = np.max(q_error)
    else:
        q_error_mean = np.inf
        q_error_max = np.inf
    
    return {
        'q_computed': q_computed,
        'q_target': q_target,
        'q_error_mean': q_error_mean,
        'q_error_max': q_error_max,
        'psi_normalized': psi_n_check
    }


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/reduced-mhd')
    
    print("="*70)
    print("Accurate Current from q-Profile - Validation Test")
    print("="*70)
    
    # Create test equilibrium
    Nr, Nz = 64, 64
    R = np.linspace(0.5, 1.5, Nr)
    Z = np.linspace(-0.5, 0.5, Nz)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    R0 = 1.0
    a = 0.4
    r = np.sqrt((RR - R0)**2 + ZZ**2)
    psi = (r / a)**2
    psi = np.clip(psi, 0, 1)
    
    print(f"\nTest grid: {Nr}×{Nz}")
    
    # Test 1: Constant q = 2.0
    print("\n" + "="*70)
    print("Test 1: Constant q = 2.0")
    print("="*70)
    
    def q_const(psi_n):
        return 2.0 * np.ones_like(psi_n)
    
    J_phi = compute_current_from_q_accurate(psi, R, Z, q_const, n_levels=30)
    
    print(f"J_φ computed:")
    print(f"  Range: [{np.min(J_phi):.3e}, {np.max(J_phi):.3e}] A/m²")
    print(f"  Total current: {np.sum(J_phi) * (R[1]-R[0]) * (Z[1]-Z[0]) * 2*np.pi:.3e} A")
    
    # Verify q
    diag = verify_q_from_current(psi, R, Z, J_phi, q_const, n_check=5)
    
    print(f"\nq-profile verification:")
    print(f"  Target: q = 2.0 (constant)")
    print(f"  Computed: q = {np.mean(diag['q_computed'][diag['q_computed']>0]):.3f} ± {np.std(diag['q_computed'][diag['q_computed']>0]):.3f}")
    print(f"  Mean error: {diag['q_error_mean']*100:.2f}%")
    print(f"  Max error: {diag['q_error_max']*100:.2f}%")
    
    if diag['q_error_mean'] < 0.05:
        print("  ✓ PASS (error < 5%)")
    else:
        print("  ⚠ FAIL (error > 5%)")
    
    print("\n" + "="*70)
    print("✅ Test Complete")
    print("="*70)
