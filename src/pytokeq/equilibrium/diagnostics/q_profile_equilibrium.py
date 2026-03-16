"""
Equilibrium construction from prescribed q-profile

Solves the inverse problem:
- Given: q(ψ)
- Find: ψ(r) such that resulting q matches

Method: Iterative integration for circular geometry

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Tuple, Callable


def integrate_psi_from_q(r: np.ndarray, 
                         q_of_r: np.ndarray,
                         F0: float = 1.0,
                         R0: float = 10.0) -> np.ndarray:
    """
    Integrate ψ(r) from q(r) profile.
    
    Uses: dψ/dr = (F0 r²) / (R0² q(r))
    
    Parameters
    ----------
    r : np.ndarray, shape (Nr,)
        Radial coordinate
    q_of_r : np.ndarray, shape (Nr,)
        Safety factor at each r
    F0 : float
        Toroidal field function (constant for zero-β)
    R0 : float
        Major radius
    
    Returns
    -------
    psi : np.ndarray, shape (Nr,)
        Poloidal flux
    """
    Nr = len(r)
    psi = np.zeros(Nr)
    
    # Integrate from axis (r=0) outward
    for i in range(1, Nr):
        dr = r[i] - r[i-1]
        
        # dψ/dr at current point
        # Use average of i-1 and i for better accuracy
        r_mid = 0.5 * (r[i-1] + r[i])
        q_mid = 0.5 * (q_of_r[i-1] + q_of_r[i])
        
        dpsi_dr = (F0 * r_mid**2) / (R0**2 * q_mid + 1e-15)  # Avoid division by zero
        
        psi[i] = psi[i-1] + dpsi_dr * dr
    
    return psi


def compute_q_from_psi(r: np.ndarray,
                       psi: np.ndarray,
                       F0: float = 1.0,
                       R0: float = 10.0) -> np.ndarray:
    """
    Compute q(r) from ψ(r).
    
    Uses: q(r) = (F0 r²) / (R0² dψ/dr)
    
    Parameters
    ----------
    r : np.ndarray
        Radial coordinate
    psi : np.ndarray
        Poloidal flux
    F0 : float
        Toroidal field function
    R0 : float
        Major radius
    
    Returns
    -------
    q : np.ndarray
        Safety factor
    """
    # Compute dψ/dr
    dpsi_dr = np.gradient(psi, r)
    
    # Avoid division by zero
    dpsi_dr = np.where(np.abs(dpsi_dr) < 1e-15, 1e-15, dpsi_dr)
    
    # q(r)
    q = (F0 * r**2) / (R0**2 * dpsi_dr)
    
    # Handle axis (r=0)
    if r[0] == 0 or r[0] < 1e-10:
        # At axis, use limit or neighboring value
        q[0] = q[1]
    
    return q


def construct_psi_from_q_profile(r: np.ndarray,
                                  q_profile: Callable,
                                  F0: float = 1.0,
                                  R0: float = 10.0,
                                  max_iter: int = 20,
                                  tol: float = 1e-4,
                                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Construct ψ(r) from prescribed q(ψ) profile via iteration.
    
    Algorithm:
    1. Start with ψ_n = (r/a)²
    2. Compute q(r) from q_profile(ψ_n)
    3. Integrate dψ/dr = (F0 r²)/(R0² q(r)) to get ψ_new
    4. Normalize: ψ_n,new = (ψ_new - ψ_axis)/(ψ_edge - ψ_axis)
    5. Check convergence: ||ψ_n,new - ψ_n|| < tol
    6. Repeat
    
    Parameters
    ----------
    r : np.ndarray, shape (Nr,)
        Radial coordinate (r[0] should be 0 or small)
    q_profile : Callable
        Function q(ψ_n) where ψ_n ∈ [0,1] is normalized flux
    F0 : float
        Toroidal field function
    R0 : float
        Major radius
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance on ||ψ_n||
    verbose : bool
        Print iteration info
    
    Returns
    -------
    psi : np.ndarray, shape (Nr,)
        Poloidal flux (dimensional)
    psi_n : np.ndarray, shape (Nr,)
        Normalized flux
    info : dict
        Convergence info
    """
    a = r[-1]  # Minor radius
    Nr = len(r)
    
    # Initial guess: ψ_n = (r/a)²
    psi_n = (r / a)**2
    
    if verbose:
        print("Iterative construction of ψ(r) from q(ψ)")
        print("-" * 60)
    
    residuals = []
    
    for n in range(max_iter):
        # 1. Compute q(r) from q(ψ_n)
        q_of_r = q_profile(psi_n)
        
        # 2. Integrate to get ψ
        psi = integrate_psi_from_q(r, q_of_r, F0, R0)
        
        # 3. Normalize
        psi_min = psi[0]
        psi_max = psi[-1]
        
        if abs(psi_max - psi_min) < 1e-15:
            if verbose:
                print(f"Iter {n+1}: ψ constant (no flux variation)")
            break
        
        psi_n_new = (psi - psi_min) / (psi_max - psi_min)
        
        # 4. Check convergence
        residual = np.linalg.norm(psi_n_new - psi_n) / np.sqrt(Nr)
        residuals.append(residual)
        
        if verbose:
            q_error = np.abs(q_of_r - q_profile(psi_n_new)).max()
            print(f"Iter {n+1:2d}: residual={residual:.3e}, q_error={q_error:.3e}")
        
        if residual < tol:
            if verbose:
                print("-" * 60)
                print(f"✓ Converged in {n+1} iterations")
            
            info = {
                'converged': True,
                'iterations': n + 1,
                'residual': residual,
                'residuals': residuals
            }
            
            return psi, psi_n_new, info
        
        # 5. Update for next iteration
        # Use relaxation to improve stability
        alpha = 0.7  # Relaxation parameter
        psi_n = alpha * psi_n_new + (1 - alpha) * psi_n
    
    # Did not converge
    if verbose:
        print("-" * 60)
        print(f"⚠ Not converged after {max_iter} iterations")
        print(f"  Final residual: {residuals[-1]:.3e}")
    
    info = {
        'converged': False,
        'iterations': max_iter,
        'residual': residuals[-1] if residuals else np.nan,
        'residuals': residuals
    }
    
    return psi, psi_n, info


def create_2d_psi_from_1d(r_1d: np.ndarray,
                          psi_1d: np.ndarray,
                          R_2d: np.ndarray,
                          Z_2d: np.ndarray,
                          R0: float) -> np.ndarray:
    """
    Create 2D ψ(R,Z) from 1D ψ(r) for circular equilibrium.
    
    Uses: r = sqrt((R-R0)² + Z²)
    
    Parameters
    ----------
    r_1d : np.ndarray, shape (Nr,)
        1D radial coordinate
    psi_1d : np.ndarray, shape (Nr,)
        1D flux
    R_2d : np.ndarray, shape (NR, NZ)
        2D major radius grid
    Z_2d : np.ndarray, shape (NR, NZ)
        2D vertical grid
    R0 : float
        Magnetic axis major radius
    
    Returns
    -------
    psi_2d : np.ndarray, shape (NR, NZ)
        2D flux
    """
    # Compute radial distance from axis
    r_2d = np.sqrt((R_2d - R0)**2 + Z_2d**2)
    
    # Interpolate ψ(r)
    psi_2d = np.interp(r_2d.flatten(), r_1d, psi_1d).reshape(R_2d.shape)
    
    return psi_2d


def validate_q_profile_construction(r: np.ndarray,
                                     psi: np.ndarray,
                                     psi_n: np.ndarray,
                                     q_target: Callable,
                                     F0: float = 1.0,
                                     R0: float = 10.0) -> dict:
    """
    Validate constructed equilibrium against target q-profile.
    
    Returns
    -------
    validation : dict
        {
            'q_computed': np.ndarray,
            'q_target': np.ndarray,
            'q_error_max': float,
            'q_error_rms': float,
            'passed': bool
        }
    """
    # Compute q from constructed ψ
    q_computed = compute_q_from_psi(r, psi, F0, R0)
    
    # Target q
    q_target_arr = q_target(psi_n)
    
    # Errors
    q_error = np.abs(q_computed - q_target_arr)
    q_error_max = q_error.max()
    q_error_rms = np.sqrt(np.mean(q_error**2))
    
    # Relative error
    q_rel_error = q_error / (np.abs(q_target_arr) + 1e-10)
    q_rel_error_max = q_rel_error.max()
    
    # Pass criterion: <5% max relative error
    passed = q_rel_error_max < 0.05
    
    return {
        'q_computed': q_computed,
        'q_target': q_target_arr,
        'q_error_max': q_error_max,
        'q_error_rms': q_error_rms,
        'q_rel_error_max': q_rel_error_max,
        'passed': passed
    }
