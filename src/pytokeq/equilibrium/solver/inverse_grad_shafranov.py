"""
Inverse Grad-Shafranov Solver

从目标q-profile构建force-balanced equilibrium

Algorithm: Picard iteration
  1. Prescribe J(ψ) = J0 × (q0/q_target(ψ))
  2. Solve Δ*ψ = -μ0 R J(ψ)
  3. Iterate until q(ψ) matches target

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Callable, Tuple, Optional
from scipy.sparse.linalg import spsolve
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.operators import build_grad_shafranov_operator, apply_grad_shafranov_operator


def normalize_flux(psi: np.ndarray) -> np.ndarray:
    """
    归一化flux到[0,1]
    
    Convention:
      ψ_n = 0 at magnetic axis (psi_min)
      ψ_n = 1 at plasma edge (psi_max)
    """
    psi_min = psi.min()
    psi_max = psi.max()
    
    psi_n = (psi - psi_min) / (psi_max - psi_min + 1e-15)
    
    return psi_n


def compute_q_from_psi_2d(psi: np.ndarray,
                          R: np.ndarray,
                          Z: np.ndarray,
                          F0: float = 1.0,
                          R0: float = 10.0,
                          q0: float = 1.75) -> np.ndarray:
    """
    从2D ψ计算safety factor q
    
    Uses cylindrical approximation:
      q(r) ≈ (F0 r²) / (R0² dψ/dr)
    
    Parameters
    ----------
    psi : np.ndarray, shape (Nr, Nz)
        Poloidal flux
    R, Z : np.ndarray
        Grid arrays
    F0 : float
        Toroidal field function
    R0 : float
        Major radius
    q0 : float
        Central q (for axis handling)
    
    Returns
    -------
    q : np.ndarray, shape (Nr, Nz)
        Safety factor
    """
    Nr, Nz = psi.shape
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    
    # Compute gradients
    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)
    
    # 2D grids
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Poloidal coordinates (r, θ from axis)
    r = np.sqrt((R_2d - R0)**2 + Z_2d**2)
    
    # Radial derivative: dψ/dr = ∂ψ/∂R cos(θ) + ∂ψ/∂Z sin(θ)
    cos_theta = (R_2d - R0) / (r + 1e-15)
    sin_theta = Z_2d / (r + 1e-15)
    
    dpsi_dr = dpsi_dR * cos_theta + dpsi_dZ * sin_theta
    
    # Safety factor (CORRECTED FORMULA!)
    # q = (F0 × r) / (R0 × dψ/dr)  (NOT r² and R0²!)
    q = (F0 * r) / (R0 * dpsi_dr + 1e-15)
    
    # Handle axis (r ≈ 0): gradient ill-defined
    axis_mask = r < 0.05  # Small radius near axis
    q[axis_mask] = q0
    
    return q


def compute_current_density(psi_n: np.ndarray,
                            q_target_func: Callable,
                            J0: float = 1.0,
                            q0: float = 1.75) -> np.ndarray:
    """
    计算prescribed current density
    
    J(ψ) = J0 × (q0 / q_target(ψ_n))
    
    Parameters
    ----------
    psi_n : np.ndarray
        Normalized flux [0,1]
    q_target_func : Callable
        Target q-profile function
    J0 : float
        Current normalization
    q0 : float
        Central safety factor
    
    Returns
    -------
    J : np.ndarray
        Current density
    """
    q_target = q_target_func(psi_n)
    
    # Avoid division by zero
    q_target = np.where(q_target < 0.1, 0.1, q_target)
    
    J = J0 * (q0 / q_target)
    
    return J


def solve_inverse_grad_shafranov(
    R: np.ndarray,
    Z: np.ndarray,
    q_target_func: Callable[[np.ndarray], np.ndarray],
    J0: float = 1.0,
    F0: float = 1.0,
    R0: float = 10.0,
    q0: float = 1.75,
    psi_init: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    alpha_relax: float = 0.5,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    解inverse Grad-Shafranov problem: 给定q(ψ)，找ψ
    
    Uses Picard iteration:
      1. Compute J = J0(q0/q_target(ψ_n))
      2. Solve Δ*ψ = -μ0 R J
      3. Iterate with relaxation
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid arrays
    q_target_func : Callable
        Target q-profile: q(ψ_n) where ψ_n ∈ [0,1]
    J0 : float
        Current normalization
    F0 : float
        Toroidal field function
    R0 : float
        Major radius
    q0 : float
        Central safety factor
    psi_init : np.ndarray, optional
        Initial guess (default: r²)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    alpha_relax : float
        Relaxation parameter (0.5 = moderate)
    verbose : bool
        Print iteration info
    
    Returns
    -------
    psi : np.ndarray, shape (Nr, Nz)
        Equilibrium flux
    info : dict
        Convergence and quality info
    """
    Nr, Nz = len(R), len(Z)
    mu0 = 4 * np.pi * 1e-7
    
    # Build Δ* operator (once)
    L = build_grad_shafranov_operator(R, Z)
    
    # Initial guess
    if psi_init is None:
        # Simple r² profile
        R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
        r = np.sqrt((R_2d - R0)**2 + Z_2d**2)
        psi = 0.5 * r**2
    else:
        psi = psi_init.copy()
    
    # Boundary mask
    boundary = np.zeros((Nr, Nz), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    
    # 2D R grid for source term
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    if verbose:
        print("=" * 70)
        print("Inverse Grad-Shafranov Solver (Picard Iteration)")
        print("=" * 70)
        print(f"Grid: {Nr}×{Nz}")
        print(f"J0={J0:.2e}, F0={F0}, R0={R0}, q0={q0}")
        print(f"Relaxation α={alpha_relax}, tol={tol:.1e}")
        print("-" * 70)
    
    residuals = []
    q_errors = []
    
    for n in range(max_iter):
        # Step 1: Normalize current flux
        psi_n = normalize_flux(psi)
        
        # Step 2: Compute q from current ψ
        q_current = compute_q_from_psi_2d(psi, R, Z, F0, R0, q0)
        
        # Step 3: Compute target q at current flux surfaces
        q_target = q_target_func(psi_n)
        
        # Step 4: Compute current density with q-feedback
        # J ~ 1/q, and we want q_current → q_target
        # So adjust J based on ratio
        J = J0 * (q0 / q_target) * (q_current[~boundary].mean() / q_target[~boundary].mean())
        
        # Step 5: Source term
        S = -mu0 * R_2d * J
        S_flat = S.flatten()
        
        # Step 6: Solve linear G-S
        psi_new_flat = spsolve(L, S_flat)
        psi_new = psi_new_flat.reshape(Nr, Nz)
        
        # Step 7: Apply boundary conditions
        psi_new[boundary] = 0
        
        # Step 8: Relaxation
        psi_relaxed = alpha_relax * psi_new + (1 - alpha_relax) * psi
        
        # Step 9: Compute q-error for convergence
        q_comp_new = compute_q_from_psi_2d(psi_relaxed, R, Z, F0, R0, q0)
        q_targ_new = q_target_func(normalize_flux(psi_relaxed))
        q_error_rms = np.sqrt(np.mean((q_comp_new[~boundary] - q_targ_new[~boundary])**2))
        q_errors.append(q_error_rms)
        
        # Step 10: Convergence check (both ψ and q)
        diff = psi_relaxed - psi
        residual = np.linalg.norm(diff) / (np.linalg.norm(psi) + 1e-15)
        residuals.append(residual)
        
        if verbose:
            q_error_max = np.abs(q_comp_new[~boundary] - q_targ_new[~boundary]).max()
            print(f"Iter {n+1:3d}: ψ_res={residual:.3e}, q_rms={q_error_rms:.3e}, q_max={q_error_max:.3e}")
        
        # Converge when BOTH ψ and q are stable
        if residual < tol and q_error_rms < 0.1:  # <10% q-error
            if verbose:
                print("-" * 70)
                print(f"✓ Converged in {n+1} iterations")
                print(f"  ψ residual: {residual:.3e}")
                print(f"  q RMS error: {q_error_rms:.3e}")
                print("=" * 70)
            
            # Final validation
            validation = verify_equilibrium_quality(
                psi_relaxed, R, Z, q_target_func, J0, F0, R0, q0
            )
            
            info = {
                'converged': True,
                'iterations': n + 1,
                'residual': residual,
                'residuals': residuals,
                'q_errors': q_errors,
                **validation
            }
            
            return psi_relaxed, info
        
        # Update for next iteration
        psi = psi_relaxed
    
    # Not converged
    if verbose:
        print("-" * 70)
        print(f"⚠ Not converged after {max_iter} iterations")
        print(f"  Final residual: {residuals[-1]:.3e}")
        print("=" * 70)
    
    validation = verify_equilibrium_quality(psi, R, Z, q_target_func, J0, F0, R0, q0)
    
    info = {
        'converged': False,
        'iterations': max_iter,
        'residual': residuals[-1] if residuals else np.nan,
        'residuals': residuals,
        **validation
    }
    
    return psi, info


def verify_equilibrium_quality(psi: np.ndarray,
                                R: np.ndarray,
                                Z: np.ndarray,
                                q_target_func: Callable,
                                J0: float,
                                F0: float,
                                R0: float,
                                q0: float) -> dict:
    """
    验证equilibrium质量
    
    Checks:
      1. q-profile match
      2. Force balance (Δ*ψ = -μ0 R J)
    
    Returns
    -------
    metrics : dict
        Quality metrics
    """
    mu0 = 4 * np.pi * 1e-7
    Nr, Nz = psi.shape
    
    # Interior points (exclude boundary)
    interior = np.ones((Nr, Nz), dtype=bool)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    
    # 1. q-profile match
    psi_n = normalize_flux(psi)
    q_computed = compute_q_from_psi_2d(psi, R, Z, F0, R0, q0)
    q_target = q_target_func(psi_n)
    
    q_error = np.abs(q_computed - q_target)
    q_error_max = q_error[interior].max()
    q_error_rms = np.sqrt(np.mean(q_error[interior]**2))
    q_rel_error_max = q_error_max / (np.abs(q_target[interior]).mean() + 1e-15)
    
    # 2. Force balance
    Delta_psi = apply_grad_shafranov_operator(psi, R, Z)
    
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    J = compute_current_density(psi_n, q_target_func, J0, q0)
    S_expected = -mu0 * R_2d * J
    
    force_error = np.abs(Delta_psi - S_expected)
    force_error_max = force_error[interior].max()
    force_error_rms = np.sqrt(np.mean(force_error[interior]**2))
    force_rel_error = force_error_rms / (np.abs(S_expected[interior]).mean() + 1e-15)
    
    return {
        'q_error_max': q_error_max,
        'q_error_rms': q_error_rms,
        'q_rel_error_max': q_rel_error_max,
        'force_error_max': force_error_max,
        'force_error_rms': force_error_rms,
        'force_rel_error': force_rel_error,
        'q_pass': q_rel_error_max < 0.10,  # <10% acceptable
        'force_pass': force_rel_error < 0.10  # <10% acceptable
    }
