#!/usr/bin/env python3
"""
Newton-Raphson Solver for Grad-Shafranov Equation

Phase 1: Minimal Working Prototype
- Finite difference Jacobian
- Direct linear solver (spsolve)
- Basic convergence checking

Solves: Δ*ψ + μ₀RJ_φ(ψ) = 0

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Callable, Tuple, Dict, Optional


MU0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]


def initialize_psi(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Create initial guess for ψ
    
    Uses simple parabolic profile: ψ ~ r²
    
    Args:
        R: Radial coordinates (1D array)
        Z: Vertical coordinates (1D array)
    
    Returns:
        psi: Initial guess (Nr×Nz)
    """
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    # Magnetic axis (assumed center)
    R_axis = (np.min(R) + np.max(R)) / 2
    Z_axis = 0.0
    
    # Radial distance from axis
    r = np.sqrt((RR - R_axis)**2 + ZZ**2)
    
    # Characteristic radius
    a = (np.max(R) - np.min(R)) / 2
    
    # Parabolic profile
    psi = np.clip((r / a)**2, 0, 1)
    
    # Enforce boundary condition: ψ=0 on boundary
    psi[0, :] = 0
    psi[-1, :] = 0
    psi[:, 0] = 0
    psi[:, -1] = 0
    
    return psi


def compute_residual(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    q_profile: Callable[[np.ndarray], np.ndarray],
    delta_star: csr_matrix,
    J0: float
) -> np.ndarray:
    """
    Compute residual F(ψ) = Δ*ψ + μ₀RJ_φ(ψ)
    
    Args:
        psi: Current solution (Nr×Nz)
        R: R coordinates
        Z: Z coordinates
        q_profile: Function q(psi_n) → q values
        delta_star: Grad-Shafranov operator
        J0: Current normalization constant
    
    Returns:
        residual: F(ψ) as flattened vector
    """
    # Assume psi already normalized to [0,1] by solver
    # (No dynamic renormalization - that breaks consistency!)
    psi_n = psi
    
    # Compute q-profile
    q = q_profile(psi_n)
    
    # Current prescription: J = J0 * q0 / q
    q0 = q_profile(np.array([0.0]))[0]  # q at magnetic axis
    
    # Avoid division by zero
    q_safe = np.where(q > 1e-6, q, 1e-6)
    J_phi = J0 * q0 / q_safe
    
    # Build source term
    RR, _ = np.meshgrid(R, Z, indexing='ij')
    source = MU0 * RR * J_phi
    
    # Apply Δ* operator
    psi_flat = psi.flatten()
    delta_psi_vec = delta_star @ psi_flat
    
    # Residual = Δ*ψ + μ₀RJ
    residual = delta_psi_vec + source.flatten()
    
    return residual


def compute_jacobian_fd(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    q_profile: Callable,
    delta_star: csr_matrix,
    J0: float,
    eps: float = 1e-6
) -> csr_matrix:
    """
    Compute Jacobian via finite differences
    
    J_ij = ∂F_i/∂ψ_j ≈ [F_i(ψ + ε·e_j) - F_i(ψ)] / ε
    
    Args:
        psi: Current solution
        ... (same as compute_residual)
        eps: Finite difference step size
    
    Returns:
        J_mat: Jacobian matrix (sparse CSR)
    """
    Nr, Nz = len(R), len(Z)
    N = Nr * Nz
    
    # Baseline residual
    F0 = compute_residual(psi, R, Z, q_profile, delta_star, J0)
    
    # Build Jacobian column by column
    J_mat = lil_matrix((N, N))
    
    psi_flat = psi.flatten()
    
    for j in range(N):
        # Perturb j-th component
        psi_pert = psi_flat.copy()
        psi_pert[j] += eps
        
        # Reshape to 2D for residual computation
        psi_2d = psi_pert.reshape((Nr, Nz))
        
        # Compute perturbed residual
        F_pert = compute_residual(psi_2d, R, Z, q_profile, delta_star, J0)
        
        # Finite difference: dF/dψ_j
        dF_dpsij = (F_pert - F0) / eps
        
        # Store as column
        J_mat[:, j] = dF_dpsij.reshape(-1, 1)
    
    return J_mat.tocsr()


def solve_newton_gs(
    R: np.ndarray,
    Z: np.ndarray,
    delta_star: csr_matrix,
    q_profile: Callable[[np.ndarray], np.ndarray],
    J0: float = 1e6,
    psi_init: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 20,
    verbose: bool = True
) -> Tuple[np.ndarray, bool, Dict]:
    """
    Newton-Raphson solver for Grad-Shafranov equation
    
    Solves: Δ*ψ + μ₀RJ_φ(ψ) = 0
    where J_φ is prescribed from q-profile
    
    Args:
        R: Radial grid coordinates (1D)
        Z: Vertical grid coordinates (1D)
        delta_star: Grad-Shafranov operator (sparse matrix)
        q_profile: Function q(psi_n) → q, where psi_n ∈ [0,1]
        J0: Current normalization (controls total plasma current)
        psi_init: Initial guess (if None, use parabolic)
        tol: Convergence tolerance on residual norm
        max_iter: Maximum Newton iterations
        verbose: Print iteration info
    
    Returns:
        psi: Converged solution (Nr×Nz)
        converged: True if converged within tol
        diagnostics: Dict with iteration history
    """
    Nr, Nz = len(R), len(Z)
    N = Nr * Nz
    
    # Initial guess
    if psi_init is None:
        psi = initialize_psi(R, Z)
    else:
        psi = psi_init.copy()
    
    # Boundary mask
    boundary_mask = np.zeros((Nr, Nz), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    
    # Storage for diagnostics
    residuals = []
    
    if verbose:
        print("="*70)
        print("Newton-Raphson Solver for Grad-Shafranov Equation")
        print("="*70)
        print(f"Grid: {Nr}×{Nz} ({N} DOF)")
        print(f"Tolerance: {tol:.1e}, Max iterations: {max_iter}")
        print()
    
    converged = False
    
    for k in range(max_iter):
        # 1. Compute residual
        R_vec = compute_residual(psi, R, Z, q_profile, delta_star, J0)
        
        # 2. Check convergence
        res_norm = np.linalg.norm(R_vec)
        residuals.append(res_norm)
        
        if verbose:
            print(f"Iter {k+1:2d}: ||F|| = {res_norm:.3e}", end="")
        
        if res_norm < tol:
            if verbose:
                print("  ✓ Converged!")
            converged = True
            break
        
        # 3. Compute Jacobian (finite difference)
        if verbose:
            print("  (computing Jacobian...)", end="", flush=True)
        
        J_mat = compute_jacobian_fd(psi, R, Z, q_profile, delta_star, J0)
        
        if verbose:
            print(" done", end="")
        
        # 4. Solve linear system: J·δψ = -R
        # Enforce boundary conditions on delta_psi
        
        # Build modified system for boundary constraints
        A = J_mat.tolil()
        b = -R_vec.copy()
        
        boundary_flat = boundary_mask.flatten()
        for i in np.where(boundary_flat)[0]:
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = 0.0  # δψ=0 on boundary
        
        A = A.tocsr()
        
        if verbose:
            print("  (solving linear system...)", end="", flush=True)
        
        delta_psi = spsolve(A, b)
        
        if verbose:
            print(" done")
        
        # 5. Update: ψ^(k+1) = ψ^k + α·δψ
        # Try full Newton step first
        alpha = 1.0
        
        psi_flat = psi.flatten()
        psi_flat += alpha * delta_psi
        psi = psi_flat.reshape((Nr, Nz))
        
        # Re-enforce boundary conditions (safety)
        psi[boundary_mask] = 0.0
        
        # CRITICAL: Rescale psi to maintain [0, 1] range
        # This keeps psi_n definition consistent across iterations
        psi_min_new = np.min(psi)
        psi_max_new = np.max(psi)
        
        if psi_max_new - psi_min_new > 1e-12:
            psi = (psi - psi_min_new) / (psi_max_new - psi_min_new)
        
        # Boundary should still be 0
        psi[boundary_mask] = 0.0
        
        if verbose and alpha < 1.0:
            print(f"  (damping: α={alpha})")
    
    if not converged and verbose:
        print()
        print(f"✗ Did not converge in {max_iter} iterations")
        print(f"  Final residual: {res_norm:.3e}")
    
    # Prepare diagnostics
    diagnostics = {
        'iterations': k + 1 if converged else max_iter,
        'residuals': residuals,
        'final_residual': res_norm,
        'converged': converged,
    }
    
    return psi, converged, diagnostics


# Test functions
def M3DC1_q_profile(psi_n: np.ndarray) -> np.ndarray:
    """
    M3D-C1 benchmark q-profile
    
    q(ψ) = q0 √(2 / (1 + 3ψ))
    
    Target: q(axis)=1.75, q(edge)≈2.5
    """
    q0 = 1.75
    return q0 * np.sqrt(2.0 / (1.0 + 3.0*psi_n + 1e-12))


def constant_q_profile(psi_n: np.ndarray, q_val: float = 2.0) -> np.ndarray:
    """Constant q-profile for testing"""
    return q_val * np.ones_like(psi_n)


if __name__ == "__main__":
    # Quick test
    print("Newton G-S Solver - Phase 1 Test\n")
    
    import sys
    sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/reduced-mhd')
    from core.operators import build_grad_shafranov_operator
    
    # Small grid for fast testing
    Nr, Nz = 32, 32
    R = np.linspace(0.5, 1.5, Nr)
    Z = np.linspace(-0.5, 0.5, Nz)
    
    print(f"Building Δ* operator ({Nr}×{Nz} grid)...")
    delta_star = build_grad_shafranov_operator(R, Z)
    print(f"  {delta_star.nnz} nonzeros\n")
    
    # Test 1: Constant q
    print("Test 1: Constant q=2.0")
    print("-"*70)
    
    def q_const(psi_n):
        return constant_q_profile(psi_n, 2.0)
    
    psi, converged, diag = solve_newton_gs(
        R, Z, delta_star, q_const,
        J0=1e6, tol=1e-6, max_iter=20, verbose=True
    )
    
    print()
    if converged:
        print("✓ Test 1 PASSED")
    else:
        print("✗ Test 1 FAILED")
    
    print("\n" + "="*70)
