"""
Differential Operators in Toroidal Geometry

Implements gradient, divergence, and Laplacian operators in toroidal
coordinates (r, θ, φ) using 2nd-order finite differences.

Coordinate system:
    r: minor radius
    θ: poloidal angle
    φ: toroidal angle (axisymmetric: ∂/∂φ = 0)

Metric tensor (orthogonal):
    g_rr = 1
    g_θθ = r²
    g_φφ = R² = (R₀ + r*cos(θ))²
    
Jacobian:
    √g = r*R

References:
    - Design doc: v1.1-toroidal-symplectic-design.md Section 1.2
    - Pyrokinetics study: notes/pyrokinetics-toroidal-study.md Section 5

Author: 小P ⚛️
Created: 2026-03-17
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid


def gradient_toroidal(f: np.ndarray, grid: ToroidalGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient ∇f in toroidal coordinates (axisymmetric).
    
    For orthogonal toroidal coordinates with axisymmetry (∂/∂φ = 0):
        ∇f = (∂f/∂r) ê_r + (1/r²)(∂f/∂θ) ê_θ
    
    Uses 2nd-order centered finite differences.
    
    Parameters
    ----------
    f : np.ndarray (nr, ntheta)
        Scalar field on toroidal grid
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    grad_r : np.ndarray (nr, ntheta)
        Radial component of gradient
    grad_theta : np.ndarray (nr, ntheta)
        Poloidal component of gradient (with metric factor 1/r²)
    
    Notes
    -----
    - Periodic boundary in θ direction
    - One-sided differences at radial boundaries
    - Accuracy: O(dr²) + O(dθ²)
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> f = grid.r_grid**2  # Test: ∇(r²) = 2r
    >>> grad_r, grad_theta = gradient_toroidal(f, grid)
    >>> assert np.allclose(grad_r, 2*grid.r_grid, atol=1e-10)
    """
    nr, ntheta = f.shape
    dr = grid.dr
    dtheta = grid.dtheta
    r_grid = grid.r_grid
    
    # Initialize
    grad_r = np.zeros_like(f)
    grad_theta = np.zeros_like(f)
    
    # Radial derivative: ∂f/∂r
    # Interior: centered difference
    grad_r[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dr)
    
    # Radial boundaries: one-sided difference
    grad_r[0, :] = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dr)
    grad_r[-1, :] = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dr)
    
    # Poloidal derivative: (1/r²) ∂f/∂θ
    # Interior: centered difference
    df_dtheta = np.zeros_like(f)
    df_dtheta[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dtheta)
    
    # Periodic boundary in θ
    df_dtheta[:, 0] = (f[:, 1] - f[:, -1]) / (2*dtheta)
    df_dtheta[:, -1] = (f[:, 0] - f[:, -2]) / (2*dtheta)
    
    # Apply metric factor 1/r²
    # Avoid division by zero at r=0 (grid.r[0] = 1e-6, safe)
    grad_theta = df_dtheta / r_grid**2
    
    return grad_r, grad_theta


def divergence_toroidal(A_r: np.ndarray, A_theta: np.ndarray, 
                        grid: ToroidalGrid) -> np.ndarray:
    """
    Compute divergence ∇·A in toroidal coordinates (axisymmetric).
    
    For axisymmetric vector field A = (A_r, A_θ, 0):
        ∇·A = (1/√g)[∂(√g A_r)/∂r + ∂(√g A_θ)/∂θ]
    
    where √g = r*R is the Jacobian.
    
    Uses 2nd-order centered finite differences.
    
    Parameters
    ----------
    A_r : np.ndarray (nr, ntheta)
        Radial component of vector field
    A_theta : np.ndarray (nr, ntheta)
        Poloidal component of vector field
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    div_A : np.ndarray (nr, ntheta)
        Divergence ∇·A
    
    Notes
    -----
    - Periodic boundary in θ direction
    - One-sided differences at radial boundaries
    - Accuracy: O(dr²) + O(dθ²)
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Constant divergence-free field
    >>> A_r = np.zeros_like(grid.r_grid)
    >>> A_theta = np.zeros_like(grid.r_grid)
    >>> div_A = divergence_toroidal(A_r, A_theta, grid)
    >>> assert np.allclose(div_A, 0.0, atol=1e-12)
    """
    nr, ntheta = A_r.shape
    dr = grid.dr
    dtheta = grid.dtheta
    J = grid.jacobian()  # √g = r*R
    
    # Multiply by Jacobian
    sqrtg_Ar = J * A_r
    sqrtg_Atheta = J * A_theta
    
    # Derivatives
    # ∂(√g A_r)/∂r
    d_sqrtg_Ar_dr = np.zeros_like(A_r)
    d_sqrtg_Ar_dr[1:-1, :] = (sqrtg_Ar[2:, :] - sqrtg_Ar[:-2, :]) / (2*dr)
    d_sqrtg_Ar_dr[0, :] = (-3*sqrtg_Ar[0, :] + 4*sqrtg_Ar[1, :] - sqrtg_Ar[2, :]) / (2*dr)
    d_sqrtg_Ar_dr[-1, :] = (3*sqrtg_Ar[-1, :] - 4*sqrtg_Ar[-2, :] + sqrtg_Ar[-3, :]) / (2*dr)
    
    # ∂(√g A_θ)/∂θ
    d_sqrtg_Atheta_dtheta = np.zeros_like(A_theta)
    d_sqrtg_Atheta_dtheta[:, 1:-1] = (sqrtg_Atheta[:, 2:] - sqrtg_Atheta[:, :-2]) / (2*dtheta)
    # Periodic
    d_sqrtg_Atheta_dtheta[:, 0] = (sqrtg_Atheta[:, 1] - sqrtg_Atheta[:, -1]) / (2*dtheta)
    d_sqrtg_Atheta_dtheta[:, -1] = (sqrtg_Atheta[:, 0] - sqrtg_Atheta[:, -2]) / (2*dtheta)
    
    # Divergence: (1/√g)[∂(√g A_r)/∂r + ∂(√g A_θ)/∂θ]
    div_A = (d_sqrtg_Ar_dr + d_sqrtg_Atheta_dtheta) / J
    
    return div_A


def laplacian_toroidal(f: np.ndarray, grid: ToroidalGrid) -> np.ndarray:
    """
    Compute Laplacian ∇²f in toroidal coordinates (axisymmetric).
    
    General formula for orthogonal coordinates:
        ∇²f = (1/√g)[∂/∂r(√g g^rr ∂f/∂r) + ∂/∂θ(√g g^θθ ∂f/∂θ)]
    
    For toroidal coordinates:
        g^rr = 1
        g^θθ = 1/r²
        √g = r*R    where R = R₀ + r*cos(θ)
    
    Therefore:
        ∇²f = (1/r*R)[∂/∂r(r*R ∂f/∂r) + ∂/∂θ(R/r ∂f/∂θ)]
    
    Uses 2nd-order centered finite differences.
    
    Parameters
    ----------
    f : np.ndarray (nr, ntheta)
        Scalar field on toroidal grid
    grid : ToroidalGrid
        Toroidal grid object
    
    Returns
    -------
    lap_f : np.ndarray (nr, ntheta)
        Laplacian ∇²f
    
    Notes
    -----
    - Periodic boundary in θ direction
    - One-sided differences at radial boundaries
    - Special handling near r=0 (currently r_min = 1e-6, no singularity)
    - Accuracy: O(dr²) + O(dθ²)
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Test 1: Laplacian of constant = 0
    >>> f_const = np.ones_like(grid.r_grid)
    >>> lap_f = laplacian_toroidal(f_const, grid)
    >>> assert np.max(np.abs(lap_f)) < 1e-12
    
    >>> # Test 2: Analytical test f = R² + Z²
    >>> f_test = grid.R_grid**2 + grid.Z_grid**2
    >>> lap_f = laplacian_toroidal(f_test, grid)
    >>> # Analytical: ∇²(R²+Z²) = 4
    >>> assert np.allclose(lap_f[5:-5, :], 4.0, atol=1e-8)
    """
    nr, ntheta = f.shape
    dr = grid.dr
    dtheta = grid.dtheta
    r_grid = grid.r_grid
    R_grid = grid.R_grid  # R = R₀ + r*cos(θ)
    sqrtg = grid.jacobian()  # √g = r*R
    
    # Contravariant metric components
    g_rr = 1.0
    g_tt = 1.0 / r_grid**2  # g^θθ = 1/r²
    
    # ∂f/∂r
    df_dr = np.zeros_like(f)
    df_dr[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dr)
    df_dr[0, :] = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dr)
    df_dr[-1, :] = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dr)
    
    # ∂f/∂θ
    df_dtheta = np.zeros_like(f)
    df_dtheta[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dtheta)
    # Periodic
    df_dtheta[:, 0] = (f[:, 1] - f[:, -1]) / (2*dtheta)
    df_dtheta[:, -1] = (f[:, 0] - f[:, -2]) / (2*dtheta)
    
    # Compute √g * g^rr * ∂f/∂r = r*R * 1 * ∂f/∂r
    term_r = sqrtg * g_rr * df_dr
    
    # Compute √g * g^θθ * ∂f/∂θ = r*R * (1/r²) * ∂f/∂θ = (R/r) * ∂f/∂θ
    term_theta = sqrtg * g_tt * df_dtheta
    
    # ∂/∂r(√g g^rr ∂f/∂r)
    d_term_r_dr = np.zeros_like(f)
    d_term_r_dr[1:-1, :] = (term_r[2:, :] - term_r[:-2, :]) / (2*dr)
    d_term_r_dr[0, :] = (-3*term_r[0, :] + 4*term_r[1, :] - term_r[2, :]) / (2*dr)
    d_term_r_dr[-1, :] = (3*term_r[-1, :] - 4*term_r[-2, :] + term_r[-3, :]) / (2*dr)
    
    # ∂/∂θ(√g g^θθ ∂f/∂θ)
    d_term_theta_dtheta = np.zeros_like(f)
    d_term_theta_dtheta[:, 1:-1] = (term_theta[:, 2:] - term_theta[:, :-2]) / (2*dtheta)
    # Periodic
    d_term_theta_dtheta[:, 0] = (term_theta[:, 1] - term_theta[:, -1]) / (2*dtheta)
    d_term_theta_dtheta[:, -1] = (term_theta[:, 0] - term_theta[:, -2]) / (2*dtheta)
    
    # Laplacian: (1/√g)[∂/∂r(...) + ∂/∂θ(...)]
    lap_f = (d_term_r_dr + d_term_theta_dtheta) / sqrtg
    
    return lap_f


def laplacian_toroidal_alternative(f: np.ndarray, grid: ToroidalGrid) -> np.ndarray:
    """
    Alternative implementation: ∇²f = ∇·(∇f).
    
    This uses gradient_toroidal() and divergence_toroidal() to compute
    Laplacian via the identity ∇²f = ∇·(∇f).
    
    Useful for validation and testing consistency of operators.
    
    Parameters
    ----------
    f : np.ndarray (nr, ntheta)
        Scalar field
    grid : ToroidalGrid
        Toroidal grid
    
    Returns
    -------
    lap_f : np.ndarray (nr, ntheta)
        Laplacian ∇²f
    
    Notes
    -----
    - Should match laplacian_toroidal() to within numerical precision
    - Slower (two operator calls) but more transparent
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> f = grid.r_grid**2 + grid.R_grid**2
    >>> lap1 = laplacian_toroidal(f, grid)
    >>> lap2 = laplacian_toroidal_alternative(f, grid)
    >>> assert np.allclose(lap1, lap2, atol=1e-10)
    """
    # Step 1: ∇f
    grad_r, grad_theta = gradient_toroidal(f, grid)
    
    # Step 2: ∇·(∇f)
    # Need to convert (grad_r, grad_theta) to contravariant components
    # For orthogonal coords: A^r = g^rr A_r = A_r (since g^rr = 1)
    #                        A^θ = g^θθ A_θ = (1/r²) A_θ
    # But gradient already returns covariant components in the right form
    # Actually, gradient_toroidal returns physical components, need to check...
    
    # Correction: gradient_toroidal returns (∂f/∂r, (1/r²)∂f/∂θ)
    # These are already scaled by metric factors
    # For divergence, we need contravariant components
    
    # Actually, let's use the direct formula: divergence of gradient
    # ∇f = (∂f/∂r) ê_r + (1/r²)(∂f/∂θ) ê_θ  (this is what gradient_toroidal returns)
    
    # To compute ∇·(∇f), we need vector field in contravariant form
    # For orthogonal coords: (∇f)^r = ∂f/∂r, (∇f)^θ = (1/r²)∂f/∂θ
    
    # But wait, let me reconsider...
    # The gradient in orthogonal curvilinear coords is:
    # ∇f = (∂f/∂r)/√g_rr ê_r + (∂f/∂θ)/√g_θθ ê_θ
    #    = (∂f/∂r) ê_r + (∂f/∂θ)/r ê_θ    (since √g_rr=1, √g_θθ=r)
    
    # So gradient_toroidal should return (∂f/∂r, ∂f/∂θ/r), not (∂f/∂r, ∂f/∂θ/r²)
    # Let me fix gradient_toroidal...
    
    # Actually, looking at the formula in gradient_toroidal:
    # It returns (∂f/∂r, (1/r²)∂f/∂θ)
    # This seems wrong. Let me recalculate.
    
    # For orthogonal toroidal coords with metric g_ij = diag(1, r², R²):
    # Contravariant metric: g^ij = diag(1, 1/r², 1/R²)
    # Gradient (contravariant): ∇f = g^rr ∂f/∂r ê^r + g^θθ ∂f/∂θ ê^θ
    #                                = ∂f/∂r ê^r + (1/r²)∂f/∂θ ê^θ
    
    # So gradient_toroidal is correct as written.
    
    # Now for ∇·(∇f), we have vector V = ∇f with components V^r = ∂f/∂r, V^θ = (1/r²)∂f/∂θ
    # Divergence: ∇·V = (1/√g)[∂(√g V^r)/∂r + ∂(√g V^θ)/∂θ]
    
    # So we can use divergence_toroidal with V^r = grad_r, V^θ = grad_theta
    
    lap_f = divergence_toroidal(grad_r, grad_theta, grid)
    
    return lap_f
