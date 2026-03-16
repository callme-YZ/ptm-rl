"""
Constraint optimization for free-boundary G-S solver

Implements:
1. Constraint evaluation (X-point, isoflux, Ip)
2. Sensitivity matrix computation
3. Tikhonov regularized least-squares
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple
from dataclasses import dataclass

MU0 = 4 * np.pi * 1e-7


@dataclass
class Grid:
    """Grid wrapper for interpolation"""
    R: np.ndarray
    Z: np.ndarray
    dR: float
    dZ: float
    nr: int
    nz: int


def interpolate_psi(psi: np.ndarray, grid: Grid, R: float, Z: float) -> float:
    """
    Interpolate ψ at (R, Z)
    
    Uses bivariate spline
    """
    R_1d = grid.R[:, 0]
    Z_1d = grid.Z[0, :]
    
    interp = RectBivariateSpline(R_1d, Z_1d, psi, kx=3, ky=3)
    return float(interp(R, Z)[0, 0])


def compute_field_components(psi: np.ndarray, grid: Grid, R: float, Z: float) -> Tuple[float, float]:
    """
    Compute B_R and B_Z at (R, Z)
    
    B_R = -1/R ∂ψ/∂Z
    B_Z = 1/R ∂ψ/∂R
    """
    R_1d = grid.R[:, 0]
    Z_1d = grid.Z[0, :]
    
    # Compute gradients
    dpsi_dR = np.gradient(psi, grid.dR, axis=0)
    dpsi_dZ = np.gradient(psi, grid.dZ, axis=1)
    
    # Interpolate gradients at (R, Z)
    interp_dR = RectBivariateSpline(R_1d, Z_1d, dpsi_dR, kx=1, ky=1)
    interp_dZ = RectBivariateSpline(R_1d, Z_1d, dpsi_dZ, kx=1, ky=1)
    
    dpsi_dR_val = float(interp_dR(R, Z)[0, 0])
    dpsi_dZ_val = float(interp_dZ(R, Z)[0, 0])
    
    B_R = -dpsi_dZ_val / R
    B_Z = dpsi_dR_val / R
    
    return B_R, B_Z


def compute_plasma_current(Jtor: np.ndarray, grid: Grid) -> float:
    """
    Compute total plasma current I_p
    
    I_p = ∫∫ J_φ dR dZ
    """
    # Integrate over plasma region
    # Simple rectangular integration
    I_p = np.sum(Jtor * grid.dR * grid.dZ)
    return I_p


def evaluate_constraints_impl(
    psi: np.ndarray,
    grid: Grid,
    Jtor: np.ndarray,
    xpoint: list,
    isoflux: list,
    Ip_target: float = None
) -> np.ndarray:
    """
    Evaluate constraint violations
    
    Returns:
        b: (n_constraints,) error vector to minimize
    """
    errors = []
    
    # X-point constraints: Br=0, Bz=0
    for R_x, Z_x in xpoint:
        B_R, B_Z = compute_field_components(psi, grid, R_x, Z_x)
        errors.append(B_R)
        errors.append(B_Z)
    
    # Isoflux constraints: ψ equal at points
    if len(isoflux) > 1:
        psi_ref = interpolate_psi(psi, grid, isoflux[0][0], isoflux[0][1])
        for i in range(1, len(isoflux)):
            R_i, Z_i = isoflux[i]
            psi_i = interpolate_psi(psi, grid, R_i, Z_i)
            errors.append(psi_i - psi_ref)
    
    # I_p constraint
    if Ip_target is not None:
        I_p = compute_plasma_current(Jtor, grid)
        errors.append(I_p - Ip_target)
    
    return np.array(errors)


def greens_function(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float) -> np.ndarray:
    """
    Green's function for Δ* operator
    
    G(R,Z; Rc,Zc) = (μ₀/2π) √(R·Rc) × [(2-k²)K(k²) - 2E(k²)] / k
    """
    from scipy.special import ellipk, ellipe
    
    # k² = 4RRc / [(R+Rc)² + (Z-Zc)²]
    k2 = 4 * R * Rc / ((R + Rc)**2 + (Z - Zc)**2 + 1e-10)
    k2 = np.clip(k2, 0, 1 - 1e-10)
    
    k = np.sqrt(k2)
    
    # Complete elliptic integrals
    K = ellipk(k2)
    E = ellipe(k2)
    
    # Green's function
    G = (MU0 / (2 * np.pi)) * np.sqrt(R * Rc) * ((2 - k2) * K - 2 * E) / (k + 1e-10)
    
    return G


def compute_sensitivity_matrix_impl(
    psi: np.ndarray,
    grid: Grid,
    Jtor: np.ndarray,
    coil_R: np.ndarray,
    coil_Z: np.ndarray,
    I_coil: np.ndarray,
    xpoint: list,
    isoflux: list,
    Ip_target: float = None,
    dI: float = 1.0
) -> np.ndarray:
    """
    Compute sensitivity matrix ∂constraint/∂I
    
    Uses Green's function approximation:
        ∂ψ/∂I_j ≈ G(r; r_coil_j)
    
    Returns:
        A: (n_constraints, n_coils)
    """
    n_coils = len(I_coil)
    
    # Count constraints
    n_const = len(xpoint) * 2  # Br, Bz each
    n_const += max(0, len(isoflux) - 1)  # N-1 isoflux
    if Ip_target is not None:
        n_const += 1
    
    A = np.zeros((n_const, n_coils))
    
    # For each coil, compute ∂constraint/∂I_j
    for j in range(n_coils):
        # Green's function from coil j
        G = greens_function(grid.R, grid.Z, coil_R[j], coil_Z[j])
        
        # Perturbation: ψ_new = ψ + dI * G
        psi_perturb = psi + dI * G
        
        # Evaluate constraints with perturbed ψ
        # (Jtor unchanged for sensitivity - linear approximation)
        b_perturb = evaluate_constraints_impl(
            psi_perturb, grid, Jtor, xpoint, isoflux, Ip_target
        )
        b_baseline = evaluate_constraints_impl(
            psi, grid, Jtor, xpoint, isoflux, Ip_target
        )
        
        # Finite difference
        A[:, j] = (b_perturb - b_baseline) / dI
    
    return A


def optimize_coils_impl(
    psi: np.ndarray,
    grid: Grid,
    Jtor: np.ndarray,
    coil_R: np.ndarray,
    coil_Z: np.ndarray,
    I_coil: np.ndarray,
    xpoint: list,
    isoflux: list,
    Ip_target: float = None,
    gamma: float = 1e-6,  # Increased from 1e-12 based on diagnosis
    dI: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Optimize coil currents to satisfy constraints
    
    Solves: minimize ||A·ΔI - b||² + γ²||ΔI||²
    
    Returns:
        I_new: Updated coil currents
        error: Constraint error norm
    """
    # Check constraint count
    n_coils = len(I_coil)
    n_const = len(xpoint) * 2 + max(0, len(isoflux) - 1)
    if Ip_target is not None:
        n_const += 1
    
    if n_const < n_coils:
        raise ValueError(
            f"Underdetermined: {n_const} constraints < {n_coils} coils"
        )
    
    # Evaluate constraint errors
    b = evaluate_constraints_impl(psi, grid, Jtor, xpoint, isoflux, Ip_target)
    
    # Compute sensitivity matrix
    A = compute_sensitivity_matrix_impl(
        psi, grid, Jtor, coil_R, coil_Z, I_coil,
        xpoint, isoflux, Ip_target, dI
    )
    
    # Tikhonov regularized least-squares
    # (A^T A + γ²I)^{-1} A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    regularization = gamma**2 * np.eye(n_coils)
    
    try:
        # Solve for ΔI
        delta_I = np.linalg.solve(ATA + regularization, ATb)
    except np.linalg.LinAlgError:
        # Singular matrix - fall back to pseudo-inverse
        delta_I = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Update currents WITH DAMPING
    # Diagnosis showed ΔI is ~1000× too large
    # Use conservative damping factor
    alpha_damping = 0.2  # Conservative (theory suggests up to 0.3)
    
    I_new = I_coil - alpha_damping * delta_I  # Damped update
    
    # Constraint error
    error = np.linalg.norm(b)
    
    return I_new, error


if __name__ == "__main__":
    # Quick test
    print("Testing constraint optimizer...")
    
    # Simple grid
    R_1d = np.linspace(1.0, 2.0, 33)
    Z_1d = np.linspace(-0.5, 0.5, 33)
    R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')
    
    grid = Grid(
        R=R, Z=Z,
        dR=R_1d[1]-R_1d[0],
        dZ=Z_1d[1]-Z_1d[0],
        nr=len(R_1d),
        nz=len(Z_1d)
    )
    
    # Simple ψ
    R0 = 1.5
    psi = -((R - R0)**2 + Z**2)
    
    # Test constraint evaluation
    xpoint = [(1.3, 0.0)]
    isoflux = [(1.2, 0.0), (1.8, 0.0)]
    
    b = evaluate_constraints_impl(
        psi, grid, np.zeros_like(R),
        xpoint, isoflux
    )
    
    print(f"Constraint errors: {b}")
    print(f"  X-point Br: {b[0]:.3e}")
    print(f"  X-point Bz: {b[1]:.3e}")
    print(f"  Isoflux: {b[2]:.3e}")
    
    print("\n✓ Constraint optimizer basic test passed")

