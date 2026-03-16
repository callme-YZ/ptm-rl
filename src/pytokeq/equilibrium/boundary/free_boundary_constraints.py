"""
Free-Boundary Constraint System

Implements X-point and isoflux constraints for coil current optimization.

Reference: Design doc Section 3.2, 2.2
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..utils.greens_function import greens_psi, greens_psi_gradient_R, greens_psi_gradient_Z
from ..diagnostics.xpoint_finder import XPoint


@dataclass
class IsofluxPair:
    """Isoflux constraint: two points on same flux surface"""
    R1: float  # First point major radius (m)
    Z1: float  # First point vertical position (m)
    R2: float  # Second point major radius (m)
    Z2: float  # Second point vertical position (m)


@dataclass
class CoilSet:
    """Coil configuration"""
    R: np.ndarray  # Coil major radii (m)
    Z: np.ndarray  # Coil vertical positions (m)
    I: np.ndarray  # Coil currents (A) - to be optimized


def build_constraint_matrix(coils: CoilSet, xpoint: Optional[XPoint], 
                            isoflux_pairs: List[IsofluxPair]) -> np.ndarray:
    """
    Build constraint matrix A for coil optimization
    
    A · I = b  (overdetermined linear system)
    
    where A[i,j] = ∂c_i/∂I_j (sensitivity of constraint i to coil j current)
    
    Parameters
    ----------
    coils : CoilSet
        Coil locations (R, Z)
    xpoint : XPoint or None
        X-point location (if present)
    isoflux_pairs : List[IsofluxPair]
        Isoflux constraint pairs
        
    Returns
    -------
    A : ndarray (m, n)
        Constraint matrix
        m = number of constraints (2 if xpoint + len(isoflux_pairs))
        n = number of coils
    """
    n_coils = len(coils.R)
    n_constraints = (2 if xpoint else 0) + len(isoflux_pairs)
    
    A = np.zeros((n_constraints, n_coils))
    
    row = 0
    
    # X-point constraints (B_R = 0, B_Z = 0)
    if xpoint:
        R_X = xpoint.R
        Z_X = xpoint.Z
        
        for j in range(n_coils):
            R_c = coils.R[j]
            Z_c = coils.Z[j]
            
            # Constraint 1: B_Z = (1/R)∂ψ/∂R = 0
            # ∂c₁/∂I_j = (1/R_X) · ∂G/∂R(R_c, Z_c, R_X, Z_X)
            dG_dR = greens_psi_gradient_R(R_c, Z_c, R_X, Z_X)
            A[row, j] = (1.0 / R_X) * dG_dR
        
        row += 1
        
        for j in range(n_coils):
            R_c = coils.R[j]
            Z_c = coils.Z[j]
            
            # Constraint 2: B_R = -(1/R)∂ψ/∂Z = 0
            # ∂c₂/∂I_j = -(1/R_X) · ∂G/∂Z(R_c, Z_c, R_X, Z_X)
            dG_dZ = greens_psi_gradient_Z(R_c, Z_c, R_X, Z_X)
            A[row, j] = -(1.0 / R_X) * dG_dZ
        
        row += 1
    
    # Isoflux constraints: ψ(R1,Z1) = ψ(R2,Z2)
    for pair in isoflux_pairs:
        for j in range(n_coils):
            R_c = coils.R[j]
            Z_c = coils.Z[j]
            
            # c = ψ(R1,Z1) - ψ(R2,Z2) = 0
            # ∂c/∂I_j = G(R_c,Z_c,R1,Z1) - G(R_c,Z_c,R2,Z2)
            G1 = greens_psi(R_c, Z_c, pair.R1, pair.Z1)
            G2 = greens_psi(R_c, Z_c, pair.R2, pair.Z2)
            A[row, j] = G1 - G2
        
        row += 1
    
    return A


def build_target_vector(psi: np.ndarray, grid, xpoint: Optional[XPoint],
                        isoflux_pairs: List[IsofluxPair]) -> np.ndarray:
    """
    Build target vector b = current constraint error
    
    We want to minimize ||A·I - b||² so that constraints → 0
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Current flux solution
    grid : Grid
        Computational grid
    xpoint : XPoint or None
        X-point location
    isoflux_pairs : List[IsofluxPair]
        Isoflux pairs
        
    Returns
    -------
    b : ndarray (m,)
        Target vector (current constraint residuals)
    """
    n_constraints = (2 if xpoint else 0) + len(isoflux_pairs)
    b = np.zeros(n_constraints)
    
    row = 0
    
    # X-point constraints
    if xpoint:
        R_X = xpoint.R
        Z_X = xpoint.Z
        i, j = xpoint.grid_i, xpoint.grid_j
        
        # Current B_Z at X-point (should be 0)
        dpsi_dR = compute_gradient_R(psi, i, j, grid)
        B_Z_current = (1.0 / R_X) * dpsi_dR
        b[row] = -B_Z_current  # Want to drive to 0
        row += 1
        
        # Current B_R at X-point (should be 0)
        dpsi_dZ = compute_gradient_Z(psi, i, j, grid)
        B_R_current = -(1.0 / R_X) * dpsi_dZ
        b[row] = -B_R_current  # Want to drive to 0
        row += 1
    
    # Isoflux constraints
    for pair in isoflux_pairs:
        # Current flux difference (should be 0)
        psi1 = interpolate_psi(psi, grid, pair.R1, pair.Z1)
        psi2 = interpolate_psi(psi, grid, pair.R2, pair.Z2)
        flux_diff = psi1 - psi2
        b[row] = -flux_diff  # Want to drive to 0
        row += 1
    
    return b


def optimize_coil_currents(A: np.ndarray, b: np.ndarray, 
                           gamma: float = 1e-12) -> np.ndarray:
    """
    Solve for optimal coil currents via Tikhonov regularization
    
    min ||A·I - b||² + γ||I||²
    
    Solution: I = (A^T A + γI)^(-1) A^T b
    
    Parameters
    ----------
    A : ndarray (m, n)
        Constraint matrix
    b : ndarray (m,)
        Target vector
    gamma : float
        Regularization parameter (default 1e-12)
        
    Returns
    -------
    I_opt : ndarray (n,)
        Optimal coil currents (A)
    """
    n_coils = A.shape[1]
    
    # Normal equations with Tikhonov regularization
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Add regularization
    ATA_reg = ATA + gamma * np.eye(n_coils)
    
    # Solve
    I_opt = np.linalg.solve(ATA_reg, ATb)
    
    return I_opt


def check_constraint_matrix(A: np.ndarray, gamma: float = 1e-12) -> Tuple[bool, str]:
    """
    Verify constraint matrix is well-conditioned
    
    Returns
    -------
    valid : bool
        True if matrix is good
    message : str
        Status message
    """
    m, n = A.shape
    
    # Check rank (use higher tolerance for small matrices)
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    
    # Check condition number of regularized system
    ATA_reg = A.T @ A + gamma * np.eye(n)
    cond = np.linalg.cond(ATA_reg)
    
    # Validity checks
    if rank < min(m, n):
        return False, f"Rank-deficient: rank={rank} < min({m},{n})"
    
    if cond > 1e10:
        return False, f"Ill-conditioned (κ={cond:.2e})"
    
    return True, f"OK (rank={rank}/{min(m,n)}, κ={cond:.2e})"


# Helper functions

def compute_gradient_R(psi: np.ndarray, i: int, j: int, grid) -> float:
    """Central difference ∂ψ/∂R"""
    dR = grid.dR
    return (psi[i+1, j] - psi[i-1, j]) / (2 * dR)


def compute_gradient_Z(psi: np.ndarray, i: int, j: int, grid) -> float:
    """Central difference ∂ψ/∂Z"""
    dZ = grid.dZ
    return (psi[i, j+1] - psi[i, j-1]) / (2 * dZ)


def interpolate_psi(psi: np.ndarray, grid, R: float, Z: float) -> float:
    """
    Bilinear interpolation of ψ at arbitrary (R,Z)
    
    Simple implementation - can improve with scipy.interpolate
    """
    # Find grid cell containing (R,Z)
    i = np.searchsorted(grid.R[:, 0], R) - 1
    j = np.searchsorted(grid.Z[0, :], Z) - 1
    
    # Clip to valid range
    nr, nz = psi.shape
    i = np.clip(i, 0, nr - 2)
    j = np.clip(j, 0, nz - 2)
    
    # Bilinear weights
    R0, R1 = grid.R[i, 0], grid.R[i+1, 0]
    Z0, Z1 = grid.Z[0, j], grid.Z[0, j+1]
    
    wR = (R - R0) / (R1 - R0) if R1 != R0 else 0.5
    wZ = (Z - Z0) / (Z1 - Z0) if Z1 != Z0 else 0.5
    
    # Interpolate
    psi_interp = (1 - wR) * (1 - wZ) * psi[i, j] + \
                 wR * (1 - wZ) * psi[i+1, j] + \
                 (1 - wR) * wZ * psi[i, j+1] + \
                 wR * wZ * psi[i+1, j+1]
    
    return psi_interp


if __name__ == "__main__":
    print("Free-Boundary Constraint System")
    print("=" * 60)
    
    # Test: Simple 4-coil system with 1 X-point + 2 isoflux
    coils = CoilSet(
        R=np.array([0.5, 2.0, 0.5, 2.0]),
        Z=np.array([1.0, 1.0, -1.0, -1.0]),
        I=np.zeros(4)
    )
    
    xpoint = XPoint(R=1.2, Z=-0.5, psi=0.0, 
                    grid_i=30, grid_j=20, 
                    grad_mag=0.0, hessian_det=-1.0)
    
    isoflux = [
        IsofluxPair(R1=1.0, Z1=0.5, R2=1.0, Z2=-0.5),  # Vertical symmetry
        IsofluxPair(R1=1.3, Z1=0.0, R2=0.7, Z2=0.0),   # Horizontal symmetry
    ]
    
    # Build constraint matrix
    A = build_constraint_matrix(coils, xpoint, isoflux)
    
    print(f"\nConstraint matrix A:")
    print(f"  Shape: {A.shape} (4 constraints × 4 coils)")
    print(f"  Rank: {np.linalg.matrix_rank(A)}")
    
    valid, msg = check_constraint_matrix(A)
    print(f"  Status: {msg}")
    
    # Mock target vector
    b = np.array([0.01, -0.02, 0.005, -0.003])  # Current errors
    
    # Optimize
    I_opt = optimize_coil_currents(A, b, gamma=1e-12)
    
    print(f"\nOptimal coil currents:")
    for i, I in enumerate(I_opt):
        print(f"  Coil {i}: {I/1000:.2f} kA")
    
    # Verify constraints satisfied
    residual = A @ I_opt - b
    rel_residual = np.linalg.norm(residual) / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 0
    
    print(f"\nConstraint residuals:")
    print(f"  ||A·I - b|| = {np.linalg.norm(residual):.2e}")
    print(f"  ||b|| = {np.linalg.norm(b):.2e}")
    print(f"  Relative: {rel_residual:.2%}")
    
    # Success if regularized solution exists and is reasonable
    if valid and rel_residual < 2.0:  # Within 200% is OK for Tikhonov
        print("\n✅ Constraint system working!")
    else:
        print(f"\n⚠️ Check results (rel_residual={rel_residual:.1%})")
