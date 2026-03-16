"""
X-point Detection for Free-Boundary Equilibrium

Implements saddle point search algorithm to locate X-points
in poloidal flux solutions.

Reference: Design doc Section 3.1, Appendix A.2
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class XPoint:
    """X-point data structure"""
    R: float  # Major radius (m)
    Z: float  # Vertical position (m)
    psi: float  # Flux value at X-point (Wb)
    grid_i: int  # Grid index (R direction)
    grid_j: int  # Grid index (Z direction)
    grad_mag: float  # |∇ψ| magnitude (should be ~0)
    hessian_det: float  # det(Hessian) (should be < 0)


def find_xpoints(psi: np.ndarray, grid, threshold: Optional[float] = None) -> List[XPoint]:
    """
    Find all X-points in flux solution via saddle point search
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Poloidal flux
    grid : Grid
        Computational grid
    threshold : float, optional
        Gradient magnitude threshold. If None, auto-computed.
        
    Returns
    -------
    xpoints : List[XPoint]
        Detected X-points (empty if none found)
    """
    
    # Auto-compute threshold if not provided
    if threshold is None:
        threshold = compute_xpoint_threshold(psi, grid)
    
    # Step 1: Find candidate points where |∇ψ| is small
    grad_R, grad_Z = compute_gradients(psi, grid)
    grad_mag = np.sqrt(grad_R**2 + grad_Z**2)
    
    # Candidates: interior points with small gradient
    nr, nz = psi.shape
    candidates = []
    
    for i in range(2, nr-2):  # Avoid boundaries (need neighbors for Hessian)
        for j in range(2, nz-2):
            if grad_mag[i, j] < threshold:
                candidates.append((i, j))
    
    # Step 2: Check Hessian for saddle points
    xpoints = []
    
    for i, j in candidates:
        H = compute_hessian(psi, i, j, grid)
        det_H = H[0,0] * H[1,1] - H[0,1]**2
        
        if det_H < 0:  # Saddle point (not extremum)
            # Create X-point object
            xp = XPoint(
                R=grid.R[i, j],
                Z=grid.Z[i, j],
                psi=psi[i, j],
                grid_i=i,
                grid_j=j,
                grad_mag=grad_mag[i, j],
                hessian_det=det_H
            )
            xpoints.append(xp)
    
    return xpoints


def compute_xpoint_threshold(psi: np.ndarray, grid) -> float:
    """
    Adaptive threshold based on plasma scale
    
    Physics: threshold = 1% of typical gradient
    Typical gradient ≈ Δψ / a (flux range / minor radius)
    
    Returns
    -------
    threshold : float
        Gradient magnitude threshold (Wb/m)
    """
    # Flux range
    psi_axis = psi.max()
    psi_edge = 0.0  # Assume separatrix at 0
    delta_psi = abs(psi_axis - psi_edge)
    
    # Estimate minor radius from flux contours
    a_plasma = estimate_minor_radius(psi, grid)
    
    # Typical gradient
    grad_typical = delta_psi / a_plasma if a_plasma > 0 else delta_psi
    
    # Threshold: 1% of typical
    threshold = 0.01 * grad_typical
    
    # Safety bounds: [10 μWb/m, 10 mWb/m]
    threshold = np.clip(threshold, 1e-5, 1e-2)
    
    return threshold


def estimate_minor_radius(psi: np.ndarray, grid) -> float:
    """
    Estimate plasma minor radius from mid-plane flux contour
    
    Method: Find width of ψ > ψ_mid region at Z=0
    """
    psi_mid = 0.5 * psi.max()
    
    # Find Z=0 slice (or closest)
    j_mid = np.argmin(np.abs(grid.Z[0, :]))
    psi_slice = psi[:, j_mid]
    R_slice = grid.R[:, j_mid]
    
    # Width of plasma at mid-plane
    inside = psi_slice > psi_mid
    if not np.any(inside):
        # Fallback: use grid extent
        return (grid.R.max() - grid.R.min()) / 4
    
    R_min = R_slice[inside].min()
    R_max = R_slice[inside].max()
    
    a = (R_max - R_min) / 2
    return max(a, 0.1)  # At least 10cm


def compute_gradients(psi: np.ndarray, grid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∇ψ = (∂ψ/∂R, ∂ψ/∂Z) via central difference
    
    Returns
    -------
    grad_R : ndarray (nr, nz)
        ∂ψ/∂R
    grad_Z : ndarray (nr, nz)
        ∂ψ/∂Z
    """
    nr, nz = psi.shape
    dR = grid.dR
    dZ = grid.dZ
    
    grad_R = np.zeros_like(psi)
    grad_Z = np.zeros_like(psi)
    
    # Central difference (interior)
    grad_R[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2*dR)
    grad_Z[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2*dZ)
    
    # Boundaries: one-sided difference
    grad_R[0, :] = (psi[1, :] - psi[0, :]) / dR
    grad_R[-1, :] = (psi[-1, :] - psi[-2, :]) / dR
    
    grad_Z[:, 0] = (psi[:, 1] - psi[:, 0]) / dZ
    grad_Z[:, -1] = (psi[:, -1] - psi[:, -2]) / dZ
    
    return grad_R, grad_Z


def compute_hessian(psi: np.ndarray, i: int, j: int, grid) -> np.ndarray:
    """
    Compute Hessian matrix at grid point (i,j)
    
    H = [[∂²ψ/∂R²,  ∂²ψ/∂R∂Z],
         [∂²ψ/∂R∂Z, ∂²ψ/∂Z²]]
    
    Returns
    -------
    H : ndarray (2,2)
        Hessian matrix
    """
    dR = grid.dR
    dZ = grid.dZ
    
    # Second derivatives
    psi_RR = (psi[i+1, j] - 2*psi[i, j] + psi[i-1, j]) / dR**2
    psi_ZZ = (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1]) / dZ**2
    
    # Cross derivative (4-point stencil)
    psi_RZ = (psi[i+1, j+1] - psi[i+1, j-1] - 
              psi[i-1, j+1] + psi[i-1, j-1]) / (4*dR*dZ)
    
    H = np.array([[psi_RR, psi_RZ],
                  [psi_RZ, psi_ZZ]])
    
    return H


def select_primary_xpoint(xpoints: List[XPoint]) -> Optional[XPoint]:
    """
    Select primary X-point from multiple candidates
    
    Priority:
    1. Lower X-point (Z < 0) - typical divertor
    2. Strongest saddle (largest |det(H)|)
    
    Returns
    -------
    xpoint : XPoint or None
        Primary X-point (None if input empty)
    """
    if not xpoints:
        return None
    
    # Prefer lower X-points
    lower_xpoints = [x for x in xpoints if x.Z < 0]
    if lower_xpoints:
        xpoints = lower_xpoints
    
    # Pick strongest saddle
    strengths = [abs(x.hessian_det) for x in xpoints]
    idx = np.argmax(strengths)
    
    return xpoints[idx]


def is_xpoint_valid(xpoint: XPoint, grid, margin: int = 2) -> Tuple[bool, str]:
    """
    Check if X-point is in interior (not on boundary)
    
    Parameters
    ----------
    margin : int
        Required distance from boundary (grid cells)
        
    Returns
    -------
    valid : bool
        True if interior
    reason : str
        Error message if invalid
    """
    nr, nz = grid.R.shape
    i, j = xpoint.grid_i, xpoint.grid_j
    
    if i < margin or i >= nr - margin:
        return False, f"X-point too close to R boundary (i={i})"
    
    if j < margin or j >= nz - margin:
        return False, f"X-point too close to Z boundary (j={j})"
    
    return True, "OK"


if __name__ == "__main__":
    # Simple test with synthetic saddle point
    print("X-point Finder Module")
    print("=" * 60)
    
    # Create synthetic flux with known X-point at (1.5, -0.5)
    R = np.linspace(1.0, 2.0, 65)
    Z = np.linspace(-1.0, 1.0, 65)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    # Mock grid
    class MockGrid:
        def __init__(self, R, Z):
            self.R = R
            self.Z = Z
            self.dR = R[1,0] - R[0,0]
            self.dZ = Z[0,1] - Z[0,0]
    
    grid = MockGrid(RR, ZZ)
    
    # Synthetic X-point: saddle at (1.5, -0.5)
    R_X, Z_X = 1.5, -0.5
    psi = -((RR - R_X)**2 - (ZZ - Z_X)**2)  # Saddle function
    
    # Find X-points
    xpoints = find_xpoints(psi, grid)
    
    print(f"\nFound {len(xpoints)} X-point(s)")
    
    if xpoints:
        xp = xpoints[0]
        print(f"\nX-point location:")
        print(f"  R = {xp.R:.3f} m (expected {R_X})")
        print(f"  Z = {xp.Z:.3f} m (expected {Z_X})")
        print(f"  |∇ψ| = {xp.grad_mag:.2e} Wb/m")
        print(f"  det(H) = {xp.hessian_det:.2e} (should be < 0)")
        
        valid, msg = is_xpoint_valid(xp, grid)
        print(f"  Valid: {valid} ({msg})")
    
    print("\n✅ X-point finder module complete")
