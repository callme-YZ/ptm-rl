"""
Δ* (Grad-Shafranov) operator implementation

Δ* ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
      = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²

This is the proper operator for axisymmetric MHD equilibrium.

Reference: Grad-Shafranov equation
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def build_delta_star_matrix(R_grid: np.ndarray, dR: float, dZ: float) -> csr_matrix:
    """
    Build sparse matrix for Δ* operator
    
    Δ* ψ = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²
    
    Using second-order finite differences:
        ∂²ψ/∂R² ≈ (ψ[i+1] - 2ψ[i] + ψ[i-1]) / dR²
        ∂ψ/∂R ≈ (ψ[i+1] - ψ[i-1]) / (2dR)
        ∂²ψ/∂Z² ≈ (ψ[j+1] - 2ψ[j] + ψ[j-1]) / dZ²
    
    Args:
        R_grid: (nr, nz) 2D R coordinates
        dR: R spacing
        dZ: Z spacing
        
    Returns:
        A: Sparse matrix for Δ* operator
        
    Notes:
        - Uses 5-point stencil
        - Assumes ψ = 0 on boundary
    """
    nr, nz = R_grid.shape
    N = nr * nz
    
    # Coefficients for each point
    # Δ* = c_R+ ψ[i+1,j] + c_R- ψ[i-1,j] + c_Z+ ψ[i,j+1] + c_Z- ψ[i,j-1] + c_0 ψ[i,j]
    
    # Initialize coefficient arrays
    c_R_plus = np.zeros((nr, nz))   # Coefficient for ψ[i+1,j]
    c_R_minus = np.zeros((nr, nz))  # Coefficient for ψ[i-1,j]
    c_Z_plus = np.zeros((nr, nz))   # Coefficient for ψ[i,j+1]
    c_Z_minus = np.zeros((nr, nz))  # Coefficient for ψ[i,j-1]
    c_center = np.zeros((nr, nz))   # Coefficient for ψ[i,j]
    
    # Compute coefficients for interior points
    for i in range(1, nr-1):
        for j in range(1, nz-1):
            R = R_grid[i, j]
            
            # ∂²ψ/∂R²: (ψ[i+1] - 2ψ[i] + ψ[i-1]) / dR²
            c_R_plus[i, j] += 1.0 / dR**2
            c_center[i, j] += -2.0 / dR**2
            c_R_minus[i, j] += 1.0 / dR**2
            
            # -(1/R) ∂ψ/∂R: -(1/R) × (ψ[i+1] - ψ[i-1]) / (2dR)
            c_R_plus[i, j] += -1.0 / (R * 2 * dR)
            c_R_minus[i, j] += 1.0 / (R * 2 * dR)
            
            # ∂²ψ/∂Z²: (ψ[j+1] - 2ψ[j] + ψ[j-1]) / dZ²
            c_Z_plus[i, j] += 1.0 / dZ**2
            c_center[i, j] += -2.0 / dZ**2
            c_Z_minus[i, j] += 1.0 / dZ**2
    
    # Build sparse matrix using diagonals
    # For 2D grid flattened to 1D: index = i * nz + j
    
    diag_main = c_center.flatten()
    diag_R_plus = c_R_plus.flatten()[:-nz]   # Shift by nz
    diag_R_minus = c_R_minus.flatten()[nz:]
    diag_Z_plus = c_Z_plus.flatten()[:-1]    # Shift by 1
    diag_Z_minus = c_Z_minus.flatten()[1:]
    
    # Create sparse matrix
    diagonals = [
        diag_main,
        diag_R_plus,
        diag_R_minus,
        diag_Z_plus,
        diag_Z_minus,
    ]
    
    offsets = [0, nz, -nz, 1, -1]
    
    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    
    # Enforce boundary conditions: ψ = 0
    # Set boundary rows to identity
    for i in range(nr):
        for j in range(nz):
            idx = i * nz + j
            if i == 0 or i == nr-1 or j == 0 or j == nz-1:
                # Boundary point: set row to [0 ... 1 ... 0]
                A.data[A.indptr[idx]:A.indptr[idx+1]] = 0
                A[idx, idx] = 1.0
    
    return A


def solve_delta_star(
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    dR: float,
    dZ: float,
    rhs: np.ndarray,
    bc_value: float = 0.0
) -> np.ndarray:
    """
    Solve Δ* ψ = rhs with boundary conditions
    
    Args:
        R_grid: (nr, nz) R coordinates
        Z_grid: (nr, nz) Z coordinates
        dR: R spacing
        dZ: Z spacing
        rhs: (nr, nz) Right-hand side (-μ₀ R J_φ)
        bc_value: Boundary condition value (default: 0)
        
    Returns:
        psi: (nr, nz) Solution
    """
    nr, nz = R_grid.shape
    
    # Build Δ* matrix
    A = build_delta_star_matrix(R_grid, dR, dZ)
    
    # Flatten RHS
    b = rhs.flatten()
    
    # Apply boundary conditions to RHS
    for i in range(nr):
        for j in range(nz):
            idx = i * nz + j
            if i == 0 or i == nr-1 or j == 0 or j == nz-1:
                b[idx] = bc_value
    
    # Solve
    psi_flat = spsolve(A, b)
    psi = psi_flat.reshape((nr, nz))
    
    return psi


def test_delta_star():
    """Test Δ* operator with known solution"""
    import sys
    sys.path.insert(0, '..')
    from picard_gs_solver import Grid
    
    # Setup
    R_1d = np.linspace(1.0, 2.0, 33)
    Z_1d = np.linspace(-0.5, 0.5, 33)
    grid = Grid.from_1d(R_1d, Z_1d)
    
    # Test with simple function: ψ = R² + Z²
    # Δ* ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
    #      = 2 - (1/R)×2R + 2
    #      = 2 - 2 + 2 = 2
    
    # So if we solve Δ* ψ = 2, should get ψ ≈ R² + Z² (up to BC)
    
    rhs = 2.0 * np.ones_like(grid.R)
    psi = solve_delta_star(grid.R, grid.Z, grid.dR, grid.dZ, rhs)
    
    # Check interior (not boundary)
    psi_expected = grid.R**2 + grid.Z**2
    psi_expected -= psi_expected[0, 0]  # Normalize
    
    interior = psi[5:-5, 5:-5]
    expected_interior = psi_expected[5:-5, 5:-5]
    
    error = np.abs(interior - expected_interior).max()
    print(f"Δ* operator test:")
    print(f"  Max error: {error:.3e}")
    print(f"  {'✓ PASS' if error < 0.1 else '✗ FAIL'}")


if __name__ == "__main__":
    test_delta_star()

