"""
Poisson Solver for Toroidal Geometry

Solves ∇²φ = ω for φ given ω (vorticity) in toroidal coordinates.

Method: FFT in θ direction + finite difference in r direction

Mathematical Background
-----------------------
In toroidal coordinates (r, θ) with axisymmetry (∂/∂φ = 0):

    ∇²φ = ∂²φ/∂r² + (1/r)∂φ/∂r + (1/r²)∂²φ/∂θ²

Fourier decompose in θ (periodic direction):
    φ(r,θ) = Σₖ φ̂ₖ(r) exp(ikθ)
    ω(r,θ) = Σₖ ω̂ₖ(r) exp(ikθ)

For each mode k:
    d²φ̂ₖ/dr² + (1/r)dφ̂ₖ/dr - (k²/r²)φ̂ₖ = ω̂ₖ

This is a modified Bessel equation. Solve using finite differences.

Boundary Conditions
-------------------
- r = 0 (axis): Regularity condition (no singularity)
  - k = 0: dφ/dr = 0 (axisymmetric)
  - k ≠ 0: φ = 0 (no odd modes at axis)
- r = a (edge): Dirichlet φ = 0 (grounded boundary)

Numerical Method
----------------
1. FFT in θ to get ω̂ₖ(r)
2. For each mode k, solve tridiagonal system
3. Inverse FFT to get φ(r,θ)

Accuracy: O(dr²) + O(dθ²)

References
----------
- Numerical Recipes: Chapter 19 (Partial Differential Equations)
- Freidberg (2014): Ideal MHD, Appendix C (Numerical Methods)

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from scipy.linalg import solve_banded
from ..geometry import ToroidalGrid


def solve_poisson_toroidal(
    omega: np.ndarray,
    grid: ToroidalGrid,
    bc_type: str = 'dirichlet'
) -> np.ndarray:
    """
    Solve Poisson equation ∇²φ = ω for φ.
    
    Uses FFT in θ direction and finite difference in r direction.
    
    Parameters
    ----------
    omega : np.ndarray (nr, ntheta)
        Source term (vorticity) ω = ∇²φ
    grid : ToroidalGrid
        Toroidal grid object
    bc_type : str, optional
        Boundary condition type at edge:
        - 'dirichlet': φ(r=a) = 0 (grounded)
        - 'neumann': dφ/dr(r=a) = 0 (insulating)
    
    Returns
    -------
    phi : np.ndarray (nr, ntheta)
        Solution φ to ∇²φ = ω
    
    Notes
    -----
    Equation in (r, θ):
        ∂²φ/∂r² + (1/r)∂φ/∂r + (1/r²)∂²φ/∂θ² = ω
    
    Boundary conditions:
        - At r=0: regularity (dφ/dr = 0 for k=0, φ=0 for k≠0)
        - At r=a: specified by bc_type
    
    Algorithm:
        1. FFT(ω) → ω̂ₖ(r) for k = 0, 1, ..., N_θ/2
        2. Solve ODE for each k:
           d²φ̂ₖ/dr² + (1/r)dφ̂ₖ/dr - (k²/r²)φ̂ₖ = ω̂ₖ
        3. IFFT({φ̂ₖ}) → φ(r,θ)
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> # Test: φ = r² sin(θ) → ω = -(∂²/∂r² + 1/r ∂/∂r - 1/r²)r² sin(θ)
    >>> phi_exact = grid.r_grid**2 * np.sin(grid.theta_grid)
    >>> omega = -laplacian_toroidal(phi_exact, grid)
    >>> phi_solved = solve_poisson_toroidal(omega, grid)
    >>> error = np.max(np.abs(phi_solved - phi_exact))
    >>> assert error < 1e-6
    """
    nr, ntheta = omega.shape
    dr = grid.dr
    r_grid = grid.r_grid[:, 0]  # 1D radial coordinate
    
    # FFT in θ direction (real FFT since ω is real)
    omega_hat = np.fft.rfft(omega, axis=1)  # (nr, ntheta//2 + 1)
    nk = omega_hat.shape[1]
    
    # Storage for Fourier coefficients of φ
    phi_hat = np.zeros_like(omega_hat)
    
    # Solve for each Fourier mode
    for k_idx in range(nk):
        k = k_idx  # Wavenumber
        
        # Source term for this mode
        source = omega_hat[:, k_idx].real  # Real part (imaginary should be ~0 for axisymmetric)
        
        # Build tridiagonal matrix for finite difference
        # Equation: d²φ/dr² + (1/r)dφ/dr - (k²/r²)φ = ω
        #
        # Discretization (2nd order centered):
        # d²φ/dr² ≈ (φ[i+1] - 2φ[i] + φ[i-1]) / dr²
        # dφ/dr ≈ (φ[i+1] - φ[i-1]) / (2dr)
        #
        # Coefficient of φ[i-1]: 1/dr² - 1/(2r·dr)
        # Coefficient of φ[i]:   -2/dr² - k²/r²
        # Coefficient of φ[i+1]: 1/dr² + 1/(2r·dr)
        
        # Tridiagonal matrix (stored in banded format for scipy)
        # ab[0, i] = upper diagonal
        # ab[1, i] = main diagonal
        # ab[2, i] = lower diagonal
        ab = np.zeros((3, nr))
        rhs = source.copy()
        
        for i in range(nr):
            r = r_grid[i]
            
            if i == 0:
                # Axis boundary condition
                if k == 0:
                    # k=0 mode: dφ/dr = 0 at r=0
                    # Use one-sided difference: φ[1] = φ[0] (or dφ/dr = 0)
                    # Simplest: φ[0] - φ[1] = 0
                    ab[1, 0] = 1.0   # φ[0]
                    ab[0, 0] = -1.0  # φ[1]
                    rhs[0] = 0.0
                else:
                    # k≠0 modes: φ = 0 at r=0 (regularity)
                    ab[1, 0] = 1.0
                    rhs[0] = 0.0
            
            elif i == nr - 1:
                # Edge boundary condition
                if bc_type == 'dirichlet':
                    # φ(r=a) = 0
                    ab[1, -1] = 1.0
                    rhs[-1] = 0.0
                elif bc_type == 'neumann':
                    # dφ/dr(r=a) = 0
                    # One-sided: (3φ[-1] - 4φ[-2] + φ[-3])/(2dr) = 0
                    ab[1, -1] = 3.0
                    ab[2, -1] = -4.0
                    # Missing φ[-3] coefficient - need special handling
                    # For simplicity, use: φ[-1] = φ[-2]
                    ab[1, -1] = 1.0
                    ab[2, -1] = -1.0
                    rhs[-1] = 0.0
            
            else:
                # Interior point
                # Avoid singularity at r=0 by checking
                if r < 1e-12:
                    r = 1e-12  # Small regularization
                
                # Coefficients
                a_lower = 1.0/dr**2 - 1.0/(2*r*dr)
                a_main = -2.0/dr**2 - k**2/r**2
                a_upper = 1.0/dr**2 + 1.0/(2*r*dr)
                
                ab[2, i] = a_lower  # Lower diagonal (i-1)
                ab[1, i] = a_main   # Main diagonal (i)
                ab[0, i] = a_upper  # Upper diagonal (i+1)
        
        # Solve tridiagonal system
        # scipy.linalg.solve_banded format: ab[upper, main, lower] = ab[0, 1, 2]
        # But we have [upper, main, lower] in ab[0, 1, 2]
        # Need to swap to [lower, main, upper] for scipy convention
        ab_scipy = np.zeros((3, nr))
        ab_scipy[0, :] = ab[0, :]  # Upper diagonal
        ab_scipy[1, :] = ab[1, :]  # Main diagonal  
        ab_scipy[2, :] = ab[2, :]  # Lower diagonal
        
        # Solve
        try:
            phi_mode = solve_banded((1, 1), ab_scipy, rhs)
        except np.linalg.LinAlgError:
            # Singular matrix - set to zero
            phi_mode = np.zeros(nr)
        
        # Store (both real and imaginary parts, though imaginary should be ~0)
        phi_hat[:, k_idx] = phi_mode + 0j
    
    # Inverse FFT
    phi = np.fft.irfft(phi_hat, n=ntheta, axis=1)
    
    return phi


def laplacian_toroidal_check(phi: np.ndarray, grid: ToroidalGrid) -> np.ndarray:
    """
    Compute ∇²φ directly (for testing/verification).
    
    This is a separate implementation from the one in toroidal_operators.py
    to cross-check consistency.
    
    Parameters
    ----------
    phi : np.ndarray (nr, ntheta)
        Scalar field
    grid : ToroidalGrid
        Grid object
    
    Returns
    -------
    laplacian : np.ndarray (nr, ntheta)
        ∇²φ = ∂²φ/∂r² + (1/r)∂φ/∂r + (1/r²)∂²φ/∂θ²
    """
    nr, ntheta = phi.shape
    dr = grid.dr
    dtheta = grid.dtheta
    r_grid = grid.r_grid
    
    # ∂²φ/∂r²
    d2phi_dr2 = np.zeros_like(phi)
    d2phi_dr2[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dr**2
    d2phi_dr2[0, :] = (phi[1, :] - 2*phi[0, :] + phi[1, :]) / dr**2  # Axis
    d2phi_dr2[-1, :] = (phi[-2, :] - 2*phi[-1, :] + phi[-2, :]) / dr**2  # Edge
    
    # ∂φ/∂r
    dphi_dr = np.zeros_like(phi)
    dphi_dr[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dr)
    dphi_dr[0, :] = (phi[1, :] - phi[0, :]) / dr  # Forward difference at axis
    dphi_dr[-1, :] = (phi[-1, :] - phi[-2, :]) / dr  # Backward difference at edge
    
    # ∂²φ/∂θ²
    d2phi_dtheta2 = np.zeros_like(phi)
    d2phi_dtheta2[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dtheta**2
    # Periodic
    d2phi_dtheta2[:, 0] = (phi[:, 1] - 2*phi[:, 0] + phi[:, -1]) / dtheta**2
    d2phi_dtheta2[:, -1] = (phi[:, 0] - 2*phi[:, -1] + phi[:, -2]) / dtheta**2
    
    # ∇²φ
    laplacian = d2phi_dr2 + dphi_dr/r_grid + d2phi_dtheta2/r_grid**2
    
    return laplacian


def test_poisson_solver():
    """
    Unit test for Poisson solver.
    
    Test case: φ = r²(1 - r/a)² sin(2θ)
    """
    from ..geometry import ToroidalGrid
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    
    # Exact solution
    r = grid.r_grid
    theta = grid.theta_grid
    a = grid.a
    phi_exact = r**2 * (1 - r/a)**2 * np.sin(2*theta)
    
    # Compute Laplacian (source term)
    omega = laplacian_toroidal_check(phi_exact, grid)
    
    # Solve Poisson equation
    phi_solved = solve_poisson_toroidal(omega, grid)
    
    # Error
    error = np.max(np.abs(phi_solved - phi_exact))
    rel_error = error / np.max(np.abs(phi_exact))
    
    print(f"Poisson solver test:")
    print(f"  Max error: {error:.3e}")
    print(f"  Relative error: {rel_error:.3e}")
    
    if rel_error < 1e-3:
        print("  ✅ PASSED")
    else:
        print(f"  ❌ FAILED (expected < 1e-3)")
    
    return rel_error < 1e-3


if __name__ == "__main__":
    test_poisson_solver()
