"""
3D Poisson Solver via Per-Mode FFT.

Solves ∇²φ = ω in 3D toroidal geometry using Fourier mode decomposition.

Algorithm (from learning notes 2.3-3d-poisson-solver.md):
1. FFT in ζ: ω(r,θ,ζ) → ω̂(r,θ,k)
2. Per-mode 2D Poisson: ∇_⊥²φ̂_k - k²φ̂_k = ω̂_k
3. Tridiagonal solve in r (per θ, per k)
4. Inverse FFT: φ̂(r,θ,k) → φ(r,θ,ζ)

Physical Interpretation:
- φ: Electrostatic potential (drives E×B flow)
- ω: Vorticity (curl of velocity)
- Each Fourier mode k solved independently → perfect parallelization

References:
- BOUT++ src/invert/laplace/impls/cyclic/cyclic_laplace.cxx
- Learning notes: 2.3-3d-poisson-solver.md
- Design doc: docs/v1.4/DESIGN.md §4.3
"""

import numpy as np
from scipy.linalg import solve_banded
from typing import Optional, Tuple, Literal
from ..operators.fft.transforms import forward_fft, inverse_fft, fft_frequencies


def solve_poisson_3d(
    omega: np.ndarray,
    grid: 'Grid3D',
    bc: Literal['dirichlet', 'neumann'] = 'dirichlet'
) -> np.ndarray:
    """
    Solve 3D Poisson equation ∇²φ = ω in cylindrical geometry.
    
    Args:
        omega: Source term (nr, nθ, nζ)
        grid: Grid3D object with r, θ, ζ coordinates
        bc: Boundary condition at r=0,a ('dirichlet' or 'neumann')
    
    Returns:
        phi: Solution (nr, nθ, nζ)
    
    Raises:
        ValueError: If omega shape doesn't match grid
        ValueError: If bc type not supported
    
    Notes:
        - Uses per-mode FFT algorithm (BOUT++ cyclic_laplace)
        - Dirichlet BC: φ(r=0) = φ(r=a) = 0
        - Neumann BC: ∂φ/∂r(r=0) = ∂φ/∂r(r=a) = 0
        - Assumes periodic BC in θ and ζ
    
    Examples:
        >>> # Test on analytical solution φ = sin(πr/a) sin(θ) cos(ζ)
        >>> grid = Grid3D(nr=32, nθ=64, nζ=64, r_max=1.0)
        >>> r, θ, ζ = grid.meshgrid()
        >>> phi_exact = np.sin(np.pi*r) * np.sin(θ) * np.cos(ζ)
        >>> 
        >>> # Compute RHS: ω = ∇²φ
        >>> omega = compute_laplacian(phi_exact, grid)
        >>> 
        >>> # Solve
        >>> phi_num = solve_poisson_3d(omega, grid, bc='dirichlet')
        >>> 
        >>> # Verify
        >>> error = np.max(np.abs(phi_num - phi_exact))
        >>> assert error < 1e-8
    """
    nr, nθ, nζ = grid.nr, grid.nθ, grid.nζ
    
    if omega.shape != (nr, nθ, nζ):
        raise ValueError(
            f"omega shape {omega.shape} doesn't match grid ({nr}, {nθ}, {nζ})"
        )
    
    if bc not in {'dirichlet', 'neumann'}:
        raise ValueError(f"BC type '{bc}' not supported. Use 'dirichlet' or 'neumann'.")
    
    # Step 1: Forward FFT in ζ direction (axis=2)
    omega_hat = forward_fft(omega, axis=2)  # (nr, nθ, nζ//2+1) complex
    phi_hat = np.zeros_like(omega_hat, dtype=complex)
    
    # Step 2: Frequency array for ζ modes
    k_ζ = fft_frequencies(nζ, domain_length=grid.Lζ)  # [0, 1, ..., nζ//2]
    
    # Step 3: Solve per-mode 2D Poisson
    for k_idx, kz in enumerate(k_ζ):
        for theta_idx in range(nθ):
            # Build tridiagonal system for this (θ, k) slice
            a, b, c, rhs_vec = _build_tridiagonal_coeffs(
                grid.r, grid.dr, kz, omega_hat[:, theta_idx, k_idx], bc
            )
            
            # Solve tridiagonal system
            phi_hat[:, theta_idx, k_idx] = _solve_tridiagonal_complex(a, b, c, rhs_vec)
    
    # Step 4: Inverse FFT to real space
    phi = inverse_fft(phi_hat, n=nζ, axis=2).real
    
    return phi


def _build_tridiagonal_coeffs(
    r: np.ndarray,
    dr: float,
    kz: float,
    rhs: np.ndarray,
    bc: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tridiagonal matrix for 1D Poisson in r direction.
    
    Equation (at fixed θ, k):
        ∇_r²φ̂_k - k_z²φ̂_k = ω̂_k
    
    Discretization (cylindrical Laplacian in r):
        (1/r) ∂/∂r(r ∂φ/∂r) - k²φ = ω
        → [1/dr² - 1/(2r·dr)]φ_{i-1} + [-2/dr² - k²]φ_i + [1/dr² + 1/(2r·dr)]φ_{i+1} = ω_i
    
    Args:
        r: Radial grid (nr,)
        dr: Grid spacing
        kz: Toroidal mode number
        rhs: RHS vector ω̂_k (nr,) complex
        bc: Boundary condition type
    
    Returns:
        a: Lower diagonal (nr-1,) complex
        b: Main diagonal (nr,) complex
        c: Upper diagonal (nr-1,) complex
        rhs_modified: RHS with BC applied (nr,)
    
    Notes:
        - Complex coefficients arise from FFT (even for real ω)
        - First/last rows modified for BC
    """
    nr = len(r)
    
    # Coefficients (vectorized)
    # Note: r[0] might be 0, handle singularity
    r_safe = np.where(r > 1e-14, r, 1e-14)  # Avoid division by zero
    
    coef_r = 1.0 / (dr**2)
    coef_k = kz**2
    
    # Tridiagonal elements (standard 3-point stencil)
    # Lower diagonal (i-1)
    a = coef_r - 1.0 / (2 * r_safe[1:] * dr)
    
    # Main diagonal (i)
    b = -2.0 * coef_r - coef_k * np.ones(nr)
    
    # Upper diagonal (i+1)
    c = coef_r + 1.0 / (2 * r_safe[:-1] * dr)
    
    # Apply boundary conditions
    rhs_modified = rhs.copy()
    
    if bc == 'dirichlet':
        # φ(r=0) = 0, φ(r=a) = 0
        # First row: φ_0 = 0 → b[0]=1, c[0]=0, rhs[0]=0
        b[0] = 1.0
        if nr > 1:
            c[0] = 0.0
        rhs_modified[0] = 0.0
        
        # Last row: φ_{nr-1} = 0
        if nr > 1:
            a[-1] = 0.0
        b[-1] = 1.0
        rhs_modified[-1] = 0.0
        
    elif bc == 'neumann':
        # ∂φ/∂r(r=0) = 0 → φ_0 = φ_1 (central difference)
        # Modify first row: -φ_0 + φ_1 = 0 (or use 2nd-order one-sided)
        b[0] = -1.0
        if nr > 1:
            c[0] = 1.0
        rhs_modified[0] = 0.0
        
        # ∂φ/∂r(r=a) = 0
        if nr > 1:
            a[-1] = 1.0
        b[-1] = -1.0
        rhs_modified[-1] = 0.0
    
    return a, b, c, rhs_modified


def _solve_tridiagonal_complex(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    rhs: np.ndarray
) -> np.ndarray:
    """
    Solve tridiagonal system with complex coefficients.
    
    Args:
        a: Lower diagonal (n-1,)
        b: Main diagonal (n,)
        c: Upper diagonal (n-1,)
        rhs: RHS vector (n,)
    
    Returns:
        x: Solution vector (n,)
    
    Notes:
        - Uses scipy.linalg.solve_banded (stable for complex matrices)
        - Format: ab[0,i] = c[i-1], ab[1,i] = b[i], ab[2,i] = a[i]
    
    References:
        - scipy.linalg.solve_banded documentation
        - BOUT++ cyclic_reduction.cxx (Thomas algorithm)
    """
    n = len(b)
    
    # scipy.linalg.solve_banded format:
    # ab[u + i - j, j] = A[i, j]  where u=1 (upper bandwidth)
    # For tridiagonal: ab[0,:] = upper, ab[1,:] = main, ab[2,:] = lower
    ab = np.zeros((3, n), dtype=complex)
    
    ab[1, :] = b           # Main diagonal
    ab[0, 1:] = c          # Upper diagonal (shifted)
    ab[2, :-1] = a         # Lower diagonal (shifted)
    
    # Solve
    x = solve_banded((1, 1), ab, rhs)
    
    return x


def compute_laplacian_3d(
    phi: np.ndarray,
    grid: 'Grid3D'
) -> np.ndarray:
    """
    Compute 3D Laplacian ∇²φ in cylindrical coordinates.
    
    ∇²φ = (1/r) ∂/∂r(r ∂φ/∂r) + (1/r²) ∂²φ/∂θ² + ∂²φ/∂ζ²
    
    Args:
        phi: Input field (nr, nθ, nζ)
        grid: Grid3D object
    
    Returns:
        laplacian: ∇²φ (nr, nθ, nζ)
    
    Notes:
        - Used for verification: compute ω = ∇²φ, then solve ∇²φ' = ω
        - Uses 2nd-order finite differences in r,θ and FFT in ζ
    
    Examples:
        >>> # Verify on φ = r² cos(θ) → ∇²φ = 4 cos(θ) - r² cos(θ)
        >>> r, θ, ζ = grid.meshgrid()
        >>> phi = r**2 * np.cos(θ)
        >>> lap = compute_laplacian_3d(phi, grid)
        >>> lap_exact = 4*np.cos(θ) - r**2*np.cos(θ)
        >>> error = np.max(np.abs(lap - lap_exact))
    """
    from ..operators.fft.derivatives import toroidal_laplacian
    
    nr, nθ, nζ = grid.nr, grid.nθ, grid.nζ
    laplacian = np.zeros_like(phi)
    
    # 1. Radial part: ∇_r²φ = ∂²φ/∂r² + (1/r) ∂φ/∂r
    # Create r array for operations
    r_1d = grid.r  # (nr,)
    r_safe_1d = np.where(r_1d > 1e-14, r_1d, 1e-14)
    
    # ∂φ/∂r (2nd-order central difference)
    dφ_dr = np.zeros_like(phi)
    dφ_dr[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * grid.dr)
    dφ_dr[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / grid.dr  # Forward at r=0
    dφ_dr[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / grid.dr  # Backward at r=a
    
    # ∂²φ/∂r² (2nd-order central difference)
    d2φ_dr2 = np.zeros_like(phi)
    d2φ_dr2[1:-1, :, :] = (phi[2:, :, :] - 2*phi[1:-1, :, :] + phi[:-2, :, :]) / (grid.dr**2)
    # Boundaries (use one-sided differences)
    d2φ_dr2[0, :, :] = (phi[2, :, :] - 2*phi[1, :, :] + phi[0, :, :]) / (grid.dr**2)
    d2φ_dr2[-1, :, :] = (phi[-1, :, :] - 2*phi[-2, :, :] + phi[-3, :, :]) / (grid.dr**2)
    
    # Radial Laplacian: ∂²φ/∂r² + (1/r) ∂φ/∂r
    # Special handling at r=0: use L'Hospital's rule or set (1/r) term to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        one_over_r = np.where(r_1d > 1e-10, 1.0/r_1d, 0.0)  # Set to 0 at r=0
    radial_laplacian = d2φ_dr2 + dφ_dr * one_over_r[:, np.newaxis, np.newaxis]
    
    # 2. Poloidal part: (1/r²) ∂²φ/∂θ²
    d2φ_dθ2 = np.zeros_like(phi)
    if nθ > 2:
        # Periodic BC in θ
        d2φ_dθ2[:, 1:-1, :] = (phi[:, 2:, :] - 2*phi[:, 1:-1, :] + phi[:, :-2, :]) / (grid.dθ**2)
        d2φ_dθ2[:, 0, :] = (phi[:, 1, :] - 2*phi[:, 0, :] + phi[:, -1, :]) / (grid.dθ**2)
        d2φ_dθ2[:, -1, :] = (phi[:, 0, :] - 2*phi[:, -1, :] + phi[:, -2, :]) / (grid.dθ**2)
    # else: d2φ_dθ2 = 0 (no poloidal variation)
    
    # Poloidal Laplacian with safe 1/r²
    with np.errstate(divide='ignore', invalid='ignore'):
        one_over_r2 = np.where(r_1d > 1e-10, 1.0/(r_1d**2), 0.0)
    poloidal_laplacian = d2φ_dθ2 * one_over_r2[:, np.newaxis, np.newaxis]
    
    # 3. Toroidal part: ∂²φ/∂ζ² (via FFT)
    toroidal_lap = toroidal_laplacian(phi, grid.dζ, axis=2)
    
    # Total Laplacian
    laplacian = radial_laplacian + poloidal_laplacian + toroidal_lap
    
    return laplacian


def verify_poisson_solver(
    grid: 'Grid3D',
    analytical_solution: callable,
    bc: str = 'dirichlet',
    tolerance: float = 1e-8
) -> dict:
    """
    Verify Poisson solver against analytical solution.
    
    Args:
        grid: Grid3D object
        analytical_solution: Function phi_exact(r, θ, ζ) → exact solution
        bc: Boundary condition
        tolerance: Acceptance tolerance
    
    Returns:
        Dictionary with verification results:
            - 'passed': bool
            - 'max_error': float
            - 'residual': float (∥∇²φ_num - ω∥)
            - 'phi_numerical': array
            - 'phi_exact': array
    
    Examples:
        >>> # Test on φ = sin(πr/a) sin(θ) cos(ζ)
        >>> def phi_exact(r, θ, ζ):
        ...     return np.sin(np.pi*r) * np.sin(θ) * np.cos(ζ)
        >>> 
        >>> result = verify_poisson_solver(grid, phi_exact, tolerance=1e-8)
        >>> print(f"Error: {result['max_error']:.2e}, Passed: {result['passed']}")
    """
    r, θ, ζ = grid.meshgrid()
    
    # Exact solution
    phi_exact = analytical_solution(r, θ, ζ)
    
    # Compute RHS: ω = ∇²φ_exact
    omega = compute_laplacian_3d(phi_exact, grid)
    
    # Solve Poisson equation
    phi_num = solve_poisson_3d(omega, grid, bc=bc)
    
    # Solution error
    solution_error = np.max(np.abs(phi_num - phi_exact))
    
    # Residual error: ∇²φ_num - ω
    lap_phi_num = compute_laplacian_3d(phi_num, grid)
    residual_error = np.max(np.abs(lap_phi_num - omega))
    
    passed = (solution_error < tolerance) and (residual_error < tolerance)
    
    return {
        'passed': passed,
        'max_error': solution_error,
        'residual': residual_error,
        'phi_numerical': phi_num,
        'phi_exact': phi_exact,
    }
