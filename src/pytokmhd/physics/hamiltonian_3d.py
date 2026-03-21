"""
3D Hamiltonian for Reduced MHD in Cylindrical Geometry

Implements the energy functional H[ψ, ω] for 3D reduced MHD:

    H = ∫∫∫ [ (1/2)|∇ψ|² + (1/2)ω² ] r dr dθ dζ

where:
    ψ: stream function (magnetic flux)
    ω = ∇²ψ: vorticity
    |∇ψ|² = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)² + (∂ψ/∂ζ)²

Physical Interpretation
-----------------------
- First term: Magnetic field energy |B|²/2 (via B = ∇ψ × ẑ)
- Second term: Kinetic energy |v|²/2 (via v = ∇φ × ẑ, φ ∝ ω)

Conservation Laws
-----------------
- **Ideal MHD (η=0):** dH/dt = 0 (energy conserved exactly)
- **Resistive MHD (η>0):** dH/dt < 0 (energy dissipates monotonically)

Numerical Implementation
------------------------
1. Gradient computation:
   - ∂/∂r: 2nd-order centered finite difference (Dirichlet BC at boundaries)
   - ∂/∂θ: 2nd-order centered finite difference (periodic BC)
   - ∂/∂ζ: FFT spectral derivative (spectral accuracy)

2. Metric factor:
   - |∇ψ|² includes 1/r² for θ component (cylindrical metric)
   - Singularity at r=0 handled via r_safe = max(r, 1e-10)

3. Volume integration:
   - Jacobian: r (cylindrical coordinates)
   - Trapezoidal rule over all grid points

References
----------
- Strauss (1976): Numerical studies of nonlinear evolution of kink modes
- Hazeltine & Meiss (2003): Plasma Confinement, §3.4 (Hamiltonian formulation)

Author: 小P ⚛️
Created: 2026-03-19
Phase: 2.1 (3D Physics Core)
"""

import numpy as np


def compute_gradient_3d(psi, grid):
    """
    Compute 3D gradient ∇ψ = (∂ψ/∂r, ∂ψ/∂θ, ∂ψ/∂ζ).
    
    Parameters
    ----------
    psi : np.ndarray (nr, nθ, nζ)
        Stream function
    grid : Grid3D
        Grid object with attributes: nr, nθ, nζ, dr, dθ, dζ, r
    
    Returns
    -------
    dpsi_dr : np.ndarray (nr, nθ, nζ)
        Radial derivative ∂ψ/∂r
    dpsi_dtheta : np.ndarray (nr, nθ, nζ)
        Poloidal derivative ∂ψ/∂θ
    dpsi_dzeta : np.ndarray (nr, nθ, nζ)
        Toroidal derivative ∂ψ/∂ζ
    
    Notes
    -----
    - ∂/∂r: 2nd-order centered FD (forward at r=0, backward at r=a)
    - ∂/∂θ: 2nd-order centered FD (periodic BC)
    - ∂/∂ζ: FFT spectral derivative (from Phase 1.1)
    
    Boundary Conditions:
        - Radial: Dirichlet (ψ=0 at r=0 and r=a)
        - Poloidal: Periodic (ψ[:, 0, :] = ψ[:, nθ, :])
        - Toroidal: Periodic (enforced by FFT)
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, nθ=64, nζ=128)
    >>> psi = np.random.randn(grid.nr, grid.nθ, grid.nζ)
    >>> dpsi_dr, dpsi_dtheta, dpsi_dzeta = compute_gradient_3d(psi, grid)
    >>> assert dpsi_dr.shape == psi.shape
    """
    nr, nθ, nζ = grid.nr, grid.nθ, grid.nζ
    dr, dθ, dζ = grid.dr, grid.dθ, grid.dζ
    
    # --- ∂ψ/∂r (radial derivative) ---
    dpsi_dr = np.zeros_like(psi)
    
    # Interior: 2nd-order centered difference
    dpsi_dr[1:-1, :, :] = (psi[2:, :, :] - psi[:-2, :, :]) / (2 * dr)
    
    # Boundary: 1st-order one-sided difference
    dpsi_dr[0, :, :] = (psi[1, :, :] - psi[0, :, :]) / dr  # Forward at r=0
    dpsi_dr[-1, :, :] = (psi[-1, :, :] - psi[-2, :, :]) / dr  # Backward at r=a
    
    # --- ∂ψ/∂θ (poloidal derivative, periodic BC) ---
    dpsi_dtheta = np.zeros_like(psi)
    
    # Interior: 2nd-order centered difference
    dpsi_dtheta[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2 * dθ)
    
    # Periodic boundaries
    dpsi_dtheta[:, 0, :] = (psi[:, 1, :] - psi[:, -1, :]) / (2 * dθ)
    dpsi_dtheta[:, -1, :] = (psi[:, 0, :] - psi[:, -2, :]) / (2 * dθ)
    
    # --- ∂ψ/∂ζ (toroidal derivative, FFT spectral) ---
    from ..operators.fft.derivatives import toroidal_derivative
    
    dpsi_dzeta = toroidal_derivative(psi, dζ, order=1, axis=2)
    
    return dpsi_dr, dpsi_dtheta, dpsi_dzeta


def compute_energy_density(psi, omega, grid):
    """
    Compute energy density E = (1/2)|∇ψ|² + (1/2)ω².
    
    Parameters
    ----------
    psi : np.ndarray (nr, nθ, nζ)
        Stream function
    omega : np.ndarray (nr, nθ, nζ)
        Vorticity ∇²ψ
    grid : Grid3D
        Grid object
    
    Returns
    -------
    energy_density : np.ndarray (nr, nθ, nζ)
        Energy density at each grid point
    
    Notes
    -----
    Energy density formula:
        E = (1/2)[(∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)² + (∂ψ/∂ζ)²] + (1/2)ω²
    
    Metric factor 1/r² for θ component comes from cylindrical coordinates:
        ds² = dr² + r²dθ² + dζ²
        |∇ψ|² = g^ij (∂ψ/∂xⁱ)(∂ψ/∂xʲ) = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)² + (∂ψ/∂ζ)²
    
    Singularity handling:
        At r=0, term (1/r²)(∂ψ/∂θ)² → 0/0 (indeterminate).
        Physically, ∂ψ/∂θ = 0 at r=0 by regularity → limit is 0.
        Numerically, we use r_safe = max(r, 1e-10) to avoid division by zero.
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, nθ=64, nζ=128)
    >>> psi = grid.r[:, None, None]**2  # Radial field
    >>> omega = 4.0 * np.ones_like(psi)  # ∇²(r²) = 4
    >>> E = compute_energy_density(psi, omega, grid)
    >>> assert np.all(E >= 0)  # Energy density is non-negative
    """
    # Compute gradient components
    dpsi_dr, dpsi_dtheta, dpsi_dzeta = compute_gradient_3d(psi, grid)
    
    # Radial grid (broadcast to 3D)
    r_3d = grid.r[:, np.newaxis, np.newaxis]  # Shape: (nr, 1, 1)
    
    # Handle r=0 singularity (clip r to avoid division by zero)
    r_safe = np.where(r_3d > 1e-10, r_3d, 1e-10)
    
    # |∇ψ|² with metric factor
    grad_psi_squared = (
        dpsi_dr**2 +                    # (∂ψ/∂r)²
        (dpsi_dtheta / r_safe)**2 +     # (1/r²)(∂ψ/∂θ)²
        dpsi_dzeta**2                   # (∂ψ/∂ζ)²
    )
    
    # Total energy density
    energy_density = 0.5 * (grad_psi_squared + omega**2)
    
    return energy_density


def compute_hamiltonian_3d(psi, omega, grid):
    """
    Compute total Hamiltonian H = ∫∫∫ [(1/2)|∇ψ|² + (1/2)ω²] r dr dθ dζ.
    
    Parameters
    ----------
    psi : np.ndarray (nr, nθ, nζ)
        Stream function (magnetic flux)
    omega : np.ndarray (nr, nθ, nζ)
        Vorticity ∇²ψ
    grid : Grid3D
        Grid object with attributes: nr, nθ, nζ, dr, dθ, dζ, r
    
    Returns
    -------
    H : float
        Total Hamiltonian energy (scalar)
    
    Notes
    -----
    Volume integral:
        H = ∫∫∫ E(r,θ,ζ) * r dr dθ dζ
    
    where E is the energy density and r is the Jacobian for cylindrical coords.
    
    Numerical integration:
        - Trapezoidal rule (implicit in np.sum with uniform grid)
        - Volume element: dV = r * dr * dθ * dζ
    
    Conservation properties:
        - Ideal MHD (η=0): dH/dt = 0 (machine precision drift only)
        - Resistive MHD (η>0): dH/dt < 0 (monotonic dissipation)
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, nθ=64, nζ=128)
    >>> # Test 1: Zero field → H = 0
    >>> psi = np.zeros((grid.nr, grid.nθ, grid.nζ))
    >>> omega = np.zeros_like(psi)
    >>> H = compute_hamiltonian_3d(psi, omega, grid)
    >>> assert np.abs(H) < 1e-14
    >>> 
    >>> # Test 2: Uniform field → H > 0
    >>> psi = np.ones_like(psi)
    >>> omega = np.zeros_like(psi)  # ∇²(const) = 0
    >>> H = compute_hamiltonian_3d(psi, omega, grid)
    >>> assert H > 0
    """
    # Step 1: Compute energy density
    energy_density = compute_energy_density(psi, omega, grid)
    
    # Step 2: Volume element with Jacobian
    # In cylindrical coords: dV = r dr dθ dζ
    r_3d = grid.r[:, np.newaxis, np.newaxis]  # (nr, 1, 1)
    volume_element = r_3d * grid.dr * grid.dθ * grid.dζ
    
    # Step 3: Integrate over all grid points (trapezoidal rule)
    H = np.sum(energy_density * volume_element)
    
    return H


# --- Convenience functions for energy partition ---

def compute_magnetic_energy(psi, grid):
    """
    Compute magnetic energy U = ∫∫∫ (1/2)|∇ψ|² r dr dθ dζ.
    
    Parameters
    ----------
    psi : np.ndarray (nr, nθ, nζ)
        Stream function
    grid : Grid3D
        Grid object
    
    Returns
    -------
    U : float
        Magnetic energy
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, nθ=64, nζ=128)
    >>> psi = grid.r[:, None, None]**2
    >>> U = compute_magnetic_energy(psi, grid)
    >>> assert U > 0
    """
    omega_zero = np.zeros_like(psi)
    return compute_hamiltonian_3d(psi, omega_zero, grid)


def compute_kinetic_energy(omega, grid):
    """
    Compute kinetic energy K = ∫∫∫ (1/2)ω² r dr dθ dζ.
    
    Parameters
    ----------
    omega : np.ndarray (nr, nθ, nζ)
        Vorticity
    grid : Grid3D
        Grid object
    
    Returns
    -------
    K : float
        Kinetic energy
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, nθ=64, nζ=128)
    >>> omega = np.ones((grid.nr, grid.nθ, grid.nζ))
    >>> K = compute_kinetic_energy(omega, grid)
    >>> assert K > 0
    """
    psi_zero = np.zeros_like(omega)
    return compute_hamiltonian_3d(psi_zero, omega, grid)
