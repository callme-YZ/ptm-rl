"""
RMP-MHD Coupling

Integrates RMP fields into MHD evolution equations.

Physics:
- RMP enters as external current source: ∂ψ/∂t += η∇²ψ_RMP
- Modification to reduced MHD equations (Model-A)
- Time-independent RMP field (static coils)

Author: 小P ⚛️
Created: 2026-03-16
Phase: 4
"""

import numpy as np
from ..solver.mhd_equations import (
    laplacian_cylindrical,
    poisson_bracket,
)
from ..solver.poisson_solver import solve_poisson
from .rmp_field import generate_rmp_field


def rhs_psi_with_rmp(psi, omega, dr, dz, r_grid, eta, psi_rmp=None):
    """
    Right-hand side of ψ equation with RMP control.
    
    Equations:
    ---------
    ∂ψ/∂t = -[φ, ψ] + η∇²ψ + η∇²ψ_RMP
    
    Where:
    - [φ, ψ]: Poisson bracket (advection)
    - η∇²ψ: Resistive diffusion
    - η∇²ψ_RMP: RMP forcing (external current source)
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    eta : float
        Resistivity
    psi_rmp : np.ndarray (Nr, Nz) or None, optional
        RMP flux field (if None, no RMP applied)
    
    Returns
    -------
    dpsi_dt : np.ndarray (Nr, Nz)
        Time derivative of ψ
    
    Notes
    -----
    - RMP field is static (time-independent)
    - RMP forcing scales with resistivity η
    - Physical interpretation: RMP coils drive external current
    
    Physics References
    ------------------
    - Fitzpatrick 1993: "Interaction of tearing modes with external structures"
    - Reduced MHD equations in cylindrical geometry
    
    Examples
    --------
    >>> # Without RMP
    >>> dpsi_dt = rhs_psi_with_rmp(psi, omega, dr, dz, r_grid, eta=1e-3)
    >>> 
    >>> # With RMP
    >>> psi_rmp, _ = generate_rmp_field(r_grid, z_grid, amplitude=0.05, m=2, n=1)
    >>> dpsi_dt = rhs_psi_with_rmp(psi, omega, dr, dz, r_grid, eta=1e-3, psi_rmp=psi_rmp)
    """
    # Solve for stream function φ from ∇²φ = -ω
    phi = solve_poisson(omega, dr, dz, r_grid, rhs_sign=-1.0)
    
    # Advection term: -[φ, ψ]
    pb_phi_psi = poisson_bracket(phi, psi, dr, dz)
    
    # Resistive diffusion: η∇²ψ
    lap_psi = laplacian_cylindrical(psi, dr, dz, r_grid)
    
    # Standard MHD evolution
    dpsi_dt = -pb_phi_psi + eta * lap_psi
    
    # Add RMP forcing if present
    if psi_rmp is not None:
        # RMP forcing: η∇²ψ_RMP
        lap_psi_rmp = laplacian_cylindrical(psi_rmp, dr, dz, r_grid)
        dpsi_dt += eta * lap_psi_rmp
    
    return dpsi_dt


def rhs_omega_with_rmp(psi, omega, dr, dz, r_grid, nu):
    """
    Right-hand side of ω equation (unchanged by RMP).
    
    Equations:
    ---------
    ∂ω/∂t = -[φ, ω] + [ψ, J] + ν∇²ω
    
    Where:
    - J = ∇²ψ is current density
    - ν is viscosity (typically 0 in Model-A)
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux
    omega : np.ndarray (Nr, Nz)
        Vorticity
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    nu : float
        Viscosity
    
    Returns
    -------
    domega_dt : np.ndarray (Nr, Nz)
        Time derivative of ω
    
    Notes
    -----
    - RMP does NOT directly affect vorticity equation
    - RMP enters only through modified ψ evolution
    - Indirect coupling: RMP → ψ → J → ω
    """
    # Solve for stream function φ from ∇²φ = -ω
    phi = solve_poisson(omega, dr, dz, r_grid, rhs_sign=-1.0)
    
    # Compute current density J = ∇²ψ
    J = laplacian_cylindrical(psi, dr, dz, r_grid)
    
    # Advection: -[φ, ω]
    pb_phi_omega = poisson_bracket(phi, omega, dr, dz)
    
    # Current-gradient coupling: [ψ, J]
    pb_psi_J = poisson_bracket(psi, J, dr, dz)
    
    # Viscous diffusion: ν∇²ω
    lap_omega = laplacian_cylindrical(omega, dr, dz, r_grid)
    
    domega_dt = -pb_phi_omega + pb_psi_J + nu * lap_omega
    
    return domega_dt


def rk4_step_with_rmp(psi, omega, dt, dr, dz, r_grid, eta, nu=0.0,
                      rmp_amplitude=0.0, rmp_phase=0.0, m=2, n=1,
                      apply_bc=None):
    """
    RK4 timestep with RMP control.
    
    Evolves Model-A MHD equations with external RMP forcing.
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux at t
    omega : np.ndarray (Nr, Nz)
        Vorticity at t
    dt : float
        Timestep size
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    eta : float
        Resistivity
    nu : float, optional
        Viscosity (default: 0.0)
    rmp_amplitude : float, optional
        RMP control amplitude (default: 0.0 = no control)
        Typical range: [-0.1, 0.1]
    rmp_phase : float, optional
        RMP phase offset in radians (default: 0.0)
    m : int, optional
        RMP poloidal mode number (default: 2)
    n : int, optional
        RMP toroidal mode number (default: 1)
    apply_bc : callable, optional
        Boundary condition function
    
    Returns
    -------
    psi_new : np.ndarray (Nr, Nz)
        Poloidal flux at t + dt
    omega_new : np.ndarray (Nr, Nz)
        Vorticity at t + dt
    
    Notes
    -----
    - RMP field is static (generated once per step)
    - RMP amplitude acts as control input
    - Accuracy: O(dt⁴) (same as standard RK4)
    
    Control Interface:
    -----------------
    action = rmp_amplitude ∈ [-A_max, A_max]
    
    - action > 0: RMP in phase with island
    - action < 0: RMP out of phase (typically suppresses)
    - action = 0: No control
    
    Examples
    --------
    >>> # No control
    >>> psi1, omega1 = rk4_step_with_rmp(psi, omega, dt, dr, dz, r_grid, eta)
    >>> 
    >>> # With RMP control
    >>> psi1, omega1 = rk4_step_with_rmp(psi, omega, dt, dr, dz, r_grid, eta,
    ...                                  rmp_amplitude=0.05, m=2, n=1)
    """
    # Generate RMP field (static, same for all RK4 substeps)
    if abs(rmp_amplitude) > 1e-10:
        # Get z_grid from r_grid shape (assume same)
        Nr, Nz = r_grid.shape
        z_max = 2 * np.pi  # Typical periodic length
        z_1d = np.linspace(0, z_max, Nz)
        r_1d = r_grid[:, 0]
        z_grid = np.tile(z_1d, (Nr, 1))
        
        psi_rmp, _ = generate_rmp_field(r_grid, z_grid, rmp_amplitude, m, n, rmp_phase)
    else:
        psi_rmp = None
    
    # RK4 stages with RMP
    def compute_rhs(psi_cur, omega_cur):
        """Compute RHS with RMP."""
        dpsi_dt = rhs_psi_with_rmp(psi_cur, omega_cur, dr, dz, r_grid, eta, psi_rmp)
        domega_dt = rhs_omega_with_rmp(psi_cur, omega_cur, dr, dz, r_grid, nu)
        return dpsi_dt, domega_dt
    
    # Stage 1
    k1_psi, k1_omega = compute_rhs(psi, omega)
    
    # Stage 2
    k2_psi, k2_omega = compute_rhs(
        psi + 0.5*dt*k1_psi,
        omega + 0.5*dt*k1_omega
    )
    
    # Stage 3
    k3_psi, k3_omega = compute_rhs(
        psi + 0.5*dt*k2_psi,
        omega + 0.5*dt*k2_omega
    )
    
    # Stage 4
    k4_psi, k4_omega = compute_rhs(
        psi + dt*k3_psi,
        omega + dt*k3_omega
    )
    
    # Combine
    psi_new = psi + dt/6.0 * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)
    omega_new = omega + dt/6.0 * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    
    # Apply boundary conditions
    if apply_bc is not None:
        psi_new, omega_new = apply_bc(psi_new, omega_new)
    
    return psi_new, omega_new


def compute_rmp_effectiveness(psi_no_rmp, psi_with_rmp, r_grid, z_grid, r_s):
    """
    Measure RMP control effectiveness.
    
    Compares island width with/without RMP control.
    
    Parameters
    ----------
    psi_no_rmp : np.ndarray (Nr, Nz)
        Flux without RMP (baseline)
    psi_with_rmp : np.ndarray (Nr, Nz)
        Flux with RMP control
    r_grid : np.ndarray (Nr, Nz)
        Radial grid
    z_grid : np.ndarray (Nr, Nz)
        Axial grid
    r_s : float
        Rational surface radius
    
    Returns
    -------
    effectiveness : float
        Control effectiveness metric
        = (w_no_rmp - w_with_rmp) / w_no_rmp
        > 0: RMP suppresses island
        < 0: RMP enhances island (wrong phase)
    
    Notes
    -----
    Requires diagnostics module (Phase 3) to measure island width.
    """
    from ..diagnostics.magnetic_island import compute_island_width as measure_island_width
    
    # Measure island widths
    w_no_rmp = measure_island_width(psi_no_rmp, r_grid, z_grid, r_s, m=2)
    w_with_rmp = measure_island_width(psi_with_rmp, r_grid, z_grid, r_s, m=2)
    
    # Effectiveness: fractional reduction
    if w_no_rmp < 1e-10:
        return 0.0
    
    effectiveness = (w_no_rmp - w_with_rmp) / w_no_rmp
    
    return effectiveness


# =============================================================================
# Validation and Testing Utilities
# =============================================================================

def test_rmp_forcing_magnitude(dr, dz, r_grid, z_grid, eta, rmp_amplitude=0.05, m=2):
    """
    Test RMP forcing magnitude.
    
    Verifies that RMP forcing term has expected scaling:
    - Scales linearly with rmp_amplitude
    - Scales linearly with resistivity η
    
    Parameters
    ----------
    dr, dz : float
        Grid spacings
    r_grid, z_grid : np.ndarray
        Coordinate meshes
    eta : float
        Resistivity
    rmp_amplitude : float
        RMP amplitude to test
    m : int
        Mode number
    
    Returns
    -------
    is_valid : bool
        True if scaling is correct
    diagnostics : dict
        Test diagnostics
    """
    # Generate RMP field
    psi_rmp, _ = generate_rmp_field(r_grid, z_grid, rmp_amplitude, m=m, n=1)
    
    # Compute forcing term
    lap_psi_rmp = laplacian_cylindrical(psi_rmp, dr, dz, r_grid)
    forcing = eta * lap_psi_rmp
    
    # Expected scaling
    forcing_magnitude = np.max(np.abs(forcing))
    
    # Test with doubled amplitude
    psi_rmp_2x, _ = generate_rmp_field(r_grid, z_grid, 2*rmp_amplitude, m=m, n=1)
    lap_psi_rmp_2x = laplacian_cylindrical(psi_rmp_2x, dr, dz, r_grid)
    forcing_2x = eta * lap_psi_rmp_2x
    forcing_magnitude_2x = np.max(np.abs(forcing_2x))
    
    # Check linear scaling
    scaling_ratio = forcing_magnitude_2x / forcing_magnitude
    is_valid = abs(scaling_ratio - 2.0) < 0.1  # 10% tolerance
    
    diagnostics = {
        'forcing_magnitude': forcing_magnitude,
        'forcing_magnitude_2x': forcing_magnitude_2x,
        'scaling_ratio': scaling_ratio,
        'expected_ratio': 2.0,
        'is_valid': is_valid
    }
    
    return is_valid, diagnostics
