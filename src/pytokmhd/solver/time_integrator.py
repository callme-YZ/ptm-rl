"""
Time Integration for MHD Evolution

Implements 4th order Runge-Kutta (RK4) time stepping.

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np
from . import mhd_equations
from . import poisson_solver


def rk4_step(psi, omega, dt, dr, dz, r_grid, eta, nu=0.0, apply_bc=None):
    """
    Single RK4 timestep for Model-A MHD equations.
    
    Evolves:
    ∂ψ/∂t = -[φ, ψ] + η∇²ψ
    ∂ω/∂t = -[φ, ω] + [ψ, J] + ν∇²ω
    
    Where φ is solved from ∇²φ = -ω at each substep.
    
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
        Viscosity (default 0.0, not used in Model-A)
    apply_bc : callable, optional
        Boundary condition function: apply_bc(psi, omega) -> (psi, omega)
    
    Returns
    -------
    psi_new : np.ndarray (Nr, Nz)
        Poloidal flux at t + dt
    omega_new : np.ndarray (Nr, Nz)
        Vorticity at t + dt
    
    Notes
    -----
    Accuracy: O(dt⁴)
    Stability: CFL condition dt < min(dr, dz) / v_max
    
    Examples
    --------
    >>> Nr, Nz = 64, 128
    >>> r = np.linspace(0, 1, Nr)
    >>> z = np.linspace(0, 6, Nz)
    >>> R, Z = np.meshgrid(r, z, indexing='ij')
    >>> psi0 = 0.1 * np.sin(2*np.pi*Z/6) * (1 - R**2)
    >>> omega0 = np.zeros_like(psi0)
    >>> psi1, omega1 = rk4_step(psi0, omega0, dt=0.001, dr=r[1]-r[0], dz=z[1]-z[0], R, eta=1e-3)
    """
    # RK4 coefficients
    def compute_rhs(psi_cur, omega_cur):
        """Compute RHS for given state."""
        # Solve for φ from ∇²φ = -ω
        phi = poisson_solver.solve_poisson(omega_cur, dr, dz, r_grid, rhs_sign=-1.0)
        
        # Compute current density
        J = mhd_equations.laplacian_cylindrical(psi_cur, dr, dz, r_grid)
        
        # ∂ψ/∂t
        pb_phi_psi = mhd_equations.poisson_bracket(phi, psi_cur, dr, dz)
        lap_psi = J
        dpsi_dt = -pb_phi_psi + eta * lap_psi
        
        # ∂ω/∂t
        pb_phi_omega = mhd_equations.poisson_bracket(phi, omega_cur, dr, dz)
        pb_psi_J = mhd_equations.poisson_bracket(psi_cur, J, dr, dz)
        lap_omega = mhd_equations.laplacian_cylindrical(omega_cur, dr, dz, r_grid)
        domega_dt = -pb_phi_omega + pb_psi_J + nu * lap_omega
        
        return dpsi_dt, domega_dt
    
    # RK4 stages
    k1_psi, k1_omega = compute_rhs(psi, omega)
    
    k2_psi, k2_omega = compute_rhs(
        psi + 0.5*dt*k1_psi,
        omega + 0.5*dt*k1_omega
    )
    
    k3_psi, k3_omega = compute_rhs(
        psi + 0.5*dt*k2_psi,
        omega + 0.5*dt*k2_omega
    )
    
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


def adaptive_timestep(psi, omega, dr, dz, cfl_target=0.5):
    """
    Compute adaptive timestep based on CFL condition.
    
    CFL = max(|v|) * dt / min(dr, dz) < cfl_target
    
    Parameters
    ----------
    psi : np.ndarray (Nr, Nz)
        Poloidal flux (to estimate velocity)
    omega : np.ndarray (Nr, Nz)
        Vorticity
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    cfl_target : float, optional
        Target CFL number (default 0.5)
    
    Returns
    -------
    dt : float
        Safe timestep
    
    Notes
    -----
    Conservative estimate: uses max gradient of psi as velocity proxy.
    """
    # Estimate velocity from gradients
    dpsi_dr = mhd_equations.gradient_r(psi, dr)
    dpsi_dz = mhd_equations.gradient_z(psi, dz)
    
    v_max = np.max(np.sqrt(dpsi_dr**2 + dpsi_dz**2))
    
    # Avoid division by zero
    if v_max < 1e-10:
        v_max = 1e-10
    
    dx_min = min(dr, dz)
    dt = cfl_target * dx_min / v_max
    
    return dt


def evolve_mhd(psi0, omega0, t_final, dr, dz, r_grid, eta, nu=0.0, 
               dt=None, cfl=0.5, apply_bc=None, callback=None):
    """
    Evolve MHD system from t=0 to t=t_final.
    
    Parameters
    ----------
    psi0 : np.ndarray (Nr, Nz)
        Initial poloidal flux
    omega0 : np.ndarray (Nr, Nz)
        Initial vorticity
    t_final : float
        Final time
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    eta : float
        Resistivity
    nu : float, optional
        Viscosity (default 0.0)
    dt : float, optional
        Fixed timestep (if None, uses adaptive)
    cfl : float, optional
        CFL target for adaptive timestep (default 0.5)
    apply_bc : callable, optional
        Boundary condition function
    callback : callable, optional
        Function called after each step: callback(t, psi, omega)
    
    Returns
    -------
    psi : np.ndarray (Nr, Nz)
        Final poloidal flux
    omega : np.ndarray (Nr, Nz)
        Final vorticity
    
    Examples
    --------
    >>> psi_final, omega_final = evolve_mhd(psi0, omega0, t_final=1.0, ...)
    """
    psi = psi0.copy()
    omega = omega0.copy()
    t = 0.0
    step = 0
    
    while t < t_final:
        # Adaptive or fixed timestep
        if dt is None:
            dt_use = adaptive_timestep(psi, omega, dr, dz, cfl_target=cfl)
            dt_use = min(dt_use, t_final - t)  # Don't overshoot
        else:
            dt_use = min(dt, t_final - t)
        
        # RK4 step
        psi, omega = rk4_step(psi, omega, dt_use, dr, dz, r_grid, eta, nu, apply_bc)
        
        t += dt_use
        step += 1
        
        # Callback (for diagnostics)
        if callback is not None:
            callback(t, psi, omega)
    
    return psi, omega
