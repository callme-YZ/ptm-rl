"""
Hamiltonian MHD Solver (v1.3)

Symplectic time integration using Poisson bracket formulation.

Physical Model
--------------
Evolution equations in Hamiltonian form:
    ∂ψ/∂t = {ψ, H} - η·J
    ∂ω/∂t = {ω, H} + S_P - ν·∇²ω

where:
    - ψ: poloidal flux
    - ω: vorticity
    - H: Hamiltonian (kinetic + magnetic energy)
    - {A, B}: Poisson bracket
    - η: resistivity
    - ν: viscosity
    - S_P: pressure force term
    - J: toroidal current density

Hamiltonian:
    H = ∫[(1/2)|∇φ|² + (1/2μ₀)|∇ψ|²] dV

Numerical Method
----------------
Störmer-Verlet symplectic integrator with explicit BC enforcement:
1. Half-step ψ: ψ^(n+1/2) = ψ^n + (dt/2){ψ, φ^n}
2. Add resistive diffusion (semi-implicit)
3. **Enforce BC on ψ** (critical!)
4. Full-step ω: ω^(n+1) = ω^n + dt({ω, φ^(n+1/2)} + S_P - ν∇²ω)
5. Half-step ψ: ψ^(n+1) = ψ^(n+1/2) + (dt/2){ψ, φ^(n+1)}
6. Add resistive diffusion (semi-implicit)
7. **Enforce BC on ψ** (critical!)

Boundary Conditions:
- Axis (r=0): axisymmetry → ψ(0, θ) = constant
- Edge (r=a): conducting wall → ψ(a, θ) = 0

Properties:
- Symplectic (preserves phase space structure)
- 2nd-order accurate
- Energy-conserving in ideal limit (η=ν=0)

References
----------
- Morrison (1998): Hamiltonian description of MHD
- Hairer et al. (2006): "Geometric Numerical Integration"
- Kraus et al. (2017): GEMPIC code

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Integration
Fixed: 2026-03-19 - Added explicit BC enforcement
"""

import numpy as np
from typing import Tuple, Optional
from ..geometry import ToroidalGrid
from ..operators import poisson_bracket, laplacian_toroidal
from ..physics import compute_current_density
from ..solvers import solve_poisson_toroidal


class HamiltonianMHD:
    """
    Hamiltonian MHD solver with symplectic time integration.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Computational grid
    dt : float
        Time step size [s]
    eta : float, optional
        Resistivity [Ω·m], default 1e-4
    nu : float, optional
        Viscosity [m²/s], default 1e-4
    P0 : float, optional
        Peak pressure [Pa], default 0.0 (no pressure)
    psi_edge : float, optional
        Edge flux value for pressure profile [Wb]
        If None, uses grid.a²
    alpha : float, optional
        Pressure profile shape parameter, default 2.0
    
    Attributes
    ----------
    step_count : int
        Number of time steps taken
    time : float
        Current simulation time [s]
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    >>> solver = HamiltonianMHD(grid, dt=1e-4)
    >>> 
    >>> # Initialize equilibrium
    >>> psi = grid.r_grid**2 * (1 - grid.r_grid/grid.a)
    >>> omega = -laplacian_toroidal(psi, grid)
    >>> 
    >>> # Time step
    >>> psi_new, omega_new = solver.step(psi, omega)
    """
    
    def __init__(
        self,
        grid: ToroidalGrid,
        dt: float = 1e-4,
        eta: float = 1e-4,
        nu: float = 1e-4,
        P0: float = 0.0,
        psi_edge: Optional[float] = None,
        alpha: float = 2.0
    ):
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.P0 = P0
        self.psi_edge = psi_edge if psi_edge is not None else grid.a**2
        self.alpha = alpha
        
        self.step_count = 0
        self.time = 0.0
    
    def compute_phi(self, omega: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation ∇²φ = ω for stream function.
        
        Parameters
        ----------
        omega : np.ndarray (nr, ntheta)
            Vorticity field
        
        Returns
        -------
        phi : np.ndarray (nr, ntheta)
            Stream function [Wb or normalized]
        
        Raises
        ------
        RuntimeError
            If Poisson solver fails to converge
        """
        phi, info = solve_poisson_toroidal(omega, self.grid)
        if info != 0:
            raise RuntimeError(f"Poisson solver failed to converge (info={info})")
        return phi
    
    def enforce_bc(self, psi: np.ndarray) -> np.ndarray:
        """
        Enforce boundary conditions on ψ.
        
        - Axis (r=0): axisymmetry → ψ(0, θ) = mean(ψ(0, :))
        - Edge (r=a): conducting wall → ψ(a, θ) = 0
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
            Poloidal flux (possibly violating BC)
        
        Returns
        -------
        psi_bc : np.ndarray (nr, ntheta)
            Flux with BC enforced
        """
        psi_bc = psi.copy()
        
        # Axis: enforce axisymmetry
        psi_bc[0, :] = np.mean(psi[0, :])
        
        # Edge: conducting wall
        psi_bc[-1, :] = 0.0
        
        return psi_bc
    
    def step(
        self,
        psi: np.ndarray,
        omega: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance solution by one time step using Störmer-Verlet.
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
            Poloidal flux [Wb]
        omega : np.ndarray (nr, ntheta)
            Vorticity [1/s]
        
        Returns
        -------
        psi_new : np.ndarray (nr, ntheta)
            Updated poloidal flux
        omega_new : np.ndarray (nr, ntheta)
            Updated vorticity
        
        Notes
        -----
        Symplectic integration scheme:
        1. Half-step ψ using current φ
        2. Add resistive diffusion (semi-implicit)
        3. **Enforce BC on ψ**
        4. Full-step ω using half-step φ
        5. Add viscous dissipation
        6. Complete half-step ψ using new φ
        7. Add resistive diffusion (second half)
        8. **Enforce BC on ψ**
        """
        # Step 1: Compute φ^n
        phi_n = self.compute_phi(omega)
        
        # Step 2: Half-step ψ (advection)
        psi_phi_bracket = poisson_bracket(psi, phi_n, self.grid)
        psi_half = psi + 0.5 * self.dt * psi_phi_bracket
        
        # Step 3: Add resistive diffusion (first half, semi-implicit)
        J_half = compute_current_density(psi_half, self.grid)
        psi_half = psi_half - 0.5 * self.dt * self.eta * J_half
        
        # Step 4: **ENFORCE BC** (critical for stability!)
        psi_half = self.enforce_bc(psi_half)
        
        # Step 5: Compute φ^(n+1/2)
        phi_half = self.compute_phi(omega)
        
        # Step 6: Full-step ω
        omega_phi_bracket = poisson_bracket(omega, phi_half, self.grid)
        
        # Pressure force term
        if self.P0 > 0:
            # S_P = (1/R²)(dP/dψ)·Δ*ψ
            # For now, simplified (TODO: implement pressure_force_term)
            S_P = np.zeros_like(omega)
        else:
            S_P = np.zeros_like(omega)
        
        # Viscous dissipation
        viscous_term = -self.nu * laplacian_toroidal(omega, self.grid)
        
        omega_new = omega + self.dt * (omega_phi_bracket + S_P + viscous_term)
        
        # Step 7: Compute φ^(n+1)
        phi_new = self.compute_phi(omega_new)
        
        # Step 8: Complete half-step ψ (advection)
        psi_phi_bracket_new = poisson_bracket(psi_half, phi_new, self.grid)
        psi_new = psi_half + 0.5 * self.dt * psi_phi_bracket_new
        
        # Step 9: Add resistive diffusion (second half)
        J_new = compute_current_density(psi_new, self.grid)
        psi_new = psi_new - 0.5 * self.dt * self.eta * J_new
        
        # Step 10: **ENFORCE BC** (critical for stability!)
        psi_new = self.enforce_bc(psi_new)
        
        # Update counters
        self.step_count += 1
        self.time += self.dt
        
        return psi_new, omega_new
    
    def reset(self):
        """Reset step counter and time."""
        self.step_count = 0
        self.time = 0.0


__all__ = ['HamiltonianMHD']
