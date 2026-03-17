"""
Toroidal MHD Solver (M3 Step 3.1 - Minimal Implementation)

Combines ToroidalGrid + SymplecticIntegrator for reduced MHD.

This is a minimal implementation focusing on framework correctness.
Physics completeness is deferred to later steps.

Author: 小P ⚛️
Created: 2026-03-17 (M3.1)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from ..geometry import ToroidalGrid
from ..integrators import SymplecticIntegrator
from ..operators import laplacian_toroidal


class ToroidalMHDSolver:
    """
    Minimal toroidal reduced MHD solver.
    
    Evolves simplified reduced MHD equations:
        ∂ψ/∂t = -η*J
        ∂ω/∂t = -ν*∇²ω
    
    where J = -∇²ψ (current density)
    
    This minimal version omits:
        - Poisson bracket terms [ψ, φ]
        - Stream function φ solver
        - Curvature terms
    
    These will be added in later steps after basic framework is validated.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Toroidal grid
    dt : float
        Time step
    eta : float, optional
        Resistivity (default: 1e-5)
    nu : float, optional
        Viscosity (default: 1e-4)
    
    Attributes
    ----------
    grid : ToroidalGrid
    integrator : SymplecticIntegrator
    eta : float
    nu : float
    time : float
    n_steps : int
    psi : np.ndarray
    omega : np.ndarray
    """
    
    def __init__(
        self,
        grid: ToroidalGrid,
        dt: float,
        eta: float = 1e-5,
        nu: float = 1e-4,
        integrator: str = 'rk4'
    ):
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.integrator_type = integrator
        
        # Create integrator (default RK4 for stability)
        if integrator == 'symplectic':
            self.integrator = SymplecticIntegrator(dt=dt)
        else:
            self.integrator = None  # Use RK4 directly
        
        # State
        self.time = 0.0
        self.n_steps = 0
        self.psi = None
        self.omega = None
    
    def initialize(self, psi0: np.ndarray, omega0: np.ndarray):
        """
        Initialize fields.
        
        Parameters
        ----------
        psi0 : np.ndarray (nr, ntheta)
            Initial poloidal flux
        omega0 : np.ndarray (nr, ntheta)
            Initial vorticity
        """
        assert psi0.shape == (self.grid.nr, self.grid.ntheta)
        assert omega0.shape == (self.grid.nr, self.grid.ntheta)
        
        self.psi = psi0.copy()
        self.omega = omega0.copy()
        self.time = 0.0
        self.n_steps = 0
    
    def compute_rhs(self, psi: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RHS of reduced MHD (simplified).
        
        ∂ψ/∂t = -η*J
        ∂ω/∂t = -ν*∇²ω
        
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
        omega : np.ndarray (nr, ntheta)
        
        Returns
        -------
        """
        
        
        
        # Resistive diffusion (CORRECTED)
        lap_psi = laplacian_toroidal(psi, self.grid)
        dpsi_dt = self.eta * lap_psi  # Positive = diffusion
        
        # Viscous diffusion (CORRECTED)
        lap_omega = laplacian_toroidal(omega, self.grid)
        domega_dt = self.nu * lap_omega  # Positive = diffusion
        
        return dpsi_dt, domega_dt
    
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single time step.
        
        Returns
        -------
        psi : np.ndarray
        omega : np.ndarray
        """
        assert self.psi is not None, "Must call initialize() first"
        
        if self.integrator_type == 'symplectic':
            # Symplectic step
            self.psi, self.omega = self.integrator.step(
                self.psi, self.omega, self.compute_rhs
            )
        else:
            # RK4 step
            self.psi, self.omega = self._rk4_step(self.psi, self.omega)
        
        self.time += self.dt
        self.n_steps += 1
        
        return self.psi.copy(), self.omega.copy()
    
    def _apply_boundary(self, field: np.ndarray) -> np.ndarray:
        """Apply Dirichlet boundary conditions (ψ=0 at edges)."""
        field = field.copy()
        field[0, :] = 0.0   # r=0 (axis)
        field[-1, :] = 0.0  # r=a (edge)
        return field
    
    def _rk4_step(self, psi: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RK4 integration step."""
        dt = self.dt
        
        # k1
        dpsi_dt1, domega_dt1 = self.compute_rhs(psi, omega)
        
        # k2
        dpsi_dt2, domega_dt2 = self.compute_rhs(
            psi + 0.5*dt*dpsi_dt1,
            omega + 0.5*dt*domega_dt1
        )
        
        # k3
        dpsi_dt3, domega_dt3 = self.compute_rhs(
            psi + 0.5*dt*dpsi_dt2,
            omega + 0.5*dt*domega_dt2
        )
        
        # k4
        dpsi_dt4, domega_dt4 = self.compute_rhs(
            psi + dt*dpsi_dt3,
            omega + dt*domega_dt3
        )
        
        # Combine
        psi_new = psi + (dt/6.0) * (dpsi_dt1 + 2*dpsi_dt2 + 2*dpsi_dt3 + dpsi_dt4)
        omega_new = omega + (dt/6.0) * (domega_dt1 + 2*domega_dt2 + 2*domega_dt3 + domega_dt4)
        
        psi_new = self._apply_boundary(psi_new)
        omega_new = self._apply_boundary(omega_new)
        
        return psi_new, omega_new
    
    def run(
        self,
        n_steps: int,
        save_interval: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        save_interval : int, optional
            Save frequency (default: 100)
        
        Returns
        -------
        history : dict
            'psi': list of psi snapshots
            'omega': list of omega snapshots
            'time': list of times
        """
        assert self.psi is not None, "Must call initialize() first"
        
        history = {
            'psi': [self.psi.copy()],
            'omega': [self.omega.copy()],
            'time': [self.time]
        }
        
        for i in range(n_steps):
            self.step()
            
            if (i + 1) % save_interval == 0:
                history['psi'].append(self.psi.copy())
                history['omega'].append(self.omega.copy())
                history['time'].append(self.time)
        
        return history
