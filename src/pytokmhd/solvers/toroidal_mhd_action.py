"""
Action-controllable Toroidal MHD Solver.

Extends ToroidalMHDSolver to accept RL actions.
"""

import numpy as np
from typing import Tuple, Optional

from .toroidal_mhd import ToroidalMHDSolver
from ..geometry import ToroidalGrid


class ActionControlledMHDSolver(ToroidalMHDSolver):
    """
    MHD solver with action-controlled parameters.
    
    v1.1 Action space: Parameter modulation
    - eta_multiplier: [0.5, 2.0]
    - nu_multiplier: [0.5, 2.0]
    
    This modulates resistivity and viscosity in time.
    Note: This is NOT realistic control (no physical actuators).
    Purpose: Framework validation only.
    
    Parameters
    ----------
    grid : ToroidalGrid
    dt : float
    eta : float
        Base resistivity
    nu : float
        Base viscosity
    integrator : str
        'rk4' or 'symplectic'
    """
    
    def __init__(
        self,
        grid: ToroidalGrid,
        dt: float,
        eta: float = 1e-5,
        nu: float = 1e-4,
        integrator: str = 'rk4'
    ):
        super().__init__(grid, dt, eta, nu, integrator)
        
        # Store base values
        self.eta_base = eta
        self.nu_base = nu
    
    def compute_rhs_with_action(
        self,
        psi: np.ndarray,
        omega: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RHS with action-modulated parameters.
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
        omega : np.ndarray (nr, ntheta)
        action : np.ndarray (2,), optional
            [eta_multiplier, nu_multiplier]
            If None, use [1.0, 1.0] (no modulation)
        
        Returns
        -------
        dpsi_dt : np.ndarray
        domega_dt : np.ndarray
        """
        # Default action: no modulation
        if action is None:
            action = np.array([1.0, 1.0])
        
        # Apply action modulation
        eta_multiplier = np.clip(action[0], 0.5, 2.0)
        nu_multiplier = np.clip(action[1], 0.5, 2.0)
        
        eta_effective = self.eta_base * eta_multiplier
        nu_effective = self.nu_base * nu_multiplier
        
        # Compute RHS with effective parameters
        from ..operators import laplacian_toroidal
        
        # Current density: J = -∇²ψ
        lap_psi = laplacian_toroidal(psi, self.grid)
        J = -lap_psi
        
        # Resistive diffusion
        dpsi_dt = -eta_effective * J
        
        # Viscous diffusion
        lap_omega = laplacian_toroidal(omega, self.grid)
        domega_dt = -nu_effective * lap_omega
        
        return dpsi_dt, domega_dt
    
    def step_with_action(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single time step with action.
        
        Parameters
        ----------
        action : np.ndarray (2,), optional
            [eta_multiplier, nu_multiplier]
        
        Returns
        -------
        psi : np.ndarray
        omega : np.ndarray
        """
        assert self.psi is not None, "Must call initialize() first"
        
        # RK4 step with action
        if self.integrator_type == 'rk4':
            psi_new, omega_new = self._rk4_step(action)
        else:
            # Symplectic (wrapper)
            psi_new, omega_new = self.integrator.step(
                self.psi, self.omega,
                lambda p, o: self.compute_rhs_with_action(p, o, action)
            )
        
        self.psi = psi_new
        self.omega = omega_new
        self.time += self.dt
        self.n_steps += 1
        
        return self.psi.copy(), self.omega.copy()
    
    def _rk4_step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 integration step.
        
        Parameters
        ----------
        action : np.ndarray (2,)
        
        Returns
        -------
        psi_new : np.ndarray
        omega_new : np.ndarray
        """
        dt = self.dt
        psi = self.psi
        omega = self.omega
        
        # k1
        dpsi_1, domega_1 = self.compute_rhs_with_action(psi, omega, action)
        
        # k2
        psi_2 = psi + 0.5 * dt * dpsi_1
        omega_2 = omega + 0.5 * dt * domega_1
        dpsi_2, domega_2 = self.compute_rhs_with_action(psi_2, omega_2, action)
        
        # k3
        psi_3 = psi + 0.5 * dt * dpsi_2
        omega_3 = omega + 0.5 * dt * domega_2
        dpsi_3, domega_3 = self.compute_rhs_with_action(psi_3, omega_3, action)
        
        # k4
        psi_4 = psi + dt * dpsi_3
        omega_4 = omega + dt * domega_3
        dpsi_4, domega_4 = self.compute_rhs_with_action(psi_4, omega_4, action)
        
        # Combine
        psi_new = psi + (dt / 6.0) * (dpsi_1 + 2*dpsi_2 + 2*dpsi_3 + dpsi_4)
        omega_new = omega + (dt / 6.0) * (domega_1 + 2*domega_2 + 2*domega_3 + domega_4)
        
        return psi_new, omega_new
