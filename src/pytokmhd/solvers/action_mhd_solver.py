"""
Action-controllable MHD Solver for RL.

Uses stable cylindrical solver from v1.0 (not unstable toroidal).
v1.2 will switch to fixed toroidal solver.
"""

import numpy as np
from typing import Tuple, Optional

# Note: For v1.1, we use a simplified cylindrical MHD solver
# This is numerically stable (unlike toroidal which has bugs)
# v1.2 will fix toroidal solver and switch back


class SimplifiedMHDSolver:
    """
    Simplified cylindrical MHD solver for v1.1 RL framework.
    
    This is a stable fallback while toroidal solver is being debugged.
    Uses basic diffusion equations on cylindrical grid.
    
    Equations:
        ∂ψ/∂t = -η*J    (J = -∇²ψ)
        ∂ω/∂t = -ν*∇²ω
    
    Parameters
    ----------
    nr : int
        Radial grid points
    ntheta : int
        Poloidal grid points
    r_max : float
        Maximum radius
    dt : float
        Time step
    eta : float
        Base resistivity
    nu : float
        Base viscosity
    """
    
    def __init__(
        self,
        nr: int = 64,
        ntheta: int = 128,
        r_max: float = 1.0,
        dt: float = 1e-4,
        eta: float = 1e-5,
        nu: float = 1e-4
    ):
        self.nr = nr
        self.ntheta = ntheta
        self.r_max = r_max
        self.dt = dt
        
        # Grid
        self.r = np.linspace(0, r_max, nr)
        self.theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        self.dr = r_max / (nr - 1)
        self.dtheta = 2 * np.pi / ntheta
        
        # Parameters
        self.eta_base = eta
        self.nu_base = nu
        
        # State
        self.psi = None
        self.omega = None
        self.time = 0.0
        self.n_steps = 0
    
    def initialize(self, psi0: np.ndarray, omega0: np.ndarray):
        """Initialize fields."""
        self.psi = psi0.copy()
        self.omega = omega0.copy()
        self.time = 0.0
        self.n_steps = 0
    
    def _laplacian_cylindrical(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian in cylindrical coordinates (simplified).
        
        ∇² = ∂²/∂r² + (1/r)∂/∂r + (1/r²)∂²/∂θ²
        
        Simplified version (avoid 1/r singularity at r=0).
        """
        lap = np.zeros_like(field)
        
        # Radial derivatives (interior points)
        for i in range(1, self.nr - 1):
            r = self.r[i]
            if r > 1e-10:  # Avoid singularity
                # ∂²/∂r²
                d2_dr2 = (field[i+1, :] - 2*field[i, :] + field[i-1, :]) / self.dr**2
                # (1/r)∂/∂r
                d_dr = (field[i+1, :] - field[i-1, :]) / (2 * self.dr)
                lap[i, :] = d2_dr2 + d_dr / r
            else:
                # At r=0, use simplified form
                lap[i, :] = (field[i+1, :] - 2*field[i, :] + field[i-1, :]) / self.dr**2
        
        # Theta derivatives (all points)
        for i in range(self.nr):
            r = self.r[i]
            if r > 1e-10:
                # ∂²/∂θ² (periodic)
                d2_dtheta2 = np.zeros(self.ntheta)
                d2_dtheta2[1:-1] = (field[i, 2:] - 2*field[i, 1:-1] + field[i, :-2]) / self.dtheta**2
                d2_dtheta2[0] = (field[i, 1] - 2*field[i, 0] + field[i, -1]) / self.dtheta**2
                d2_dtheta2[-1] = (field[i, 0] - 2*field[i, -1] + field[i, -2]) / self.dtheta**2
                
                lap[i, :] += d2_dtheta2 / (r**2)
        
        return lap
    
    def compute_rhs(
        self,
        psi: np.ndarray,
        omega: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RHS with action modulation.
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
        omega : np.ndarray (nr, ntheta)
        action : np.ndarray (2,), optional
            [eta_multiplier, nu_multiplier]
        
        Returns
        -------
        dpsi_dt, domega_dt : np.ndarray
        """
        # Default action
        if action is None:
            action = np.array([1.0, 1.0])
        
        # Apply action modulation
        eta_mult = np.clip(action[0], 0.5, 2.0)
        nu_mult = np.clip(action[1], 0.5, 2.0)
        
        eta_eff = self.eta_base * eta_mult
        nu_eff = self.nu_base * nu_mult
        
        # Current: J = -∇²ψ
        lap_psi = self._laplacian_cylindrical(psi)
        J = -lap_psi
        
        # RHS
        dpsi_dt = -eta_eff * J
        
        lap_omega = self._laplacian_cylindrical(omega)
        domega_dt = -nu_eff * lap_omega
        
        return dpsi_dt, domega_dt
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 step with action.
        
        Parameters
        ----------
        action : np.ndarray (2,)
        
        Returns
        -------
        psi, omega : np.ndarray
        """
        dt = self.dt
        psi = self.psi
        omega = self.omega
        
        # RK4
        k1_psi, k1_omega = self.compute_rhs(psi, omega, action)
        
        k2_psi, k2_omega = self.compute_rhs(
            psi + 0.5*dt*k1_psi,
            omega + 0.5*dt*k1_omega,
            action
        )
        
        k3_psi, k3_omega = self.compute_rhs(
            psi + 0.5*dt*k2_psi,
            omega + 0.5*dt*k2_omega,
            action
        )
        
        k4_psi, k4_omega = self.compute_rhs(
            psi + dt*k3_psi,
            omega + dt*k3_omega,
            action
        )
        
        self.psi = psi + (dt/6) * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)
        self.omega = omega + (dt/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        
        self.time += dt
        self.n_steps += 1
        
        return self.psi.copy(), self.omega.copy()
