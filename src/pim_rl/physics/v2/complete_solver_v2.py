"""
Complete v2.0 Solver with Pluggable Integrators

Issue #26: Refactored to use TimeIntegrator interface

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
from typing import Optional

from .elsasser_bracket import ElsasserState, functional_derivative
from .toroidal_bracket import ToroidalMorrisonBracket
from .toroidal_hamiltonian import toroidal_hamiltonian
from .resistive_dynamics import resistive_mhd_rhs
from .time_integrators import TimeIntegrator, RK2Integrator


class CompleteMHDSolver:
    """
    Complete MHD solver with pluggable time integrators.
    
    Physics:
    - Ideal MHD: {z±, H} (Morrison bracket)
    - Resistive: η∇²B (magnetic diffusion)
    - Pressure: -∇p/ρ (ballooning drive)
    
    Time integration:
    - Pluggable via TimeIntegrator interface
    - Default: RK2 (backward compatible)
    - Alternative: Symplectic (structure-preserving)
    
    Parameters
    ----------
    grid_shape : tuple
        (Nr, Ntheta, Nz)
    dr, dtheta, dz : float
        Grid spacing
    epsilon : float
        Inverse aspect ratio (default: 0.3)
    eta : float
        Resistivity (default: 0.01)
    pressure_scale : float
        Pressure gradient strength (default: 0.2)
    integrator : TimeIntegrator, optional
        Time integrator (default: RK2Integrator())
    """
    
    def __init__(
        self,
        grid_shape: tuple,
        dr: float,
        dtheta: float,
        dz: float,
        epsilon: float = 0.3,
        eta: float = 0.01,
        pressure_scale: float = 0.2,
        integrator: Optional[TimeIntegrator] = None
    ):
        """Initialize solver."""
        
        self.grid = ToroidalMorrisonBracket(grid_shape, dr, dtheta, dz, epsilon)
        self.epsilon = epsilon
        self.eta = eta
        self.pressure_scale = pressure_scale
        
        # Integrator (default: RK2 for backward compatibility)
        if integrator is None:
            integrator = RK2Integrator()
        self.integrator = integrator
        
        print("CompleteMHDSolver initialized:")
        print(f"  Grid: {grid_shape}")
        print(f"  ε: {epsilon}")
        print(f"  η: {eta}")
        print(f"  ∇p scale: {pressure_scale}")
        print(f"  Integrator: {self.integrator.name} (order {self.integrator.order})")
        if self.integrator.is_symplectic:
            print(f"  Structure-preserving: ✅ Symplectic")
        else:
            print(f"  Structure-preserving: ❌ Not symplectic")
    
    def set_eta(self, eta: float):
        """
        Update resistivity parameter (for RL control).
        
        Parameters
        ----------
        eta : float
            New resistivity value
        """
        self.eta = eta
    
    def hamiltonian(self, state: ElsasserState) -> float:
        """Compute Hamiltonian (energy)."""
        return toroidal_hamiltonian(state, self.grid, self.epsilon)
    
    def rhs(self, state: ElsasserState) -> ElsasserState:
        """
        Compute complete RHS: dz±/dt.
        
        RHS = {z±, H} + η∇²B - ∇p/ρ
        
        Returns
        -------
        dstate : ElsasserState
            Time derivative dz±/dt
        """
        # Ideal bracket
        def H(s, g):
            return toroidal_hamiltonian(s, g, self.epsilon)
        
        dH = functional_derivative(H, state, self.grid)
        ideal_bracket = self.grid.bracket(
            ElsasserState(z_plus=state.z_plus, z_minus=state.z_minus, P=state.P),
            dH
        )
        
        # Add resistive + pressure
        total_rhs = resistive_mhd_rhs(
            state, self.grid, ideal_bracket,
            self.eta, self.pressure_scale
        )
        
        return total_rhs
    
    def step(self, state: ElsasserState, dt: float) -> ElsasserState:
        """
        Single timestep using configured integrator.
        
        Parameters
        ----------
        state : ElsasserState
            Current state
        dt : float
            Timestep
            
        Returns
        -------
        state_new : ElsasserState
            State at t + dt
        """
        return self.integrator.step(state, self.rhs, dt)
    
    def step_multi(
        self,
        state: ElsasserState,
        dt: float,
        n_substeps: int = 1
    ) -> ElsasserState:
        """
        Multiple substeps (for RL environment).
        
        Useful when RL timestep dt_RL > physics timestep dt_physics.
        
        Parameters
        ----------
        state : ElsasserState
            Current state
        dt : float
            Total time to advance
        n_substeps : int
            Number of physics substeps (default: 1)
            
        Returns
        -------
        state_new : ElsasserState
            State after time dt
        """
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            state = self.step(state, dt_sub)
        
        return state
    
    # Backward compatibility methods
    def step_rk2(self, state: ElsasserState, dt: float) -> ElsasserState:
        """
        Backward compatibility: RK2 step.
        
        Deprecated: Use step() with RK2Integrator instead.
        """
        if not isinstance(self.integrator, RK2Integrator):
            print("Warning: step_rk2() called but integrator is not RK2. "
                  "Using configured integrator instead.")
        return self.step(state, dt)
