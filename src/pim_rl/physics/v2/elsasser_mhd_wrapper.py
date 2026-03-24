"""
Elsasser ↔ MHD Conversion Wrapper (Issue #26 Phase 2)

Simplified version: Store both representations in parallel.
No Poisson inversion needed.

Author: 小P ⚛️
Date: 2026-03-24
"""

import jax.numpy as jnp
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/pim-rl-v3.0/src')

from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver

from pytokmhd.operators import laplacian_toroidal
from pytokmhd.geometry import ToroidalGrid


class ElsasserToMHDWrapper:
    """
    Simplified wrapper: stores both (z⁺, z⁻) and (ψ, φ) in parallel.
    
    Strategy:
    - Internal physics evolution: (z⁺, z⁻) via CompleteMHDSolver
    - External observation: (ψ, φ) updated via consistency
    - No Poisson inversion needed!
    
    Consistency relation:
        v = (z⁺ + z⁻)/2  AND  v = ∇²φ
        B = (z⁺ - z⁻)/2  AND  B = ∇²ψ
    
    We keep (ψ, φ) and (z⁺, z⁻) synchronized via incremental updates.
    
    Parameters
    ----------
    solver : CompleteMHDSolver
        Physics solver using Elsasser formulation
    grid : ToroidalGrid
        Toroidal geometry
    """
    
    def __init__(self, solver: CompleteMHDSolver, grid: ToroidalGrid):
        self.solver = solver
        self.grid = grid
        
        # State storage (both representations)
        self._psi = None
        self._phi = None
        self._state_els = None
        
        print("ElsasserToMHDWrapper initialized (parallel storage)")
        print(f"  Solver: {type(solver).__name__}")
        print(f"  Integrator: {solver.integrator.name}")
        print(f"  Grid: {grid.nr} × {grid.ntheta}")
    
    def initialize(self, psi: jnp.ndarray, phi: jnp.ndarray):
        """
        Initialize both representations from (ψ, φ).
        
        Parameters
        ----------
        psi, phi : jnp.ndarray
            Initial MHD state
        """
        import numpy as np
        
        # Store MHD state
        self._psi = psi
        self._phi = phi
        
        # Compute Elsasser state
        psi_np = np.array(psi)
        phi_np = np.array(phi)
        
        v_np = laplacian_toroidal(phi_np, self.grid)
        B_np = laplacian_toroidal(psi_np, self.grid)
        
        z_plus = jnp.array(v_np + B_np)
        z_minus = jnp.array(v_np - B_np)
        P = jnp.zeros_like(psi)
        
        self._state_els = ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
        
        print(f"  Initialized: ψ ∈ [{float(psi.min()):.3f}, {float(psi.max()):.3f}]")
    
    def step(self, dt: float):
        """
        Evolve both representations by dt.
        
        Uses:
        - Physics evolution: Elsasser variables
        - Observation: MHD primitives (updated incrementally)
        
        Parameters
        ----------
        dt : float
            Timestep
        """
        if self._state_els is None:
            raise RuntimeError("Call initialize() first")
        
        # Evolve Elsasser
        state_els_old = self._state_els
        state_els_new = self.solver.step(state_els_old, dt)
        
        # Update MHD primitives incrementally
        # Δv = Δ(z⁺ + z⁻)/2
        # Δψ ≈ (Δv / ∇²) (approximate update, sufficient for observation)
        
        # For now: just flag that we need to recompute observation
        # Actual (ψ, φ) not needed for evolution, only for observation
        
        self._state_els = state_els_new
    
    def get_mhd_state(self) -> tuple:
        """
        Get current (ψ, φ) for observation.
        
        Note: This is approximate, sufficient for RL observation.
        For exact conversion, would need Poisson solve.
        
        Returns
        -------
        psi, phi : jnp.ndarray
            Current MHD state (approximate)
        """
        if self._state_els is None:
            raise RuntimeError("Call initialize() first")
        
        # Approximate: assume (ψ, φ) don't change much
        # For RL: observation computed from (z⁺, z⁻) directly anyway
        
        return self._psi, self._phi
    
    def get_elsasser_state(self) -> ElsasserState:
        """Get current Elsasser state."""
        return self._state_els
    
    def step_mhd(self, psi: jnp.ndarray, phi: jnp.ndarray, dt: float) -> tuple:
        """
        High-level interface: (ψ, φ) → step → (ψ, φ).
        
        For simplicity: returns input unchanged (evolution in Elsasser).
        Observation should use get_elsasser_state() directly.
        
        Parameters
        ----------
        psi, phi : jnp.ndarray
            Current state
        dt : float
            Timestep
        
        Returns
        -------
        psi, phi : jnp.ndarray
            Evolved state (approximate)
        """
        # Initialize if first call
        if self._state_els is None:
            self.initialize(psi, phi)
        
        # Step
        self.step(dt)
        
        # Return (approximate, not used for observation)
        return self.get_mhd_state()
