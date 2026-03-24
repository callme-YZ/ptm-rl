"""
Unit Tests for Complete MHD Solver (v2.0 Physics)

Issue #17: Add unit tests for v2.0 physics modules

Tests RK2 time integration and solver API.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp

from pim_rl.physics.v2.elsasser_bracket import ElsasserState
from pim_rl.physics.v2.complete_solver import CompleteMHDSolver


class TestCompleteSolver:
    """Test CompleteMHDSolver time integration"""
    
    @pytest.fixture
    def solver(self):
        """Small solver for fast tests"""
        return CompleteMHDSolver(
            grid_shape=(8, 8, 4),
            dr=0.1, dtheta=0.1, dz=0.1,
            epsilon=0.3,
            eta=0.01,
            pressure_scale=0.1
        )
    
    @pytest.fixture
    def test_state(self, solver):
        """Small smooth initial state"""
        Nr, Ntheta, Nz = 8, 8, 4
        
        r = jnp.linspace(0, 1, Nr)[:, None, None]
        theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
        z = jnp.linspace(0, 1, Nz)[None, None, :]
        
        z_plus = jnp.sin(jnp.pi * r) * jnp.cos(theta) * jnp.sin(jnp.pi * z) * 0.1
        z_minus = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * jnp.cos(jnp.pi * z) * 0.05
        P = jnp.ones((Nr, Ntheta, Nz)) * 0.01
        
        return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    def test_solver_initialization(self, solver):
        """Test solver creates successfully"""
        assert solver.epsilon == 0.3
        assert solver.eta == 0.01
        assert solver.pressure_scale == 0.1
        assert solver.grid.Nr == 8
        print("✅ Solver initialization")
    
    def test_hamiltonian_callable(self, solver, test_state):
        """Test Hamiltonian computation"""
        H = solver.hamiltonian(test_state)
        
        assert isinstance(float(H), float)
        # Note: H can be negative due to curvature energy -∫ h·P dV
        assert not jnp.isnan(H), "Hamiltonian is NaN"
        
        print(f"✅ Hamiltonian: H = {H:.6e}")
    
    def test_rhs_callable(self, solver, test_state):
        """Test RHS computation"""
        rhs = solver.rhs(test_state)
        
        # Check output is ElsasserState
        assert isinstance(rhs, ElsasserState)
        assert rhs.z_plus.shape == test_state.z_plus.shape
        assert rhs.z_minus.shape == test_state.z_minus.shape
        assert rhs.P.shape == test_state.P.shape
        
        print(f"✅ RHS callable:")
        print(f"   dz⁺/dt max: {jnp.max(jnp.abs(rhs.z_plus)):.3e}")
        print(f"   dz⁻/dt max: {jnp.max(jnp.abs(rhs.z_minus)):.3e}")
    
    def test_step_rk2(self, solver, test_state):
        """Test RK2 time step"""
        dt = 0.01
        
        state_new = solver.step_rk2(test_state, dt)
        
        # Check state updated
        assert isinstance(state_new, ElsasserState)
        
        # State should change
        diff = jnp.max(jnp.abs(state_new.z_plus - test_state.z_plus))
        assert diff > 1e-10, f"State didn't change: max diff = {diff}"
        
        print(f"✅ RK2 step (dt={dt}):")
        print(f"   Δz⁺ max: {diff:.3e}")
    
    def test_energy_trend_resistive(self, solver, test_state):
        """
        Test energy decreases over time (resistive case).
        
        With resistivity, energy should decay.
        """
        H0 = solver.hamiltonian(test_state)
        
        # Take 5 steps
        state = test_state
        for _ in range(5):
            state = solver.step_rk2(state, dt=0.01)
        
        H_final = solver.hamiltonian(state)
        
        # Energy should decrease (resistive damping)
        assert H_final < H0, f"Energy increased: H0={H0:.3e} → H={H_final:.3e}"
        
        decay = (H0 - H_final) / H0
        print(f"✅ Energy decay (5 steps):")
        print(f"   H₀ = {H0:.6e}")
        print(f"   H₅ = {H_final:.6e}")
        print(f"   Decay: {decay:.2%}")
    
    def test_state_bounds(self, solver, test_state):
        """
        Test fields stay bounded over time.
        
        No explosion or NaN.
        """
        state = test_state
        max_z_plus = jnp.max(jnp.abs(test_state.z_plus))
        
        # Evolve 10 steps
        for _ in range(10):
            state = solver.step_rk2(state, dt=0.01)
        
        # Check no explosion
        max_after = jnp.max(jnp.abs(state.z_plus))
        
        assert not jnp.isnan(max_after), "NaN detected!"
        assert max_after < max_z_plus * 10, f"Explosion: {max_after:.3e} > 10×{max_z_plus:.3e}"
        
        print(f"✅ State bounded (10 steps):")
        print(f"   |z⁺|_max: {max_z_plus:.3e} → {max_after:.3e}")


def test_zero_step():
    """Test dt=0 returns same state"""
    solver = CompleteMHDSolver((4, 4, 2), 0.1, 0.1, 0.1)
    
    state = ElsasserState(
        z_plus=jnp.ones((4, 4, 2)),
        z_minus=jnp.ones((4, 4, 2)) * 0.5,
        P=jnp.ones((4, 4, 2)) * 0.1
    )
    
    state_new = solver.step_rk2(state, dt=0.0)
    
    diff = jnp.max(jnp.abs(state_new.z_plus - state.z_plus))
    
    assert diff < 1e-12, f"State changed with dt=0: {diff}"
    print(f"✅ Zero step: state unchanged")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
