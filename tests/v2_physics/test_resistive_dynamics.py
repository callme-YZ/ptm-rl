"""
Unit Tests for Resistive Dynamics (v2.0 Physics)

Issue #17: Add unit tests for v2.0 physics modules

Tests resistivity and pressure gradient terms.

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp

from pim_rl.physics.v2.elsasser_bracket import ElsasserState, MorrisonBracket
from pim_rl.physics.v2.resistive_dynamics import (
    add_resistive_diffusion,
    add_pressure_gradient_force,
    resistive_mhd_rhs
)


class TestResistiveDynamics:
    """Test resistive MHD terms"""
    
    @pytest.fixture
    def grid(self):
        """Small test grid"""
        return MorrisonBracket((8, 8, 4), dr=0.1, dtheta=0.1, dz=0.1)
    
    @pytest.fixture
    def test_state(self, grid):
        """State with gradients"""
        Nr, Ntheta, Nz = 8, 8, 4
        
        r = jnp.linspace(0, 1, Nr)[:, None, None]
        
        z_plus = jnp.sin(jnp.pi * r) * jnp.ones((1, Ntheta, Nz))
        z_minus = jnp.cos(jnp.pi * r) * jnp.ones((1, Ntheta, Nz)) * 0.5
        P = 1.0 - r**2  # Radial pressure gradient
        
        return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    def test_resistive_diffusion_callable(self, grid, test_state):
        """Test resistive diffusion computation"""
        eta = 0.01
        
        resistive = add_resistive_diffusion(test_state, grid, eta)
        
        # Check output
        assert isinstance(resistive, ElsasserState)
        assert resistive.z_plus.shape == test_state.z_plus.shape
        
        # Should be non-zero (there's curvature)
        max_val = jnp.max(jnp.abs(resistive.z_plus))
        assert max_val > 1e-10, f"Resistive term too small: {max_val}"
        
        print(f"✅ Resistive diffusion:")
        print(f"   η = {eta}")
        print(f"   |dz⁺/dt| max = {max_val:.3e}")
    
    def test_resistive_antisymmetry(self, grid, test_state):
        """
        Test dz⁺ = -dz⁻ for resistive term.
        
        Resistivity acts on B = (z⁺-z⁻)/2, so dz⁺ = -dz⁻.
        """
        eta = 0.01
        resistive = add_resistive_diffusion(test_state, grid, eta)
        
        # Check antisymmetry
        max_diff = jnp.max(jnp.abs(resistive.z_plus + resistive.z_minus))
        
        assert max_diff < 1e-8, f"Antisymmetry violated: |dz⁺ + dz⁻|_max = {max_diff:.3e}"
        print(f"✅ Resistive antisymmetry: |dz⁺ + dz⁻|_max = {max_diff:.3e}")
    
    def test_pressure_gradient_callable(self, grid, test_state):
        """Test pressure gradient force"""
        p_scale = 0.1
        
        pressure = add_pressure_gradient_force(test_state, grid, p_scale)
        
        # Check output
        assert isinstance(pressure, ElsasserState)
        
        # Should be non-zero (P has gradient)
        max_val = jnp.max(jnp.abs(pressure.z_plus))
        assert max_val > 1e-10, f"Pressure term too small: {max_val}"
        
        print(f"✅ Pressure gradient:")
        print(f"   scale = {p_scale}")
        print(f"   |force| max = {max_val:.3e}")
    
    def test_pressure_symmetry(self, grid, test_state):
        """
        Test dz⁺ = dz⁻ for pressure term.
        
        Pressure acts on v = (z⁺+z⁻)/2, so both get same force.
        """
        p_scale = 0.1
        pressure = add_pressure_gradient_force(test_state, grid, p_scale)
        
        # Check symmetry
        max_diff = jnp.max(jnp.abs(pressure.z_plus - pressure.z_minus))
        
        assert max_diff < 1e-10, f"Symmetry violated: |dz⁺ - dz⁻|_max = {max_diff:.3e}"
        print(f"✅ Pressure symmetry: |dz⁺ - dz⁻|_max = {max_diff:.3e}")
    
    def test_resistive_mhd_rhs(self, grid, test_state):
        """Test complete RHS combination"""
        # Dummy bracket RHS
        bracket_rhs = ElsasserState(
            z_plus=jnp.ones_like(test_state.z_plus) * 0.01,
            z_minus=jnp.ones_like(test_state.z_minus) * 0.02,
            P=jnp.zeros_like(test_state.P)
        )
        
        eta = 0.01
        p_scale = 0.1
        
        total_rhs = resistive_mhd_rhs(test_state, grid, bracket_rhs, eta, p_scale)
        
        # Check output
        assert isinstance(total_rhs, ElsasserState)
        
        # Should include all contributions
        assert jnp.max(jnp.abs(total_rhs.z_plus)) > jnp.max(jnp.abs(bracket_rhs.z_plus))
        
        print(f"✅ Complete RHS:")
        print(f"   |total| = {jnp.max(jnp.abs(total_rhs.z_plus)):.3e}")
        print(f"   |bracket| = {jnp.max(jnp.abs(bracket_rhs.z_plus)):.3e}")
    
    def test_zero_eta(self, grid, test_state):
        """Test η=0 gives zero resistive term"""
        resistive = add_resistive_diffusion(test_state, grid, eta=0.0)
        
        max_val = jnp.max(jnp.abs(resistive.z_plus))
        
        assert max_val < 1e-12, f"Non-zero with η=0: {max_val}"
        print(f"✅ Zero resistivity: max = {max_val:.3e}")
    
    def test_zero_pressure_scale(self, grid, test_state):
        """Test scale=0 gives zero pressure force"""
        pressure = add_pressure_gradient_force(test_state, grid, pressure_gradient_scale=0.0)
        
        max_val = jnp.max(jnp.abs(pressure.z_plus))
        
        assert max_val < 1e-12, f"Non-zero with scale=0: {max_val}"
        print(f"✅ Zero pressure scale: max = {max_val:.3e}")


def test_flat_pressure():
    """Test uniform pressure gives zero force"""
    grid = MorrisonBracket((4, 4, 2), 0.1, 0.1, 0.1)
    
    # Uniform pressure
    state = ElsasserState(
        z_plus=jnp.ones((4, 4, 2)),
        z_minus=jnp.ones((4, 4, 2)) * 0.5,
        P=jnp.ones((4, 4, 2)) * 0.5  # Constant
    )
    
    pressure = add_pressure_gradient_force(state, grid, 0.1)
    
    max_force = jnp.max(jnp.abs(pressure.z_plus))
    
    # Should be ~0 (only boundary errors from roll)
    assert max_force < 1e-8, f"Force from flat P: {max_force}"
    print(f"✅ Flat pressure: force = {max_force:.3e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
