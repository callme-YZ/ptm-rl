"""
Tests for Elsasser ↔ MHD Wrapper (Issue #26 Phase 2)

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp
import numpy as np

from pim_rl.physics.v2.elsasser_mhd_wrapper import ElsasserToMHDWrapper
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver
from pim_rl.physics.v2.time_integrators import RK2Integrator

from pytokmhd.geometry import ToroidalGrid


class TestElsasserMHDWrapper:
    """Test conversion wrapper."""
    
    @pytest.fixture
    def setup(self):
        """Create wrapper."""
        # Grid parameters (must match!)
        nr, ntheta = 32, 64  # ToroidalGrid requirement: nr >= 32
        nz = 8
        
        dr, dtheta, dz = 0.01, 0.1, 0.2
        
        # Solver
        solver = CompleteMHDSolver(
            (nr, ntheta, nz), dr, dtheta, dz,
            epsilon=0.3, eta=0.01,
            integrator=RK2Integrator()
        )
        
        # Geometry for Poisson solver
        toro_grid = ToroidalGrid(R0=1.0, a=0.3, nr=nr, ntheta=ntheta)
        
        # Wrapper
        wrapper = ElsasserToMHDWrapper(solver, toro_grid)
        
        return wrapper, toro_grid
    
    def test_round_trip_conversion(self, setup):
        """Test: (ψ,φ) → (z⁺,z⁻) → (ψ,φ) ≈ (ψ,φ)."""
        
        wrapper, grid = setup
        
        # Test MHD state
        r = grid.r_grid
        theta = grid.theta_grid
        
        psi_0 = r**2 * np.sin(theta)
        phi_0 = r**2 * np.cos(theta)
        
        # Convert to JAX
        psi_jax = jnp.array(psi_0)
        phi_jax = jnp.array(phi_0)
        
        # Round-trip
        state_els = wrapper.mhd_to_elsasser(psi_jax, phi_jax)
        psi_recovered, phi_recovered = wrapper.elsasser_to_mhd(state_els)
        
        # Error
        psi_error = float(jnp.max(jnp.abs(psi_recovered - psi_jax)))
        phi_error = float(jnp.max(jnp.abs(phi_recovered - phi_jax)))
        
        psi_rel = psi_error / float(jnp.max(jnp.abs(psi_jax)))
        phi_rel = phi_error / float(jnp.max(jnp.abs(phi_jax)))
        
        print(f"\nRound-trip conversion test:")
        print(f"  ψ error: {psi_error:.3e} (rel: {psi_rel*100:.2f}%)")
        print(f"  φ error: {phi_error:.3e} (rel: {phi_rel*100:.2f}%)")
        
        # Tolerance: ~5% (Poisson solver + FD errors)
        assert psi_rel < 0.1, f"ψ round-trip error too large: {psi_rel*100:.1f}%"
        assert phi_rel < 0.1, f"φ round-trip error too large: {phi_rel*100:.1f}%"
        
        print("  ✅ PASSED")
    
    def test_step_mhd_interface(self, setup):
        """Test: step_mhd() runs without error."""
        
        wrapper, grid = setup
        
        # Initial state
        r = grid.r_grid
        theta = grid.theta_grid
        
        psi_0 = jnp.array(r**2 * np.sin(theta))
        phi_0 = jnp.array(r**2 * np.cos(theta))
        
        # Step
        dt = 0.01
        psi_new, phi_new = wrapper.step_mhd(psi_0, phi_0, dt)
        
        # Check output shape
        assert psi_new.shape == psi_0.shape
        assert phi_new.shape == phi_0.shape
        
        # Check no NaN/Inf
        assert jnp.all(jnp.isfinite(psi_new))
        assert jnp.all(jnp.isfinite(phi_new))
        
        print(f"\nstep_mhd() test:")
        print(f"  Input shape: {psi_0.shape}")
        print(f"  Output shape: {psi_new.shape}")
        print(f"  All finite: ✅")
        
        print("  ✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
