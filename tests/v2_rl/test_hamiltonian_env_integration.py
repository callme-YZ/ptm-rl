"""
Test HamiltonianMHDEnv integration with ElsasserMHDSolver

Issue #26 Phase 2: Replace dummy solver with real MHD physics

Author: 小A 🤖
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp
import numpy as np

from pim_rl.physics.v2.elsasser_mhd_solver import ElsasserMHDSolver
from pim_rl.physics.v2.complete_solver_v2 import CompleteMHDSolver
from pim_rl.physics.v2.time_integrators import RK2Integrator, make_integrator

from pytokmhd.geometry import ToroidalGrid


class TestElsasserMHDIntegration:
    """Test integration of ElsasserMHDSolver for RL environment."""
    
    @pytest.fixture
    def setup(self):
        """Create solver with RL-appropriate parameters."""
        # Grid parameters (match HamiltonianMHDEnv defaults)
        R0 = 1.5  # Major radius [m]
        a = 0.5   # Minor radius [m]
        nr = 32   # Radial resolution
        ntheta = 64  # Poloidal resolution
        nz = 8    # Toroidal resolution (for 3D solver)
        
        # Grid spacing
        dr = a / (nr - 1)
        dtheta = 2 * np.pi / ntheta
        Lz = 2 * np.pi * R0  # Toroidal length
        dz = Lz / nz
        
        # Physics parameters
        epsilon = a / R0  # Inverse aspect ratio
        eta = 1e-5  # Resistivity
        nu = 1e-4   # Viscosity (not used in Model-A)
        
        # Create CompleteMHDSolver (3D)
        physics_solver = CompleteMHDSolver(
            grid_shape=(nr, ntheta, nz),
            dr=dr,
            dtheta=dtheta,
            dz=dz,
            epsilon=epsilon,
            eta=eta,
            pressure_scale=0.2,
            integrator=RK2Integrator()
        )
        
        # Create ToroidalGrid (2D, for observation)
        grid_2d = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=ntheta)
        
        # Wrapper
        solver = ElsasserMHDSolver(physics_solver, grid_2d)
        
        return solver, grid_2d
    
    def test_initialization_from_rl_state(self, setup):
        """Test initialization from typical RL initial state."""
        solver, grid = setup
        
        # Create initial state (tearing mode perturbation)
        r = grid.r_grid
        theta = grid.theta_grid
        
        # Poloidal flux (m=1 mode)
        psi_0 = jnp.array(0.01 * r**2 * (1 - r**2) * np.sin(theta))
        
        # Stream function (initially zero)
        phi_0 = jnp.zeros_like(psi_0)
        
        # Initialize
        solver.initialize(psi_0, phi_0)
        
        # Should not crash
        assert solver._state_els is not None
        
        print("✅ Initialization from RL state successful")
    
    def test_rl_step_and_observation(self, setup):
        """Test RL step: evolve + get observation."""
        solver, grid = setup
        
        # Initialize
        r = grid.r_grid
        theta = grid.theta_grid
        psi_0 = jnp.array(0.01 * r**2 * (1 - r**2) * np.sin(theta))
        phi_0 = jnp.zeros_like(psi_0)
        
        solver.initialize(psi_0, phi_0)
        
        # RL timestep
        dt_rl = 1e-4
        
        # Evolve
        solver.step(dt_rl)
        
        # Get observation
        psi_new, phi_new = solver.get_mhd_state()
        
        # Check shapes
        assert psi_new.shape == (grid.nr, grid.ntheta)
        assert phi_new.shape == (grid.nr, grid.ntheta)
        
        # Check no NaN/Inf
        assert jnp.all(jnp.isfinite(psi_new))
        assert jnp.all(jnp.isfinite(phi_new))
        
        print(f"✅ RL step successful:")
        print(f"  ψ range: [{float(psi_new.min()):.3e}, {float(psi_new.max()):.3e}]")
        print(f"  φ range: [{float(phi_new.min()):.3e}, {float(phi_new.max()):.3e}]")
    
    def test_multi_step_stability(self, setup):
        """Test multiple RL steps (stability check)."""
        solver, grid = setup
        
        # Initialize
        r = grid.r_grid
        theta = grid.theta_grid
        psi_0 = jnp.array(0.01 * r**2 * (1 - r**2) * np.sin(theta))
        phi_0 = jnp.zeros_like(psi_0)
        
        solver.initialize(psi_0, phi_0)
        
        # Run 100 steps
        dt_rl = 1e-4
        n_steps = 100
        
        for i in range(n_steps):
            solver.step(dt_rl)
            
            # Check every 20 steps
            if i % 20 == 0:
                psi, phi = solver.get_mhd_state()
                
                # Should remain finite
                assert jnp.all(jnp.isfinite(psi))
                assert jnp.all(jnp.isfinite(phi))
        
        # Final state
        psi_final, phi_final = solver.get_mhd_state()
        
        print(f"✅ {n_steps} steps stable:")
        print(f"  Final ψ range: [{float(psi_final.min()):.3e}, {float(psi_final.max()):.3e}]")
        print(f"  Final φ range: [{float(phi_final.min()):.3e}, {float(phi_final.max()):.3e}]")
    
    def test_integrator_comparison(self, setup):
        """Test RK2 vs Symplectic integrator (when available)."""
        solver, grid = setup
        
        # Initialize
        r = grid.r_grid
        theta = grid.theta_grid
        psi_0 = jnp.array(0.01 * r**2 * (1 - r**2) * np.sin(theta))
        phi_0 = jnp.zeros_like(psi_0)
        
        solver.initialize(psi_0, phi_0)
        
        # Check integrator
        integrator_name = solver.solver.integrator.name
        is_symplectic = solver.solver.integrator.is_symplectic
        
        print(f"✅ Using integrator: {integrator_name}")
        print(f"  Symplectic: {is_symplectic}")
        
        # Run a few steps
        for _ in range(10):
            solver.step(1e-4)
        
        psi, phi = solver.get_mhd_state()
        
        # Should be finite
        assert jnp.all(jnp.isfinite(psi))
        assert jnp.all(jnp.isfinite(phi))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
