"""
M3 Step 3.2.2: Action-Solver Integration Tests (Simplified)

Validation of action integration with Symplectic solver.

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest
from pytokmhd.operators import laplacian_toroidal


class TestActionBasics:
    """Test 1: Basic action acceptance."""
    
    def test_solver_accepts_action(self):
        """Solver should accept action parameter in step()."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators import SymplecticIntegrator
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        
        # Initialize with consistent IC
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        solver.initialize(psi0, omega0)
        
        print(f"\nSolver initialized:")
        print(f"  grid: {grid.nr} × {grid.ntheta}")
        print(f"  eta: {solver.eta}")
        print(f"  nu: {solver.nu}")
        
        # Step without action (should work)
        solver.step()
        print(f"Step without action: OK ✅")
        
        # Step with identity action
        action_identity = np.array([1.0, 1.0])
        solver.step(action=action_identity)
        print(f"Step with identity action: OK ✅")
        
        # Step with modulated action
        action_modulated = np.array([1.5, 0.8])
        solver.step(action=action_modulated)
        print(f"Step with modulated action: OK ✅")
        
        assert solver.t == 3 * solver.dt
    
    def test_extreme_actions(self):
        """Extreme action values should be accepted."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators import SymplecticIntegrator
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        solver.initialize(psi0, omega0)
        
        # Extreme actions (within [0.5, 2.0] bounds)
        extreme_actions = [
            np.array([0.5, 0.5]),   # Min values
            np.array([2.0, 2.0]),   # Max values
            np.array([0.5, 2.0]),   # Mixed
            np.array([2.0, 0.5]),   # Mixed
        ]
        
        print(f"\nTesting extreme actions:")
        for i, action in enumerate(extreme_actions):
            solver.step(action=action)
            print(f"  Action {i+1}: {action} → OK ✅")
        
        assert solver.t == len(extreme_actions) * solver.dt


class TestDefaultBehavior:
    """Test 2: Default behavior consistency."""
    
    def test_no_action_equals_identity(self):
        """step() without action should equal step(action=[1,1])."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators import SymplecticIntegrator
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        solver1 = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        solver2 = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        solver1.initialize(psi0, omega0)
        solver2.initialize(psi0, omega0)
        
        # Solver1: No action
        for _ in range(5):
            solver1.step()
        
        # Solver2: Identity action
        action_identity = np.array([1.0, 1.0])
        for _ in range(5):
            solver2.step(action=action_identity)
        
        # Results should be identical
        psi_diff = np.linalg.norm(solver1.psi - solver2.psi)
        omega_diff = np.linalg.norm(solver1.omega - solver2.omega)
        
        print(f"\nDefault behavior test:")
        print(f"  |psi_no_action - psi_identity|:   {psi_diff:.3e}")
        print(f"  |omega_no_action - omega_identity|: {omega_diff:.3e}")
        
        # Should be exactly the same (machine precision)
        assert psi_diff < 1e-14
        assert omega_diff < 1e-14


class TestActionIntegration:
    """Test 3: Integration with MHDAction class."""
    
    def test_mhd_action_handler_integration(self):
        """MHDAction handler should work with solver."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators import SymplecticIntegrator
        from pytokmhd.rl.actions import MHDAction
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        
        # Create action handler
        action_handler = MHDAction(eta_base=solver.eta, nu_base=solver.nu)
        
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        solver.initialize(psi0, omega0)
        
        print(f"\nMHDAction integration test:")
        print(f"  Action handler: eta_base={action_handler.eta_base}, nu_base={action_handler.nu_base}")
        
        # Sample actions
        actions = [
            np.array([1.0, 1.0]),   # Identity
            np.array([1.5, 0.8]),   # Modulated
            np.array([0.7, 1.3]),   # Modulated
        ]
        
        for i, action in enumerate(actions):
            eta_eff, nu_eff = action_handler.apply(action)
            solver.step(action=action)
            print(f"  Step {i+1}: action={action}, eta_eff={eta_eff:.2e}, nu_eff={nu_eff:.2e} ✅")
        
        assert solver.t == len(actions) * solver.dt


class TestActionEffects:
    """Test 4: Physical effects of actions."""
    
    def test_different_actions_give_different_results(self):
        """Different actions should produce different trajectories."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators import SymplecticIntegrator
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        solver_low = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        solver_high = SymplecticIntegrator(grid, dt=1e-4, eta=1e-5, nu=1e-4)
        
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        solver_low.initialize(psi0, omega0)
        solver_high.initialize(psi0, omega0)
        
        # Run with different actions
        action_low = np.array([0.5, 0.5])   # Low dissipation
        action_high = np.array([2.0, 2.0])  # High dissipation
        
        for _ in range(10):
            solver_low.step(action=action_low)
            solver_high.step(action=action_high)
        
        # Results should be different
        psi_diff = np.linalg.norm(solver_low.psi - solver_high.psi)
        omega_diff = np.linalg.norm(solver_low.omega - solver_high.omega)
        
        print(f"\nDifferent actions test:")
        print(f"  Low dissipation ([0.5, 0.5])")
        print(f"  High dissipation ([2.0, 2.0])")
        print(f"  |psi_low - psi_high|:   {psi_diff:.3e}")
        print(f"  |omega_low - omega_high|: {omega_diff:.3e}")
        
        # Should be noticeably different (not machine precision)
        assert psi_diff > 1e-10
        assert omega_diff > 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
