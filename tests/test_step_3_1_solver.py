"""
M3 Step 3.1: Basic Solver Integration Test

Minimal test to verify ToroidalMHDSolver framework works.
"""

import numpy as np
import pytest


class TestToroidalMHDSolverBasic:
    """Basic integration test for ToroidalMHDSolver."""
    
    def test_solver_instantiation(self):
        """Test that solver can be created."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4)
        
        assert solver.grid is grid
        assert solver.dt == 1e-4
        assert solver.eta == 1e-5
        assert solver.nu == 1e-4
        
        print("✅ Solver instantiation: OK")
    
    def test_solver_initialization(self):
        """Test that solver can be initialized with fields."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4)
        
        # Create simple initial state
        psi0 = np.zeros((grid.nr, grid.ntheta))
        omega0 = np.zeros((grid.nr, grid.ntheta))
        
        solver.initialize(psi0, omega0)
        
        assert solver.psi is not None
        assert solver.omega is not None
        assert solver.time == 0.0
        assert solver.n_steps == 0
        
        print("✅ Solver initialization: OK")
    
    def test_solver_single_step(self):
        """Test that solver can execute single time step."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4)
        
        # Initialize with zeros (equilibrium)
        psi0 = np.zeros((grid.nr, grid.ntheta))
        omega0 = np.zeros((grid.nr, grid.ntheta))
        solver.initialize(psi0, omega0)
        
        # Single step
        psi, omega = solver.step()
        
        # Check outputs
        assert psi.shape == (grid.nr, grid.ntheta)
        assert omega.shape == (grid.nr, grid.ntheta)
        assert solver.time == 1e-4
        assert solver.n_steps == 1
        
        # For zero initial state, should remain zero (within numerical precision)
        assert np.max(np.abs(psi)) < 1e-12
        assert np.max(np.abs(omega)) < 1e-12
        
        print("✅ Solver single step: OK")
    
    def test_solver_run(self):
        """Test that solver can run multiple steps."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4)
        
        # Initialize
        psi0 = np.zeros((grid.nr, grid.ntheta))
        omega0 = np.zeros((grid.nr, grid.ntheta))
        solver.initialize(psi0, omega0)
        
        # Run 1000 steps
        history = solver.run(n_steps=1000, save_interval=100)
        
        # Check history
        assert 'psi' in history
        assert 'omega' in history
        assert 'time' in history
        
        # Should have 11 snapshots (initial + 10 saves)
        assert len(history['psi']) == 11
        assert len(history['omega']) == 11
        assert len(history['time']) == 11
        
        # Final time
        assert np.isclose(history['time'][-1], 0.1, rtol=1e-10)  # 1000 * 1e-4
        
        print("✅ Solver run: OK")
        print(f"  Final time: {history['time'][-1]}")
        print(f"  Snapshots saved: {len(history['psi'])}")
    
    def test_solver_compute_rhs(self):
        """Test that RHS computation works."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4)
        
        # Create test fields
        psi = np.random.randn(grid.nr, grid.ntheta) * 0.01
        omega = np.random.randn(grid.nr, grid.ntheta) * 0.01
        
        # Compute RHS
        dpsi_dt, domega_dt = solver.compute_rhs(psi, omega)
        
        # Check shape
        assert dpsi_dt.shape == (grid.nr, grid.ntheta)
        assert domega_dt.shape == (grid.nr, grid.ntheta)
        
        # Check finite
        assert np.all(np.isfinite(dpsi_dt))
        assert np.all(np.isfinite(domega_dt))
        
        print("✅ RHS computation: OK")
        print(f"  dpsi_dt range: [{np.min(dpsi_dt):.2e}, {np.max(dpsi_dt):.2e}]")
        print(f"  domega_dt range: [{np.min(domega_dt):.2e}, {np.max(domega_dt):.2e}]")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
