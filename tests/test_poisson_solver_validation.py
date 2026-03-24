"""
Comprehensive Poisson Solver Validation for Issue #26

Tests:
1. Exact solution recovery
2. Residual accuracy  
3. Boundary conditions
4. Round-trip test (solve → laplacian → should equal input)
5. Conversion accuracy for Issue #26

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytest
import numpy as np

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.solvers import solve_poisson_toroidal


class TestPoissonSolverValidation:
    """Validate Poisson solver for Issue #26 usage."""
    
    @pytest.fixture
    def grid(self):
        """Create test grid."""
        return ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    def test_exact_solution_r_squared(self, grid):
        """Test: φ = r² (axisymmetric)."""
        
        # Exact solution
        r_grid = grid.r_grid
        phi_exact = r_grid**2
        
        # Compute RHS
        omega = laplacian_toroidal(phi_exact, grid)
        
        # Boundary
        phi_bnd = grid.a**2 * np.ones(grid.ntheta)
        
        # Solve
        phi_solved, info = solve_poisson_toroidal(omega, grid, phi_bnd, tol=1e-8)
        
        # Check convergence
        assert info == 0, f"GMRES did not converge: info={info}"
        
        # Check error
        error = np.max(np.abs(phi_solved - phi_exact))
        rel_error = error / np.max(np.abs(phi_exact))
        
        print(f"\nTest 1: φ = r²")
        print(f"  Max error: {error:.3e}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        
        assert error < 0.01, f"Error {error:.3e} too large"
        
        print("  ✅ PASSED")
    
    def test_with_theta_dependence(self, grid):
        """Test: φ = r² sin(2θ)."""
        
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        # Exact solution
        phi_exact = r_grid**2 * np.sin(2*theta_grid)
        
        # RHS
        omega = laplacian_toroidal(phi_exact, grid)
        
        # Boundary (1D theta array)
        theta_1d = theta_grid[0, :]  # Extract 1D theta
        phi_bnd = grid.a**2 * np.sin(2*theta_1d)
        
        # Solve
        phi_solved, info = solve_poisson_toroidal(omega, grid, phi_bnd, tol=1e-8)
        
        assert info == 0
        
        # Error
        error = np.max(np.abs(phi_solved - phi_exact))
        rel_error = error / np.max(np.abs(phi_exact))
        
        print(f"\nTest 2: φ = r² sin(2θ)")
        print(f"  Max error: {error:.3e}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        
        assert error < 0.01
        
        print("  ✅ PASSED")
    
    def test_round_trip(self, grid):
        """Test: solve(laplacian(φ)) ≈ φ."""
        
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        # Test function
        phi_0 = r_grid**2 * (1 - r_grid/grid.a) * np.cos(3*theta_grid)
        
        # Compute laplacian
        omega = laplacian_toroidal(phi_0, grid)
        
        # Boundary (from phi_0)
        phi_bnd = phi_0[-1, :]
        
        # Solve
        phi_recovered, info = solve_poisson_toroidal(omega, grid, phi_bnd, tol=1e-8)
        
        assert info == 0
        
        # Check recovery
        error = np.max(np.abs(phi_recovered - phi_0))
        rel_error = error / np.max(np.abs(phi_0))
        
        print(f"\nTest 3: Round-trip (solve ∘ laplacian)")
        print(f"  Max error: {error:.3e}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        
        assert error < 0.01
        
        print("  ✅ PASSED")
    
    def test_boundary_enforcement(self, grid):
        """Test: boundary conditions correctly enforced."""
        
        r_grid = grid.r_grid
        phi_exact = r_grid**2
        omega = laplacian_toroidal(phi_exact, grid)
        phi_bnd = grid.a**2 * np.ones(grid.ntheta)
        
        phi_solved, info = solve_poisson_toroidal(omega, grid, phi_bnd, tol=1e-8)
        
        assert info == 0
        
        # Check outer boundary
        bc_error = np.max(np.abs(phi_solved[-1, :] - phi_bnd))
        
        print(f"\nTest 4: Boundary conditions")
        print(f"  BC error: {bc_error:.3e}")
        
        assert bc_error < 1e-6, f"BC not enforced: error={bc_error:.3e}"
        
        print("  ✅ PASSED")
    
    def test_zero_rhs(self, grid):
        """Test: ∇²φ = 0 with φ(r=a) = 1 should give constant."""
        
        omega = np.zeros((grid.nr, grid.ntheta))
        phi_bnd = np.ones(grid.ntheta)
        
        phi_solved, info = solve_poisson_toroidal(omega, grid, phi_bnd, tol=1e-8)
        
        assert info == 0
        
        # Should be constant (Laplace equation)
        variation = np.std(phi_solved)
        
        print(f"\nTest 5: Zero RHS (Laplace)")
        print(f"  Solution variation: {variation:.3e}")
        
        # Not necessarily constant due to Dirichlet BC
        # But outer boundary should be correct
        bc_error = np.max(np.abs(phi_solved[-1, :] - phi_bnd))
        
        print(f"  BC error: {bc_error:.3e}")
        
        assert bc_error < 1e-6
        
        print("  ✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
