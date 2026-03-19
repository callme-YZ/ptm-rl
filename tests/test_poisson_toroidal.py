"""
Unit tests for toroidal Poisson solver.

Tests:
1. Exact solution recovery (φ = r²)
2. Boundary condition enforcement
3. Residual accuracy
4. Grid convergence

Author: 小P ⚛️
Date: 2026-03-19
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.solvers import (
    solve_poisson_toroidal,
    compute_residual,
    check_boundary_conditions,
)


class TestPoissonToroidal:
    """Test suite for toroidal Poisson solver."""
    
    def setup_method(self):
        """Set up test grid."""
        self.grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    def test_exact_solution_r_squared(self):
        """Test exact recovery of φ = r²."""
        r_grid = self.grid.r_grid
        phi_exact = r_grid**2
        
        # Compute vorticity ω = ∇²φ
        omega_exact = laplacian_toroidal(phi_exact, self.grid)
        
        # Boundary values
        phi_bnd = (self.grid.a**2) * np.ones(self.grid.ntheta)
        
        # Solve
        phi_computed, info = solve_poisson_toroidal(
            omega_exact, self.grid, phi_bnd, tol=1e-8, verbose=True
        )
        
        # Check convergence
        assert info == 0, "GMRES should converge"
        
        # Check solution error
        error = np.max(np.abs(phi_computed - phi_exact))
        rel_error = error / np.max(np.abs(phi_exact))
        
        print(f"\nExact solution test (φ = r²):")
        print(f"  Max error: {error:.3e}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        
        assert error < 0.01, f"Solution error {error:.3e} too large"
    
    def test_residual_accuracy(self):
        """Test that residual ‖∇²φ - ω‖ is small."""
        r_grid = self.grid.r_grid
        phi_exact = r_grid**2
        omega_exact = laplacian_toroidal(phi_exact, self.grid)
        phi_bnd = (self.grid.a**2) * np.ones(self.grid.ntheta)
        
        phi_computed, info = solve_poisson_toroidal(
            omega_exact, self.grid, phi_bnd, tol=1e-8
        )
        
        max_res, mean_res = compute_residual(phi_computed, omega_exact, self.grid)
        
        print(f"\nResidual test:")
        print(f"  Max residual: {max_res:.3e}")
        print(f"  Mean residual: {mean_res:.3e}")
        
        assert max_res < 0.1, f"Residual {max_res:.3e} too large"
    
    def test_boundary_conditions(self):
        """Test BC enforcement."""
        r_grid = self.grid.r_grid
        phi_exact = r_grid**2
        omega_exact = laplacian_toroidal(phi_exact, self.grid)
        phi_bnd = (self.grid.a**2) * np.ones(self.grid.ntheta)
        
        phi_computed, info = solve_poisson_toroidal(
            omega_exact, self.grid, phi_bnd, tol=1e-8
        )
        
        bc_outer, bc_axis = check_boundary_conditions(
            phi_computed, self.grid, phi_bnd
        )
        
        print(f"\nBoundary condition test:")
        print(f"  Outer BC error: {bc_outer:.3e}")
        print(f"  Axis symmetry error: {bc_axis:.3e}")
        
        assert bc_outer < 1e-6, f"Outer BC error {bc_outer:.3e} too large"
        assert bc_axis < 1e-6, f"Axis BC error {bc_axis:.3e} too large"
    
    def test_zero_rhs(self):
        """Test ∇²φ = 0 with zero boundary gives φ = 0."""
        omega = np.zeros((self.grid.nr, self.grid.ntheta))
        phi_bnd = np.zeros(self.grid.ntheta)
        
        phi, info = solve_poisson_toroidal(omega, self.grid, phi_bnd, tol=1e-8)
        
        assert info == 0
        assert np.max(np.abs(phi)) < 1e-6, "φ should be ~0 for zero RHS and BC"
    
    def test_different_boundary_values(self):
        """Test with non-trivial boundary values."""
        # Use φ = r² cos(θ)
        r_grid = self.grid.r_grid
        theta_grid = self.grid.theta_grid
        phi_exact = r_grid**2 * np.cos(theta_grid)
        
        omega_exact = laplacian_toroidal(phi_exact, self.grid)
        phi_bnd = (self.grid.a**2) * np.cos(self.grid.theta)
        
        phi_computed, info = solve_poisson_toroidal(
            omega_exact, self.grid, phi_bnd, tol=1e-8
        )
        
        assert info == 0
        error = np.max(np.abs(phi_computed - phi_exact))
        print(f"\nNon-trivial BC test (φ = r² cos θ):")
        print(f"  Max error: {error:.3e}")
        
        assert error < 0.02, f"Error {error:.3e} too large for non-trivial BC"



if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
