"""
M3 Step 1.5: Boundary Conditions Validation

Tests boundary condition implementation in toroidal geometry:
1. Periodic boundary conditions (θ direction)
2. Dirichlet boundary conditions (radial)
3. Flux conservation

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest


class TestPeriodicBoundaryConditions:
    """Test 1: Periodic BC in θ direction."""
    
    def test_gradient_theta_periodic(self):
        """
        Gradient operator should respect periodic BC in θ.
        
        For periodic function f(θ+2π) = f(θ):
            ∂f/∂θ|_{θ=0} should equal ∂f/∂θ|_{θ=2π}
        
        Validation: Use smooth periodic test function.
        """
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.operators import gradient_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        
        # Periodic test function: f = r² * sin(3θ)
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        f = r_grid**2 * np.sin(3 * theta_grid)
        
        # Compute gradient
        grad_r, grad_theta = gradient_toroidal(f, grid)
        
        # Check periodicity: grad_theta[:,0] ≈ grad_theta[:,-1]
        diff_boundary = np.abs(grad_theta[:, 0] - grad_theta[:, -1])
        max_diff = np.max(diff_boundary)
        
        # Relative error
        grad_magnitude = np.max(np.abs(grad_theta))
        relative_diff = max_diff / (grad_magnitude + 1e-12)
        
        print(f"\n✅ Periodic BC test (gradient):")
        print(f"  |∇_θ f(θ=0) - ∇_θ f(θ=2π)| max = {max_diff:.3e}")
        print(f"  Relative: {relative_diff:.3e}")
        
        # Periodic BC uses one-sided differences at boundaries
        # Tolerance should account for finite difference errors
        assert max_diff < 0.1, \
            f"Gradient not periodic: max diff {max_diff:.3e}"
        assert relative_diff < 0.02, \
            f"Gradient relative error {relative_diff:.3e} > 2%"
        
        print(f"  ✅ PASS: Gradient respects periodic BC")
    
    def test_laplacian_theta_periodic(self):
        """
        Laplacian should also respect periodic BC.
        
        For periodic f: ∇²f should also be periodic.
        """
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.operators import laplacian_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        
        # Periodic test function
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        f = r_grid**2 * (1 + 0.1 * np.cos(2 * theta_grid))
        
        # Compute Laplacian
        lap_f = laplacian_toroidal(f, grid)
        
        # Check periodicity
        diff_boundary = np.abs(lap_f[:, 0] - lap_f[:, -1])
        max_diff = np.max(diff_boundary)
        
        # Relative error
        lap_magnitude = np.max(np.abs(lap_f))
        relative_diff = max_diff / (lap_magnitude + 1e-12)
        
        print(f"\n✅ Periodic BC test (Laplacian):")
        print(f"  |∇²f(θ=0) - ∇²f(θ=2π)| max = {max_diff:.3e}")
        print(f"  Relative: {relative_diff:.3e}")
        
        assert max_diff < 0.01, \
            f"Laplacian not periodic: max diff {max_diff:.3e}"
        assert relative_diff < 0.01, \
            f"Laplacian relative error {relative_diff:.3e} > 1%"
        
        print(f"  ✅ PASS: Laplacian respects periodic BC")


class TestDirichletBoundaryConditions:
    """Test 2: Dirichlet BC at radial boundaries."""
    
    def test_boundary_enforcement(self):
        """
        Solver should enforce ψ=0 at r_min and r=a.
        
        Validation:
        - Initialize with non-zero boundary
        - Step solver
        - Check boundary is forced to zero
        """
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4, eta=1e-6, nu=1e-6)
        
        # Initialize with non-zero values everywhere (including boundaries)
        psi0 = np.ones((grid.nr, grid.ntheta))
        omega0 = np.zeros_like(psi0)
        
        solver.initialize(psi0, omega0)
        
        # Take one step
        solver.step()
        
        # Check boundaries are zero
        psi_boundary_inner = solver.psi[0, :]
        psi_boundary_outer = solver.psi[-1, :]
        
        max_inner = np.max(np.abs(psi_boundary_inner))
        max_outer = np.max(np.abs(psi_boundary_outer))
        
        print(f"\n✅ Dirichlet BC test:")
        print(f"  |ψ(r_min)| max = {max_inner:.3e}")
        print(f"  |ψ(r=a)| max = {max_outer:.3e}")
        
        assert max_inner < 1e-14, \
            f"Inner boundary not zero: {max_inner:.3e}"
        assert max_outer < 1e-14, \
            f"Outer boundary not zero: {max_outer:.3e}"
        
        print(f"  ✅ PASS: Dirichlet BC enforced")


class TestFluxConservation:
    """Test 3: Poloidal flux conservation."""
    
    def test_total_flux_conservation(self):
        """
        Total poloidal flux should be conserved (or controlled).
        
        For equilibrium with no sources:
            Φ(t) = ∫∫ ψ dV should not change
        
        Note: With Dirichlet BC and diffusion, flux may decrease.
        We test that it's controlled (not wild growth).
        """
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.solvers import ToroidalMHDSolver
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        solver = ToroidalMHDSolver(grid, dt=1e-4, eta=1e-6, nu=1e-6)
        
        # Initialize with smooth equilibrium (compatible with BC)
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)  # Zero at boundaries
        omega0 = np.zeros_like(psi0)
        
        solver.initialize(psi0, omega0)
        
        # Compute initial flux (volume integral)
        # dV = r * R * dr * dθ (toroidal volume element)
        jacobian = grid.jacobian()
        dr = grid.dr
        dtheta = grid.dtheta
        
        def compute_flux(psi):
            return np.sum(psi * jacobian) * dr * dtheta
        
        flux_0 = compute_flux(solver.psi)
        
        # Evolve 100 steps
        for _ in range(100):
            solver.step()
        
        flux_final = compute_flux(solver.psi)
        
        # With dissipation, flux should decrease or stay same
        flux_change = flux_final - flux_0
        relative_change = abs(flux_change) / (abs(flux_0) + 1e-12)
        
        print(f"\n✅ Flux conservation test:")
        print(f"  Initial flux: {flux_0:.6e}")
        print(f"  Final flux: {flux_final:.6e}")
        print(f"  Change: {flux_change:.6e} ({relative_change*100:.2f}%)")
        
        # Flux should not wildly increase
        # (Decrease is OK due to boundary dissipation)
        assert flux_change <= 1e-10, \
            f"Flux increased: {flux_change:.3e}"
        
        # Flux should not decrease too fast (sanity check)
        assert relative_change < 0.5, \
            f"Flux changed too much: {relative_change*100:.1f}%"
        
        print(f"  ✅ PASS: Flux controlled (no spurious generation)")


class TestBoundaryGradients:
    """Test 4: Boundary gradients (∇ψ·n̂ = 0 for no-flux condition)."""
    
    def test_no_flux_through_boundary(self):
        """
        For Dirichlet BC (ψ=0 at boundary), normal derivative should be well-defined.
        
        This tests that the boundary condition is compatible with
        the differential operators.
        
        Note: With ψ=0 at boundary, we have a flux condition,
        not strictly ∇ψ·n̂=0. This test checks compatibility.
        """
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.operators import gradient_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        
        # Create field with Dirichlet BC
        r_grid = grid.r_grid
        psi = r_grid**2 * (1 - r_grid / grid.a)
        
        # Compute gradient
        grad_r, grad_theta = gradient_toroidal(psi, grid)
        
        # At outer boundary (r=a), ∂ψ/∂r should be ~ -a (from BC)
        grad_r_outer = grad_r[-1, :]
        
        # Check it's non-zero and reasonable
        mean_grad = np.mean(np.abs(grad_r_outer))
        
        print(f"\n✅ Boundary gradient test:")
        print(f"  |∂ψ/∂r| at r=a: {mean_grad:.6f}")
        
        assert mean_grad > 0.01, \
            f"Boundary gradient suspiciously small: {mean_grad:.6f}"
        assert mean_grad < 10.0, \
            f"Boundary gradient suspiciously large: {mean_grad:.6f}"
        
        print(f"  ✅ PASS: Boundary gradients well-behaved")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
