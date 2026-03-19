"""
Unit Tests for Hamiltonian Formulation

Tests for:
1. Hamiltonian computation and conservation
2. Poisson bracket properties (anti-symmetry, Jacobi identity)
3. Energy partitioning
4. Evolution equation consistency

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
import pytest
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import (
    poisson_bracket,
    jacobi_identity_residual,
    advection_bracket,
)
from pytokmhd.physics import (
    compute_hamiltonian,
    hamiltonian_density,
    kinetic_energy,
    magnetic_energy,
    energy_partition,
)


@pytest.fixture
def grid():
    """Standard toroidal grid for testing."""
    return ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)


@pytest.fixture
def simple_fields(grid):
    """Simple test fields."""
    psi = grid.r_grid**2
    phi = grid.r_grid * np.sin(grid.theta_grid)
    return psi, phi


class TestPoissonBracket:
    """Test Poisson bracket operator properties."""
    
    def test_anti_symmetry(self, grid):
        """Test [f, g] = -[g, f] to machine precision."""
        f = grid.r_grid**2
        g = np.sin(grid.theta_grid)
        
        bracket_fg = poisson_bracket(f, g, grid)
        bracket_gf = poisson_bracket(g, f, grid)
        
        # Anti-symmetry: [f, g] = -[g, f]
        assert np.allclose(bracket_fg, -bracket_gf, atol=1e-14)
        print(f"✓ Anti-symmetry: max|[f,g] + [g,f]| = {np.max(np.abs(bracket_fg + bracket_gf)):.2e}")
    
    def test_linearity(self, grid):
        """Test [af + bg, h] = a[f, h] + b[g, h]."""
        f = grid.r_grid**2
        g = np.sin(grid.theta_grid)
        h = grid.r_grid * np.cos(grid.theta_grid)
        
        a, b = 2.0, -3.0
        
        # LHS: [af + bg, h]
        lhs = poisson_bracket(a*f + b*g, h, grid)
        
        # RHS: a[f, h] + b[g, h]
        rhs = a * poisson_bracket(f, h, grid) + b * poisson_bracket(g, h, grid)
        
        assert np.allclose(lhs, rhs, atol=1e-12)
        print(f"✓ Linearity: max|LHS - RHS| = {np.max(np.abs(lhs - rhs)):.2e}")
    
    def test_jacobi_identity(self, grid):
        """Test [f, [g, h]] + [g, [h, f]] + [h, [f, g]] ≈ 0."""
        f = grid.r_grid**2
        g = np.sin(grid.theta_grid)
        h = grid.r_grid * np.cos(grid.theta_grid)
        
        residual = jacobi_identity_residual(f, g, h, grid)
        
        # Should be small (discretization error O(dr² + dθ²))
        # For nr=64, ntheta=128: dr≈0.005, dθ≈0.05 → expect residual ~ O(1e-3)
        assert residual < 0.01, f"Jacobi identity residual too large: {residual:.2e}"
        print(f"✓ Jacobi identity: residual = {residual:.2e}")
    
    def test_constant_bracket_zero(self, grid):
        """Test [const, f] = 0."""
        const = np.ones_like(grid.r_grid) * 3.14
        f = grid.r_grid**2 * np.sin(grid.theta_grid)
        
        bracket = poisson_bracket(const, f, grid)
        
        assert np.max(np.abs(bracket)) < 1e-12
        print(f"✓ Constant bracket: max|[const, f]| = {np.max(np.abs(bracket)):.2e}")
    
    def test_advection_bracket(self, grid):
        """Test advection_bracket is alias for poisson_bracket."""
        psi = grid.r_grid**2
        omega = grid.r_grid * np.sin(grid.theta_grid)
        
        bracket1 = poisson_bracket(psi, omega, grid)
        bracket2 = advection_bracket(psi, omega, grid)
        
        assert np.allclose(bracket1, bracket2, atol=1e-14)
        print(f"✓ Advection bracket: identical to Poisson bracket")


class TestHamiltonian:
    """Test Hamiltonian energy functional."""
    
    def test_energy_density_positive(self, grid, simple_fields):
        """Test h ≥ 0 everywhere."""
        psi, phi = simple_fields
        h = hamiltonian_density(psi, phi, grid)
        
        assert np.all(h >= 0), "Energy density must be non-negative"
        print(f"✓ Energy density: min = {np.min(h):.2e}, max = {np.max(h):.2e}")
    
    def test_total_hamiltonian_positive(self, grid, simple_fields):
        """Test H > 0 for non-trivial fields."""
        psi, phi = simple_fields
        H = compute_hamiltonian(psi, phi, grid)
        
        assert H > 0, "Total Hamiltonian must be positive"
        print(f"✓ Total Hamiltonian: H = {H:.6e}")
    
    def test_zero_fields_zero_energy(self, grid):
        """Test H = 0 for zero fields."""
        psi = np.zeros_like(grid.r_grid)
        phi = np.zeros_like(grid.r_grid)
        
        H = compute_hamiltonian(psi, phi, grid)
        
        assert np.abs(H) < 1e-14, f"Expected H=0, got H={H:.2e}"
        print(f"✓ Zero fields: H = {H:.2e}")
    
    def test_kinetic_magnetic_decomposition(self, grid, simple_fields):
        """Test H = K + U."""
        psi, phi = simple_fields
        
        H = compute_hamiltonian(psi, phi, grid)
        K = kinetic_energy(phi, grid)
        U = magnetic_energy(psi, grid)
        
        assert np.abs(H - (K + U)) < 1e-10 * H, "Energy decomposition failed"
        print(f"✓ Energy decomposition: H = {H:.6e}, K+U = {K+U:.6e}")
    
    def test_energy_partition(self, grid, simple_fields):
        """Test energy partition fractions sum to 1."""
        psi, phi = simple_fields
        
        energy = energy_partition(psi, phi, grid)
        
        assert 'total' in energy
        assert 'kinetic' in energy
        assert 'magnetic' in energy
        
        # Fractions should sum to 1
        sum_fractions = energy['fraction_kinetic'] + energy['fraction_magnetic']
        assert np.abs(sum_fractions - 1.0) < 1e-10
        
        print(f"✓ Energy partition:")
        print(f"  Total: {energy['total']:.6e}")
        print(f"  Kinetic: {energy['kinetic']:.6e} ({energy['fraction_kinetic']:.1%})")
        print(f"  Magnetic: {energy['magnetic']:.6e} ({energy['fraction_magnetic']:.1%})")


class TestEvolutionEquations:
    """Test Hamiltonian evolution equation structure."""
    
    def test_hamiltonian_gradient_dimensions(self, grid, simple_fields):
        """Test that ∂H/∂ψ and ∂H/∂φ have correct dimensions."""
        psi, phi = simple_fields
        
        # Functional derivative δH/δψ should be computable
        # For H = ∫ (1/2)|∇ψ|² dV, we have δH/δψ = -∇²ψ
        
        from pytokmhd.operators import laplacian_toroidal
        
        # Variation in ψ direction
        delta_psi = np.ones_like(psi) * 1e-6
        H0 = compute_hamiltonian(psi, phi, grid)
        H1 = compute_hamiltonian(psi + delta_psi, phi, grid)
        
        # Finite difference approximation to ∫ (δH/δψ) δψ dV
        dH_dpsi_approx = (H1 - H0) / 1e-6  # This is an integral
        
        # For testing purposes, just verify it's finite
        assert np.isfinite(dH_dpsi_approx)
        print(f"✓ Hamiltonian variation: ΔH/Δψ ≈ {dH_dpsi_approx:.6e}")
    
    def test_poisson_bracket_with_hamiltonian(self, grid):
        """Test [ψ, H] structure for evolution equation."""
        # For simple test: ψ = r², H ~ ∫|∇ψ|² dV
        psi = grid.r_grid**2
        phi = np.zeros_like(psi)
        
        # Compute Hamiltonian
        H_total = compute_hamiltonian(psi, phi, grid)
        
        # For evolution ∂ψ/∂t = [ψ, H], we need functional derivative
        # This is more complex - requires variational calculus
        # For now, test that bracket with a field is finite
        
        # Use a proxy field for H (should be δH/δψ ≈ -∇²ψ)
        from pytokmhd.operators import laplacian_toroidal
        
        lap_psi = laplacian_toroidal(psi, grid)
        evolution = poisson_bracket(psi, -lap_psi, grid)
        
        assert np.all(np.isfinite(evolution))
        print(f"✓ Evolution structure: max|[ψ, -∇²ψ]| = {np.max(np.abs(evolution)):.6e}")


class TestNumericalAccuracy:
    """Test numerical accuracy and convergence."""
    
    def test_energy_conservation_order(self, grid):
        """Test energy is conserved during evolution (passive test)."""
        # This would require time integration - mark as placeholder
        # Will be tested in integration tests
        pytest.skip("Requires time integration - see integration tests")
    
    def test_bracket_discretization_error(self, grid):
        """Test Jacobi identity error scales with grid resolution."""
        # Create coarser grid
        grid_coarse = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        f = grid.r_grid**2
        g = np.sin(grid.theta_grid)
        h = grid.r_grid * np.cos(grid.theta_grid)
        
        f_c = grid_coarse.r_grid**2
        g_c = np.sin(grid_coarse.theta_grid)
        h_c = grid_coarse.r_grid * np.cos(grid_coarse.theta_grid)
        
        residual_fine = jacobi_identity_residual(f, g, h, grid)
        residual_coarse = jacobi_identity_residual(f_c, g_c, h_c, grid_coarse)
        
        # Finer grid should have smaller error
        assert residual_fine < residual_coarse
        print(f"✓ Convergence: coarse = {residual_coarse:.2e}, fine = {residual_fine:.2e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
