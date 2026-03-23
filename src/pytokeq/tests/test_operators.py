"""
Unit tests for operators

Tests:
1. Grad-Shafranov operator (Δ*)
2. Poisson bracket ([f,g])

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop')

from pytokeq.core.operators import (
    build_grad_shafranov_operator,
    apply_grad_shafranov_operator,
    poisson_bracket
)


class TestGradShafranovOperator:
    """Tests for Δ* operator"""
    
    def setup_method(self):
        """Setup test grids"""
        # Simple uniform grid
        self.Nr = 32
        self.Nz = 32
        self.R = np.linspace(1.0, 2.0, self.Nr)  # Avoid R=0
        self.Z = np.linspace(-0.5, 0.5, self.Nz)
        
        # Build operator matrix
        self.L = build_grad_shafranov_operator(self.R, self.Z)
    
    def test_matrix_shape(self):
        """Test matrix has correct shape"""
        N = self.Nr * self.Nz
        assert self.L.shape == (N, N)
    
    def test_matrix_sparsity(self):
        """Test matrix is sparse (5-point stencil)"""
        N = self.Nr * self.Nz
        nnz = self.L.nnz
        
        # Interior points: 5 nonzeros
        # Boundary points: 3-4 nonzeros
        # Average ~5 per row
        expected_nnz = 5 * N  # Upper bound
        assert nnz <= expected_nnz
        
        # Sparsity ratio
        density = nnz / N**2
        assert density < 0.01  # Less than 1%
        print(f"Matrix density: {density*100:.3f}%")
    
    def test_analytic_R_squared(self):
        """
        Test: Δ*(R²) = 2
        
        For ψ = R², we have:
        ∂²ψ/∂R² = 2
        ∂ψ/∂R = 2R
        ∂²ψ/∂Z² = 0
        
        So Δ*ψ = 2 - (1/R)×2R + 0 = 2 - 2 = 0
        
        Wait, that's wrong! Let me recalculate:
        Δ*(R²) = ∂²(R²)/∂R² - (1/R)∂(R²)/∂R + ∂²(R²)/∂Z²
               = 2 - (1/R)×2R + 0
               = 2 - 2 = 0
        
        Hmm, R² gives 0. Let's use a different test function.
        """
        # Actually, let's test ψ = R² + Z²
        # ∂²/∂R²(R²+Z²) = 2
        # ∂/∂R(R²+Z²) = 2R
        # ∂²/∂Z²(R²+Z²) = 2
        # Δ*(R²+Z²) = 2 - 2R/(R) + 2 = 2 - 2 + 2 = 2
        
        psi = self.R[:, None]**2 + self.Z[None, :]**2
        psi_flat = psi.flatten()
        
        # Matrix-based
        result_matrix = self.L @ psi_flat
        result_matrix = result_matrix.reshape(self.Nr, self.Nz)
        
        # Direct FD
        result_fd = apply_grad_shafranov_operator(psi, self.R, self.Z)
        
        # Expected (interior points)
        expected = 2.0
        
        # Check interior points (boundaries have errors due to one-sided derivatives)
        interior = result_matrix[1:-1, 1:-1]
        
        # Should be close to 2.0
        error = np.abs(interior - expected).max()
        print(f"Max error for Δ*(R²+Z²): {error:.2e}")
        assert error < 1e-10, f"Error too large: {error}"
        
        # Check matrix vs FD match
        # With one-sided differences at boundaries, expect O(1e-12) roundoff
        error_fd = np.abs(result_matrix[1:-1, 1:-1] - result_fd[1:-1, 1:-1]).max()
        assert error_fd < 1e-11, f"Matrix vs FD mismatch: {error_fd}"
    
    def test_analytic_constant(self):
        """
        Test: Δ*(const) = 0
        
        All derivatives of constant are zero.
        """
        psi = 5.0 * np.ones((self.Nr, self.Nz))
        psi_flat = psi.flatten()
        
        result = self.L @ psi_flat
        result = result.reshape(self.Nr, self.Nz)
        
        # Should be zero everywhere (including boundaries)
        # Note: With one-sided differences at boundaries, expect O(1e-12) error
        error = np.abs(result).max()
        assert error < 1e-11, f"Error: {error:.3e}"
    
    def test_symmetry(self):
        """
        Test matrix is symmetric for this operator.
        
        Wait, is Δ* symmetric? Let me think...
        Δ* = ∂²/∂R² - (1/R)∂/∂R + ∂²/∂Z²
        
        The (1/R)∂/∂R term breaks symmetry!
        So this test should FAIL, which is correct.
        """
        # Check if symmetric
        diff = self.L - self.L.T
        is_symmetric = diff.nnz == 0
        
        # Δ* is NOT symmetric due to 1/R term
        assert not is_symmetric, "Δ* should not be symmetric"
        print("✓ Correctly identified as non-symmetric")
    
    def test_linear_function(self):
        """
        Test: Δ*(aR + bZ + c) = -a/R
        
        For ψ = aR + bZ + c:
        ∂²ψ/∂R² = 0
        ∂ψ/∂R = a
        ∂²ψ/∂Z² = 0
        
        Δ*ψ = 0 - (1/R)×a + 0 = -a/R
        """
        a, b, c = 2.0, 3.0, 5.0
        psi = a * self.R[:, None] + b * self.Z[None, :] + c
        psi_flat = psi.flatten()
        
        result = self.L @ psi_flat
        result = result.reshape(self.Nr, self.Nz)
        
        # Expected: -a/R
        expected = -a / self.R[:, None] * np.ones((self.Nr, self.Nz))
        
        # Interior points
        error = np.abs(result[1:-1, 1:-1] - expected[1:-1, 1:-1]).max()
        print(f"Max error for linear function: {error:.2e}")
        assert error < 1e-10


class TestPoissonBracket:
    """Tests for Poisson bracket [f,g]"""
    
    def setup_method(self):
        """Setup test grids"""
        self.Nr = 32
        self.Nz = 32
        self.R = np.linspace(0.5, 1.5, self.Nr)
        self.Z = np.linspace(-0.5, 0.5, self.Nz)
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        
        # Create 2D grids for tests
        self.R_grid, self.Z_grid = np.meshgrid(self.R, self.Z, indexing='ij')
    
    def test_antisymmetry(self):
        """
        Test: [f,g] = -[g,f]
        """
        f = self.R[:, None]**2 + self.Z[None, :]**2
        g = np.sin(2*np.pi*self.R[:, None]) * np.cos(2*np.pi*self.Z[None, :])
        
        bracket_fg = poisson_bracket(f, g, self.dR, self.dZ)
        bracket_gf = poisson_bracket(g, f, self.dR, self.dZ)
        
        # Should be antisymmetric
        error = np.abs(bracket_fg + bracket_gf).max()
        print(f"Antisymmetry error: {error:.2e}")
        assert error < 1e-13, f"Error: {error:.3e}"
    
    def test_linearity(self):
        """
        Test: [af + bg, h] = a[f,h] + b[g,h]
        """
        a, b = 2.0, 3.0
        f = self.R_grid**2
        g = self.Z_grid**2
        h = np.sin(np.pi * self.R_grid) * np.sin(np.pi * self.Z_grid)
        
        # LHS: [af + bg, h]
        lhs = poisson_bracket(a*f + b*g, h, self.dR, self.dZ)
        
        # RHS: a[f,h] + b[g,h]
        rhs = a * poisson_bracket(f, h, self.dR, self.dZ) + b * poisson_bracket(g, h, self.dR, self.dZ)
        
        error = np.abs(lhs - rhs).max()
        print(f"Linearity error: {error:.2e}")
        assert error < 1e-13, f"Error: {error:.3e}"
    
    def test_constant_gives_zero(self):
        """
        Test: [const, g] = 0 and [f, const] = 0
        """
        const = 5.0 * np.ones((self.Nr, self.Nz))
        g = self.R[:, None] * self.Z[None, :]
        
        bracket1 = poisson_bracket(const, g, self.dR, self.dZ)
        bracket2 = poisson_bracket(g, const, self.dR, self.dZ)
        
        assert np.abs(bracket1).max() < 1e-14
        assert np.abs(bracket2).max() < 1e-14
    
    def test_sum_is_zero(self):
        """
        Test: ∫∫ [f,g] dR dZ = 0 (conservation)
        
        This is a key property of Arakawa scheme.
        """
        f = np.sin(2*np.pi*self.R[:, None]) * np.sin(2*np.pi*self.Z[None, :])
        g = np.cos(np.pi*self.R[:, None]) * np.cos(np.pi*self.Z[None, :])
        
        bracket = poisson_bracket(f, g, self.dR, self.dZ)
        
        # Integrate (simple sum × grid spacing)
        integral = np.sum(bracket) * self.dR * self.dZ
        
        print(f"Integral of [f,g]: {integral:.2e}")
        
        # Should be very small (machine precision)
        # Note: Not exactly zero due to boundary effects
        assert abs(integral) < 1e-12
    
    def test_known_analytic(self):
        """
        Test against known analytic case.
        
        For f = R, g = Z:
        [R, Z] = ∂R/∂R × ∂Z/∂Z - ∂R/∂Z × ∂Z/∂R
               = 1 × 1 - 0 × 0 = 1
        """
        f = self.R[:, None] * np.ones((self.Nr, self.Nz))
        g = np.ones((self.Nr, self.Nz)) * self.Z[None, :]
        
        bracket = poisson_bracket(f, g, self.dR, self.dZ)
        
        # Interior should be 1.0
        interior = bracket[2:-2, 2:-2]  # Avoid boundaries
        expected = 1.0
        
        error = np.abs(interior - expected).max()
        print(f"Error for [R, Z] = 1: {error:.2e}")
        assert error < 1e-10


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
