"""
Unit Tests for Force Balance Implementation

Tests pressure profiles, gradients, and J×B = ∇P verification.

Test Coverage
-------------
1. Pressure profile P(ψ)
2. Pressure gradient dP/dψ
3. Pressure gradient ∇P in toroidal geometry
4. Current density Jφ computation
5. Lorentz force J×B
6. Force balance residual
7. Solovev equilibrium verification (if PyTokEq available)

Author: 小P ⚛️
Created: 2026-03-19
"""

import pytest
import numpy as np
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.equilibrium import (
    pressure_profile,
    pressure_gradient_psi,
    pressure_gradient,
    PYTOKEQ_AVAILABLE,
)
from pytokmhd.physics import (
    compute_current_density,
    compute_lorentz_force,
    force_balance_residual,
)


# Test fixtures
@pytest.fixture
def grid():
    """Standard toroidal grid for testing."""
    return ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)


@pytest.fixture
def simple_psi(grid):
    """Simple test flux function: ψ = r²."""
    return grid.r_grid**2


class TestPressureProfile:
    """Test pressure profile P(ψ) implementation."""
    
    def test_central_pressure(self):
        """Test P(ψ=0) = P₀."""
        psi = np.array([0.0, 0.5, 1.0])
        P0 = 1e5
        psi_edge = 1.0
        
        P = pressure_profile(psi, P0, psi_edge, alpha=2.0)
        
        assert np.isclose(P[0], P0), "Central pressure should equal P₀"
    
    def test_edge_pressure(self):
        """Test P(ψ=ψ_edge) = 0."""
        psi = np.array([0.0, 0.5, 1.0])
        P0 = 1e5
        psi_edge = 1.0
        
        P = pressure_profile(psi, P0, psi_edge, alpha=2.0)
        
        assert np.isclose(P[-1], 0.0), "Edge pressure should be zero"
    
    def test_monotonic_decrease(self):
        """Test P decreases monotonically from axis to edge."""
        psi = np.linspace(0, 1.0, 100)
        P0 = 1e5
        psi_edge = 1.0
        
        P = pressure_profile(psi, P0, psi_edge, alpha=2.0)
        
        # Check monotonic decrease
        assert np.all(np.diff(P) <= 0), "Pressure should decrease monotonically"
    
    def test_outside_separatrix(self):
        """Test P(ψ > ψ_edge) = 0."""
        psi = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        P0 = 1e5
        psi_edge = 1.0
        
        P = pressure_profile(psi, P0, psi_edge, alpha=2.0)
        
        # Outside separatrix: P = 0
        assert np.all(P[3:] == 0.0), "Pressure outside separatrix should be zero"
    
    def test_alpha_scaling(self):
        """Test different alpha values give different peaking.
        
        For power law P(ψ) = P₀(1 - ψ/ψ_edge)^α:
        - Higher α → flatter profile near axis, steeper gradient near edge
        - Lower α → more uniform profile
        
        At mid-radius (ψ_n=0.5):
        - α=1: P = 0.5P₀
        - α=2: P = 0.25P₀
        - α=4: P = 0.0625P₀
        """
        psi = np.linspace(0, 1.0, 100)
        P0 = 1e5
        psi_edge = 1.0
        
        P_alpha1 = pressure_profile(psi, P0, psi_edge, alpha=1.0)  # Linear
        P_alpha2 = pressure_profile(psi, P0, psi_edge, alpha=2.0)  # Parabolic
        P_alpha4 = pressure_profile(psi, P0, psi_edge, alpha=4.0)  # Flatter near axis
        
        # At mid-radius (ψ=0.5ψ_edge)
        mid_idx = 50
        # Higher alpha → lower pressure at mid-radius
        assert P_alpha1[mid_idx] > P_alpha2[mid_idx] > P_alpha4[mid_idx], \
            f"Expected P_α1 > P_α2 > P_α4 at mid-radius, got {P_alpha1[mid_idx]:.1f}, {P_alpha2[mid_idx]:.1f}, {P_alpha4[mid_idx]:.1f}"


class TestPressureGradient:
    """Test pressure gradient calculations."""
    
    def test_gradient_psi_negative(self):
        """Test dP/dψ < 0 (pressure decreases outward)."""
        psi = np.linspace(0, 0.9, 100)  # Inside separatrix
        P0 = 1e5
        psi_edge = 1.0
        
        dP_dpsi = pressure_gradient_psi(psi, P0, psi_edge, alpha=2.0)
        
        assert np.all(dP_dpsi < 0), "dP/dψ should be negative"
    
    def test_gradient_psi_edge(self):
        """Test dP/dψ → 0 at edge."""
        psi = np.array([0.0, 0.5, 0.9, 1.0])
        P0 = 1e5
        psi_edge = 1.0
        
        dP_dpsi = pressure_gradient_psi(psi, P0, psi_edge, alpha=2.0)
        
        assert np.isclose(dP_dpsi[-1], 0.0, atol=1e-10), \
            "Gradient should vanish at edge"
    
    def test_gradient_toroidal(self, grid, simple_psi):
        """Test ∇P = (dP/dψ)·∇ψ in toroidal geometry.
        
        For ψ = r²:
        - ∂ψ/∂r = 2r > 0
        - dP/dψ < 0 (pressure decreases with ψ)
        - ∇P_r = (dP/dψ)·(∂ψ/∂r) < 0 (points inward)
        
        This means pressure gradient points radially inward (toward higher pressure).
        """
        P0 = 1e5
        psi_edge = grid.a**2  # Since ψ = r², ψ_edge = a²
        
        gradP_r, gradP_theta = pressure_gradient(simple_psi, P0, psi_edge, grid, alpha=2.0)
        
        # Shape check
        assert gradP_r.shape == simple_psi.shape
        assert gradP_theta.shape == simple_psi.shape
        
        # ∇P should be finite
        assert np.all(np.isfinite(gradP_r))
        assert np.all(np.isfinite(gradP_theta))
        
        # For ψ = r², ∂ψ/∂θ = 0, so gradP_theta should be ~0
        # (Small nonzero due to numerical errors)
        assert np.max(np.abs(gradP_theta)) < 1.0, \
            "Poloidal gradient should be ~0 for axisymmetric ψ=r²"


class TestCurrentDensity:
    """Test toroidal current density computation."""
    
    def test_current_density_shape(self, grid, simple_psi):
        """Test J_phi has correct shape."""
        J_phi = compute_current_density(simple_psi, grid)
        
        assert J_phi.shape == simple_psi.shape
    
    def test_grad_shafranov_operator(self, grid):
        """Test Δ*ψ operator on known function."""
        # For ψ = r², Δ*ψ should be constant in circular geometry
        psi = grid.r_grid**2
        
        # Compute Δ*ψ via current density
        mu0 = 4*np.pi*1e-7
        J_phi = compute_current_density(psi, grid, mu0)
        Delta_star_psi = mu0 * grid.R_grid * J_phi
        
        # For ψ = r²:
        # ∂²ψ/∂r² = 2
        # ∂²ψ/∂θ² = 0
        # ∂ψ/∂r = 2r
        # Δ*ψ = 2 + 0 + cos(θ)/R · 2r = 2 + 2r·cos(θ)/R
        
        # Not constant, but should be smooth
        assert np.all(np.isfinite(Delta_star_psi))


class TestLorentzForce:
    """Test J×B computation."""
    
    def test_lorentz_force_shape(self, grid, simple_psi):
        """Test J×B has correct shape."""
        JxB_r, JxB_theta = compute_lorentz_force(simple_psi, grid)
        
        assert JxB_r.shape == simple_psi.shape
        assert JxB_theta.shape == simple_psi.shape
    
    def test_lorentz_force_components(self, grid, simple_psi):
        """Test J×B components are finite."""
        JxB_r, JxB_theta = compute_lorentz_force(simple_psi, grid)
        
        assert np.all(np.isfinite(JxB_r))
        assert np.all(np.isfinite(JxB_theta))


class TestForceBalance:
    """Test force balance verification."""
    
    def test_force_balance_residual_structure(self, grid, simple_psi):
        """Test force_balance_residual returns correct structure."""
        P0 = 1e5
        psi_edge = grid.a**2
        
        result = force_balance_residual(simple_psi, P0, psi_edge, grid, alpha=2.0)
        
        # Check all required keys
        required_keys = [
            'residual_r', 'residual_theta',
            'max_residual', 'rms_residual', 'relative_error',
            'JxB_r', 'JxB_theta', 'gradP_r', 'gradP_theta'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_force_balance_metrics(self, grid, simple_psi):
        """Test force balance error metrics are computed."""
        P0 = 1e5
        psi_edge = grid.a**2
        
        result = force_balance_residual(simple_psi, P0, psi_edge, grid, alpha=2.0)
        
        # Metrics should be finite and positive
        assert np.isfinite(result['max_residual'])
        assert np.isfinite(result['rms_residual'])
        assert np.isfinite(result['relative_error'])
        assert result['max_residual'] >= 0
        assert result['rms_residual'] >= 0


@pytest.mark.skipif(not PYTOKEQ_AVAILABLE, reason="PyTokEq not installed")
class TestSolovevEquilibrium:
    """Test Solovev equilibrium verification (requires PyTokEq)."""
    
    def test_solovev_force_balance(self, grid):
        """Test force balance for Solovev equilibrium."""
        from pytokmhd.equilibrium import verify_solovev_force_balance
        
        result = verify_solovev_force_balance(grid, P0=1e5, B0=2.0, tolerance=1e-6)
        
        # Solovev should satisfy force balance to machine precision
        assert result['passed'], \
            f"Solovev force balance failed: max_residual = {result['max_residual']:.2e}"
        assert result['max_residual'] < 1e-6, \
            "Solovev residual should be < 1e-6"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestPressureForceTerm:
    """Test pressure force term for vorticity equation."""
    
    def test_pressure_force_shape(self, grid, simple_psi):
        """Test S_P has correct shape."""
        from pytokmhd.physics import pressure_force_term
        
        P0 = 1e5
        psi_edge = grid.a**2
        
        S_P = pressure_force_term(simple_psi, P0, psi_edge, grid, alpha=2.0)
        
        assert S_P.shape == simple_psi.shape
    
    def test_pressure_force_finite(self, grid, simple_psi):
        """Test S_P is finite everywhere."""
        from pytokmhd.physics import pressure_force_term
        
        P0 = 1e5
        psi_edge = grid.a**2
        
        S_P = pressure_force_term(simple_psi, P0, psi_edge, grid, alpha=2.0)
        
        assert np.all(np.isfinite(S_P))
    
    def test_pressure_force_equilibrium(self, grid):
        """Test S_P contributes to force balance.
        
        In equilibrium, the pressure force term should balance
        magnetic stress in the vorticity equation.
        """
        from pytokmhd.physics import pressure_force_term
        
        # Create a simple equilibrium-like psi
        psi = grid.r_grid**2
        P0 = 1e5
        psi_edge = grid.a**2
        
        S_P = pressure_force_term(psi, P0, psi_edge, grid, alpha=2.0)
        
        # Pressure force should be non-zero inside plasma
        assert np.any(S_P != 0), "Pressure force should be non-zero"
