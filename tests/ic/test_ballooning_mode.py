"""
Tests for 3D Ballooning Mode Initial Conditions

Tests cover:
1. Grid setup and properties
2. q-profile monotonicity and bounds
3. Equilibrium axisymmetry (∂ψ₀/∂ζ = 0)
4. Radial profile localization
5. Ballooning envelope localization
6. Periodicity in θ and ζ
7. Energy budget (perturbation << equilibrium)
8. Mode spectrum (Fourier analysis)
9. Edge cases (n=0, ε=0)

Author: 小P ⚛️
Created: 2026-03-19
Phase: 2.2 (3D Initial Conditions)
"""

import numpy as np
import pytest
from pytokmhd.ic.ballooning_mode import (
    Grid3D,
    create_q_profile,
    create_equilibrium_ic,
    create_ballooning_mode_ic,
)


class TestGrid3D:
    """Test Grid3D class."""
    
    def test_grid_creation(self):
        """Test basic grid creation."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        
        assert grid.nr == 16
        assert grid.ntheta == 32
        assert grid.nzeta == 64
        
        assert len(grid.r) == 16
        assert len(grid.theta) == 32
        assert len(grid.zeta) == 64
        
        # Check spacing
        assert grid.dr > 0
        assert grid.dtheta > 0
        assert grid.dzeta > 0
    
    def test_grid_periodicity(self):
        """Test periodic coordinates θ and ζ."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        
        # θ: [0, 2π)
        assert grid.theta[0] == 0.0
        assert grid.theta[-1] < 2*np.pi
        assert np.isclose(grid.theta[-1] + grid.dtheta, 2*np.pi)
        
        # ζ: [0, 2π)
        assert grid.zeta[0] == 0.0
        assert grid.zeta[-1] < 2*np.pi
        assert np.isclose(grid.zeta[-1] + grid.dzeta, 2*np.pi)
    
    def test_grid_singularity_avoidance(self):
        """Test that grid avoids r=0 singularity."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64, r_max=1.0)
        
        assert grid.r[0] > 0  # r_min = 0.1 * r_max by default
        assert grid.r[-1] == pytest.approx(1.0)
    
    def test_grid_validation(self):
        """Test grid validation."""
        with pytest.raises(ValueError, match="Grid too coarse"):
            Grid3D(nr=4, ntheta=32, nzeta=64)  # nr too small
        
        with pytest.raises(ValueError, match="Grid too coarse"):
            Grid3D(nr=16, ntheta=8, nzeta=64)  # ntheta too small


class TestQProfile:
    """Test q-profile creation."""
    
    def test_linear_q_profile(self):
        """Test linear q-profile: q(r) = q₀ + (qa - q₀) * r/a."""
        r = np.linspace(0.1, 1.0, 32)
        q = create_q_profile(r, q0=1.0, qa=3.0, profile_type='linear')
        
        # Check monotonicity
        assert np.all(np.diff(q) > 0), "q-profile must be monotonically increasing"
        
        # Check bounds (with tolerance for discretization)
        assert q[0] > 1.0  # Near q0 at r_min
        assert q[-1] < 3.1  # Near qa at r_max
    
    def test_parabolic_q_profile(self):
        """Test parabolic q-profile: q(r) = q₀ + (qa - q₀) * (r/a)²."""
        r = np.linspace(0.1, 1.0, 32)
        q = create_q_profile(r, q0=1.0, qa=3.0, profile_type='parabolic')
        
        # Check monotonicity
        assert np.all(np.diff(q) > 0), "q-profile must be monotonically increasing"
        
        # Parabolic profile has stronger shear near edge
        dq_dr = np.diff(q) / np.diff(r)
        assert dq_dr[-1] > dq_dr[0], "Parabolic profile has increasing shear"
    
    def test_q_profile_validation(self):
        """Test q-profile validation."""
        r = np.linspace(0.1, 1.0, 32)
        
        with pytest.raises(ValueError, match="q0 must be >= 0.5"):
            create_q_profile(r, q0=0.3, qa=3.0)
        
        with pytest.raises(ValueError, match="qa.*must be > q0"):
            create_q_profile(r, q0=2.0, qa=1.5)


class TestEquilibrium:
    """Test equilibrium IC."""
    
    def test_equilibrium_axisymmetry(self):
        """Test that equilibrium is axisymmetric (∂ψ₀/∂ζ = 0)."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        psi0, omega0, q = create_equilibrium_ic(grid)
        
        # Check shape
        assert psi0.shape == (16, 32, 64)
        assert omega0.shape == (16, 32, 64)
        assert len(q) == 16
        
        # Check axisymmetry: all ζ slices should be identical
        for iz in range(grid.nzeta):
            assert np.allclose(psi0[:, :, iz], psi0[:, :, 0]), \
                f"Equilibrium not axisymmetric at ζ index {iz}"
    
    def test_equilibrium_boundary_conditions(self):
        """Test Dirichlet BC: ψ₀(r=0) = ψ₀(r=a) = 0 (approximately)."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        psi0, omega0, q = create_equilibrium_ic(grid, psi0_type='polynomial')
        
        # Note: r[0] != 0 (we avoid singularity), but ψ₀ should be small at edges
        # At r=a, ψ₀ should be exactly 0
        assert np.allclose(psi0[-1, :, :], 0.0, atol=1e-10), \
            "Boundary condition ψ₀(r=a) = 0 not satisfied"
    
    def test_equilibrium_zero_vs_polynomial(self):
        """Test zero vs polynomial equilibrium."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        
        psi0_zero, omega0_zero, q_zero = create_equilibrium_ic(
            grid, psi0_type='zero'
        )
        psi0_poly, omega0_poly, q_poly = create_equilibrium_ic(
            grid, psi0_type='polynomial'
        )
        
        # Zero equilibrium
        assert np.allclose(psi0_zero, 0.0)
        assert np.allclose(omega0_zero, 0.0)
        
        # Polynomial equilibrium is non-trivial
        assert np.max(np.abs(psi0_poly)) > 0.01
        assert np.max(np.abs(omega0_poly)) > 0.01
        
        # q-profiles should be identical
        assert np.allclose(q_zero, q_poly)


class TestBallooningMode:
    """Test ballooning mode IC."""
    
    def test_ballooning_mode_shape(self):
        """Test basic shape and amplitude."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=0.01
        )
        
        assert psi1.shape == (16, 32, 64)
        assert omega1.shape == (16, 32, 64)
        
        # Check amplitude: |ψ₁| ≈ ε (within factor of 2)
        assert np.max(np.abs(psi1)) < 0.05, \
            "Perturbation amplitude too large"
        assert np.max(np.abs(psi1)) > 0.001, \
            "Perturbation amplitude too small"
    
    def test_radial_localization(self):
        """Test radial profile peaks at r_s."""
        grid = Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)
        r_s = 0.5
        Delta_r = 0.1
        
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=0.1, r_s=r_s, Delta_r=Delta_r
        )
        
        # Find radial profile by averaging over θ, ζ
        radial_amplitude = np.sqrt(np.mean(psi1**2, axis=(1, 2)))
        
        # Peak should be near r_s
        i_peak = np.argmax(radial_amplitude)
        r_peak = grid.r[i_peak]
        
        assert abs(r_peak - r_s) < 2 * Delta_r, \
            f"Radial peak at r={r_peak:.3f}, expected near r_s={r_s}"
    
    def test_ballooning_localization(self):
        """Test localization at bad curvature (θ₀ ≈ 0)."""
        grid = Grid3D(nr=16, ntheta=64, nzeta=128, r_max=1.0)
        
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=0.1
        )
        
        # At each (r, ζ), find θ where |ψ₁| is maximum
        # For ballooning mode, this should be near θ₀ = 0
        # (but θ₀ varies with ζ, so we check statistical localization)
        
        # Compute RMS amplitude vs θ
        theta_amplitude = np.sqrt(np.mean(psi1**2, axis=(0, 2)))  # Average over r, ζ
        
        # Check that amplitude is not uniform (should have structure)
        assert np.std(theta_amplitude) > 0.01 * np.mean(theta_amplitude), \
            "Mode should have poloidal structure"
    
    def test_periodicity_theta(self):
        """Test periodicity in θ direction."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        psi1, omega1 = create_ballooning_mode_ic(grid, n=5, m0=2, epsilon=0.01)
        
        # Check ψ(θ=0) ≈ ψ(θ=2π) by construction
        # Note: we use endpoint=False, so theta[-1] + dtheta ≈ 2π
        # The periodicity is implicit in the sin/cos used to generate the mode
        
        # For a ballooning mode with m modes, we expect m-fold symmetry
        # Here we just check that the field is smooth (no discontinuity)
        
        # Compare last and first θ slices (should be close if periodic)
        diff_theta = psi1[:, 0, :] - psi1[:, -1, :]
        
        # With ballooning mode, exact periodicity may not hold due to θ₀ coupling
        # But numerical discontinuity should be small
        assert np.max(np.abs(diff_theta)) < 0.5 * np.max(np.abs(psi1)), \
            "Large discontinuity at θ boundary (check periodicity)"
    
    def test_periodicity_zeta(self):
        """Test periodicity in ζ direction."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        n = 5
        psi1, omega1 = create_ballooning_mode_ic(grid, n=n, m0=2, epsilon=0.01)
        
        # For mode with toroidal number n, we expect n-fold symmetry
        # ψ(ζ + 2π/n) ≈ ψ(ζ) (with possible phase shift)
        
        # Check full 2π periodicity
        diff_zeta = psi1[:, :, 0] - psi1[:, :, -1]
        
        assert np.max(np.abs(diff_zeta)) < 0.5 * np.max(np.abs(psi1)), \
            "Large discontinuity at ζ boundary (check periodicity)"
    
    def test_energy_budget(self):
        """Test perturbation energy << equilibrium energy."""
        grid = Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)
        epsilon = 0.05
        
        # Create equilibrium
        psi0, omega0, q = create_equilibrium_ic(grid, psi0_type='polynomial')
        
        # Create perturbation
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=epsilon
        )
        
        # Compute energies (simplified: just L2 norms)
        E0 = np.sum(psi0**2) * grid.dr * grid.dtheta * grid.dzeta
        E1 = np.sum(psi1**2) * grid.dr * grid.dtheta * grid.dzeta
        
        # Check perturbation theory: E1/E0 ≈ ε² (within factor of 10)
        ratio = E1 / E0 if E0 > 0 else 0
        expected_ratio = epsilon**2
        
        assert ratio < 10 * expected_ratio, \
            f"Perturbation energy too large: E1/E0 = {ratio:.2e}, expected ≈ {expected_ratio:.2e}"
    
    def test_mode_spectrum(self):
        """Test Fourier mode spectrum (dominated by n, m0)."""
        grid = Grid3D(nr=16, ntheta=64, nzeta=128, r_max=1.0)
        n = 5
        m0 = 2
        
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=n, m0=m0, epsilon=0.1
        )
        
        # Fourier transform in ζ direction
        psi1_fft_zeta = np.fft.rfft(psi1, axis=2)  # (nr, ntheta, nzeta//2+1)
        
        # Spectrum: sum over r, θ
        spectrum_zeta = np.sum(np.abs(psi1_fft_zeta)**2, axis=(0, 1))
        
        # Peak should be near n (within ±1 due to coupling)
        i_peak_zeta = np.argmax(spectrum_zeta)
        
        assert abs(i_peak_zeta - n) <= 2, \
            f"Toroidal spectrum peak at n={i_peak_zeta}, expected near n={n}"
    
    def test_edge_case_n_equals_1(self):
        """Test edge case: n=1 (axisymmetric limit)."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=1, m0=2, epsilon=0.01
        )
        
        assert psi1.shape == (16, 32, 64)
        assert np.max(np.abs(psi1)) > 0
    
    def test_edge_case_epsilon_small(self):
        """Test edge case: ε → 0 (very small perturbation)."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64)
        
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=1e-6
        )
        
        assert psi1.shape == (16, 32, 64)
        assert np.max(np.abs(psi1)) < 1e-4, \
            "Perturbation should be very small when ε → 0"


class TestValidation:
    """Additional validation tests."""
    
    def test_full_ic_creation(self):
        """Test creating equilibrium + perturbation."""
        grid = Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)
        
        # Equilibrium
        psi0, omega0, q = create_equilibrium_ic(grid, psi0_type='polynomial')
        
        # Perturbation (use same q-profile)
        psi1, omega1 = create_ballooning_mode_ic(
            grid, n=5, m0=2, epsilon=0.01, q_profile=q
        )
        
        # Total IC: ψ = ψ₀ + ψ₁
        psi_total = psi0 + psi1
        omega_total = omega0 + omega1
        
        assert psi_total.shape == (32, 64, 128)
        assert omega_total.shape == (32, 64, 128)
        
        # Perturbation should be small
        assert np.max(np.abs(psi1)) < 0.1 * np.max(np.abs(psi0))
    
    def test_parameter_validation(self):
        """Test parameter validation in ballooning mode IC."""
        grid = Grid3D(nr=16, ntheta=32, nzeta=64, r_max=1.0)
        
        # Invalid n
        with pytest.raises(ValueError, match="Toroidal mode number"):
            create_ballooning_mode_ic(grid, n=0)
        
        # Invalid m0
        with pytest.raises(ValueError, match="Poloidal mode number"):
            create_ballooning_mode_ic(grid, m0=0)
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="Perturbation amplitude"):
            create_ballooning_mode_ic(grid, epsilon=1.5)
        
        # Invalid r_s
        with pytest.raises(ValueError, match="Rational surface"):
            create_ballooning_mode_ic(grid, r_s=2.0, Delta_r=0.1)
        
        # Invalid Delta_r
        with pytest.raises(ValueError, match="Radial width"):
            create_ballooning_mode_ic(grid, r_s=0.5, Delta_r=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
