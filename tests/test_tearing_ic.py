"""
Tests for Tearing Mode Initial Conditions

Author: 小P ⚛️
Date: 2026-03-24
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pim_rl.physics.v2.tearing_ic import (
    psi_harris_sheet,
    current_harris_sheet,
    psi_tearing_perturbation,
    phi_tearing_perturbation,
    create_tearing_ic,
    get_expected_growth_rate,
    compute_m1_amplitude,
    MODERATE_GROWTH
)


class TestHarrisSheetEquilibrium:
    """Test Harris sheet equilibrium functions."""
    
    def test_psi_shape(self):
        """ψ should match input shape."""
        r = np.linspace(0, 1, 32)[:, None]
        psi = psi_harris_sheet(r)
        assert psi.shape == r.shape
        
    def test_psi_symmetry(self):
        """ψ symmetric around r0."""
        r = np.linspace(0, 1, 100)[:, None]
        r0 = 0.5
        psi = psi_harris_sheet(r, r0=r0)
        
        # Check symmetry
        mid_idx = 50
        left = psi[:mid_idx]
        right = psi[mid_idx:][::-1]
        
        # Should be approximately symmetric
        # (Not exact due to discrete grid)
        assert np.allclose(left, right, rtol=0.1)
        
    def test_current_peaked_at_r0(self):
        """Current should peak at r0."""
        r = np.linspace(0, 1, 100)[:, None]
        r0 = 0.5
        J = current_harris_sheet(r, r0=r0)
        
        # Find peak location
        peak_idx = np.argmax(np.abs(J))
        peak_r = r[peak_idx, 0]
        
        assert np.abs(peak_r - r0) < 0.02  # Within 2% of domain
        
    def test_current_width(self):
        """Current width ~ λ."""
        r = np.linspace(0, 1, 200)[:, None]
        lam = 0.1
        J = current_harris_sheet(r, lam=lam)
        
        # Half-width at half-maximum
        J_max = np.abs(J).max()
        above_half = np.abs(J) > J_max / 2
        width = r[above_half].ptp()
        
        # Should be ~ 2λ (FWHM of sech²)
        assert 0.5 < width / (2*lam) < 2.0  # Rough check


class TestTearingPerturbation:
    """Test tearing mode perturbation."""
    
    def test_perturbation_shape(self):
        """δψ should be (nr, ntheta)."""
        r = np.linspace(0, 1, 32)[:, None]
        theta = np.linspace(0, 2*np.pi, 64)[None, :]
        
        delta_psi = psi_tearing_perturbation(r, theta)
        assert delta_psi.shape == (32, 64)
        
    def test_m1_structure(self):
        """δψ should have m=1 structure (sin θ)."""
        r = np.linspace(0, 1, 32)[:, None]
        theta = np.linspace(0, 2*np.pi, 128, endpoint=False)[None, :]
        
        delta_psi = psi_tearing_perturbation(r, theta, m=1)
        
        # FFT to check mode content
        psi_fft = np.fft.fft(delta_psi, axis=1)
        amplitudes = np.abs(psi_fft).mean(axis=0)
        
        # m=1 should dominate
        m1_amp = amplitudes[1]
        other_amps = np.concatenate([amplitudes[2:10]])
        
        assert m1_amp > 10 * other_amps.max()  # m=1 >> others
        
    def test_phi_phase_shift(self):
        """φ should be π/2 shifted from ψ."""
        r = np.linspace(0, 1, 32)[:, None]
        theta = np.linspace(0, 2*np.pi, 64)[None, :]
        
        delta_psi = psi_tearing_perturbation(r, theta)
        delta_phi = phi_tearing_perturbation(r, theta)
        
        # At resonance, ψ ~ sin(θ), φ ~ cos(θ)
        r_idx = 16  # Middle
        
        psi_profile = delta_psi[r_idx, :]
        phi_profile = delta_phi[r_idx, :]
        
        # Check orthogonality (rough test)
        dot_product = np.sum(psi_profile * phi_profile)
        assert np.abs(dot_product) < 0.1 * np.sum(psi_profile**2)


class TestCombinedIC:
    """Test complete IC generation."""
    
    def test_create_ic_shape(self):
        """IC should match grid shape."""
        psi, phi = create_tearing_ic(nr=32, ntheta=64)
        
        assert psi.shape == (32, 64)
        assert phi.shape == (32, 64)
        
    def test_ic_has_perturbation(self):
        """IC should have nonzero perturbation."""
        psi, phi = create_tearing_ic(nr=32, ntheta=64, eps=0.01)
        
        # Check θ-variation (perturbation)
        psi_np = np.array(psi)
        theta_variance = psi_np.var(axis=1).mean()
        
        assert theta_variance > 1e-6  # Nonzero perturbation
        
    def test_m1_amplitude(self):
        """m=1 amplitude should match ε."""
        eps = 0.01
        psi, phi = create_tearing_ic(nr=32, ntheta=64, eps=eps)
        
        m1_amp = compute_m1_amplitude(np.array(psi))
        
        # Should be ~ ε/2 (Gaussian envelope reduces amplitude)
        assert 0.3 * eps < m1_amp < 1.0 * eps
        
    def test_parameter_sets(self):
        """Test predefined parameter sets."""
        # Moderate growth (default)
        params = {k: v for k, v in MODERATE_GROWTH.items() if k not in ['r0', 'lam', 'B0']}
        psi, phi = create_tearing_ic(nr=32, ntheta=64, **params)
        m1 = compute_m1_amplitude(np.array(psi))
        
        assert 0.003 < m1 < 0.02  # Reasonable amplitude


class TestGrowthRateFormula:
    """Test theoretical growth rate."""
    
    def test_growth_rate_scaling(self):
        """γ should scale as η^0.6 / λ^0.8."""
        eta = 0.05
        lam = 0.1
        
        gamma = get_expected_growth_rate(lam, eta)
        
        # Should be positive
        assert gamma > 0
        
        # Check scaling
        gamma2 = get_expected_growth_rate(lam, 2*eta)
        ratio = gamma2 / gamma
        expected_ratio = (2)**0.6
        
        assert 0.8 * expected_ratio < ratio < 1.2 * expected_ratio
        
    def test_reasonable_magnitude(self):
        """Growth rate should be reasonable."""
        # Default parameters
        gamma = get_expected_growth_rate(lam=0.1, eta=0.05)
        
        # Should be 0.1 - 10 s⁻¹ range
        assert 0.1 < gamma < 10.0


class TestM1Extraction:
    """Test m=1 amplitude extraction."""
    
    def test_extract_pure_m1(self):
        """Should correctly extract pure m=1 mode."""
        nr, ntheta = 32, 64
        r = np.linspace(0, 1, nr)[:, None]
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)[None, :]
        
        # Pure m=1 mode
        amplitude = 0.05
        psi = amplitude * r * (1 - r) * np.sin(theta)
        
        m1 = compute_m1_amplitude(psi)
        
        # Should match amplitude at peak r (r=0.5, gives 0.25)
        # But FFT normalization gives 0.25/2 = 0.125
        expected = amplitude * 0.25 / 2
        assert 0.8 * expected < m1 < 1.2 * expected
        
    def test_zero_for_m0(self):
        """m=1 should be ~0 for m=0 field."""
        nr, ntheta = 32, 64
        r = np.linspace(0, 1, nr)[:, None]
        
        # m=0 (axisymmetric)
        psi = r**2 * (1 - r)
        psi_2d = np.broadcast_to(psi, (nr, ntheta))
        
        m1 = compute_m1_amplitude(psi_2d)
        
        # Should be very small (numerical noise)
        assert m1 < 1e-10


@pytest.mark.slow
class TestIntegration:
    """Integration tests (require full solver, marked slow)."""
    
    def test_ic_works_with_solver(self):
        """IC should work with CompleteMHDSolver."""
        pytest.skip("Requires solver integration - do in Phase 3")
        
    def test_growth_measurement(self):
        """Measure actual growth rate."""
        pytest.skip("Requires time evolution - do in Phase 3")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-m", "not slow"])
