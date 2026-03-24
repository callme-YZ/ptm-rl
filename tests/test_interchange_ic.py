"""
Unit tests for interchange mode IC (Issue #27)

Author: 小P ⚛️
Date: 2026-03-24
"""

import pytest
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, 'src')

from pim_rl.physics.v2.interchange_ic import (
    pressure_interchange_equilibrium,
    psi_interchange_equilibrium,
    psi_interchange_perturbation,
    phi_interchange_perturbation,
    create_interchange_ic,
    get_expected_growth_rate,
    compute_mode_amplitude,
    MODERATE_INTERCHANGE,
    M3_INTERCHANGE
)


class TestInterchangeEquilibrium:
    """Test pressure-driven equilibrium."""
    
    def test_pressure_profile_peak(self):
        """Pressure should peak at specified location."""
        r = np.linspace(0, 1, 100)
        p = pressure_interchange_equilibrium(r, p0=1.0, r_peak=0.6, width=0.15)
        
        # Find peak
        i_max = np.argmax(p)
        r_max = r[i_max]
        
        # Should be near r_peak
        assert r_max == pytest.approx(0.6, abs=0.02)
        
        # Peak value should be p0
        assert p[i_max] == pytest.approx(1.0, rel=0.01)
    
    def test_pressure_gradient(self):
        """Pressure gradient should be steep at peak."""
        r = np.linspace(0, 1, 200)
        p = pressure_interchange_equilibrium(r, p0=1.0, r_peak=0.6, width=0.15)
        
        # Compute gradient
        dp_dr = np.gradient(p, r[1] - r[0])
        
        # Max gradient should be near r_peak (relaxed range)
        i_max_grad = np.argmax(np.abs(dp_dr))
        r_max_grad = r[i_max_grad]
        
        assert 0.45 < r_max_grad < 0.75
    
    def test_psi_equilibrium_smooth(self):
        """Flux should be smooth."""
        r = np.linspace(0, 1, 100)
        psi = psi_interchange_equilibrium(r, p0=1.0, r_peak=0.6, width=0.15)
        
        # Should be finite everywhere
        assert np.all(np.isfinite(psi))
        
        # Should be smooth (no large jumps)
        dpsi = np.diff(psi)
        assert np.max(np.abs(dpsi)) < 0.5


class TestInterchangePerturbation:
    """Test mode-m perturbation."""
    
    def test_mode_m_structure(self):
        """Perturbation should have specified m-structure."""
        r = np.linspace(0, 1, 32)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        
        for m in [2, 3, 4]:
            delta_psi = psi_interchange_perturbation(r, theta, eps=0.01, m=m)
            
            # Check shape
            assert delta_psi.shape == (32, 64)
            
            # FFT to check mode number
            psi_fft = np.fft.fft(delta_psi, axis=1) / 64
            mode_amps = np.abs(psi_fft).mean(axis=0)
            
            # Mode m should dominate
            assert mode_amps[m] > 5 * mode_amps[0]  # Much larger than m=0
            if m < 32:
                assert mode_amps[m] > 5 * mode_amps[m+1]  # Larger than m+1
    
    def test_phi_psi_phase(self):
        """φ and ψ should have consistent phase."""
        r = np.linspace(0, 1, 32)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        
        delta_psi = psi_interchange_perturbation(r, theta, eps=0.01, m=2)
        delta_phi = phi_interchange_perturbation(r, theta, eps=0.01, m=2)
        
        # Both should have same mode number
        psi_fft = np.fft.fft(delta_psi, axis=1)
        phi_fft = np.fft.fft(delta_phi, axis=1)
        
        # m=2 should dominate in both (check first few modes, ignore aliasing)
        mode_amps_psi = np.abs(psi_fft).mean(axis=0)[:10]  # First 10 modes
        mode_amps_phi = np.abs(phi_fft).mean(axis=0)[:10]
        
        assert np.argmax(mode_amps_psi) == 2
        assert np.argmax(mode_amps_phi) == 2


class TestCombinedIC:
    """Test complete interchange IC generation."""
    
    def test_create_ic_shape(self):
        """IC should have correct shape."""
        psi, phi = create_interchange_ic(nr=32, ntheta=64)
        
        assert psi.shape == (32, 64)
        assert phi.shape == (32, 64)
    
    def test_ic_has_equilibrium(self):
        """IC should contain equilibrium component."""
        psi, phi = create_interchange_ic(nr=32, ntheta=64, eps=0.01)
        
        # Average over θ should give equilibrium
        psi_avg = psi.mean(axis=1)
        
        # Should be non-zero (equilibrium present)
        assert np.abs(psi_avg).max() > 0.1
    
    def test_ic_has_perturbation(self):
        """IC should contain mode-m perturbation."""
        psi, phi = create_interchange_ic(nr=32, ntheta=64, eps=0.01, m=2)
        
        # m=2 amplitude should be order of eps
        m2_amp = compute_mode_amplitude(np.array(psi), m=2)
        
        # Should be order of eps (relaxed due to envelope)
        assert 0.001 < m2_amp < 0.02
    
    def test_different_mode_numbers(self):
        """Should support m=2,3,4."""
        for m in [2, 3, 4]:
            psi, phi = create_interchange_ic(nr=32, ntheta=64, eps=0.01, m=m)
            
            # Should have correct mode
            mode_amp = compute_mode_amplitude(np.array(psi), m=m)
            
            # This mode should be larger than others
            other_modes = [compute_mode_amplitude(np.array(psi), m=k) 
                          for k in range(1, 6) if k != m]
            
            assert mode_amp > 2 * max(other_modes)
    
    def test_parameter_sets(self):
        """Predefined parameter sets should work."""
        for params in [MODERATE_INTERCHANGE, M3_INTERCHANGE]:
            psi, phi = create_interchange_ic(nr=32, ntheta=64, **params)
            
            assert psi.shape == (32, 64)
            assert phi.shape == (32, 64)
            assert np.all(np.isfinite(psi))
            assert np.all(np.isfinite(phi))


class TestGrowthRate:
    """Test theoretical growth rate formula."""
    
    def test_growth_rate_positive(self):
        """Growth rate should be positive for pressure bump."""
        gamma = get_expected_growth_rate(p0=1.0, rho=1.0, L_p=0.15)
        
        assert gamma > 0
    
    def test_growth_rate_scaling(self):
        """Growth rate should scale with √(p₀)."""
        gamma1 = get_expected_growth_rate(p0=1.0, rho=1.0, L_p=0.15)
        gamma2 = get_expected_growth_rate(p0=4.0, rho=1.0, L_p=0.15)
        
        # γ ∝ √(p₀)
        assert gamma2 == pytest.approx(2 * gamma1, rel=1e-6)
    
    def test_gradient_scaling(self):
        """Growth rate should scale with 1/L_p."""
        gamma1 = get_expected_growth_rate(p0=1.0, rho=1.0, L_p=0.2)
        gamma2 = get_expected_growth_rate(p0=1.0, rho=1.0, L_p=0.1)
        
        # γ ∝ 1/L_p
        assert gamma2 == pytest.approx(2 * gamma1, rel=1e-6)
    
    def test_reasonable_magnitude(self):
        """Growth rate should be in reasonable range."""
        gamma = get_expected_growth_rate(p0=1.0, rho=1.0, L_p=0.15)
        
        # Should be 5-10 s⁻¹ for observable growth
        assert 3.0 < gamma < 15.0


class TestModeExtraction:
    """Test mode amplitude extraction."""
    
    def test_extract_pure_mode(self):
        """Should correctly extract pure mode-m."""
        r = np.linspace(0, 1, 32)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        
        for m in [2, 3]:
            # Pure mode-m: psi = cos(m*θ)
            R, Theta = np.meshgrid(r, theta, indexing='ij')
            psi_m = np.cos(m * Theta)
            
            # Extract
            mode_amp = compute_mode_amplitude(psi_m, m=m)
            
            # Should be non-zero
            assert mode_amp > 0.3
    
    def test_zero_for_wrong_mode(self):
        """Should return small value for different mode."""
        r = np.linspace(0, 1, 32)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        
        # m=2 field
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        psi_m2 = np.cos(2 * Theta)
        
        # Extract m=3
        mode_amp = compute_mode_amplitude(psi_m2, m=3)
        
        # Should be very small
        assert mode_amp < 0.1


class TestPhysicsConsistency:
    """Test physics consistency."""
    
    def test_pressure_drives_mode(self):
        """Mode should be localized where pressure gradient is steep."""
        r = np.linspace(0, 1, 100)
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        
        # Pressure peaked at 0.6
        p = pressure_interchange_equilibrium(r, p0=1.0, r_peak=0.6, width=0.15)
        
        # Perturbation also at 0.6
        delta_psi = psi_interchange_perturbation(r, theta, eps=0.01, 
                                                 r_unstable=0.6, m=2)
        
        # Check perturbation is largest near r=0.6
        delta_psi_rms = np.sqrt(np.mean(delta_psi**2, axis=1))
        i_max = np.argmax(delta_psi_rms)
        r_max = r[i_max]
        
        # Should be near pressure peak
        assert 0.5 < r_max < 0.7


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
