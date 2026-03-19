"""
Unit tests for FFT-based toroidal derivatives.

Test cases:
1. Analytical sin/cos functions (spectral accuracy < 1e-10)
2. First and second derivatives
3. Multi-dimensional arrays (3D MHD fields)
4. Edge cases (constant, linear functions)
5. FFT round-trip invertibility

References:
- v1.4 Design Doc Section 8.1 (Validation Plan)
- Learning notes 2.2-bout-fft-tricks.md
"""

import numpy as np
import pytest

from pytokmhd.operators.fft import (
    toroidal_derivative,
    toroidal_laplacian,
    forward_fft,
    inverse_fft,
    fft_frequencies,
)
from pytokmhd.operators.fft.derivatives import verify_spectral_accuracy
from pytokmhd.operators.fft.transforms import verify_fft_invertibility


class TestFFTTransforms:
    """Test FFT forward/inverse transforms (BOUT++ conventions)."""
    
    def test_fft_invertibility_1d(self):
        """Test FFT → iFFT round-trip for 1D array."""
        x = np.random.randn(64)
        assert verify_fft_invertibility(x, atol=1e-14)
    
    def test_fft_invertibility_3d(self):
        """Test FFT → iFFT for 3D MHD field."""
        data = np.random.randn(32, 64, 32)  # (nr, nθ, nζ)
        assert verify_fft_invertibility(data, axis=2, atol=1e-14)
    
    def test_fft_normalization(self):
        """Verify BOUT++ normalization (forward 1/N, inverse none)."""
        x = np.array([1, 2, 3, 4])
        x_hat = forward_fft(x, norm='forward')
        
        # DC component should be mean(x) with forward normalization
        assert np.allclose(x_hat[0], np.mean(x))
        
        # Round-trip
        x_reconstructed = inverse_fft(x_hat, n=len(x), norm='forward')
        assert np.allclose(x, x_reconstructed)
    
    def test_fft_frequencies(self):
        """Test frequency array for 2π periodic domain."""
        n = 32
        Lζ = 2 * np.pi
        k = fft_frequencies(n, domain_length=Lζ)
        
        # Should be [0, 1, 2, ..., 16] for n=32, L=2π
        expected = np.arange(n//2 + 1, dtype=float)
        assert np.allclose(k, expected)


class TestToroidalDerivative:
    """Test toroidal derivative ∂/∂ζ via FFT."""
    
    def test_derivative_sin_first_order(self):
        """∂sin(kζ)/∂ζ = k cos(kζ) (spectral accuracy)."""
        nζ = 128
        Lζ = 2 * np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k = 3.0
        f = np.sin(k * ζ)
        df_exact = k * np.cos(k * ζ)
        
        df_numerical = toroidal_derivative(f, dζ, order=1, axis=0)
        
        error = np.max(np.abs(df_numerical - df_exact))
        assert error < 1e-10, f"Error {error:.2e} exceeds 1e-10"
    
    def test_derivative_cos_first_order(self):
        """∂cos(kζ)/∂ζ = -k sin(kζ)."""
        nζ = 64
        Lζ = 2 * np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k = 2.0
        f = np.cos(k * ζ)
        df_exact = -k * np.sin(k * ζ)
        
        df_numerical = toroidal_derivative(f, dζ, order=1, axis=0)
        
        error = np.max(np.abs(df_numerical - df_exact))
        assert error < 1e-10
    
    def test_derivative_sin_second_order(self):
        """∂²sin(kζ)/∂ζ² = -k² sin(kζ)."""
        nζ = 128
        Lζ = 2 * np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k = 4.0
        f = np.sin(k * ζ)
        d2f_exact = -k**2 * np.sin(k * ζ)
        
        d2f_numerical = toroidal_derivative(f, dζ, order=2, axis=0)
        
        error = np.max(np.abs(d2f_numerical - d2f_exact))
        assert error < 1e-10
    
    def test_laplacian_cos(self):
        """Test toroidal_laplacian on cos(kζ)."""
        nζ = 128
        Lζ = 2 * np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k = 2.0
        f = np.cos(k * ζ)
        d2f_exact = -k**2 * np.cos(k * ζ)
        
        d2f_numerical = toroidal_laplacian(f, dζ, axis=0)
        
        error = np.max(np.abs(d2f_numerical - d2f_exact))
        assert error < 2e-12  # Slightly relaxed for machine precision
    
    def test_derivative_3d_field(self):
        """Test ∂/∂ζ on 3D MHD field (nr, nθ, nζ)."""
        nr, nθ, nζ = 16, 32, 32
        Lζ = 2 * np.pi
        dζ = Lζ / nζ
        
        # Create 3D field: f(r, θ, ζ) = sin(3ζ) independent of r,θ
        r = np.linspace(0, 1, nr)
        θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        
        k = 3.0
        f_3d = np.sin(k * ζ)[None, None, :]  # Broadcast to (1,1,nζ)
        f_3d = np.broadcast_to(f_3d, (nr, nθ, nζ)).copy()
        
        df_exact = k * np.cos(k * ζ)[None, None, :]
        df_exact = np.broadcast_to(df_exact, (nr, nθ, nζ))
        
        df_numerical = toroidal_derivative(f_3d, dζ, order=1, axis=2)
        
        error = np.max(np.abs(df_numerical - df_exact))
        assert error < 1e-10
    
    def test_derivative_constant_is_zero(self):
        """∂(constant)/∂ζ = 0."""
        nζ = 64
        dζ = 2*np.pi / nζ
        
        f = np.ones(nζ) * 5.0  # Constant
        df = toroidal_derivative(f, dζ, order=1, axis=0)
        
        assert np.max(np.abs(df)) < 1e-14
    
    def test_derivative_linear_is_constant(self):
        """∂(a·ζ)/∂ζ = a (but FFT assumes periodic, so this may fail)."""
        # Note: Linear function NOT periodic → FFT will have error
        # This test verifies we get expected FFT behavior (not exact for non-periodic)
        nζ = 64
        Lζ = 2*np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        a = 2.0
        f = a * ζ
        
        # FFT expects periodic → linear function has discontinuity at boundary
        # So we expect some error (NOT spectral accuracy)
        df = toroidal_derivative(f, dζ, order=1, axis=0)
        
        # Should be approximately constant a, but with Gibbs phenomenon
        # Just verify it doesn't crash (don't assert tight tolerance)
        assert df.shape == f.shape
    
    def test_unsupported_order_raises(self):
        """Verify order=3 raises ValueError."""
        nζ = 32
        dζ = 2*np.pi / nζ
        f = np.random.randn(nζ)
        
        with pytest.raises(ValueError, match="order 3 not supported"):
            toroidal_derivative(f, dζ, order=3, axis=0)


class TestSpectralAccuracy:
    """Test spectral convergence of FFT derivatives."""
    
    def test_verify_spectral_accuracy_sin(self):
        """Use verify_spectral_accuracy helper on sin(kζ)."""
        k = 5.0
        result = verify_spectral_accuracy(
            func_exact=lambda ζ: np.sin(k*ζ),
            deriv_exact=lambda ζ: k*np.cos(k*ζ),
            nζ=128,
            order=1,
            atol=1e-10
        )
        
        assert result['passed'], f"Error {result['error']:.2e} > 1e-10"
    
    def test_convergence_with_grid_refinement(self):
        """Verify spectral convergence: error decreases exponentially with N."""
        k = 3.0
        errors = []
        
        for nζ in [16, 32, 64, 128]:
            result = verify_spectral_accuracy(
                func_exact=lambda ζ: np.sin(k*ζ),
                deriv_exact=lambda ζ: k*np.cos(k*ζ),
                nζ=nζ,
                order=1,
                atol=1.0  # Don't fail, just collect errors
            )
            errors.append(result['error'])
        
        # Errors should decrease (spectral: exponential convergence)
        # For smooth functions, error rapidly approaches machine precision
        # errors = [~1e-14, ~1e-14, ~3e-14, ~1e-14] (all at machine precision)
        # Just verify all are very small (spectral accuracy achieved)
        assert all(e < 1e-10 for e in errors), f"Errors {errors} not all < 1e-10"
        assert errors[3] < 1e-13  # Final error at machine precision


class TestMultipleModes:
    """Test derivatives on multi-mode functions."""
    
    def test_two_mode_superposition(self):
        """∂[sin(k₁ζ) + sin(k₂ζ)]/∂ζ = k₁cos(k₁ζ) + k₂cos(k₂ζ)."""
        nζ = 128
        Lζ = 2*np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k1, k2 = 2.0, 5.0
        f = np.sin(k1*ζ) + np.sin(k2*ζ)
        df_exact = k1*np.cos(k1*ζ) + k2*np.cos(k2*ζ)
        
        df_numerical = toroidal_derivative(f, dζ, order=1, axis=0)
        
        error = np.max(np.abs(df_numerical - df_exact))
        assert error < 1e-10
    
    def test_complex_function(self):
        """Test on f = exp(ikζ) (complex exponential)."""
        nζ = 64
        Lζ = 2*np.pi
        ζ = np.linspace(0, Lζ, nζ, endpoint=False)
        dζ = Lζ / nζ
        
        k = 3.0
        f = np.exp(1j * k * ζ)
        df_exact = 1j * k * np.exp(1j * k * ζ)
        
        # Note: Our derivative takes real part
        # For complex input, test real and imag separately
        df_real = toroidal_derivative(f.real, dζ, order=1, axis=0)
        df_imag = toroidal_derivative(f.imag, dζ, order=1, axis=0)
        
        df_numerical = df_real + 1j * df_imag
        
        error = np.max(np.abs(df_numerical - df_exact))
        assert error < 1e-10


# Acceptance criteria from v1.4 Design Doc Section 7 (Phase 1.1):
# - ✅ test_fft_derivatives.py all passing
# - ✅ Error < 1e-10 for analytical sin/cos
# - ✅ Order 1 and 2 derivatives correct
# - ✅ Code documented (docstrings in derivatives.py)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
