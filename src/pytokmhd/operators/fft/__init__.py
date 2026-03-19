"""
FFT-based operators for 3D toroidal MHD.

This module provides:
- Toroidal derivatives (∂/∂ζ) via FFT
- De-aliasing for nonlinear terms
- FFT transform utilities (BOUT++ conventions)
"""

from .derivatives import toroidal_derivative, toroidal_laplacian
from .transforms import forward_fft, inverse_fft, fft_frequencies

__all__ = [
    'toroidal_derivative',
    'toroidal_laplacian',
    'forward_fft',
    'inverse_fft',
    'fft_frequencies',
]
