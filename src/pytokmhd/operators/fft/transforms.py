"""
FFT transform utilities following BOUT++ conventions.

References:
- Learning notes: 2.2-bout-fft-tricks.md
- BOUT++ source: src/fft.cxx

Key conventions (BOUT++ vs NumPy):
- BOUT++: Forward FFT normalized (1/N), Inverse NOT
- NumPy: Forward NOT normalized, Inverse normalized (1/N)
- We follow BOUT++ for easier mode interpretation
"""

import numpy as np
from typing import Optional


def forward_fft(
    data: np.ndarray,
    axis: int = -1,
    norm: str = 'forward'
) -> np.ndarray:
    """
    Forward FFT with BOUT++ normalization (1/N).
    
    Args:
        data: Input array (real)
        axis: FFT axis (default: last dimension)
        norm: 'forward' (BOUT++), 'backward' (NumPy default), 'ortho'
    
    Returns:
        Complex FFT coefficients (half spectrum for real input)
    
    Notes:
        - Uses rfft for real-to-complex transform
        - Output size: N//2 + 1 for N input points
        - Frequency ordering: [0, 1, ..., N//2]
    
    Examples:
        >>> x = np.sin(2*np.pi*np.arange(32)/32)
        >>> x_hat = forward_fft(x)
        >>> x_hat.shape
        (17,)  # 32//2 + 1
    """
    N = data.shape[axis]
    
    # NumPy rfft (no normalization by default)
    x_hat = np.fft.rfft(data, axis=axis)
    
    # Apply BOUT++ convention: divide by N
    if norm == 'forward':
        x_hat = x_hat / N
    elif norm == 'ortho':
        x_hat = x_hat / np.sqrt(N)
    # 'backward' = NumPy default (no scaling)
    
    return x_hat


def inverse_fft(
    data_hat: np.ndarray,
    n: Optional[int] = None,
    axis: int = -1,
    norm: str = 'forward'
) -> np.ndarray:
    """
    Inverse FFT with BOUT++ convention (no normalization).
    
    Args:
        data_hat: Complex FFT coefficients
        n: Output size (if None, inferred from input)
        axis: FFT axis
        norm: 'forward' (BOUT++), 'backward' (NumPy), 'ortho'
    
    Returns:
        Real array (reconstructed from FFT coefficients)
    
    Notes:
        - Inverse of forward_fft with same norm
        - Automatically handles real output (irfft)
    
    Examples:
        >>> x_hat = forward_fft(x)
        >>> x_reconstructed = inverse_fft(x_hat, n=len(x))
        >>> np.allclose(x, x_reconstructed)
        True
    """
    # NumPy irfft (default: normalize by N)
    x = np.fft.irfft(data_hat, n=n, axis=axis)
    
    # BOUT++ convention: no normalization on inverse
    # NumPy already normalized, so multiply back
    if norm == 'forward':
        if n is None:
            n = 2 * (data_hat.shape[axis] - 1)  # Infer from rfft output
        x = x * n
    elif norm == 'ortho':
        if n is None:
            n = 2 * (data_hat.shape[axis] - 1)
        x = x * np.sqrt(n)
    # 'backward' = no additional scaling needed
    
    return x


def fft_frequencies(
    n: int,
    d: float = 1.0,
    domain_length: Optional[float] = None
) -> np.ndarray:
    """
    Frequency array for FFT (BOUT++ convention).
    
    Args:
        n: Number of grid points
        d: Grid spacing (default: 1.0)
        domain_length: Physical domain length (if provided, overrides d)
    
    Returns:
        k: Frequency array [0, 1, ..., n//2] * (2π / L)
    
    Notes:
        - For periodic domain [0, L], use domain_length=L
        - Returns angular frequencies k (rad/length)
        - Compatible with np.fft.rfftfreq but uses 2π convention
    
    Examples:
        >>> k = fft_frequencies(32, domain_length=2*np.pi)
        >>> k
        array([0, 1, 2, ..., 16])  # Mode numbers for 2π domain
    """
    if domain_length is not None:
        d = domain_length / n
    
    # NumPy rfftfreq returns f = [0, 1/N, 2/N, ..., 1/2]
    # We want k = 2π * f / d
    freq = np.fft.rfftfreq(n, d=d)
    k = 2 * np.pi * freq
    
    return k


def verify_fft_invertibility(
    data: np.ndarray,
    axis: int = -1,
    atol: float = 1e-14
) -> bool:
    """
    Verify FFT → iFFT recovers original data.
    
    Args:
        data: Input array
        axis: FFT axis
        atol: Absolute tolerance
    
    Returns:
        True if invertible within tolerance
    
    Raises:
        AssertionError: If round-trip error exceeds tolerance
    
    Examples:
        >>> x = np.random.randn(64)
        >>> verify_fft_invertibility(x)
        True
    """
    data_hat = forward_fft(data, axis=axis)
    data_reconstructed = inverse_fft(data_hat, n=data.shape[axis], axis=axis)
    
    error = np.max(np.abs(data - data_reconstructed))
    
    if error > atol:
        raise AssertionError(
            f"FFT round-trip error {error:.2e} exceeds tolerance {atol:.2e}"
        )
    
    return True
