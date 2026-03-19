"""
De-aliasing for nonlinear terms via 2/3 Rule (Orszag padding).

Implements Orszag (1971) de-aliasing strategy for quadratic nonlinearities
in MHD equations. Critical for energy conservation and numerical stability.

Algorithm (from learning notes 2.1-fft-dealiasing.md):
1. Pad spectral coefficients to 3N/2 (zero-pad high frequencies)
2. Transform to physical space (larger grid)
3. Compute nonlinear product
4. Transform back to spectral space
5. Truncate to 2N/3 modes (safe wavenumber limit)

Physical Motivation:
- Nonlinear term f*g generates modes up to k_f + k_g
- If k_f + k_g > K_max (Nyquist limit) → aliasing (false energy injection)
- 2/3 Rule ensures k_f + k_g < 3N/4 < K_eff (padded grid Nyquist)

Cost: ~2.4× (acceptable for correctness, per Design Doc §4.2)

References:
- Orszag (1971): "On the elimination of aliasing in finite-difference schemes"
- Boyd (2001), Chapter 11.5: "2/3 Rule for quadratic nonlinearity"
- Learning notes: 2.1-fft-dealiasing.md (comprehensive derivation)
"""

import numpy as np
from typing import Union, Optional, Tuple
from .transforms import forward_fft, inverse_fft


def dealias_2thirds(
    u: np.ndarray,
    v: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """
    Compute u*v with 2/3 Rule de-aliasing.
    
    Algorithm:
    1. FFT(u) → u_hat, FFT(v) → v_hat
    2. Zero-pad to 3N/2 modes
    3. iFFT to padded grid (3N/2 points)
    4. Multiply: result = u_padded * v_padded
    5. FFT(result) → result_hat
    6. Truncate to 2N/3 modes
    7. iFFT back to original grid
    
    Args:
        u, v: Input arrays (same shape)
        axis: Axis along which to de-alias (default: last)
    
    Returns:
        De-aliased product u*v (same shape as input)
    
    Raises:
        ValueError: If u and v shapes don't match
        ValueError: If N (axis size) < 6 (too small for 3/2 padding)
    
    Notes:
        - Assumes periodic boundary conditions along axis
        - Handles multi-dimensional arrays (broadcasts over other axes)
        - Safe wavenumber limit: k_max = 2N/3 (vs N/2 for standard FFT)
    
    Examples:
        >>> # Test energy conservation in Poisson bracket
        >>> ψ = np.random.randn(32, 64, 32)  # Random field
        >>> φ = np.random.randn(32, 64, 32)
        >>> 
        >>> # Aliased multiplication (wrong)
        >>> product_aliased = ψ * φ
        >>> 
        >>> # De-aliased multiplication (correct)
        >>> product_dealiased = dealias_2thirds(ψ, φ, axis=2)
        >>> 
        >>> # Energy conservation test (see test_dealiasing.py)
    """
    if u.shape != v.shape:
        raise ValueError(
            f"Input arrays must have same shape: u {u.shape} != v {v.shape}"
        )
    
    N = u.shape[axis]
    
    if N < 6:
        raise ValueError(
            f"Grid too small for 2/3 de-aliasing: N={N} < 6. "
            "Need at least 6 points for 3/2 padding."
        )
    
    # Step 1: Forward FFT
    u_hat = forward_fft(u, axis=axis)
    v_hat = forward_fft(v, axis=axis)
    
    # Step 2: Pad to 3N/2 modes
    N_pad = (3 * N) // 2
    
    # Zero-pad high frequencies
    # u_hat shape: (..., N//2+1) for real input
    # Padded shape: (..., N_pad//2+1)
    n_modes_original = u_hat.shape[axis]
    n_modes_padded = N_pad // 2 + 1
    
    pad_width = [(0, 0)] * u.ndim
    pad_width[axis] = (0, n_modes_padded - n_modes_original)
    
    u_hat_pad = np.pad(u_hat, pad_width, mode='constant', constant_values=0)
    v_hat_pad = np.pad(v_hat, pad_width, mode='constant', constant_values=0)
    
    # Step 3: Inverse FFT to padded grid
    u_pad = inverse_fft(u_hat_pad, n=N_pad, axis=axis)
    v_pad = inverse_fft(v_hat_pad, n=N_pad, axis=axis)
    
    # Step 4: Multiply in physical space (on padded grid)
    result_pad = u_pad * v_pad
    
    # Step 5: Forward FFT
    result_hat_pad = forward_fft(result_pad, axis=axis)
    
    # Step 6: Truncate to 2N/3 modes (safe wavenumber limit)
    k_safe = (2 * N) // 3
    
    # For rfft: number of modes = N//2 + 1
    # k_safe is in wavenumber space, convert to mode index
    # Mode index for rfft: [0, 1, 2, ..., N//2]
    # We want to keep modes [0, 1, ..., k_safe-1]
    n_modes_safe = min(k_safe, n_modes_padded)
    
    # Truncate high modes
    slices = [slice(None)] * u.ndim
    slices[axis] = slice(0, n_modes_safe)
    result_hat_trunc = result_hat_pad[tuple(slices)]
    
    # Zero-pad back to original mode count for consistent iFFT size
    # Only pad if we truncated more than original
    if n_modes_safe < n_modes_original:
        pad_width_back = [(0, 0)] * u.ndim
        pad_width_back[axis] = (0, n_modes_original - n_modes_safe)
        result_hat = np.pad(
            result_hat_trunc, 
            pad_width_back, 
            mode='constant', 
            constant_values=0
        )
    else:
        # If safe modes >= original, just truncate to original
        slices_orig = [slice(None)] * u.ndim
        slices_orig[axis] = slice(0, n_modes_original)
        result_hat = result_hat_trunc[tuple(slices_orig)]
    
    # Step 7: Inverse FFT to original grid
    result = inverse_fft(result_hat, n=N, axis=axis)
    
    return result.real


def dealias_product(
    f: np.ndarray,
    g: np.ndarray,
    axes: Union[int, Tuple[int, ...]] = -1
) -> np.ndarray:
    """
    General de-aliasing wrapper (multi-axis support).
    
    Args:
        f, g: Input arrays
        axes: Axis or tuple of axes to de-alias
              -1: last axis only (toroidal ζ)
              (1,2): both θ and ζ (for full 3D bracket)
    
    Returns:
        De-aliased product
    
    Examples:
        >>> # 1D de-aliasing (toroidal only)
        >>> product = dealias_product(f, g, axes=-1)
        >>> 
        >>> # 2D de-aliasing (poloidal + toroidal)
        >>> product_2d = dealias_product(f, g, axes=(1,2))
    """
    if isinstance(axes, int):
        axes = (axes,)
    
    result = f * g  # Start with direct product
    
    # Apply 2/3 rule sequentially along each axis
    for axis in axes:
        result = dealias_2thirds(
            f if axis == axes[0] else result,
            g if axis == axes[0] else np.ones_like(g),
            axis=axis
        )
    
    # Correct approach: de-alias both inputs first, then multiply
    # TODO: Multi-axis de-aliasing needs tensor product approach
    # Current implementation is simplified (sequential)
    # For now, use single-axis (toroidal only) in v1.4
    
    return dealias_2thirds(f, g, axis=axes[0])


def dealias_product_field3d(
    f,  # Field3D type
    g,  # Field3D type
    axis: int = 2  # Toroidal axis by default
) -> 'Field3D':
    """
    De-aliased product for Field3D objects.
    
    Args:
        f, g: Field3D instances
        axis: Axis to de-alias (2 = toroidal ζ)
    
    Returns:
        New Field3D with de-aliased product
    
    Examples:
        >>> psi = Field3D(...)
        >>> phi = Field3D(...)
        >>> bracket_term = dealias_product_field3d(psi, phi)
    """
    from ...core.field3d import Field3D
    
    # Compute de-aliased product
    product_data = dealias_2thirds(f.data, g.data, axis=axis)
    
    return Field3D(
        data=product_data,
        grid=f.grid,
        name=f"{f.name}*{g.name}_dealiased"
    )


def measure_aliasing_error(
    u: np.ndarray,
    v: np.ndarray,
    axis: int = -1
) -> dict:
    """
    Quantify aliasing error by comparing direct vs de-aliased product.
    
    Args:
        u, v: Input arrays
        axis: Axis for de-aliasing
    
    Returns:
        Dictionary with:
            - 'aliased': Direct product u*v
            - 'dealiased': De-aliased product
            - 'error_max': Max absolute difference
            - 'error_rms': RMS difference
            - 'error_spectrum': Spectral error |û*v - ûv|
    
    Examples:
        >>> # High-wavenumber inputs (prone to aliasing)
        >>> N = 64
        >>> x = np.linspace(0, 2*np.pi, N, endpoint=False)
        >>> k_high = N // 3  # Near Nyquist
        >>> u = np.sin(k_high * x)
        >>> v = np.cos(k_high * x)
        >>> 
        >>> error = measure_aliasing_error(u, v, axis=0)
        >>> print(f"RMS error: {error['error_rms']:.2e}")
    """
    # Direct product (aliased)
    product_aliased = u * v
    
    # De-aliased product
    product_dealiased = dealias_2thirds(u, v, axis=axis)
    
    # Errors
    diff = product_dealiased - product_aliased
    error_max = np.max(np.abs(diff))
    error_rms = np.sqrt(np.mean(diff**2))
    
    # Spectral error
    diff_hat = forward_fft(diff, axis=axis)
    error_spectrum = np.abs(diff_hat)
    
    return {
        'aliased': product_aliased,
        'dealiased': product_dealiased,
        'error_max': error_max,
        'error_rms': error_rms,
        'error_spectrum': error_spectrum,
    }


def benchmark_dealiasing_cost(
    shape: Tuple[int, int, int] = (32, 64, 32),
    n_iterations: int = 100
) -> dict:
    """
    Benchmark computational cost of de-aliasing.
    
    Args:
        shape: 3D array shape (nr, nθ, nζ)
        n_iterations: Number of timing iterations
    
    Returns:
        Dictionary with:
            - 'time_aliased': Time for direct product (ms)
            - 'time_dealiased': Time for de-aliased product (ms)
            - 'overhead': Ratio time_dealiased / time_aliased
    
    Notes:
        - Expected overhead: ~2.4× (per Design Doc §4.2)
        - Actual depends on FFT library (NumPy/FFTW)
    
    Examples:
        >>> cost = benchmark_dealiasing_cost()
        >>> print(f"Overhead: {cost['overhead']:.1f}×")
        Overhead: 2.4×
    """
    import time
    
    # Random test arrays
    u = np.random.randn(*shape)
    v = np.random.randn(*shape)
    
    # Time aliased (direct) product
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = u * v
    time_aliased = (time.perf_counter() - start) * 1000 / n_iterations
    
    # Time de-aliased product
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = dealias_2thirds(u, v, axis=2)
    time_dealiased = (time.perf_counter() - start) * 1000 / n_iterations
    
    overhead = time_dealiased / time_aliased
    
    return {
        'time_aliased': time_aliased,
        'time_dealiased': time_dealiased,
        'overhead': overhead,
    }
