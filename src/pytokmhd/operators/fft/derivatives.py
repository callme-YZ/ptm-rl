"""
Toroidal derivatives via FFT.

Implements ∂/∂ζ and ∂²/∂ζ² for 3D MHD fields.

Algorithm (from learning notes 2.2-bout-fft-tricks.md):
1. Forward FFT (normalized 1/N)
2. Multiply by i*k (first derivative) or -k² (second derivative)
3. Inverse FFT (no normalization)

References:
- BOUT++ src/mesh/difops.cxx: DDZ, D2DZ2
- Learning notes: 2.1-fft-dealiasing.md (spectral accuracy)
"""

import numpy as np
from typing import Union, Optional
from .transforms import forward_fft, inverse_fft, fft_frequencies


def toroidal_derivative(
    data: np.ndarray,
    dζ: float,
    order: int = 1,
    axis: int = 2
) -> np.ndarray:
    """
    Compute ∂^n/∂ζ^n via FFT (spectral method).
    
    Args:
        data: Input array (nr, nθ, nζ) or any shape with ζ along axis
        dζ: Grid spacing in ζ direction
        order: Derivative order (1 or 2)
        axis: ζ axis (default: 2 for standard 3D MHD arrays)
    
    Returns:
        Derivative array (same shape as input)
    
    Raises:
        ValueError: If order not in {1, 2}
    
    Notes:
        - Spectral accuracy: error ~ machine precision for smooth functions
        - Assumes periodic boundary conditions in ζ
        - Handles multi-dimensional arrays (broadcasts over other axes)
    
    Examples:
        >>> # Test on sin(kζ)
        >>> ζ = np.linspace(0, 2*np.pi, 64, endpoint=False)
        >>> k = 3.0
        >>> f = np.sin(k * ζ)
        >>> df_dζ = toroidal_derivative(f, dζ=2*np.pi/64, order=1, axis=0)
        >>> df_exact = k * np.cos(k * ζ)
        >>> error = np.max(np.abs(df_dζ - df_exact))
        >>> assert error < 1e-10
    """
    if order not in {1, 2}:
        raise ValueError(f"Derivative order {order} not supported. Use 1 or 2.")
    
    nζ = data.shape[axis]
    
    # Step 1: Forward FFT (BOUT++ convention: normalized 1/N)
    data_hat = forward_fft(data, axis=axis)
    
    # Step 2: Frequency array
    Lζ = nζ * dζ  # Domain length
    k = fft_frequencies(nζ, domain_length=Lζ)
    
    # Broadcast k to match data_hat shape
    k_shape = [1] * data.ndim
    k_shape[axis] = len(k)
    k_broadcast = k.reshape(k_shape)
    
    # Step 3: Derivative in frequency domain
    if order == 1:
        # ∂/∂ζ → multiply by i*k
        deriv_hat = 1j * k_broadcast * data_hat
    elif order == 2:
        # ∂²/∂ζ² → multiply by -k²
        deriv_hat = -(k_broadcast**2) * data_hat
    
    # Step 4: Inverse FFT (BOUT++ convention: no normalization)
    deriv = inverse_fft(deriv_hat, n=nζ, axis=axis)
    
    # Real part only (imaginary should be ~0 for real input)
    return deriv.real


def toroidal_laplacian(
    data: np.ndarray,
    dζ: float,
    axis: int = 2
) -> np.ndarray:
    """
    Compute ∂²/∂ζ² via FFT.
    
    Equivalent to toroidal_derivative(..., order=2) but clearer naming.
    
    Args:
        data: Input array
        dζ: Grid spacing in ζ
        axis: ζ axis
    
    Returns:
        Second derivative ∂²/∂ζ²
    
    Examples:
        >>> # Laplacian of cos(kζ) = -k² cos(kζ)
        >>> ζ = np.linspace(0, 2*np.pi, 128, endpoint=False)
        >>> k = 2.0
        >>> f = np.cos(k * ζ)
        >>> d2f = toroidal_laplacian(f, dζ=2*np.pi/128, axis=0)
        >>> d2f_exact = -k**2 * np.cos(k * ζ)
        >>> error = np.max(np.abs(d2f - d2f_exact))
        >>> assert error < 1e-12
    """
    return toroidal_derivative(data, dζ, order=2, axis=axis)


def toroidal_derivative_field3d(
    field,  # Field3D type (not imported to avoid circular dependency)
    order: int = 1
) -> 'Field3D':
    """
    Compute toroidal derivative for Field3D objects.
    
    Args:
        field: Field3D instance with .data and .grid attributes
        order: Derivative order (1 or 2)
    
    Returns:
        New Field3D with derivative
    
    Notes:
        - Automatically extracts dζ from field.grid
        - Preserves grid and metadata
        - Name updated to reflect derivative
    
    Examples:
        >>> psi = Field3D(data=..., grid=grid, name="psi")
        >>> dpsi_dζ = toroidal_derivative_field3d(psi, order=1)
        >>> dpsi_dζ.name
        "dpsi/dζ"
    """
    # Extract grid spacing
    dζ = field.grid.dζ
    
    # Compute derivative on raw data
    deriv_data = toroidal_derivative(field.data, dζ, order=order, axis=2)
    
    # Create new Field3D (avoid circular import)
    # Assumes Field3D(data, grid, name) constructor
    from ...core.field3d import Field3D
    
    # Generate derivative name
    if order == 1:
        name = f"d{field.name}/dζ"
    elif order == 2:
        name = f"d²{field.name}/dζ²"
    else:
        name = f"d^{order}{field.name}/dζ^{order}"
    
    return Field3D(data=deriv_data, grid=field.grid, name=name)


def verify_spectral_accuracy(
    func_exact: callable,
    deriv_exact: callable,
    nζ: int = 128,
    Lζ: float = 2*np.pi,
    order: int = 1,
    atol: float = 1e-10
) -> dict:
    """
    Verify spectral accuracy of FFT derivative on analytical function.
    
    Args:
        func_exact: Function f(ζ) → exact values
        deriv_exact: Function df^n/dζ^n → exact derivative
        nζ: Number of grid points
        Lζ: Domain length
        order: Derivative order
        atol: Absolute tolerance
    
    Returns:
        Dictionary with:
            - 'error': Max absolute error
            - 'passed': bool (error < atol)
            - 'ζ': Grid points
            - 'numerical': Numerical derivative
            - 'exact': Exact derivative
    
    Examples:
        >>> # Test ∂sin(kζ)/∂ζ = k cos(kζ)
        >>> k = 3.0
        >>> result = verify_spectral_accuracy(
        ...     func_exact=lambda ζ: np.sin(k*ζ),
        ...     deriv_exact=lambda ζ: k*np.cos(k*ζ),
        ...     order=1
        ... )
        >>> result['passed']
        True
    """
    # Grid
    ζ = np.linspace(0, Lζ, nζ, endpoint=False)
    dζ = Lζ / nζ
    
    # Function values
    f = func_exact(ζ)
    df_exact = deriv_exact(ζ)
    
    # Numerical derivative
    df_numerical = toroidal_derivative(f, dζ, order=order, axis=0)
    
    # Error
    error = np.max(np.abs(df_numerical - df_exact))
    passed = error < atol
    
    return {
        'error': error,
        'passed': passed,
        'ζ': ζ,
        'numerical': df_numerical,
        'exact': df_exact,
    }
