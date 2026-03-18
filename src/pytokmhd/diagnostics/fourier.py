"""
Fourier Decomposition for MHD Fields

Extracts dominant Fourier modes from 2D toroidal fields for RL observations.

Author: 小P ⚛️
Date: 2026-03-18
"""

import numpy as np
from typing import Tuple


def fourier_decompose(psi: np.ndarray, grid, n_modes: int = 8) -> np.ndarray:
    """
    Extract dominant Fourier modes from poloidal flux ψ.
    
    Performs FFT along poloidal angle θ at mid-radius to capture
    the main mode structure (m-numbers).
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux field
    grid : ToroidalGrid
        Grid object (provides geometry)
    n_modes : int
        Number of modes to extract (default: 8)
        
    Returns
    -------
    modes : np.ndarray (2*n_modes,)
        Real and imaginary parts of first n_modes:
        [Re(m=0), Im(m=0), Re(m=1), Im(m=1), ..., Re(m=n_modes-1), Im(m=n_modes-1)]
        
    Notes
    -----
    - Extracts modes at mid-radius (r = a/2) for better signal
    - m=0: Axisymmetric component
    - m=1: Kink mode
    - m=2: Tearing mode (most important)
    - m≥3: Higher harmonics
    
    Physics:
    ψ(r,θ) ≈ Σₘ ψₘ(r) exp(imθ)
    
    For RL observation, we capture radial structure at r=a/2.
    """
    # Input validation
    if psi.shape != (grid.nr, grid.ntheta):
        raise ValueError(f"psi shape {psi.shape} != grid shape ({grid.nr}, {grid.ntheta})")
    
    if n_modes > grid.ntheta // 2:
        raise ValueError(f"n_modes={n_modes} too large for ntheta={grid.ntheta}")
    
    # Extract mid-radius slice
    mid_idx = grid.nr // 2
    psi_mid = psi[mid_idx, :]  # Shape: (ntheta,)
    
    # FFT along θ (periodic direction)
    psi_fft = np.fft.fft(psi_mid)
    
    # Extract first n_modes (m = 0, 1, 2, ..., n_modes-1)
    # FFT output: [DC, pos freqs, Nyquist, neg freqs]
    # We want: m=0,1,2,...,n_modes-1
    modes_complex = psi_fft[:n_modes]
    
    # Normalize by ntheta (FFT normalization)
    modes_complex = modes_complex / grid.ntheta
    
    # Flatten to real vector: [Re(m=0), Im(m=0), Re(m=1), Im(m=1), ...]
    modes_real = np.zeros(2 * n_modes, dtype=np.float64)
    for i, mode in enumerate(modes_complex):
        modes_real[2*i] = mode.real
        modes_real[2*i + 1] = mode.imag
    
    # Note: m=0 should have Im=0 (axisymmetric)
    # but we keep it for generality
    
    return modes_real


def reconstruct_from_modes(modes: np.ndarray, grid, r_idx: int = None) -> np.ndarray:
    """
    Reconstruct ψ(θ) from Fourier modes.
    
    Inverse operation of fourier_decompose for visualization/validation.
    
    Parameters
    ----------
    modes : np.ndarray (2*n_modes,)
        Real/imag parts from fourier_decompose
    grid : ToroidalGrid
        Grid object
    r_idx : int, optional
        Radial index to reconstruct at (default: mid-radius)
        
    Returns
    -------
    psi_reconstructed : np.ndarray (ntheta,)
        Reconstructed ψ(θ) at specified radius
        
    Notes
    -----
    Used for validation that modes capture structure correctly.
    """
    n_modes = len(modes) // 2
    
    if r_idx is None:
        r_idx = grid.nr // 2
    
    # Reconstruct complex modes
    modes_complex = np.zeros(n_modes, dtype=np.complex128)
    for i in range(n_modes):
        modes_complex[i] = modes[2*i] + 1j * modes[2*i + 1]
    
    # Build full FFT array (only positive frequencies for real signal)
    # For real signal: negative freqs are complex conjugates
    psi_fft_full = np.zeros(grid.ntheta, dtype=np.complex128)
    psi_fft_full[:n_modes] = modes_complex * grid.ntheta  # Undo normalization
    
    # Nyquist and negative frequencies (complex conjugate for real signal)
    if grid.ntheta % 2 == 0:
        # Even ntheta: Nyquist freq exists
        for m in range(1, n_modes):
            psi_fft_full[-m] = np.conj(psi_fft_full[m])
    else:
        # Odd ntheta: no Nyquist
        for m in range(1, n_modes):
            psi_fft_full[-m] = np.conj(psi_fft_full[m])
    
    # Inverse FFT
    psi_reconstructed = np.fft.ifft(psi_fft_full).real
    
    return psi_reconstructed


def compute_mode_amplitudes(psi: np.ndarray, grid, n_modes: int = 8) -> np.ndarray:
    """
    Compute mode amplitudes |ψₘ| for each m-number.
    
    Useful for tracking tearing mode growth (m=2 amplitude).
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux field
    grid : ToroidalGrid
        Grid object
    n_modes : int
        Number of modes
        
    Returns
    -------
    amplitudes : np.ndarray (n_modes,)
        |ψₘ| for m = 0, 1, 2, ..., n_modes-1
    """
    modes = fourier_decompose(psi, grid, n_modes)
    
    # Compute |ψₘ| = sqrt(Re²+ Im²) for each mode
    amplitudes = np.zeros(n_modes)
    for m in range(n_modes):
        re = modes[2*m]
        im = modes[2*m + 1]
        amplitudes[m] = np.sqrt(re**2 + im**2)
    
    return amplitudes


def compute_dominant_mode(psi: np.ndarray, grid, n_modes: int = 8) -> Tuple[int, float]:
    """
    Find dominant mode (largest amplitude) excluding m=0.
    
    Useful for identifying tearing mode activity.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux field
    grid : ToroidalGrid
        Grid object
    n_modes : int
        Number of modes to analyze
        
    Returns
    -------
    m_dominant : int
        m-number of dominant mode (1 to n_modes-1)
    amplitude : float
        Amplitude of dominant mode
        
    Notes
    -----
    - Excludes m=0 (axisymmetric background)
    - Typically m=2 for tearing modes
    - m=1 for kink modes
    """
    amplitudes = compute_mode_amplitudes(psi, grid, n_modes)
    
    # Exclude m=0 (axisymmetric)
    amplitudes_nonzero = amplitudes[1:]
    
    # Find max
    m_dominant = np.argmax(amplitudes_nonzero) + 1  # +1 because we excluded m=0
    amplitude = amplitudes_nonzero[m_dominant - 1]
    
    return m_dominant, amplitude
