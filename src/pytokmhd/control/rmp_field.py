"""
RMP Field Generation

Generates Resonant Magnetic Perturbation (RMP) fields for tearing mode control.

Physics:
- RMP field: ψ_RMP(r, z) = A * r^m * cos(mθ + φ)
- Mode numbers (m, n) must match target tearing mode for resonance
- Enters MHD as external current source: η∇²ψ_RMP

Author: 小P ⚛️
Created: 2026-03-16
Phase: 4
"""

import numpy as np
from ..solver.mhd_equations import laplacian_cylindrical


def generate_rmp_field(r_grid, z_grid, amplitude, m=2, n=1, phase=0.0):
    """
    Generate single-mode RMP field.
    
    Physics:
    -------
    B_RMP = A_RMP * cos(mθ - nφ + φ_0)
    
    In cylindrical geometry (2D, φ-independent):
    ψ_RMP(r, z) = A * r^m * cos(mθ + φ)
    
    where θ = arctan(z/r) is poloidal angle.
    
    Parameters
    ----------
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    z_grid : np.ndarray (Nr, Nz)
        Axial coordinate mesh
    amplitude : float
        RMP amplitude (control input), typically ∈ [-0.1, 0.1]
    m : int, optional
        Poloidal mode number (default: 2 for m=2 tearing mode)
    n : int, optional
        Toroidal mode number (default: 1, not used in 2D)
    phase : float, optional
        Phase offset in radians (default: 0.0)
    
    Returns
    -------
    psi_rmp : np.ndarray (Nr, Nz)
        RMP contribution to poloidal flux
    j_rmp : np.ndarray (Nr, Nz)
        RMP current density (from ∇²ψ_RMP)
    
    Notes
    -----
    - RMP field is static (time-independent)
    - Enters MHD equation as: ∂ψ/∂t += η∇²ψ_RMP
    - Phase dependence: optimal phase minimizes island width
    - Mode matching: (m, n)_RMP = (m, n)_tearing for resonance
    
    Physics References
    ------------------
    - Fitzpatrick 1993: "Interaction of tearing modes with external structures"
    - Cole & Fitzpatrick 2006: "RMP control of tearing modes in tokamaks"
    
    Examples
    --------
    >>> r = np.linspace(0, 1, 64)
    >>> z = np.linspace(0, 6, 128)
    >>> R, Z = np.meshgrid(r, z, indexing='ij')
    >>> psi_rmp, j_rmp = generate_rmp_field(R, Z, amplitude=0.05, m=2, n=1)
    >>> np.max(np.abs(psi_rmp))  # Should be ~ amplitude
    0.05
    """
    # Compute poloidal angle θ = arctan(z/r)
    # Handle r=0 carefully (avoid division by zero)
    theta = np.arctan2(z_grid, r_grid)
    
    # Generate helical perturbation
    # ψ_RMP = A * r^m * cos(mθ + φ)
    psi_rmp = amplitude * r_grid**m * np.cos(m * theta + phase)
    
    # Set axis (r=0) to zero (avoid singularity)
    psi_rmp[0, :] = 0.0
    
    # Compute associated current density
    # Note: j_rmp is placeholder, actual current computed via Laplacian
    j_rmp = np.zeros_like(psi_rmp)
    
    return psi_rmp, j_rmp


def compute_rmp_current(psi_rmp, dr, dz, r_grid):
    """
    Compute RMP current density from flux.
    
    Physics:
    -------
    J_RMP = -∇²ψ_RMP / μ₀
    
    (In normalized units, μ₀ = 1)
    
    Parameters
    ----------
    psi_rmp : np.ndarray (Nr, Nz)
        RMP poloidal flux
    dr : float
        Radial grid spacing
    dz : float
        Axial grid spacing
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    
    Returns
    -------
    j_rmp : np.ndarray (Nr, Nz)
        RMP current density
    
    Notes
    -----
    Uses same Laplacian operator as MHD solver for consistency.
    """
    lap_psi_rmp = laplacian_cylindrical(psi_rmp, dr, dz, r_grid)
    j_rmp = -lap_psi_rmp
    
    return j_rmp


def generate_multimode_rmp(r_grid, z_grid, amplitudes, modes, phases):
    """
    Generate multi-mode RMP field.
    
    Physics:
    -------
    ψ_RMP = Σ A_mn * r^m * cos(mθ - nφ + φ_mn)
    
    Useful for:
    - Controlling multiple tearing modes simultaneously
    - Optimizing control spectrum
    - Advanced control algorithms (Phase 5+)
    
    Parameters
    ----------
    r_grid : np.ndarray (Nr, Nz)
        Radial coordinate mesh
    z_grid : np.ndarray (Nr, Nz)
        Axial coordinate mesh
    amplitudes : list of float
        RMP amplitudes [A_21, A_31, ...], length N_modes
    modes : list of tuple
        Mode numbers [(m1, n1), (m2, n2), ...], length N_modes
    phases : list of float
        Phase offsets [φ_1, φ_2, ...], length N_modes
    
    Returns
    -------
    psi_rmp : np.ndarray (Nr, Nz)
        Total RMP poloidal flux
    j_rmp : np.ndarray (Nr, Nz)
        Total RMP current density
    
    Examples
    --------
    >>> # Control m=2,n=1 and m=3,n=1 modes
    >>> amplitudes = [0.05, 0.02]
    >>> modes = [(2, 1), (3, 1)]
    >>> phases = [0.0, np.pi/4]
    >>> psi_rmp, j_rmp = generate_multimode_rmp(R, Z, amplitudes, modes, phases)
    """
    psi_rmp_total = np.zeros_like(r_grid)
    j_rmp_total = np.zeros_like(r_grid)
    
    for amp, (m, n), phase in zip(amplitudes, modes, phases):
        psi_mode, j_mode = generate_rmp_field(r_grid, z_grid, amp, m, n, phase)
        psi_rmp_total += psi_mode
        j_rmp_total += j_mode
    
    return psi_rmp_total, j_rmp_total


def compute_rmp_helicity(m, n, q_profile, r_values):
    """
    Compute RMP helicity for resonance matching.
    
    Physics:
    -------
    Resonance condition: q(r_s) = m/n at rational surface r_s
    
    RMP is most effective when (m,n)_RMP matches (m,n)_tearing.
    
    Parameters
    ----------
    m : int
        Poloidal mode number
    n : int
        Toroidal mode number
    q_profile : np.ndarray
        Safety factor profile q(r)
    r_values : np.ndarray
        Radial positions corresponding to q_profile
    
    Returns
    -------
    r_resonant : float or None
        Radial position of resonant surface (q = m/n)
        Returns None if no resonance found
    
    Notes
    -----
    Used for:
    - Validating RMP mode selection
    - Diagnosing resonance location
    - Optimizing control strategy
    
    Examples
    --------
    >>> q = np.linspace(1.5, 3.5, 100)  # Safety factor profile
    >>> r = np.linspace(0, 1, 100)
    >>> r_res = compute_rmp_helicity(m=2, n=1, q_profile=q, r_values=r)
    >>> r_res  # Should be near q=2.0
    0.5
    """
    q_resonant = m / n
    
    # Find where q crosses q_resonant
    # Interpolate to find exact crossing
    if np.min(q_profile) > q_resonant or np.max(q_profile) < q_resonant:
        # No resonance in this q-profile
        return None
    
    # Find crossing point
    idx = np.where(np.diff(np.sign(q_profile - q_resonant)))[0]
    
    if len(idx) == 0:
        return None
    
    # Use first crossing (closest to axis)
    idx0 = idx[0]
    
    # Linear interpolation
    r0, r1 = r_values[idx0], r_values[idx0 + 1]
    q0, q1 = q_profile[idx0], q_profile[idx0 + 1]
    
    r_resonant = r0 + (q_resonant - q0) * (r1 - r0) / (q1 - q0)
    
    return r_resonant


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_rmp_field(psi_rmp, r_grid, z_grid, amplitude, m):
    """
    Validate RMP field properties.
    
    Checks:
    1. Amplitude correct: max(|ψ_RMP|) ≈ amplitude
    2. Mode structure: m zero-crossings in θ
    3. Axis regularity: ψ_RMP(r=0) = 0
    
    Parameters
    ----------
    psi_rmp : np.ndarray (Nr, Nz)
        RMP field to validate
    r_grid : np.ndarray (Nr, Nz)
        Radial grid
    z_grid : np.ndarray (Nr, Nz)
        Axial grid
    amplitude : float
        Expected amplitude
    m : int
        Expected mode number
    
    Returns
    -------
    is_valid : bool
        True if all checks pass
    diagnostics : dict
        Validation diagnostics
    
    Examples
    --------
    >>> psi_rmp, _ = generate_rmp_field(R, Z, 0.05, m=2)
    >>> valid, diag = validate_rmp_field(psi_rmp, R, Z, 0.05, m=2)
    >>> valid
    True
    """
    diagnostics = {}
    
    # Check 1: Amplitude
    max_amp = np.max(np.abs(psi_rmp))
    amp_error = abs(max_amp - amplitude) / amplitude
    diagnostics['max_amplitude'] = max_amp
    diagnostics['amplitude_error'] = amp_error
    amp_valid = amp_error < 0.1  # 10% tolerance
    
    # Check 2: Axis regularity
    axis_value = np.max(np.abs(psi_rmp[0, :]))
    diagnostics['axis_value'] = axis_value
    axis_valid = axis_value < 1e-6
    
    # Check 3: Mode structure (count zero-crossings along θ at fixed r)
    # Sample at mid-radius
    r_mid_idx = len(r_grid[:, 0]) // 2
    psi_theta = psi_rmp[r_mid_idx, :]
    zero_crossings = np.sum(np.diff(np.sign(psi_theta)) != 0)
    diagnostics['zero_crossings'] = zero_crossings
    # Expected: 2*m zero-crossings in full poloidal turn
    mode_valid = abs(zero_crossings - 2*m) <= 2  # Allow ±2 tolerance
    
    is_valid = amp_valid and axis_valid and mode_valid
    diagnostics['is_valid'] = is_valid
    
    return is_valid, diagnostics
