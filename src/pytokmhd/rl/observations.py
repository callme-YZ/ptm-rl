"""
Observation extraction for RL environment.

Extracts physics quantities from MHD solver state.
"""

import numpy as np
from typing import Dict, Tuple
from ..geometry import ToroidalGrid


def fourier_decompose_2d(field: np.ndarray, n_modes: int = 8) -> np.ndarray:
    """
    Fourier decomposition of 2D field.
    
    Parameters
    ----------
    field : np.ndarray (nr, ntheta)
        2D field to decompose
    n_modes : int
        Number of Fourier modes to extract
    
    Returns
    -------
    modes : np.ndarray (n_modes,)
        Fourier mode amplitudes
    """
    # FFT along theta direction
    fft_theta = np.fft.fft(field, axis=1)
    
    # Extract amplitudes (normalized)
    amplitudes = np.abs(fft_theta[:, :n_modes])
    
    # Average over radial direction
    modes = np.mean(amplitudes, axis=0)
    
    # Normalize to [-1, 1] roughly
    modes = modes / (np.max(modes) + 1e-10)
    
    return modes


def compute_energy(psi: np.ndarray, omega: np.ndarray, grid: ToroidalGrid) -> float:
    """
    Compute total MHD energy.
    
    E = E_magnetic + E_kinetic
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux
    omega : np.ndarray (nr, ntheta)
        Vorticity
    grid : ToroidalGrid
    
    Returns
    -------
    E : float
        Total energy
    """
    from ..operators import laplacian_toroidal
    
    # Magnetic energy: ∝ ∫ |∇ψ|² dV
    # Approximation: use Laplacian
    lap_psi = laplacian_toroidal(psi, grid)
    E_mag = 0.5 * np.sum(lap_psi**2) * grid.dr * grid.dtheta
    
    # Kinetic energy: ∝ ∫ ω² dV
    E_kin = 0.5 * np.sum(omega**2) * grid.dr * grid.dtheta
    
    return E_mag + E_kin


def compute_div_B_max(psi: np.ndarray, grid: ToroidalGrid) -> float:
    """
    Compute max|∇·B|.
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
    grid : ToroidalGrid
    
    Returns
    -------
    div_B_max : float
        Maximum divergence of B
    """
    from ..operators import B_poloidal_from_psi, divergence_toroidal
    
    # Compute B from psi
    B_r, B_theta = B_poloidal_from_psi(psi, grid)
    
    # Compute divergence
    div_B = divergence_toroidal(B_r, B_theta, grid)
    
    return np.max(np.abs(div_B))


def extract_observation(
    psi: np.ndarray,
    omega: np.ndarray,
    grid: ToroidalGrid,
    E_eq: float
) -> Dict[str, np.ndarray]:
    """
    Extract 11D observation from solver state.
    
    Observation components:
    - psi_modes: 8D Fourier modes
    - energy: 1D (E - E_eq) / E_eq
    - energy_drift: 1D |E - E_eq| / E_eq
    - div_B_max: 1D max|∇·B| / 1e-6
    
    Total: 11D
    
    Parameters
    ----------
    psi : np.ndarray (nr, ntheta)
    omega : np.ndarray (nr, ntheta)
    grid : ToroidalGrid
    E_eq : float
        Equilibrium energy
    
    Returns
    -------
    obs : Dict[str, np.ndarray]
        Observation dictionary
    """
    # Fourier modes
    psi_modes = fourier_decompose_2d(psi, n_modes=8)
    
    # Energy
    E = compute_energy(psi, omega, grid)
    energy_rel = (E - E_eq) / (E_eq + 1e-10)
    energy_drift = np.abs(energy_rel)
    
    # Divergence of B
    div_B = compute_div_B_max(psi, grid)
    div_B_normalized = div_B / 1e-6  # Normalize by threshold
    
    # Assemble observation
    obs = {
        'psi_modes': psi_modes,              # (8,)
        'energy': np.array([energy_rel]),    # (1,)
        'energy_drift': np.array([energy_drift]),  # (1,)
        'div_B_max': np.array([div_B_normalized])  # (1,)
    }
    
    return obs


def observation_to_array(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert observation dict to flat array.
    
    Parameters
    ----------
    obs : dict
        Observation dictionary
    
    Returns
    -------
    obs_array : np.ndarray (11,)
        Flattened observation
    """
    return np.concatenate([
        obs['psi_modes'],      # 8D
        obs['energy'],         # 1D
        obs['energy_drift'],   # 1D
        obs['div_B_max']       # 1D
    ])  # Total: 11D
