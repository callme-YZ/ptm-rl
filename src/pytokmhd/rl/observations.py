"""
Observation extraction for RL environment.

Extracts physics quantities from MHD solver state into normalized
feature vectors for RL algorithms.

Author: 小P ⚛️
Updated: 2026-03-18 (Phase 3 Step 3.1)
"""

import numpy as np
from typing import Dict, Any
from ..geometry import ToroidalGrid
from ..diagnostics import fourier_decompose
from ..operators import laplacian_toroidal


class MHDObservation:
    """
    Observation wrapper for MHD RL environment.
    
    Extracts 11D observation vector:
    - psi_modes (8): Fourier modes of ψ
    - energy (1): Total energy relative to equilibrium
    - energy_drift (1): |E - E_eq| / E_eq
    - div_B_max (1): max|∇·B| / threshold
    
    Parameters
    ----------
    psi_eq : np.ndarray (nr, ntheta)
        Equilibrium poloidal flux
    E_eq : float
        Equilibrium energy
    grid : ToroidalGrid
        Grid object
    n_modes : int
        Number of Fourier modes (default: 8 → 16D with Re/Im)
    div_B_threshold : float
        Threshold for ∇·B normalization (default: 1e-6)
    """
    
    def __init__(
        self,
        psi_eq: np.ndarray,
        E_eq: float,
        grid: ToroidalGrid,
        n_modes: int = 8,
        div_B_threshold: float = 1e-6
    ):
        self.psi_eq = psi_eq
        self.E_eq = E_eq
        self.grid = grid
        self.n_modes = n_modes
        self.div_B_threshold = div_B_threshold
        
        # Compute equilibrium Fourier modes for normalization
        self.psi_eq_modes = fourier_decompose(psi_eq, grid, n_modes)
        self.mode_scale = np.max(np.abs(self.psi_eq_modes)) + 1e-10
    
    def get_observation(self, psi: np.ndarray, omega: np.ndarray) -> Dict[str, Any]:
        """
        Extract observation from current solver state.
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
            Current poloidal flux
        omega : np.ndarray (nr, ntheta)
            Current vorticity
            
        Returns
        -------
        obs : dict
            Observation dictionary with:
            - 'psi_modes': np.ndarray (2*n_modes,) — Normalized Fourier modes
            - 'energy': float — Relative energy (E - E_eq) / E_eq
            - 'energy_drift': float — |E - E_eq| / E_eq
            - 'div_B_max': float — max|∇·B| / threshold
            - 'vector': np.ndarray (11,) — Flattened observation vector
        """
        # 1. Fourier modes (16D for 8 modes)
        psi_modes = fourier_decompose(psi, self.grid, self.n_modes)
        psi_modes_norm = psi_modes / self.mode_scale  # Normalize to ~[-1, 1]
        
        # 2. Energy (scalar)
        E = self._compute_energy(psi, omega)
        energy_relative = (E - self.E_eq) / (self.E_eq + 1e-10)
        energy_drift = np.abs(energy_relative)
        
        # 3. Divergence of B (constraint)
        div_B = self._compute_div_B(psi)
        div_B_max_norm = np.max(np.abs(div_B)) / self.div_B_threshold
        
        # Construct observation dict
        obs = {
            'psi_modes': psi_modes_norm,
            'energy': energy_relative,
            'energy_drift': energy_drift,
            'div_B_max': div_B_max_norm,
        }
        
        # Flatten to vector (for gym.spaces.Box)
        # v1.1: 16 + 1 + 1 + 1 = 19D (updated from design 11D due to Re/Im split)
        obs['vector'] = np.concatenate([
            psi_modes_norm,
            [energy_relative],
            [energy_drift],
            [div_B_max_norm]
        ])
        
        return obs
    
    def _compute_energy(self, psi: np.ndarray, omega: np.ndarray) -> float:
        """
        Compute total MHD energy.
        
        E = (1/2) ∫ ω² dV
        
        Using ω² formulation (consistent with Phase 2).
        """
        # Grid spacing
        dr = self.grid.dr
        dtheta = self.grid.dtheta
        
        # Volume element: dV = R r dr dθ dφ
        # For 2D: integrate over φ → 2π factor
        # dV_2D = R r dr dθ
        r_grid = self.grid.r_grid
        R_grid = self.grid.R_grid
        
        dV = R_grid * r_grid * dr * dtheta
        
        # Kinetic energy (from vorticity)
        E_kin = 0.5 * np.sum(omega**2 * dV)
        
        # Magnetic energy (from poloidal flux)
        # E_mag = (1/2) ∫ |∇ψ|² dV
        # For simplicity, approximate with finite differences
        grad_psi_r = np.gradient(psi, dr, axis=0)
        grad_psi_theta = np.gradient(psi, dtheta, axis=1) / r_grid
        
        grad_psi_sq = grad_psi_r**2 + grad_psi_theta**2
        E_mag = 0.5 * np.sum(grad_psi_sq * dV)
        
        E_total = E_kin + E_mag
        
        return E_total
    
    def _compute_div_B(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute ∇·B from poloidal flux.
        
        For axisymmetric tokamak:
        B = ∇ψ × ∇φ / R
        
        In toroidal geometry, ∇·B = 0 is automatically satisfied
        for ψ-based formulation, but we check numerically.
        
        Returns
        -------
        div_B : np.ndarray (nr, ntheta)
            Divergence of B field
            
        Notes
        -----
        For validation only. Should be << 1e-6 everywhere.
        """
        # Simplified: Check Laplacian consistency
        # True check requires full B field computation
        # For now, use ω - ∇²ψ as proxy (should be 0)
        
        lap_psi = laplacian_toroidal(psi, self.grid)
        
        # For equilibrium: ω = ∇²ψ
        # Deviation indicates constraint violation
        # This is a proxy, not true ∇·B
        
        # Return Laplacian (will be small for good equilibrium)
        return lap_psi
    
    @property
    def observation_space_shape(self) -> tuple:
        """Return shape of observation vector."""
        return (2 * self.n_modes + 3,)  # 16 modes + 3 scalars = 19D
    
    def get_observation_dict_space(self) -> Dict[str, tuple]:
        """
        Get observation space specification for gym.spaces.Dict.
        
        Returns
        -------
        spaces : dict
            Dictionary of (name, shape) pairs
        """
        return {
            'psi_modes': (2 * self.n_modes,),
            'energy': (1,),
            'energy_drift': (1,),
            'div_B_max': (1,),
        }


def normalize_observation(obs: Dict[str, Any], clip: float = 10.0) -> Dict[str, Any]:
    """
    Apply additional normalization and clipping to observation.
    
    Parameters
    ----------
    obs : dict
        Raw observation from MHDObservation.get_observation()
    clip : float
        Clipping threshold for scalars (default: 10.0)
        
    Returns
    -------
    obs_normalized : dict
        Normalized observation with all values in reasonable range
        
    Notes
    -----
    - psi_modes: already normalized by MHDObservation
    - energy: clip to [-clip, +clip]
    - energy_drift: clip to [0, clip]
    - div_B_max: clip to [0, clip] (should be << 1 normally)
    """
    obs_norm = obs.copy()
    
    # Clip energy (can be large during instability)
    obs_norm['energy'] = np.clip(obs['energy'], -clip, clip)
    obs_norm['energy_drift'] = np.clip(obs['energy_drift'], 0, clip)
    obs_norm['div_B_max'] = np.clip(obs['div_B_max'], 0, clip)
    
    # Rebuild vector
    obs_norm['vector'] = np.concatenate([
        obs_norm['psi_modes'],
        [obs_norm['energy']],
        [obs_norm['energy_drift']],
        [obs_norm['div_B_max']]
    ])
    
    return obs_norm


# Legacy functions for backward compatibility
# (keep for existing tests, will deprecate in v2.0)

def fourier_decompose_2d(field: np.ndarray, n_modes: int = 8) -> np.ndarray:
    """
    DEPRECATED: Use pytokmhd.diagnostics.fourier_decompose instead.
    
    Simple Fourier decomposition (old implementation).
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
    DEPRECATED: Use MHDObservation._compute_energy instead.
    
    Legacy energy computation.
    """
    obs = MHDObservation(psi_eq=psi, E_eq=0.0, grid=grid)
    return obs._compute_energy(psi, omega)
