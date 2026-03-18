"""
Action Space for MHD RL Environment

Handles action processing and parameter modulation.

v1.1: Parameter modulation (eta, nu multipliers)
v1.2: Spatial current drive (J_ext, heating)

Author: 小P ⚛️
Created: 2026-03-18 (Phase 3 Step 3.2)
"""

import numpy as np
from typing import Tuple, Dict, Any
import gymnasium as gym


class MHDAction:
    """
    Action handler for MHD control.
    
    v1.1: Parameter modulation (2D continuous)
    - eta_multiplier: Resistivity modulation [0.5, 2.0]
    - nu_multiplier: Viscosity modulation [0.5, 2.0]
    
    Purpose: Framework validation (NOT realistic control)
    v1.2 will add spatial current drive.
    
    Parameters
    ----------
    eta_base : float
        Base resistivity
    nu_base : float
        Base viscosity
    action_bounds : tuple
        (low, high) for multipliers (default: (0.5, 2.0))
        
    Notes
    -----
    v1.1 Limitation:
    - Parameter modulation is NOT physical actuator
    - Cannot transfer learned policy to v1.2
    - Use only for framework validation
    
    Physics:
    - Increase eta → faster resistive diffusion
    - Decrease eta → slower diffusion (more current)
    - Similar for nu (viscosity)
    """
    
    def __init__(
        self,
        eta_base: float,
        nu_base: float,
        action_bounds: Tuple[float, float] = (0.5, 2.0)
    ):
        self.eta_base = eta_base
        self.nu_base = nu_base
        self.action_bounds = action_bounds
        
        # Validate inputs
        if eta_base <= 0:
            raise ValueError(f"eta_base must be > 0, got {eta_base}")
        if nu_base <= 0:
            raise ValueError(f"nu_base must be > 0, got {nu_base}")
        if action_bounds[0] >= action_bounds[1]:
            raise ValueError(f"Invalid bounds: {action_bounds}")
    
    def apply(self, action: np.ndarray) -> Tuple[float, float]:
        """
        Apply action to get effective parameters.
        
        Parameters
        ----------
        action : np.ndarray (2,)
            [eta_multiplier, nu_multiplier]
            
        Returns
        -------
        eta_effective : float
            Modulated resistivity
        nu_effective : float
            Modulated viscosity
            
        Examples
        --------
        >>> handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        >>> action = np.array([1.5, 0.8])
        >>> eta_eff, nu_eff = handler.apply(action)
        >>> eta_eff  # 1.5 * 1e-5 = 1.5e-5
        >>> nu_eff   # 0.8 * 1e-4 = 8e-5
        """
        # Clip action to valid range
        action_clipped = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        # Apply modulation
        eta_effective = self.eta_base * action_clipped[0]
        nu_effective = self.nu_base * action_clipped[1]
        
        return eta_effective, nu_effective
    
    def get_action_space(self) -> gym.spaces.Box:
        """
        Get Gymnasium action space.
        
        Returns
        -------
        action_space : gym.spaces.Box
            Continuous 2D action space
        """
        low = np.array([self.action_bounds[0], self.action_bounds[0]], dtype=np.float32)
        high = np.array([self.action_bounds[1], self.action_bounds[1]], dtype=np.float32)
        
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def get_default_action(self) -> np.ndarray:
        """
        Get default action (no modulation).
        
        Returns
        -------
        action : np.ndarray (2,)
            [1.0, 1.0] (identity multipliers)
        """
        return np.array([1.0, 1.0], dtype=np.float32)
    
    def normalize_from_unit(self, action_unit: np.ndarray) -> np.ndarray:
        """
        Normalize action from [-1, 1] to [low, high].
        
        Useful for RL algorithms that output [-1, 1].
        
        Parameters
        ----------
        action_unit : np.ndarray (2,)
            Action in [-1, 1] range
            
        Returns
        -------
        action_scaled : np.ndarray (2,)
            Action in [low, high] range
            
        Examples
        --------
        >>> handler = MHDAction(1e-5, 1e-4, bounds=(0.5, 2.0))
        >>> action_unit = np.array([-1.0, 1.0])  # Full range
        >>> action_scaled = handler.normalize_from_unit(action_unit)
        >>> action_scaled  # [0.5, 2.0]
        """
        low, high = self.action_bounds
        
        # Map [-1, 1] → [low, high]
        # Formula: (x + 1) / 2 * (high - low) + low
        action_scaled = (action_unit + 1.0) / 2.0 * (high - low) + low
        
        return action_scaled.astype(np.float32)
    
    def get_action_info(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Get human-readable action information.
        
        Parameters
        ----------
        action : np.ndarray (2,)
            [eta_multiplier, nu_multiplier]
            
        Returns
        -------
        info : dict
            Action interpretation
        """
        eta_eff, nu_eff = self.apply(action)
        
        info = {
            'eta_multiplier': action[0],
            'nu_multiplier': action[1],
            'eta_effective': eta_eff,
            'nu_effective': nu_eff,
            'eta_change': (action[0] - 1.0) * 100,  # Percentage change
            'nu_change': (action[1] - 1.0) * 100,
        }
        
        return info


class SpatialCurrentDrive:
    """
    Spatial current drive action (v1.2+).
    
    NOT IMPLEMENTED in v1.1 (solver limitation).
    Placeholder for future development.
    
    Physics:
    ∂ψ/∂t = [ψ,φ] - η*J + J_ext(r,θ)
    
    Action:
    - J_ext: Spatial current profile (nr × ntheta)
    - Or: Parametric (Gaussian basis, etc.)
    """
    
    def __init__(self, grid, n_basis: int = 8):
        """
        Parameters
        ----------
        grid : ToroidalGrid
            Grid object
        n_basis : int
            Number of radial basis functions
        """
        raise NotImplementedError(
            "SpatialCurrentDrive deferred to v1.2 "
            "(requires solver extension)"
        )


# Convenience functions

def create_action_handler(
    eta: float = 1e-5,
    nu: float = 1e-4,
    action_type: str = "parameter_modulation"
) -> MHDAction:
    """
    Factory function to create action handler.
    
    Parameters
    ----------
    eta : float
        Base resistivity (default: 1e-5)
    nu : float
        Base viscosity (default: 1e-4)
    action_type : str
        'parameter_modulation' (v1.1) or 'spatial_current' (v1.2)
        
    Returns
    -------
    handler : MHDAction
        Action handler instance
    """
    if action_type == "parameter_modulation":
        return MHDAction(eta, nu)
    elif action_type == "spatial_current":
        raise NotImplementedError("spatial_current deferred to v1.2")
    else:
        raise ValueError(f"Unknown action_type: {action_type}")


def get_action_space_v1_1() -> gym.spaces.Box:
    """
    Get v1.1 action space (parameter modulation).
    
    Returns
    -------
    action_space : gym.spaces.Box
        2D continuous space [0.5, 2.0]²
    """
    return gym.spaces.Box(
        low=np.array([0.5, 0.5], dtype=np.float32),
        high=np.array([2.0, 2.0], dtype=np.float32),
        dtype=np.float32
    )
