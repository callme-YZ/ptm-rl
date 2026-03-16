"""
MHD Tearing Mode Control Environment for Reinforcement Learning.

This module provides a Gymnasium-compatible environment for training RL agents
to control tearing modes in tokamak plasmas using RMP (Resonant Magnetic Perturbation).

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
Status: Phase 5 Step 2.5 - Gymnasium Migration + Parameterization
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Literal

# Phase 1-4 imports (available after PyTokMHD integration)
# from pytokmhd.solver import rk4_step_with_rmp, setup_tearing_mode, create_equilibrium_cache
# from pytokmhd.diagnostics import TearingModeMonitor, find_rational_surface


class MHDTearingControlEnv(gym.Env):
    """
    Gymnasium environment for RL-based tearing mode control.
    
    **Configurable Parameters:**
    - equilibrium_type: 'simple' (sin profile) or 'solovev' (真实)
    - grid_size: Spatial resolution (default: 64)
    - action_smoothing: Alpha for exponential smoothing (default: 0.3)
    - max_psi: Early termination threshold (default: 10)
    
    **Observation Space (25D):**
    - w: Island width (primary control target)
    - gamma: Growth rate (stability indicator)
    - x_o, z_o: Island center position
    - psi_samples: 9 magnetic flux samples (3x3 grid)
    - omega_samples: 9 vorticity samples (3x3 grid)
    - energy: Total magnetic energy (normalized)
    - helicity: Magnetic helicity (normalized)
    - previous_action: Last applied RMP amplitude
    
    **Action Space (1D):**
    - RMP amplitude ∈ [-1, 1]
    
    **Reward:**
    - r = -w - 0.1*|gamma| - 0.01*|action|
    
    **Grid Attributes:**
    - nx, ny, nz: Grid size (Cartesian naming)
    - Nr, Nphi, Nz: Grid size (toroidal naming, aliases for nx, ny, nz)
    
    **Physics Parameters:**
    - eta: Resistivity (controls tearing mode growth)
    - nu: Viscosity (controls vorticity dissipation)
    - RMP: External resonant magnetic perturbation
    
    **Termination Conditions:**
    - Episode length >= max_steps (200 default)
    - psi exceeds threshold (CFL violation prevention)
    
    **Known Limitations:**
    - Simplified equilibrium (Phase 5 Step 2)
    - Phase 4 RMP verification incomplete (fast action changes not tested)
    - Action smoothing (alpha=0.3) reduces responsiveness but ensures stability
    
    For details see: PHASE5_STEP2_NUMERICAL_STABILITY_FIX.md
    """
    
    metadata = {'render.modes': []}
    
    def __init__(
        self,
        equilibrium_type: Literal['simple', 'solovev'] = 'simple',
        grid_size: int = 64,
        action_smoothing_alpha: float = 0.3,
        max_psi_threshold: float = 10.0,
        max_steps: int = 200,
        dt: float = 0.01,
        eta: float = 1e-5,
        nu: float = 1e-6,
        # Solovev equilibrium parameters (used when equilibrium_type='solovev')
        R0: float = 1.0,
        a: float = 0.3,
        kappa: float = 1.0,
        delta: float = 0.0,
    ):
        """
        Initialize MHD tearing mode control environment.
        
        Parameters
        ----------
        equilibrium_type : {'simple', 'solovev'}
            Type of equilibrium initialization
            - 'simple': psi = 0.1*sin(z) (fast, for testing)
            - 'solovev': PyTokEq Solovev equilibrium (realistic, requires integration)
        grid_size : int
            Spatial grid resolution (default: 64)
        action_smoothing_alpha : float
            Exponential smoothing factor for RMP actions ∈ [0,1]
            - 0: No smoothing (instantaneous response, may cause instability)
            - 1: Maximum smoothing (very slow response)
            - 0.3 (default): Physical RMP coil inductance approximation
        max_psi_threshold : float
            Early termination threshold for |psi| to prevent CFL violation
        max_steps : int
            Maximum episode length
        dt : float
            Time step for MHD evolution
        eta : float
            Plasma resistivity
        nu : float
            Plasma viscosity
        R0 : float
            Major radius (for Solovev equilibrium)
        a : float
            Minor radius (for Solovev equilibrium)
        kappa : float
            Elongation (for Solovev equilibrium)
        delta : float
            Triangularity (for Solovev equilibrium)
        """
        super().__init__()
        
        # Store configuration
        self.equilibrium_type = equilibrium_type
        self.grid_size = grid_size
        self.alpha_smooth = action_smoothing_alpha
        self.max_psi = max_psi_threshold
        self.max_steps = max_steps
        self.dt = dt
        self.eta = eta
        self.nu = nu
        
        # Solovev parameters
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        
        # Observation space: 25D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )
        
        # Action space: RMP amplitude ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.nx = grid_size
        self.ny = grid_size
        self.nz = grid_size // 2
        
        # Aliases for consistency with Phase 4 naming (Nr, Nz)
        self.Nr = self.nx  # Radial direction
        self.Nphi = self.ny  # Toroidal direction  
        self.Nz = self.nz  # Vertical direction
        
        self.x = np.linspace(0, 2*np.pi, self.nx)
        self.y = np.linspace(0, 2*np.pi, self.ny)
        self.z = np.linspace(0, 2*np.pi, self.nz)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # Runtime state
        self.psi = None
        self.omega = None
        self.step_count = 0
        self.current_action = 0.0
        self.previous_action = 0.0
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        
    def _initialize_fields(self):
        """Initialize magnetic flux and vorticity fields."""
        if self.equilibrium_type == 'simple':
            # Simple sinusoidal profile (fast, for testing)
            self.psi = 0.1 * np.sin(self.z[None, None, :])
            self.psi = np.broadcast_to(self.psi, (self.nx, self.ny, self.nz)).copy()
        
        elif self.equilibrium_type == 'solovev':
            # Realistic Solovev equilibrium via PyTokEq
            import sys
            import os
            
            # Add PyTokEq to path
            pytokeq_path = os.path.join(os.path.dirname(__file__), '..', '..')
            if pytokeq_path not in sys.path:
                sys.path.insert(0, pytokeq_path)
            
            try:
                from pytokeq.equilibrium.profiles.solovev_solution import SolovevSolution
                
                # Create Solovev analytical solution
                eq = SolovevSolution(
                    R0=self.R0,
                    eps=self.a / self.R0,  # Inverse aspect ratio
                    kappa=self.kappa,
                    delta=self.delta,
                    A=0.1  # Shafranov shift parameter
                )
                
                # Generate grid in cylindrical coordinates
                # Convert Cartesian (x,y,z) to (R,Z,phi)
                # For now, approximate R ~ x, Z ~ z (near axis)
                R_grid = self.R0 + (self.x - np.pi) * self.a / np.pi
                Z_grid = (self.z - np.pi/2) * self.a / (np.pi/2)
                
                # Compute psi on 2D grid
                R_2d, Z_2d = np.meshgrid(R_grid, Z_grid, indexing='ij')
                psi_2d = eq.psi(R_2d, Z_2d)
                
                # Extend to 3D (toroidally symmetric)
                self.psi = np.zeros((self.nx, self.ny, self.nz))
                for iy in range(self.ny):
                    self.psi[:, iy, :] = psi_2d
                
                # Normalize to reasonable amplitude
                self.psi = self.psi / (np.max(np.abs(self.psi)) + 1e-10) * 0.1
                
            except ImportError as e:
                raise RuntimeError(
                    f"equilibrium_type='solovev' requires PyTokEq. "
                    f"Import error: {e}"
                )
        else:
            raise ValueError(f"Unknown equilibrium_type: {self.equilibrium_type}")
        
        # Initialize vorticity (zero everywhere)
        self.omega = np.zeros_like(self.psi)
        
        # Add small perturbation to seed tearing mode
        perturbation = 0.01 * np.random.randn(*self.psi.shape)
        self.psi += perturbation
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options
        
        Returns
        -------
        observation : np.ndarray
            Initial observation (25D)
        info : dict
            Diagnostic information
        """
        super().reset(seed=seed)
        
        # Initialize fields
        self._initialize_fields()
        
        # Reset episode counters
        self.step_count = 0
        self.current_action = 0.0
        self.previous_action = 0.0
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Compute initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns
        -------
        obs : np.ndarray, shape (25,)
            [w, gamma, x_o, z_o, psi[9], omega[9], energy, helicity, prev_action]
        """
        # Simplified diagnostics (Phase 5 Step 2)
        # In Step 3, replace with TearingModeMonitor
        
        # Island width (mock for now)
        w = np.std(self.psi)
        
        # Growth rate (mock)
        gamma = 0.0
        
        # Island center (mock)
        x_o = np.pi
        z_o = np.pi / 2
        
        # Sample psi and omega at 3x3 grid (9 samples each for 25D total)
        step = max(1, self.nx // 3)
        psi_samples = self.psi[::step, ::step, 0].flatten()[:9]
        if len(psi_samples) < 9:
            psi_samples = np.pad(psi_samples, (0, 9 - len(psi_samples)))
        
        omega_samples = self.omega[::step, ::step, 0].flatten()[:9]
        if len(omega_samples) < 9:
            omega_samples = np.pad(omega_samples, (0, 9 - len(omega_samples)))
        
        # Normalize energy and helicity
        energy = np.sum(self.psi**2) * self.dx * self.dy * self.dz
        energy_norm = energy / (self.nx * self.ny * self.nz)
        
        helicity = np.sum(self.psi * self.omega) * self.dx * self.dy * self.dz
        helicity_norm = helicity / (self.nx * self.ny * self.nz)
        
        obs = np.concatenate([
            [w, gamma, x_o, z_o],
            psi_samples,
            omega_samples,
            [energy_norm, helicity_norm, self.previous_action]
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get diagnostic information."""
        return {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
            'psi_max': float(np.max(np.abs(self.psi))),
            'omega_max': float(np.max(np.abs(self.omega))),
            'equilibrium_type': self.equilibrium_type,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of environment.
        
        Parameters
        ----------
        action : np.ndarray, shape (1,)
            RMP amplitude ∈ [-1, 1]
        
        Returns
        -------
        observation : np.ndarray
            New observation (25D)
        reward : float
            Reward signal
        terminated : bool
            Episode termination flag
        truncated : bool
            Episode truncation flag (max steps)
        info : dict
            Diagnostic information
        """
        action_value = float(action[0])
        
        # Apply action smoothing (physical RMP coil inductance)
        smoothed_action = (
            self.alpha_smooth * action_value +
            (1 - self.alpha_smooth) * self.current_action
        )
        self.current_action = smoothed_action
        
        # Evolve MHD fields (simplified for Phase 5 Step 2)
        # In Step 3, replace with rk4_step_with_rmp
        self._evolve_simplified(smoothed_action)
        
        self.step_count += 1
        self.episode_steps += 1
        
        # Compute observation
        obs = self._get_observation()
        
        # Compute reward
        w = obs[0]
        gamma = obs[1]
        reward = -w - 0.1 * np.abs(gamma) - 0.01 * np.abs(smoothed_action)
        
        self.episode_reward += reward
        self.previous_action = smoothed_action
        
        # Check termination
        terminated = False
        truncated = False
        
        # Early termination (CFL violation prevention)
        if np.max(np.abs(self.psi)) > self.max_psi:
            terminated = True
        
        # Max steps truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _evolve_simplified(self, rmp_amplitude: float):
        """
        Simplified MHD evolution (Phase 5 Step 2).
        
        In Step 3, replace with:
        - rk4_step_with_rmp(self.psi, self.omega, rmp_amplitude, ...)
        """
        # Mock evolution (gradual growth)
        self.psi += 0.001 * self.eta * np.random.randn(*self.psi.shape)
        self.omega += 0.0001 * self.nu * np.random.randn(*self.omega.shape)
        
        # Mock RMP effect
        self.psi -= 0.0001 * rmp_amplitude * np.sin(self.z[None, None, :])
        
    def render(self, mode='human'):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
