"""
Basic MHD Control Environment (v1.3 Proof-of-Concept)

Simple Gym environment for RL proof-of-concept:
- Observation: island width + energy
- Action: RMP coil current (scalar)
- Reward: -island_width (minimize tearing mode)

Author: 小A 🤖
Date: 2026-03-19
Phase: v1.3 RL Integration (basic proof)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from ..geometry import ToroidalGrid
from ..solvers import HamiltonianMHDIMEX
from ..physics import compute_hamiltonian
from ..operators import laplacian_toroidal


class MHDControlEnv(gym.Env):
    """
    Basic MHD control environment for tearing mode suppression.
    
    Observation Space
    -----------------
    Box(2,):
        [0] island_width (normalized [0, 1])
        [1] energy (normalized [0, 1])
    
    Action Space
    ------------
    Box(1,):
        [0] RMP coil current [-1, 1] (normalized)
    
    Reward
    ------
    reward = -island_width (minimize island)
    
    Episode
    -------
    - Initial condition: equilibrium + perturbation
    - Max steps: 100
    - Termination: island_width > threshold OR energy diverges
    
    Usage
    -----
    >>> env = MHDControlEnv()
    >>> obs, info = env.reset()
    >>> for _ in range(100):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         break
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        nr: int = 32,
        ntheta: int = 64,
        R0: float = 1.0,
        a: float = 0.3,
        dt: float = 1e-4,
        eta: float = 1e-4,
        max_steps: int = 100,
        island_threshold: float = 0.5,
        rmp_strength: float = 0.01,
    ):
        """
        Initialize MHD control environment.
        
        Parameters
        ----------
        nr, ntheta : int
            Grid resolution (radial × poloidal)
        R0, a : float
            Major radius and minor radius [m]
        dt : float
            Timestep [s]
        eta : float
            Resistivity (normalized)
        max_steps : int
            Maximum episode length
        island_threshold : float
            Termination threshold for island width
        rmp_strength : float
            RMP coil coupling strength
        """
        super().__init__()
        
        # Physics parameters
        self.nr = nr
        self.ntheta = ntheta
        self.R0 = R0
        self.a = a
        self.dt = dt
        self.eta = eta
        self.max_steps = max_steps
        self.island_threshold = island_threshold
        self.rmp_strength = rmp_strength
        
        # Grid
        self.grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=ntheta)
        
        # Solver
        self.solver = HamiltonianMHDIMEX(
            grid=self.grid,
            dt=dt,
            eta=eta,
            nu=0.0,
            use_imex=True,
            verbose=False
        )
        
        # Observation space: [island_width, energy] (normalized)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: [rmp_current] (normalized [-1, 1])
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # State
        self.psi = None
        self.omega = None
        self.phi = None
        self.step_count = 0
        self.initial_energy = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial condition.
        
        Returns
        -------
        observation : np.ndarray
            Initial observation [island_width, energy]
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        
        # Initialize equilibrium
        r = self.grid.r_grid
        theta = self.grid.theta_grid
        
        # Simple equilibrium: radial profile
        psi_eq = r**2 * (1 - r/self.a)**2
        
        # Add perturbation (seed tearing mode)
        m = 2  # Poloidal mode number
        n = 1  # Toroidal mode number (axisymmetric, so n=1 means θ dependence)
        pert_amplitude = 0.01
        perturbation = pert_amplitude * r**2 * (1 - r/self.a)**2 * np.sin(m * theta)
        
        self.psi = psi_eq + perturbation
        
        # Enforce boundary conditions
        self.psi[-1, :] = 0.0  # Conducting wall
        self.psi[0, :] = np.mean(self.psi[0, :])  # Axis
        
        # Compute omega
        self.omega = -laplacian_toroidal(self.psi, self.grid)
        
        # Initialize phi (will be computed in _get_obs)
        self.phi = np.zeros_like(self.psi)
        
        # Compute initial energy
        self.initial_energy = compute_hamiltonian(self.psi, self.phi, self.grid)
        
        # Reset counter
        self.step_count = 0
        
        # Get observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one timestep in the environment.
        
        Parameters
        ----------
        action : np.ndarray
            RMP coil current (normalized [-1, 1])
        
        Returns
        -------
        observation : np.ndarray
            New observation
        reward : float
            Reward signal
        terminated : bool
            Episode terminated (island too large or energy diverged)
        truncated : bool
            Episode truncated (max steps reached)
        info : dict
            Additional information
        """
        # Apply RMP forcing (simplified: add to omega equation)
        rmp_current = float(action[0])  # Normalized [-1, 1]
        rmp_force = self._compute_rmp_force(rmp_current)
        
        # Add RMP to omega (external drive)
        self.omega = self.omega + self.dt * rmp_force
        
        # Evolve physics
        self.psi, self.omega = self.solver.step(self.psi, self.omega)
        
        # Increment counter
        self.step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        island_width = obs[0]  # Normalized island width
        reward = -island_width  # Minimize island
        
        # Check termination
        terminated = self._check_terminated(obs)
        truncated = self.step_count >= self.max_steps
        
        # Info
        info = self._get_info()
        info['rmp_current'] = rmp_current
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """
        Compute observation from current state.
        
        Returns
        -------
        obs : np.ndarray
            [island_width_normalized, energy_normalized]
        """
        # Compute island width (simplified: max amplitude of m=2 component)
        m = 2
        psi_m2 = np.abs(np.mean(self.psi * np.sin(m * self.grid.theta_grid), axis=1))
        island_width_raw = np.max(psi_m2)
        
        # Normalize (assume max possible island ~ 0.1)
        island_width_norm = np.clip(island_width_raw / 0.1, 0.0, 1.0)
        
        # Compute energy
        energy = compute_hamiltonian(self.psi, self.phi, self.grid)
        
        # Normalize energy (relative to initial)
        if self.initial_energy > 0:
            energy_norm = np.clip(energy / self.initial_energy, 0.0, 2.0) / 2.0
        else:
            energy_norm = 0.5
        
        obs = np.array([island_width_norm, energy_norm], dtype=np.float32)
        return obs
    
    def _compute_rmp_force(self, rmp_current: float) -> np.ndarray:
        """
        Compute RMP forcing term.
        
        Simplified model: RMP creates m=2 poloidal perturbation.
        
        Parameters
        ----------
        rmp_current : float
            Normalized RMP current [-1, 1]
        
        Returns
        -------
        force : np.ndarray (nr, ntheta)
            RMP forcing term to add to omega equation
        """
        r = self.grid.r_grid
        theta = self.grid.theta_grid
        
        # RMP creates m=2 component
        m = 2
        
        # Radial profile (localized near edge)
        radial_profile = np.exp(-((r - 0.8*self.a) / (0.1*self.a))**2)
        
        # Poloidal structure
        poloidal_structure = np.cos(m * theta)
        
        # Force amplitude (proportional to current)
        force = self.rmp_strength * rmp_current * radial_profile * poloidal_structure
        
        return force
    
    def _check_terminated(self, obs: np.ndarray) -> bool:
        """
        Check if episode should terminate.
        
        Termination conditions:
        - Island width exceeds threshold
        - Energy diverges (>2× initial)
        
        Parameters
        ----------
        obs : np.ndarray
            Current observation
        
        Returns
        -------
        terminated : bool
        """
        island_width = obs[0]
        energy_norm = obs[1]
        
        # Terminate if island too large
        if island_width > self.island_threshold:
            return True
        
        # Terminate if energy diverges
        if energy_norm > 1.0:  # >2× initial energy
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """
        Get additional information.
        
        Returns
        -------
        info : dict
            - step: current step count
            - psi_max: max |psi|
            - omega_max: max |omega|
        """
        return {
            'step': self.step_count,
            'psi_max': float(np.max(np.abs(self.psi))),
            'omega_max': float(np.max(np.abs(self.omega))),
        }
