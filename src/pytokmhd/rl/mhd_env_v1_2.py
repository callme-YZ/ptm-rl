"""
Gym environment for MHD control task (v1.2 - Toroidal Symplectic).

v1.2 Upgrade:
- Uses SymplecticIntegrator (structure-preserving)
- MHDObservation (19D with phase-resolved modes)
- MHDAction (parameter modulation)

Author: 小P ⚛️
Created: 2026-03-18 (Phase 3 Step 3.2.3)
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional

from ..geometry import ToroidalGrid
from ..integrators import SymplecticIntegrator
from ..operators import laplacian_toroidal
from .observations import MHDObservation
from .actions import MHDAction


class ToroidalMHDEnv(gym.Env):
    """
    MHD control environment with Symplectic integrator (v1.2).
    
    Observation: 19D (MHDObservation)
        - psi_modes: 16D (8 modes × Re/Im)
        - energy: 1D (relative to equilibrium)
        - energy_drift: 1D (absolute drift)
        - div_B_max: 1D (constraint violation proxy)
    
    Action: 2D continuous [eta_multiplier, nu_multiplier] ∈ [0.5, 2.0]
        v1.2: Parameter modulation (framework validation)
        v2.0: Spatial current drive J_ext(r,θ)
    
    Reward: Multi-objective
        reward = -w_energy * energy_drift 
                 -w_action * ||action - 1||²
                 -w_constraint * div_B_max (v1.2: proxy)
    
    Parameters
    ----------
    R0 : float
        Major radius (default: 1.0)
    a : float
        Minor radius (default: 0.3)
    nr : int
        Radial grid points (default: 32)
    ntheta : int
        Poloidal grid points (default: 64)
    dt : float
        Time step (default: 1e-4)
    max_steps : int
        Episode length (default: 1000)
    eta : float
        Base resistivity (default: 1e-6)
    nu : float
        Base viscosity (default: 1e-6)
    w_energy : float
        Energy weight (default: 1.0)
    w_action : float
        Action penalty (default: 0.01)
    w_constraint : float
        Constraint weight (default: 0.1, v1.2 low due to proxy)
    
    Notes
    -----
    v1.2 Limitations:
    - div_B is Laplacian proxy (not true ∇·B)
    - Action is parameter modulation (not physical)
    - Purpose: Framework validation
    
    v2.0 will add:
    - True ∇·B constraint
    - Spatial current drive J_ext(r,θ)
    - Realistic actuator models
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        R0: float = 1.0,
        a: float = 0.3,
        nr: int = 32,
        ntheta: int = 64,
        dt: float = 1e-4,
        max_steps: int = 1000,
        eta: float = 1e-6,
        nu: float = 1e-6,
        w_energy: float = 1.0,
        w_action: float = 0.01,
        w_constraint: float = 0.1  # v1.2: Low weight (proxy only)
    ):
        super().__init__()
        
        # Create grid
        self.grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=ntheta)
        
        # Create solver
        self.solver = SymplecticIntegrator(
            self.grid, dt=dt, eta=eta, nu=nu,
            operator_splitting=True
        )
        
        # Create action handler
        self.action_handler = MHDAction(eta_base=eta, nu_base=nu)
        
        # Episode config
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward weights
        self.w_energy = w_energy
        self.w_action = w_action
        self.w_constraint = w_constraint
        
        # Spaces (v1.2: 19D observation, 2D action)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,), dtype=np.float32
        )
        self.action_space = self.action_handler.get_action_space()
        
        # Observation handler (created after reset)
        self.obs_handler = None
        
        # Equilibrium state for normalization
        self.psi_eq = None
        self.omega_eq = None
        self.E_eq = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.
        
        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            - 'perturbation_amplitude': float (default: 0.01)
            - 'perturbation_mode': int (default: 2)
        
        Returns
        -------
        obs : np.ndarray (19,)
            Initial observation
        info : dict
            Metadata
        """
        super().reset(seed=seed)
        
        # Parse options
        if options is None:
            options = {}
        pert_amp = options.get('perturbation_amplitude', 0.01)
        pert_mode = options.get('perturbation_mode', 2)
        
        # Initialize with equilibrium + perturbation
        if seed is not None:
            np.random.seed(seed)
        
        # Equilibrium: r²(1-r/a) profile
        r_grid = self.grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / self.grid.a)
        
        # Add perturbation (toroidal mode m)
        theta_grid = self.grid.theta_grid
        psi_pert = pert_amp * r_grid * np.sin(pert_mode * theta_grid)
        
        psi0 = psi_eq + psi_pert
        
        # Consistent vorticity
        omega0 = laplacian_toroidal(psi0, self.grid)
        
        # Initialize solver
        self.solver.initialize(psi0, omega0)
        
        # Store equilibrium
        self.psi_eq = psi_eq
        self.omega_eq = laplacian_toroidal(psi_eq, self.grid)
        
        # Compute equilibrium energy for observation handler
        # E = ∫ [1/2 ω² + 1/2 |∇ψ|²] dV
        from ..operators import gradient_toroidal
        grad_psi_r, grad_psi_theta = gradient_toroidal(psi_eq, self.grid)
        e_mag = 0.5 * (grad_psi_r**2 + grad_psi_theta**2)
        jacobian = self.grid.jacobian()
        dV = jacobian * self.grid.dr * self.grid.dtheta
        E_kin = 0.5 * np.sum(self.omega_eq**2 * dV)
        E_mag_total = np.sum(e_mag * dV)
        self.E_eq = E_kin + E_mag_total
        
        # Create observation handler (now that we have psi_eq and E_eq)
        self.obs_handler = MHDObservation(self.psi_eq, self.E_eq, self.grid)
        
        # Reset counter
        self.current_step = 0
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            'E_eq': self.E_eq,
            'perturbation_amplitude': pert_amp,
            'perturbation_mode': pert_mode
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.
        
        Parameters
        ----------
        action : np.ndarray (2,)
            [eta_multiplier, nu_multiplier] ∈ [0.5, 2.0]
        
        Returns
        -------
        obs : np.ndarray (19,)
            Observation after step
        reward : float
            Reward signal
        terminated : bool
            Episode terminated (failure)
        truncated : bool
            Episode truncated (max_steps)
        info : dict
            Metadata
        """
        # Apply action through handler (clips to bounds)
        eta_eff, nu_eff = self.action_handler.apply(action)
        
        # Advance solver
        self.solver.step(action=action)
        
        # Get observation
        obs = self._get_observation()
        
        # Increment step counter BEFORE checks
        self.current_step += 1
        
        # Compute reward
        reward, reward_info = self._compute_reward(obs, action)
        
        # Check termination
        terminated = self._check_failure(obs)
        truncated = (self.current_step >= self.max_steps)
        
        # Terminal bonus/penalty
        if terminated:
            reward += -10.0  # Failure penalty
        elif truncated and self._check_success(obs):
            reward += 10.0  # Success bonus
        
        # Info
        info = {
            **reward_info,
            'time': self.solver.t,
            'eta_effective': eta_eff,
            'nu_effective': nu_eff,
            'terminated': terminated,
            'truncated': truncated,
            'current_step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation from current solver state.
        
        Returns
        -------
        obs : np.ndarray (19,)
            MHDObservation vector
        """
        obs_dict = self.obs_handler.get_observation(
            self.solver.psi,
            self.solver.omega
        )
        # Extract vector from dict and convert to float32
        obs = obs_dict['vector'].astype(np.float32)
        return obs
    
    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute reward.
        
        reward = -w_energy * energy_drift
                 -w_action * ||action - 1||²
                 -w_constraint * div_B_max
        
        Parameters
        ----------
        obs : np.ndarray (19,)
            Observation vector
        action : np.ndarray (2,)
            Action vector
        
        Returns
        -------
        reward : float
            Total reward
        info : dict
            Reward components
        """
        # Unpack observation
        # obs = [psi_modes(16), energy(1), energy_drift(1), div_B_max(1)]
        energy_drift = obs[17]  # Second-to-last
        div_B_max = obs[18]     # Last
        
        # Components
        r_energy = -self.w_energy * energy_drift
        r_action = -self.w_action * np.linalg.norm(action - 1.0)**2
        r_constraint = -self.w_constraint * div_B_max
        
        reward = r_energy + r_action + r_constraint
        
        info = {
            'reward_energy': r_energy,
            'reward_action': r_action,
            'reward_constraint': r_constraint,
            'reward_total': reward,
            'energy_drift': energy_drift,
            'div_B_max': div_B_max
        }
        
        return reward, info
    
    def _check_failure(self, obs: np.ndarray) -> bool:
        """
        Check failure condition.
        
        Failure: Energy exploded (>100% deviation)
        
        Parameters
        ----------
        obs : np.ndarray (19,)
            Observation vector
        
        Returns
        -------
        failed : bool
            True if episode should terminate
        """
        energy_drift = obs[17]
        
        # Failure: Energy exploded (>100% deviation)
        if energy_drift > 1.0:
            return True
        
        # v1.2: Could add div_B threshold, but proxy unreliable
        # if div_B_max > threshold: return True
        
        return False
    
    def _check_success(self, obs: np.ndarray) -> bool:
        """
        Check success condition.
        
        Success: Energy drift < 1%
        
        Parameters
        ----------
        obs : np.ndarray (19,)
            Observation vector
        
        Returns
        -------
        success : bool
            True if successfully controlled
        """
        energy_drift = obs[17]
        
        # Success: Energy drift < 1%
        return energy_drift < 0.01
    
    def render(self):
        """Rendering not implemented in v1.2."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
