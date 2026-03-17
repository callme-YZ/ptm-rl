"""
Gym environment for MHD control task.

v1.1: Simplified cylindrical solver, energy-only control.
v1.2: Will use fixed toroidal solver with full physics.

v1.1 LIMITATIONS:
- NO div_B constraint (cylindrical simplification)
- Energy control only (framework validation)
- Parameter modulation action (not realistic)
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional

from ..solvers.action_mhd_solver import SimplifiedMHDSolver


class ToroidalMHDEnv(gym.Env):
    """
    MHD control environment for RL (v1.1 simplified).
    
    Observation: 9D (psi_modes(8) + energy_drift(1))
        - Removed: energy_rel, div_B (v1.1 simplification)
    
    Action: 2D continuous [eta_multiplier, nu_multiplier] ∈ [0.5, 2.0]
    
    Reward: Energy-based only
        reward = -energy_drift - action_penalty
    
    v1.1 Focus: Framework validation, not physics accuracy.
    v1.2 will add: div_B constraint, realistic control.
    
    Parameters
    ----------
    nr : int
        Radial grid points
    ntheta : int
        Poloidal grid points
    dt : float
        Time step
    max_steps : int
        Episode length
    eta : float
        Base resistivity
    nu : float
        Base viscosity
    w_energy : float
        Energy weight (default: 1.0)
    w_action : float
        Action penalty (default: 0.01)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        nr: int = 32,
        ntheta: int = 64,
        dt: float = 1e-4,
        max_steps: int = 1000,
        eta: float = 1e-5,
        nu: float = 1e-4,
        w_energy: float = 1.0,
        w_action: float = 0.01
    ):
        super().__init__()
        
        # Create solver
        self.solver = SimplifiedMHDSolver(
            nr=nr, ntheta=ntheta,
            dt=dt, eta=eta, nu=nu
        )
        
        # Episode config
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reward weights (v1.1: no constraint term)
        self.w_energy = w_energy
        self.w_action = w_action
        
        # Spaces (v1.1: 9D observation)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(9,), dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([0.5, 0.5]),
            high=np.array([2.0, 2.0]),
            shape=(2,), dtype=np.float32
        )
        
        # Reference equilibrium energy
        self.E_eq = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Initialize with small random perturbation
        if seed is not None:
            np.random.seed(seed)
        
        psi0 = np.random.randn(self.solver.nr, self.solver.ntheta) * 0.01
        omega0 = np.random.randn(self.solver.nr, self.solver.ntheta) * 0.01
        
        self.solver.initialize(psi0, omega0)
        
        # Compute equilibrium energy
        self.E_eq = self._compute_energy(psi0, omega0)
        
        # Reset counter
        self.current_step = 0
        
        # Get observation
        obs = self._get_observation()
        info = {'E_eq': self.E_eq}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Clip action to valid range
        action = np.clip(action, [0.5, 0.5], [2.0, 2.0])
        
        # Advance solver
        psi, omega = self.solver.step(action)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward (v1.1: energy-only)
        reward, reward_info = self._compute_reward(obs, action)
        
        # Check termination (v1.1: only energy explosion)
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
            'time': self.solver.time,
            'n_steps': self.solver.n_steps,
            'terminated': terminated,
            'truncated': truncated
        }
        
        self.current_step += 1
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Extract 9D observation from solver state."""
        psi = self.solver.psi
        omega = self.solver.omega
        
        # Fourier modes (8D)
        fft_theta = np.fft.fft(psi, axis=1)
        amplitudes = np.abs(fft_theta[:, :8])
        psi_modes = np.mean(amplitudes, axis=0)
        psi_modes = psi_modes / (np.max(psi_modes) + 1e-10)
        
        # Energy drift (1D)
        E = self._compute_energy(psi, omega)
        energy_drift = np.abs((E - self.E_eq) / (self.E_eq + 1e-10))
        
        # Assemble 9D observation
        obs = np.concatenate([
            psi_modes,               # 8D
            [energy_drift]           # 1D
        ])
        
        return obs.astype(np.float32)
    
    def _compute_energy(self, psi: np.ndarray, omega: np.ndarray) -> float:
        """Compute total energy."""
        E_mag = 0.5 * np.sum(psi**2) * self.solver.dr * self.solver.dtheta
        E_kin = 0.5 * np.sum(omega**2) * self.solver.dr * self.solver.dtheta
        return E_mag + E_kin
    
    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute reward (v1.1: energy-only).
        
        reward = -w_energy * energy_drift - w_action * ||action - 1||²
        """
        # Unpack observation
        energy_drift = obs[8]  # Last element
        
        # Components
        r_energy = -self.w_energy * energy_drift
        r_action = -self.w_action * np.linalg.norm(action - 1.0)**2
        
        reward = r_energy + r_action
        
        info = {
            'reward_energy': r_energy,
            'reward_action': r_action,
            'reward_total': reward,
            'energy_drift': energy_drift
        }
        
        return reward, info
    
    def _check_failure(self, obs: np.ndarray) -> bool:
        """Check failure (v1.1: only energy explosion)."""
        energy_drift = obs[8]
        
        # Failure: Energy exploded (>1000% deviation)
        if energy_drift > 10.0:
            return True
        
        return False
    
    def _check_success(self, obs: np.ndarray) -> bool:
        """Check success (v1.1: energy well controlled)."""
        energy_drift = obs[8]
        
        # Success: Energy drift < 1%
        return energy_drift < 0.01
