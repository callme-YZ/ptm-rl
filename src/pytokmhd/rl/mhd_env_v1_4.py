"""
3D MHD Gym Environment for RL Control (v1.4)

Implements Gym-compatible RL environment wrapping the 3D IMEX solver
for tokamak MHD control using external coil currents.

Architecture:
- Observation: Dict with magnetic/kinetic state + diagnostics
- Action: 5 coil currents (simplified Gaussian profiles)
- Reward: Energy conservation penalty -|ΔE/E₀|

Physics:
- 3D Reduced MHD: ∂ψ/∂t = [φ, ψ] + η∇²ψ + J_ext
                  ∂ω/∂t = [ψ, ω] + η∇²ω
- Coil response: J_ext = Σᵢ Iᵢ·exp(-(r-rᵢ)²/σ²)·δ(θ-θᵢ)

Author: 小A 🤖
Created: 2026-03-20
Phase: 3 (RL Environment Implementation)
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional

from ..solvers.imex_3d import evolve_3d_imex
from ..ic.ballooning_mode import Grid3D, create_ballooning_mode_ic, create_equilibrium_ic
from ..physics.hamiltonian_3d import compute_hamiltonian_3d


class MHDEnv3D(gym.Env):
    """
    3D MHD control environment with external coil currents.
    
    **Observation Space:** Dict with 5 keys
        - 'psi': (nr, nθ, nζ) normalized stream function ψ/ψ_max
        - 'omega': (nr, nθ, nζ) normalized vorticity ω/ω_max
        - 'energy': float, relative energy E/E₀
        - 'max_psi': float, max|ψ|/ψ_max
        - 'max_omega': float, max|ω|/ω_max
    
    **Action Space:** Box(5,) in [-1, 1]
        - 5 coil currents scaled to [-I_max, I_max]
        - Coils evenly spaced in θ (poloidal angle)
    
    **Reward:**
        r(t) = -|ΔE/E₀| = -|E(t) - E(t-1)|/E₀
    
    **Episode:**
        - Initial condition: Ballooning mode (n=5, m0=2, ε=0.01)
        - Episode length: 50 steps (dt=0.01, T=0.5s)
        - Termination: None (fixed horizon)
    
    Parameters
    ----------
    grid_size : tuple, optional
        (nr, nθ, nζ) grid resolution, default (32, 64, 32)
    eta : float, optional
        Resistivity, default 1e-4
    dt : float, optional
        Time step, default 0.01
    max_steps : int, optional
        Episode length, default 50
    I_max : float, optional
        Maximum coil current, default 1.0
    coil_sigma : float, optional
        Radial width of coil Gaussian, default 0.05
    n_coils : int, optional
        Number of coils, default 5
    
    Examples
    --------
    >>> env = MHDEnv3D(grid_size=(32, 64, 32))
    >>> obs, info = env.reset()
    >>> for _ in range(10):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    >>> print(f"Final energy drift: {info['energy_drift']:.2e}")
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (32, 64, 32),
        eta: float = 1e-4,
        dt: float = 0.01,
        max_steps: int = 50,
        I_max: float = 1.0,
        coil_sigma: float = 0.05,
        n_coils: int = 5,
    ):
        super().__init__()
        
        # Grid setup
        nr, ntheta, nzeta = grid_size
        self.grid = Grid3D(nr=nr, ntheta=ntheta, nzeta=nzeta, r_max=1.0, R0=3.0)
        
        # Physics parameters
        self.eta = eta
        self.dt = dt
        self.max_steps = max_steps
        
        # Coil configuration
        self.I_max = I_max
        self.n_coils = n_coils
        self.coil_sigma = coil_sigma
        
        # Coil positions (evenly spaced in θ)
        self.coil_theta = np.linspace(0, 2*np.pi, n_coils, endpoint=False)
        self.coil_r = 0.7 * self.grid.r_max  # Place coils at r=0.7a
        
        # Normalization factors (set in reset)
        self.psi_max = None
        self.omega_max = None
        self.E0 = None
        
        # Current state
        self.psi = None
        self.omega = None
        self.current_step = 0
        self.energy_prev = None
        
        # Define observation and action spaces
        # Note: Gym doesn't support nested Dict spaces well, so we'll use a flat Dict
        self.observation_space = gym.spaces.Dict({
            'psi': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(nr, ntheta, nzeta), dtype=np.float32
            ),
            'omega': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(nr, ntheta, nzeta), dtype=np.float32
            ),
            'energy': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(), dtype=np.float32
            ),
            'max_psi': gym.spaces.Box(
                low=0, high=np.inf, shape=(), dtype=np.float32
            ),
            'max_omega': gym.spaces.Box(
                low=0, high=np.inf, shape=(), dtype=np.float32
            ),
        })
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_coils,), dtype=np.float32
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to ballooning mode initial condition.
        
        Parameters
        ----------
        seed : int, optional
            Random seed (for future stochastic ICs)
        options : dict, optional
            Reset options (reserved for future use)
        
        Returns
        -------
        obs : dict
            Initial observation
        info : dict
            Metadata: {'E0', 'psi_max', 'omega_max'}
        """
        super().reset(seed=seed)
        
        # Create equilibrium + ballooning mode perturbation
        # Note: epsilon reduced from spec (0.01) to prevent immediate instability
        # at dt=0.01. Ballooning modes are physically unstable and will grow.
        # RL agent's task is to control this growth via external currents.
        psi0, omega0, q_profile = create_equilibrium_ic(self.grid)
        psi1, omega1 = create_ballooning_mode_ic(
            self.grid,
            n=5,
            m0=2,
            epsilon=0.0001,  # Small perturbation for numerical stability
            q_profile=q_profile
        )
        
        # Combine equilibrium + perturbation
        self.psi = psi0 + psi1
        self.omega = omega0 + omega1
        
        # Compute normalization factors from initial condition
        self.psi_max = np.max(np.abs(self.psi))
        self.omega_max = np.max(np.abs(self.omega))
        self.E0 = compute_hamiltonian_3d(self.psi, self.omega, self.grid)
        
        # Initialize step counter and energy tracker
        self.current_step = 0
        self.energy_prev = self.E0
        
        # Get initial observation
        obs = self._compute_observation()
        
        info = {
            'E0': self.E0,
            'psi_max': self.psi_max,
            'omega_max': self.omega_max,
            'q_profile': q_profile,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step with external coil currents.
        
        Parameters
        ----------
        action : np.ndarray (n_coils,)
            Coil currents in [-1, 1], scaled to [-I_max, I_max]
        
        Returns
        -------
        obs : dict
            Observation after step
        reward : float
            Energy conservation reward
        terminated : bool
            Always False (no failure condition)
        truncated : bool
            True if current_step >= max_steps
        info : dict
            Diagnostics: energy, energy_drift, max fields, etc.
        """
        # Scale action to physical coil currents
        coil_currents = action * self.I_max
        
        # Compute external current density J_ext
        J_ext = self._compute_coil_response(coil_currents)
        
        # Evolve MHD system for 1 time step
        psi_hist, omega_hist, diagnostics = evolve_3d_imex(
            psi_init=self.psi,
            omega_init=self.omega,
            grid=self.grid,
            eta=self.eta,
            dt=self.dt,
            n_steps=1,
            J_ext=J_ext,
            store_interval=1,
            verbose=False
        )
        
        # Update state (take last time step)
        self.psi = psi_hist[-1]
        self.omega = omega_hist[-1]
        
        # Compute reward (energy conservation)
        energy_current = diagnostics['energy'][-1]
        energy_change = abs(energy_current - self.energy_prev)
        reward = -energy_change / self.E0
        
        # Update energy tracker
        self.energy_prev = energy_current
        
        # Increment step counter
        self.current_step += 1
        
        # Check termination
        terminated = False  # No failure condition in this version
        truncated = (self.current_step >= self.max_steps)
        
        # Get observation
        obs = self._compute_observation()
        
        # Info dict
        info = {
            'time': self.current_step * self.dt,
            'energy': energy_current,
            'E0': self.E0,  # Include E0 for normalization
            'energy_drift': abs(energy_current - self.E0) / self.E0,
            'energy_change': energy_change / self.E0,
            'max_psi': diagnostics['max_psi'][-1],
            'max_omega': diagnostics['max_omega'][-1],
            'coil_currents': coil_currents,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_observation(self) -> Dict:
        """
        Compute normalized observation from current state.
        
        Returns
        -------
        obs : dict
            Normalized observation with keys:
            - 'psi': ψ/ψ_max
            - 'omega': ω/ω_max
            - 'energy': E/E₀
            - 'max_psi': max|ψ|/ψ_max
            - 'max_omega': max|ω|/ω_max
        """
        # Compute current energy
        E_current = compute_hamiltonian_3d(self.psi, self.omega, self.grid)
        
        # Normalize fields
        psi_norm = self.psi / self.psi_max if self.psi_max > 0 else self.psi
        omega_norm = self.omega / self.omega_max if self.omega_max > 0 else self.omega
        
        obs = {
            'psi': psi_norm.astype(np.float32),
            'omega': omega_norm.astype(np.float32),
            'energy': np.array(E_current / self.E0, dtype=np.float32),
            'max_psi': np.array(np.max(np.abs(self.psi)) / self.psi_max, dtype=np.float32),
            'max_omega': np.array(np.max(np.abs(self.omega)) / self.omega_max, dtype=np.float32),
        }
        
        return obs
    
    def _compute_coil_response(self, coil_currents: np.ndarray) -> np.ndarray:
        """
        Map coil currents to external current density J_ext(r, θ, ζ).
        
        Simplified model:
            J_ext(r, θ, ζ) = α · Σᵢ Iᵢ · G(r, rᵢ) · δ(θ - θᵢ)
        
        where:
            - α: Small coupling coefficient (0.01) to prevent instability
            - G(r, rᵢ) = exp(-(r - rᵢ)²/σ²): Gaussian radial profile
            - δ(θ - θᵢ): Coil localized at θᵢ (approximated by narrow Gaussian)
            - Constant in ζ (axisymmetric approximation)
        
        Parameters
        ----------
        coil_currents : np.ndarray (n_coils,)
            Coil currents in physical units
        
        Returns
        -------
        J_ext : np.ndarray (nr, nθ, nζ)
            External current density field
        
        Notes
        -----
        The coupling coefficient α=0.01 is chosen to prevent numerical instability.
        Strong external currents can drive CFL violations and exponential growth.
        """
        nr, ntheta, nzeta = self.grid.nr, self.grid.ntheta, self.grid.nzeta
        J_ext = np.zeros((nr, ntheta, nzeta))
        
        # Coupling coefficient (prevent instability)
        alpha = 0.01
        
        # Build 3D grid arrays
        r_3d = self.grid.r[:, np.newaxis, np.newaxis]  # (nr, 1, 1)
        theta_3d = self.grid.theta[np.newaxis, :, np.newaxis]  # (1, nθ, 1)
        
        # Radial Gaussian centered at coil position
        G_r = np.exp(-(r_3d - self.coil_r)**2 / self.coil_sigma**2)
        
        # Add contribution from each coil
        for i, (I_coil, theta_coil) in enumerate(zip(coil_currents, self.coil_theta)):
            # Poloidal Gaussian (narrow, width ~ 2 grid cells)
            theta_width = 2 * self.grid.dtheta
            
            # Handle periodic wrapping: compute distance on circle
            dtheta = theta_3d - theta_coil
            dtheta = np.angle(np.exp(1j * dtheta))  # Wrap to [-π, π]
            G_theta = np.exp(-dtheta**2 / theta_width**2)
            
            # Combine radial and poloidal profiles with coupling
            # Replicate along toroidal direction (axisymmetric)
            J_coil = alpha * I_coil * G_r * G_theta
            J_ext += J_coil
        
        return J_ext
    
    def render(self):
        """Rendering not implemented."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


# Convenience function for creating environment
def make_env(**kwargs) -> MHDEnv3D:
    """
    Create MHD environment with default parameters.
    
    Parameters
    ----------
    **kwargs : dict
        Override default parameters
    
    Returns
    -------
    env : MHDEnv3D
        Gym environment instance
    
    Examples
    --------
    >>> env = make_env(grid_size=(32, 64, 32), dt=0.01)
    >>> obs, info = env.reset()
    """
    return MHDEnv3D(**kwargs)
