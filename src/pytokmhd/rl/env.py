"""
MHD Tearing Mode Control Environment for Reinforcement Learning.

This module provides a Gym-compatible environment for training RL agents
to control tearing modes in tokamak plasmas using RMP (Resonant Magnetic Perturbation).

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
Status: Phase 5 Week 1 Implementation
"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Optional

# Phase 1-4 imports (will be available after integration)
# from pytokmhd.solver import rk4_step_with_rmp, setup_tearing_mode, create_equilibrium_cache
# from pytokmhd.diagnostics import TearingModeMonitor, find_rational_surface


class MHDTearingControlEnv(gym.Env):
    """
    Gym environment for RL-based tearing mode control.
    
    **Observation Space (25D):**
    - w: Island width (primary control target)
    - gamma: Growth rate (stability indicator)
    - x_o, z_o: Island center position
    - psi_samples: 8 magnetic flux samples
    - omega_samples: 8 vorticity samples
    - energy: Total MHD energy (normalized)
    - mag_helicity: Magnetic helicity (normalized)
    - prev_action: Previous RMP amplitude
    - t: Normalized time
    - dt_since_reset: Time since episode start
    
    **Action Space:**
    - Continuous: RMP amplitude ∈ [-1, 1] (scaled to [-0.1, 0.1] internally)
    - Action smoothing applied (alpha=0.3): Reflects physical RMP coil inductance
    
    **Reward Function:**
    - reward = -w - 0.1*|gamma| - 0.01*|action| + convergence_bonus
    
    **Episode Termination:**
    - Max steps reached (default 200)
    - Numerical instability detected (NaN/Inf)
    - MHD field overflow (|psi| > 10 or |omega| > 100)
    
    **Numerical Stability Constraints:**
    - Action smoothing (alpha=0.3): Reflects physical RMP coil inductance (~0.03s time constant)
    - Early termination (psi_max=10, omega_max=100): Prevents solver CFL violation
    - These are NOT workarounds - they reflect physical/numerical reality
    
    **Known Limitations (Honest Documentation):**
    - MHD solver (Phase 4) stable for slow RMP changes (Δt ~ dt)
    - Fast RMP changes may violate CFL condition → early termination
    - Phase 4 validation focused on fixed RMP, not rapid changes (technical debt)
    - Parameters (alpha=0.3, psi_max=10) empirically chosen, need sensitivity analysis
    - May require re-tuning when upgrading to PyTokEq (Step 3)
    
    **Physics Review:** APPROVED by 小P ⚛️ (2026-03-16)
    **Quality Assessment:** Scientific rigor + engineering pragmatism (YZ, 小A, 小P 2026-03-16)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        Nr: int = 64,
        Nz: int = 128,
        dt: float = 0.01,
        eta: float = 1e-3,
        nu: float = 1e-3,
        m: int = 2,
        n: int = 1,
        A_max: float = 0.1,
        max_steps: int = 200,
        w_0: float = 0.01,
        convergence_threshold: float = 0.005,
        use_phase4_api: bool = True  # Use Phase 1-4 real MHD solver
    ):
        """
        Initialize MHD Tearing Control Environment.
        
        Args:
            Nr: Radial grid points
            Nz: Toroidal grid points
            dt: Time step size
            eta: Resistivity
            nu: Viscosity
            m: Poloidal mode number
            n: Toroidal mode number
            A_max: Maximum RMP amplitude
            max_steps: Maximum steps per episode
            w_0: Initial island width
            convergence_threshold: Threshold for convergence bonus
            use_phase4_api: If True, use Phase 1-4 API (requires integration)
        """
        super().__init__()
        
        # Environment configuration
        self.Nr = Nr
        self.Nz = Nz
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.m = m
        self.n = n
        self.A_max = A_max
        self.max_steps = max_steps
        self.w_0 = w_0
        self.convergence_threshold = convergence_threshold
        self.use_phase4_api = use_phase4_api
        
        # Grid setup
        self.r = np.linspace(0.1, 1.0, Nr)
        self.z = np.linspace(0, 2*np.pi, Nz)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]
        R, Z = np.meshgrid(self.r, self.z, indexing='ij')
        self.r_grid = R
        self.z_grid = Z
        
        # Phase 1-4 integration placeholders
        self.eq_cache = None
        self.monitor = None
        
        if use_phase4_api:
            # Import Phase 4 real MHD solver
            import sys
            from pathlib import Path
            phase4_path = Path(__file__).parent.parent.parent.parent / "src"
            if str(phase4_path) not in sys.path:
                sys.path.insert(0, str(phase4_path))
            
            from pytokmhd.diagnostics import TearingModeMonitor
            self.monitor = TearingModeMonitor(m=m, n=n, track_every=1)  # ✅ Bug fix: track every step for RL
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),  # 4 + 8 + 8 + 2 + 3 = 25D (removed energy_drift)
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Environment state
        self.psi = None  # Magnetic flux
        self.omega = None  # Vorticity
        self.t = 0.0
        self.step_count = 0
        self.prev_action = 0.0
        self.smoothed_action = 0.0  # For action smoothing (alpha=0.3)
        
        # History for diagnostics
        self.w_history = []
        self.gamma_history = []
        self.energy_initial = None
        self.last_diagnostics = None  # Store latest diagnostics for info dict
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial tearing mode state.
        
        Returns:
            obs: Initial observation (26D)
            info: Information dict with diagnostics
        """
        if self.use_phase4_api:
            # ✅ Phase 4 simplified initialization (verified stable 200+ steps)
            # Use simple tearing mode perturbation, NOT Solovev equilibrium
            # This avoids numerical overflow and is verified in Phase 4 tests
            
            # Grid dimensions
            Lr = self.r[-1] - self.r[0]  # Radial extent
            Lz = self.z[-1] - self.z[0]  # Toroidal extent (2π)
            
            # Simplified initial state from Phase 4 tests
            # psi: small amplitude sinusoidal perturbation
            self.psi = 0.1 * np.sin(2 * np.pi * self.z_grid / Lz) * (1 - self.r_grid**2)
            
            # omega: start from zero (as in Phase 4 tests)
            self.omega = np.zeros_like(self.psi)
            
            # Rational surface location (approximate)
            self.rational_surface_r = 0.5  # Fixed for (m,n)=(2,1)
        else:
            # Standalone mode: simplified initial state
            self._reset_standalone()
        
        # Reset counters
        self.t = 0.0
        self.step_count = 0
        self.prev_action = 0.0
        self.smoothed_action = 0.0  # Reset smoothed action
        self.w_history = []
        self.gamma_history = []
        
        # Compute initial observation
        obs = self._get_observation()
        self.energy_initial = obs[20]  # Store initial energy (correct index)
        
        # Info dict (same format as step())
        info = {
            'w': float(obs[0]),
            'gamma': float(obs[1]),
            'x_o': float(obs[2]),
            'z_o': float(obs[3]),
            't': self.t,
            'step': self.step_count,
            'rmp_amplitude': 0.0,
            'diagnostics': self.last_diagnostics  # Complete diagnostics
        }
        
        return obs, info
    
    def _reset_standalone(self):
        """
        Standalone reset for testing without Phase 1-4 API.
        
        Creates a simplified tearing mode initial state:
        - Equilibrium: linear q-profile
        - Perturbation: localized island structure
        """
        # Simple equilibrium: psi ~ r^2
        r_2d = self.r_grid
        self.psi = r_2d**2
        
        # Add tearing mode perturbation
        r_s = 0.5  # Rational surface at r=0.5
        z_2d = np.tile(self.z, (self.Nr, 1))
        
        # Island structure: Gaussian envelope × cos(m*theta - n*phi)
        theta = np.arctan2(z_2d, r_2d - r_s)
        perturbation = (
            self.w_0 * 
            np.exp(-((r_2d - r_s) / 0.1)**2) *
            np.cos(self.m * theta - self.n * z_2d)
        )
        self.psi += perturbation
        
        # Vorticity: initialize from psi
        self.omega = self._compute_omega_from_psi(self.psi)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of MHD evolution with RMP control.
        
        Args:
            action: RMP amplitude ∈ [-1, 1] (scaled internally to [-A_max, A_max])
        
        Returns:
            obs: Observation after step (25D)
            reward: Scalar reward
            done: Episode termination flag
            info: Additional information dict
        """
        # Action smoothing (low-pass filter)
        # Physical basis: RMP coils have inductance, cannot change instantaneously
        # alpha=0.3: coil time constant ~ 3*dt (0.03s for dt=0.01s)
        # Chosen to balance responsiveness vs numerical stability
        alpha = 0.3
        self.smoothed_action = alpha * float(action[0]) + (1 - alpha) * self.smoothed_action
        
        # Scale smoothed action to physical range
        rmp_amplitude = self.smoothed_action * self.A_max
        
        if self.use_phase4_api:
            # Phase 4 API step: use real MHD evolution with RMP
            from pytokmhd.control.rmp_coupling import rk4_step_with_rmp
            
            self.psi, self.omega = rk4_step_with_rmp(
                self.psi, self.omega,
                self.dt, self.dr, self.dz,
                self.r_grid,
                self.eta, self.nu,
                rmp_amplitude=rmp_amplitude,
                m=self.m, n=self.n
            )
        else:
            # Standalone mode: simplified MHD evolution
            self._step_standalone(rmp_amplitude)
        
        # Update time and counters
        self.t += self.dt
        self.step_count += 1
        self.prev_action = rmp_amplitude
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, rmp_amplitude)
        
        # Check termination
        done = self._check_done(obs)
        
        # Info dict
        info = {
            'w': float(obs[0]),
            'gamma': float(obs[1]),
            'x_o': float(obs[2]),
            'z_o': float(obs[3]),
            't': self.t,
            'step': self.step_count,
            'rmp_amplitude': rmp_amplitude,
            'diagnostics': self.last_diagnostics  # ✅ Add complete diagnostics
        }
        
        return obs, reward, done, info
    
    def _step_standalone(self, rmp_amplitude: float):
        """
        Standalone MHD evolution (simplified, for testing).
        
        Implements a toy model:
        - dψ/dt = -η * ∇²ψ + RMP forcing
        - dω/dt = -ν * ∇²ω + coupling terms
        
        This is NOT the full reduced MHD! Just for environment testing.
        Real physics will use Phase 1-4 API.
        """
        # Laplacian approximation (finite difference)
        laplacian_psi = self._laplacian(self.psi)
        laplacian_omega = self._laplacian(self.omega)
        
        # RMP forcing (m=2, n=1 resonant perturbation)
        r_s = 0.5
        r_2d = self.r_grid
        z_2d = np.tile(self.z, (self.Nr, 1))
        theta = np.arctan2(z_2d, r_2d - r_s)
        
        rmp_forcing = (
            rmp_amplitude *
            np.exp(-((r_2d - r_s) / 0.1)**2) *
            np.cos(self.m * theta - self.n * z_2d)
        )
        
        # Time evolution (simple Euler)
        dpsi_dt = -self.eta * laplacian_psi + rmp_forcing
        domega_dt = -self.nu * laplacian_omega
        
        self.psi += dpsi_dt * self.dt
        self.omega += domega_dt * self.dt
    
    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian in cylindrical coordinates (simplified).
        
        ∇² = ∂²/∂r² + (1/r)∂/∂r + ∂²/∂z²
        """
        laplacian = np.zeros_like(field)
        
        # ∂²/∂r²
        laplacian[1:-1, :] += (
            (field[2:, :] - 2*field[1:-1, :] + field[:-2, :]) / self.dr**2
        )
        
        # (1/r)∂/∂r
        for i in range(1, self.Nr-1):
            laplacian[i, :] += (
                (field[i+1, :] - field[i-1, :]) / (2 * self.dr * self.r[i])
            )
        
        # ∂²/∂z² (periodic)
        laplacian[:, 1:-1] += (
            (field[:, 2:] - 2*field[:, 1:-1] + field[:, :-2]) / self.dz**2
        )
        # Periodic BC
        laplacian[:, 0] += (field[:, 1] - 2*field[:, 0] + field[:, -1]) / self.dz**2
        laplacian[:, -1] += (field[:, 0] - 2*field[:, -1] + field[:, -2]) / self.dz**2
        
        return laplacian
    
    def _compute_omega_from_psi(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute vorticity from magnetic flux: ω = ∇²ψ
        """
        return self._laplacian(psi)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct 25D observation vector.
        
        Returns:
            obs: [w, gamma, x_o, z_o, psi×8, omega×8, energy, helicity, prev_action, t, dt]
                 (4 + 8 + 8 + 2 + 3 = 25 dimensions)
        """
        # Diagnostics (w, gamma, x_o, z_o)
        if self.use_phase4_api:
            # Use Phase 3 diagnostics (real island monitoring)
            q_profile = 1.0 + self.r  # Approximate q-profile
            
            diag = self.monitor.update(
                self.psi, self.omega, self.t,
                self.r, self.z,
                q_profile=q_profile
            )
            
            # ✅ Store diagnostics for info dict
            self.last_diagnostics = diag
            
            if diag is None:
                # No island detected
                w, gamma, x_o, z_o = 0.0, 0.0, 0.5, 0.0
            else:
                # ✅ Phase 3 diagnostics returns: w, r_s, phase, gamma, sigma
                # NOTE: x_o, z_o not directly provided, use r_s and phase
                w = diag['w']
                gamma = diag['gamma'] if diag['gamma'] is not None else 0.0
                
                # Reconstruct island center from rational surface radius and phase
                r_s = diag['r_s']
                phase = diag['phase']
                
                # Island center in (r, z) coordinates
                x_o = r_s  # Radial position = rational surface
                z_o = phase  # Toroidal angle = phase
        else:
            # Standalone: estimate from psi field
            w, gamma, x_o, z_o = self._estimate_island_diagnostics()
        
        # Sample MHD state (8 psi + 8 omega)
        r_samples = np.linspace(0, self.Nr-1, 8, dtype=int)
        z_sample = self.Nz // 2  # Mid-plane
        
        psi_samples = self.psi[r_samples, z_sample]
        omega_samples = self.omega[r_samples, z_sample]
        
        # Conservation quantities (normalized by grid size)
        energy_raw = self._compute_energy()
        helicity_raw = self._compute_helicity()
        
        # Normalize by number of grid points to keep values O(1)
        grid_size = self.Nr * self.Nz
        energy = energy_raw / grid_size
        mag_helicity = helicity_raw / grid_size
        
        # Context
        prev_action_norm = self.prev_action / self.A_max
        t_norm = self.t / (self.max_steps * self.dt)
        dt_since_reset = self.step_count * self.dt
        
        # Construct observation (25D) - energy_drift removed
        obs = np.array([
            w, gamma, x_o, z_o,  # 4D
            *psi_samples,  # 8D
            *omega_samples,  # 8D
            energy, mag_helicity,  # 2D (normalized, no drift)
            prev_action_norm, t_norm, dt_since_reset  # 3D
        ], dtype=np.float32)
        
        return obs
    
    def _estimate_island_diagnostics(self) -> Tuple[float, float, float, float]:
        """
        Estimate island diagnostics from psi field (standalone mode).
        
        Returns:
            w: Island width
            gamma: Growth rate
            x_o: Island x-center
            z_o: Island z-center
        """
        # Find O-point and X-point (simplified)
        r_s_idx = np.argmin(np.abs(self.r - 0.5))  # Rational surface at r=0.5
        psi_slice = self.psi[r_s_idx, :]
        
        # O-point: local maximum
        o_idx = np.argmax(psi_slice)
        # X-point: local minimum
        x_idx = np.argmin(psi_slice)
        
        # Island width: |psi_O - psi_X|
        w = abs(psi_slice[o_idx] - psi_slice[x_idx])
        
        # Island center
        x_o = self.r[r_s_idx]
        z_o = self.z[o_idx]
        
        # Growth rate: estimate from history
        self.w_history.append(w)
        if len(self.w_history) >= 2:
            gamma = (self.w_history[-1] - self.w_history[-2]) / self.dt
        else:
            gamma = 0.0
        
        self.gamma_history.append(gamma)
        
        return w, gamma, x_o, z_o
    
    def _compute_reward(self, obs: np.ndarray, action: float) -> float:
        """
        Compute reward function.
        
        Baseline reward:
            reward = -w - 0.1*|gamma| - 0.01*|action| + convergence_bonus
        
        Args:
            obs: Current observation
            action: Scaled RMP amplitude
        
        Returns:
            reward: Scalar reward
        """
        w = obs[0]
        gamma = obs[1]
        
        # Main penalties
        width_penalty = -w
        growth_penalty = -0.1 * abs(gamma)
        effort_penalty = -0.01 * abs(action)
        
        # Convergence bonus
        if w < self.convergence_threshold and abs(gamma) < 0.01:
            convergence_bonus = 1.0
        else:
            convergence_bonus = 0.0
        
        reward = width_penalty + growth_penalty + effort_penalty + convergence_bonus
        
        return float(reward)
    
    def _check_done(self, obs: np.ndarray) -> bool:
        """
        Check episode termination conditions.
        
        Returns:
            done: True if episode should terminate
        """
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # Numerical instability check (NaN/Inf in observation)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            return True
        
        # Early termination for MHD field overflow
        # Physical basis: Solver designed for |psi| < 10, |omega| < 100
        # Beyond these values, numerical scheme violates CFL condition
        # This is NOT a bug fix - it's acknowledging solver's valid operating range
        # Similar to disruption detection in real tokamaks
        psi_max = np.max(np.abs(self.psi))
        omega_max = np.max(np.abs(self.omega))
        
        if psi_max > 10.0:
            return True  # psi field too large (solver unstable)
        
        if omega_max > 100.0:
            return True  # vorticity too large (solver unstable)
        
        return False
    
    def _compute_energy(self) -> float:
        """
        Compute total MHD energy.
        
        E = ∫ (|∇ψ|² + |ω|²) dV
        """
        grad_psi_r = np.gradient(self.psi, self.dr, axis=0)
        grad_psi_z = np.gradient(self.psi, self.dz, axis=1)
        
        energy = np.sum(grad_psi_r**2 + grad_psi_z**2 + self.omega**2)
        return float(energy)
    
    def _compute_helicity(self) -> float:
        """
        Compute magnetic helicity.
        
        K = ∫ A·B dV ≈ ∫ ψ·ω dV
        """
        helicity = np.sum(self.psi * self.omega)
        return float(helicity)
    
    def render(self, mode: str = 'human'):
        """
        Render environment state (optional).
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
        """
        if mode == 'human':
            # Text output
            obs = self._get_observation()
            print(f"Step {self.step_count}: w={obs[0]:.4f}, gamma={obs[1]:.4f}, "
                  f"energy={obs[8]:.2e}, action={self.prev_action:.3f}")
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")
    
    def close(self):
        """Clean up environment resources."""
        pass


# Expose in module API
__all__ = ['MHDTearingControlEnv']
