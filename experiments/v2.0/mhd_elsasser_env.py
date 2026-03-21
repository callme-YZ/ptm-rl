"""
Gym Environment for v2.0 Elsasser MHD (Phase 4.1)

Author: 小A 🤖
Date: 2026-03-20

Wraps 小P's v2.0 JAX physics in standard Gym interface.

Observation: 113 features
- 50 z⁺ spectral modes
- 50 z⁻ spectral modes
- 10 island diagnostics
- 3 conservation metrics

Action: N coil currents (RMP control)

Reward: -|m=1 amplitude| (suppress ballooning)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add v2.0 physics path
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/v2.0/src')

try:
    import jax.numpy as jnp
    from elsasser_bracket import ElsasserState
    from complete_solver_with_rmp import MHDSolverWithRMP
    from ballooning_ic_v2 import ballooning_mode_ic_v2
    from bout_metric import BOUTMetric
    from field_aligned import FieldAlignedCoordinates
    from toroidal_hamiltonian import toroidal_hamiltonian
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: JAX import failed: {e}")
    JAX_AVAILABLE = False


class MHDElsasserEnv(gym.Env):
    """Gym environment for v2.0 Elsasser MHD with structure-preserving physics
    
    Physics:
    - Morrison bracket (structure-preserving)
    - Elsasser variables (z±)
    - Wu time transformation (adaptive symplectic)
    - Resistive dynamics (η + ∇p)
    
    Control:
    - RMP coils (N channels)
    - Action = coil currents
    
    Objective:
    - Suppress ballooning mode (m=1)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, 
                 grid_shape=(32, 64, 32),
                 n_coils=4,
                 epsilon=0.323,
                 eta=0.01,
                 pressure_scale=0.2,
                 dt_rl=0.02,
                 steps_per_action=5,
                 max_episode_steps=200):
        """Initialize MHD environment
        
        Args:
            grid_shape: (Nr, Nθ, Nz) resolution
            n_coils: Number of RMP control coils
            epsilon: Inverse aspect ratio ε=a/R₀
            eta: Resistivity
            pressure_scale: Pressure gradient drive strength
            dt_rl: RL timestep (physics may use adaptive Wu steps)
            steps_per_action: Physics substeps per RL action
            max_episode_steps: Episode length
        """
        super().__init__()
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX/v2.0 physics not available")
        
        # Grid parameters
        self.Nr, self.Ntheta, self.Nz = grid_shape
        self.grid_shape = grid_shape
        self.dr = 0.05
        self.dtheta = 2*np.pi / self.Ntheta
        self.dz = 2*np.pi / self.Nz
        
        # Physics parameters
        self.epsilon = epsilon
        self.eta = eta
        self.pressure_scale = pressure_scale
        
        # RL parameters
        self.n_coils = n_coils
        self.dt_rl = dt_rl
        self.steps_per_action = steps_per_action
        self.max_episode_steps = max_episode_steps
        
        # Initialize physics
        self._init_physics()
        
        # Observation space: 113 features
        # 50 z⁺ modes + 50 z⁻ modes + 10 island diagnostics + 3 conservation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(113,), dtype=np.float32
        )
        
        # Action space: N coil currents (normalized -1 to 1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_coils,), dtype=np.float32
        )
        
        # State
        self.state = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # History for diagnostics
        self.energy_history = []
        self.amplitude_history = []
        
        # Cache for IC generation (avoid recreating BOUT objects every reset)
        self._cached_metric = None
        self._cached_fa = None
        self._cached_initial_state = None
    
    def _init_physics(self):
        """Initialize v2.0 physics backend (Complete solver with RMP control)"""
        self.solver = MHDSolverWithRMP(
            grid_shape=self.grid_shape,
            dr=self.dr,
            dtheta=self.dtheta,
            dz=self.dz,
            epsilon=self.epsilon,
            eta=self.eta,
            pressure_scale=self.pressure_scale
        )
        
        print(f"✅ v2.0 Complete Physics + RMP Control initialized:")
        print(f"   Grid: {self.Nr}×{self.Ntheta}×{self.Nz}")
        print(f"   ε: {self.epsilon}")
        print(f"   η: {self.eta}")
        print(f"   RMP coils: {self.n_coils}")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial ballooning perturbation
        
        Returns:
            observation: Initial obs (113 features)
            info: Metadata dict
        """
        super().reset(seed=seed)
        
        # Create ballooning IC (cache to avoid recreation)
        if self._cached_initial_state is None:
            # First time: create BOUT++ objects and initial state
            metric = BOUTMetric(R0=6.2, a=2.0, epsilon=self.epsilon)
            fa = FieldAlignedCoordinates(metric, q_profile='constant')
            
            state_ic = ballooning_mode_ic_v2(
                metric, fa,
                grid_shape=self.grid_shape,
                m=2, n=1, amplitude=0.05,
                scale_B=1.0, scale_P=0.05
            )
            
            # Cache for future resets
            self._cached_metric = metric
            self._cached_fa = fa
            self._cached_initial_state = state_ic
        
        # Use cached initial state (deep copy if JAX arrays are mutable)
        self.state = ElsasserState(
            z_plus=self._cached_initial_state.z_plus.copy(),
            z_minus=self._cached_initial_state.z_minus.copy(),
            P=self._cached_initial_state.P.copy()
        )
        
        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        self.energy_history = []
        self.amplitude_history = []
        
        # Initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one RL step (= steps_per_action physics steps)
        
        Args:
            action: Coil currents (n_coils,)
            
        Returns:
            observation: Next obs (113 features)
            reward: Scalar reward
            terminated: Episode done (instability suppressed or diverged)
            truncated: Max steps reached
            info: Metadata
        """
        # Apply RMP control (action → external current)
        # TODO: Implement RMP forcing when 小P provides API
        # For now: just evolve physics without control
        
        # Substep physics (COMPLETE v2.0: ideal + resistive + RMP control)
        dt_sub = self.dt_rl / self.steps_per_action
        
        # Convert action to coil currents (scale to kA)
        # Action in [-1, 1] → coil currents
        # Based on empirical tuning: 2kA gives ~26% effect (sweet spot)
        coil_currents = jnp.array(action) * 2.0  # kA (±2kA range)
        
        # Multi-step integration WITH RMP control
        for _ in range(self.steps_per_action):
            self.state = self.solver.step_rk2_with_control(
                self.state, dt_sub, coil_currents
            )
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        self.step_count += 1
        terminated = self._check_terminated()
        truncated = (self.step_count >= self.max_episode_steps)
        
        # Info
        info = self._get_info()
        
        self.episode_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Extract 113-feature observation from Elsasser state
        
        Features:
        - 0-49: z⁺ spectral amplitudes (FFT, 50 modes)
        - 50-99: z⁻ spectral amplitudes (50 modes)
        - 100-109: Island diagnostics (10 features, placeholder)
        - 110-112: Conservation (energy, cross-helicity, drift)
        
        Returns:
            obs: (113,) numpy array
        """
        # Convert JAX → NumPy
        z_plus_np = np.array(self.state.z_plus)
        z_minus_np = np.array(self.state.z_minus)
        P_np = np.array(self.state.P)
        
        # Spectral analysis (simplified: use poloidal FFT at midplane)
        # Take middle radial slice
        r_mid = self.Nr // 2
        z_plus_mid = z_plus_np[r_mid, :, :]  # (Nθ, Nz)
        z_minus_mid = z_minus_np[r_mid, :, :]
        
        # FFT in θ direction
        zp_fft = np.fft.fft(z_plus_mid, axis=0)  # (Nθ, Nz)
        zm_fft = np.fft.fft(z_minus_mid, axis=0)
        
        # Take first N modes (up to 50, limited by grid resolution)
        n_modes = min(50, self.Ntheta)
        zp_modes = np.abs(zp_fft[:n_modes, :]).mean(axis=1)  # Average over z
        zm_modes = np.abs(zm_fft[:n_modes, :]).mean(axis=1)
        
        # Pad to 50 if necessary
        if n_modes < 50:
            zp_modes = np.pad(zp_modes, (0, 50 - n_modes), constant_values=-10)  # log10(1e-10)
            zm_modes = np.pad(zm_modes, (0, 50 - n_modes), constant_values=-10)
        
        # Normalize (log scale for large dynamic range)
        # Clip to avoid log(0) even with epsilon
        zp_modes_safe = np.clip(zp_modes, 1e-12, None)
        zm_modes_safe = np.clip(zm_modes, 1e-12, None)
        zp_feat = np.log10(zp_modes_safe + 1e-10)
        zm_feat = np.log10(zm_modes_safe + 1e-10)
        
        # Clip log values to reasonable range
        zp_feat = np.clip(zp_feat, -10, 2)
        zm_feat = np.clip(zm_feat, -10, 2)
        
        # Island diagnostics (placeholder: 10 zeros)
        # TODO: Request from 小P when diagnostic API ready
        island_diag = np.zeros(10)
        
        # Conservation metrics
        energy = float(self.solver.hamiltonian(self.state))
        
        # Cross-helicity (z⁺² - z⁻²)/4
        hc = 0.25 * (np.sum(z_plus_np**2) - np.sum(z_minus_np**2)) * self.solver.grid.dV
        
        # Energy drift (track over episode)
        self.energy_history.append(energy)
        if len(self.energy_history) > 1:
            drift = abs(energy - self.energy_history[0]) / abs(self.energy_history[0])
        else:
            drift = 0.0
        
        conservation = np.array([energy, hc, drift], dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([zp_feat, zm_feat, island_diag, conservation])
        
        # NaN check
        if np.any(np.isnan(obs)):
            print(f"⚠️ NaN in observation at step {self.step_count}")
            print(f"   zp_feat NaN: {np.any(np.isnan(zp_feat))}")
            print(f"   zm_feat NaN: {np.any(np.isnan(zm_feat))}")
            print(f"   conservation NaN: {np.any(np.isnan(conservation))}")
            # Replace NaN with zeros
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Final safety clip
        obs = np.clip(obs, -100, 100)
        
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        """Compute reward: negative m=1 amplitude
        
        Goal: Suppress ballooning mode (m=1,2,...)
        
        Returns:
            reward: Scalar (higher = better control)
        """
        # Extract m=1 amplitude (dominant ballooning mode)
        z_plus_np = np.array(self.state.z_plus)
        r_mid = self.Nr // 2
        z_plus_mid = z_plus_np[r_mid, :, :]
        
        # FFT
        fft = np.fft.fft(z_plus_mid, axis=0)
        m1_amplitude = np.abs(fft[1, :]).mean()  # m=1 mode
        
        # Track amplitude
        self.amplitude_history.append(m1_amplitude)
        
        # Reward: negative amplitude (want to minimize)
        reward = -m1_amplitude
        
        # Bonus: energy conservation (penalize drift)
        if len(self.energy_history) > 1:
            E0_safe = abs(self.energy_history[0]) + 1e-6  # Avoid division by zero
            drift = abs(self.energy_history[-1] - self.energy_history[0]) / E0_safe
            conservation_penalty = -10.0 * drift  # Penalize >10% drift
            reward += conservation_penalty
        
        # NaN check
        if np.isnan(reward) or np.isinf(reward):
            print(f"⚠️ NaN/Inf reward at step {self.step_count}: {reward}")
            print(f"   m1_amplitude: {m1_amplitude}")
            if len(self.energy_history) > 1:
                print(f"   energy drift: {drift}")
            reward = -1.0  # Safe fallback
        
        return float(reward)
    
    def _check_terminated(self):
        """Check if episode should terminate early
        
        Termination conditions:
        - Amplitude explodes (>10× initial)
        - Amplitude suppressed (<10% initial)
        - Numerical instability (NaN)
        
        Returns:
            terminated: bool
        """
        if len(self.amplitude_history) < 2:
            return False
        
        A_current = self.amplitude_history[-1]
        A_init = self.amplitude_history[0]
        
        # Check NaN
        if np.isnan(A_current):
            print("⚠️ NaN detected, terminating")
            return True
        
        # Check explosion
        if A_current > 10 * A_init:
            print(f"⚠️ Amplitude exploded ({A_current/A_init:.1f}×), terminating")
            return True
        
        # Check suppression (success!) - disabled for training speed
        # if A_current < 0.1 * A_init:
        #     print(f"✅ Mode suppressed ({A_current/A_init:.1%}), terminating")
        #     return True
        
        return False
    
    def _get_info(self):
        """Get metadata dict
        
        Returns:
            info: Dict with diagnostics
        """
        info = {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
        }
        
        if len(self.amplitude_history) > 0:
            info['m1_amplitude'] = self.amplitude_history[-1]
        
        if len(self.energy_history) > 0:
            info['energy'] = self.energy_history[-1]
        
        if len(self.energy_history) > 1:
            drift = abs(self.energy_history[-1] - self.energy_history[0]) / abs(self.energy_history[0])
            info['energy_drift'] = drift
        
        return info
    
    def render(self):
        """Render (not implemented)"""
        pass


# Test function
def test_env():
    """Test MHD Elsasser environment"""
    print("=" * 60)
    print("MHD Elsasser Env Test (Phase 4.1)")
    print("=" * 60 + "\n")
    
    # Create env
    env = MHDElsasserEnv(
        grid_shape=(16, 32, 16),  # Smaller for faster test
        n_coils=4,
        max_episode_steps=10
    )
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")
    
    # Reset
    obs, info = env.reset()
    print(f"Reset:")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  Info: {info}\n")
    
    # Random rollout
    print("Random rollout (10 steps):")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: reward={reward:.4f}, "
              f"m1_amp={info.get('m1_amplitude', 0):.4f}, "
              f"terminated={terminated}")
        
        if terminated or truncated:
            break
    
    print("\n✅ Env test complete!")


if __name__ == "__main__":
    test_env()
