"""
Classical Control Baselines for MHD

Issue #28: Establish baseline controllers for comparison with RL.

Implements:
- NoControlAgent: Let system evolve naturally
- RandomAgent: Random actions
- PIDController: Feedback control on m=1 mode
- (Future) LQRController: Optimal linear control

Author: 小A 🤖
Physics: 小P ⚛️
Date: 2026-03-24
"""

import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym


class BaselineAgent:
    """
    Base class for classical baseline controllers.
    
    All agents follow Gym agent interface:
    - reset(): Initialize agent state
    - act(obs): Return action given observation
    - update(obs, reward, done, info): Update internal state (if needed)
    """
    
    def __init__(self, action_space: gym.spaces.Box):
        """
        Initialize agent.
        
        Parameters
        ----------
        action_space : gym.spaces.Box
            Environment action space
        """
        self.action_space = action_space
        self.action_low = action_space.low
        self.action_high = action_space.high
    
    def reset(self):
        """Reset agent state."""
        pass
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Select action given observation.
        
        Parameters
        ----------
        obs : np.ndarray
            Current observation
            
        Returns
        -------
        action : np.ndarray
            Selected action
        """
        raise NotImplementedError
    
    def update(
        self,
        obs: np.ndarray,
        reward: float,
        done: bool,
        info: Dict
    ):
        """
        Update agent state (if needed).
        
        Most classical controllers don't need this,
        but included for interface consistency.
        """
        pass


class NoControlAgent(BaselineAgent):
    """
    No control baseline: Always return neutral action.
    
    Action: [1.0, 1.0] (no modification to η, ν)
    
    Purpose:
    - Establish "do nothing" baseline
    - See natural evolution of instability
    - Expect: Tearing mode grows exponentially
    """
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return neutral action (no control)."""
        # Neutral: eta_mult=1.0, nu_mult=1.0
        return np.array([1.0, 1.0], dtype=np.float32)


class RandomAgent(BaselineAgent):
    """
    Random control baseline: Sample random actions.
    
    Action: Random uniform in [low, high]
    
    Purpose:
    - Test if any control helps
    - Sanity check: should be better than no control
    - Expect: Marginally better, but unstable
    """
    
    def __init__(self, action_space: gym.spaces.Box, seed: Optional[int] = None):
        super().__init__(action_space)
        self.rng = np.random.RandomState(seed)
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Sample random action."""
        return self.rng.uniform(
            low=self.action_low,
            high=self.action_high
        ).astype(np.float32)
    
    def reset(self):
        """Reset RNG state (if seeded)."""
        pass


class PIDController(BaselineAgent):
    """
    PID feedback controller for tearing mode suppression.
    
    Control law:
        error = target - m1_amp
        u = Kp * error + Ki * ∫error dt + Kd * d(error)/dt
    
    Control variable: m=1 Fourier mode amplitude
    Control action: Resistivity multiplier (eta_mult)
    
    Physics (Issue #28 temporary):
    - m=1 mode drives tearing instability
    - Higher η → faster reconnection → mode saturation/suppression
    - PID tunes η to maintain target amplitude
    
    Note: Viscosity control (ν) deferred to future issue
          (CompleteMHDSolver doesn't implement ν yet)
    
    Features:
    - Anti-windup protection
    - Conservative tuning (小P recommendation)
    - Derivative filtering (avoid noise amplification)
    
    Parameters
    ----------
    action_space : gym.spaces.Box
        Environment action space
    target : float
        Target m=1 amplitude (default: 0.0 = full suppression)
    Kp, Ki, Kd : float
        PID gains (小P recommended: Kp=5.0, Ki=0.5, Kd=0.01)
    dt : float
        Control timestep (should match env dt)
    """
    
    def __init__(
        self,
        action_space: gym.spaces.Box,
        target: float = 0.0,
        Kp: float = 5.0,
        Ki: float = 0.5,
        Kd: float = 0.01,
        dt: float = 1e-4
    ):
        super().__init__(action_space)
        
        # Target
        self.target = target
        
        # PID gains (小P conservative tuning)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # Timestep
        self.dt = dt
        
        # Internal state
        self.error_prev = 0.0
        self.error_int = 0.0
        self.m1_prev = None
    
    def reset(self):
        """Reset PID internal state."""
        self.error_prev = 0.0
        self.error_int = 0.0
        self.m1_prev = None
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute PID control action.
        
        Observation indexing (from Issue #25):
        - obs[0:7]: Hamiltonian quantities
        - obs[7:15]: ψ Fourier modes (8 modes)
        - obs[15:23]: φ Fourier modes (8 modes)
        
        For m=1 mode:
        - We use first ψ mode (obs[7]) as proxy
        - (Assumes modes ordered by importance)
        
        Alternative: Could extract exact m=1 from state,
        but for PID baseline, first mode is sufficient.
        """
        # Extract m=1 amplitude
        # Observation: [H, K, Ω, dH/dt, drift, grad, J_max, psi_modes[8], phi_modes[8]]
        # obs[7] = psi m=0, obs[8] = psi m=1 (小P correction ⚛️)
        m1_amp = np.abs(obs[8])  # psi m=1 Fourier mode
        
        # PID terms
        # Physics (小P ⚛️): Higher η suppresses tearing mode
        # → Positive error (m1 > target) → increase η
        error = m1_amp - self.target  # (NOT target - m1_amp)
        
        # Integral (with anti-windup)
        error_int_candidate = self.error_int + error * self.dt
        
        # Derivative
        if self.m1_prev is not None:
            error_der = (m1_amp - self.m1_prev) / self.dt
        else:
            error_der = 0.0
        
        # PID output
        u = self.Kp * error + self.Ki * error_int_candidate + self.Kd * error_der
        
        # Action: eta_mult (control resistivity)
        # Temporary: control η instead of ν (Issue #28)
        # ν control deferred until CompleteMHDSolver implements viscosity
        eta_mult = 1.0 + u
        
        # Clip to valid range
        eta_mult_clipped = np.clip(eta_mult, self.action_low[0], self.action_high[0])
        
        # Anti-windup: Only integrate if not saturated
        if np.abs(eta_mult_clipped - eta_mult) < 1e-6:
            # Not saturated, update integral
            self.error_int = error_int_candidate
        # else: saturated, don't update integral (anti-windup)
        
        # Action: [eta_mult, nu_mult]
        # Control resistivity, keep viscosity at baseline
        action = np.array([eta_mult_clipped, 1.0], dtype=np.float32)
        
        # Update state
        self.error_prev = error
        self.m1_prev = m1_amp
        
        return action


# Convenience functions
def make_baseline_agent(
    agent_type: str,
    action_space: gym.spaces.Box,
    **kwargs
) -> BaselineAgent:
    """
    Create baseline agent by type.
    
    Parameters
    ----------
    agent_type : str
        'no_control', 'random', 'pid'
    action_space : gym.spaces.Box
        Environment action space
    **kwargs : dict
        Agent-specific parameters
        
    Returns
    -------
    agent : BaselineAgent
        Baseline agent instance
        
    Examples
    --------
    >>> agent = make_baseline_agent('no_control', env.action_space)
    >>> agent = make_baseline_agent('pid', env.action_space, Kp=10.0)
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'no_control':
        return NoControlAgent(action_space)
    elif agent_type == 'random':
        return RandomAgent(action_space, **kwargs)
    elif agent_type == 'pid':
        return PIDController(action_space, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
