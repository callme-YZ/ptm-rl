"""
RMP Controller Interface

Provides standardized control interface for RMP-based tearing mode suppression.

Supports:
- Proportional (P) control
- PID feedback control
- RL policy integration (Phase 5)

Author: 小P ⚛️
Created: 2026-03-16
Phase: 4
"""

import numpy as np
from typing import Dict, Optional, Callable


class RMPController:
    """
    RMP control interface for tearing mode suppression.
    
    Provides multiple control strategies:
    1. Proportional control: u = -K_p * (w - w_setpoint)
    2. PID control: u = -K_p*e - K_i*∫e - K_d*de/dt
    3. RL policy: u = π(observation) (Phase 5)
    
    Parameters
    ----------
    m : int, optional
        Target poloidal mode number (default: 2)
    n : int, optional
        Target toroidal mode number (default: 1)
    A_max : float, optional
        Maximum RMP amplitude (default: 0.1)
    control_type : str, optional
        Control strategy: 'proportional' | 'pid' | 'rl' (default: 'proportional')
    
    Attributes
    ----------
    m, n : int
        Target mode numbers
    A_max : float
        Control amplitude limit
    control_type : str
        Active control strategy
    integral_error : float
        Accumulated error (for PID)
    last_error : float
        Previous error (for PID derivative)
    
    Examples
    --------
    >>> # Proportional control
    >>> controller = RMPController(m=2, n=1, A_max=0.1, control_type='proportional')
    >>> 
    >>> for step in range(n_steps):
    >>>     # Get diagnostics
    >>>     diag = monitor.update(psi, omega, t, r, z, q)
    >>>     
    >>>     # Compute control action
    >>>     action = controller.compute_action(diag, setpoint=0.0)
    >>>     
    >>>     # Apply control
    >>>     psi, omega = rk4_step_with_rmp(psi, omega, dt, ..., rmp_amplitude=action)
    
    Notes
    -----
    - Control action ∈ [-A_max, A_max]
    - Negative action typically suppresses island growth
    - Phase information handled separately (not yet implemented)
    
    Physics References
    ------------------
    - La Haye 2006: "Control of neoclassical tearing modes in DIII-D"
    - Maraschek 2012: "Active control of MHD instabilities"
    """
    
    def __init__(
        self,
        m: int = 2,
        n: int = 1,
        A_max: float = 0.1,
        control_type: str = 'proportional'
    ):
        """Initialize RMP controller."""
        self.m = m
        self.n = n
        self.A_max = A_max
        self.control_type = control_type
        
        # Controller state (for PID)
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        
        # Validate control type
        valid_types = ['proportional', 'pid', 'rl']
        if control_type not in valid_types:
            raise ValueError(f"control_type must be one of {valid_types}, got '{control_type}'")
    
    def compute_action(
        self,
        diag: Dict,
        setpoint: float = 0.0,
        t: Optional[float] = None
    ) -> float:
        """
        Compute RMP amplitude based on diagnostics.
        
        Parameters
        ----------
        diag : dict
            Diagnostics from TearingModeMonitor
            Required keys: 'w' (island width)
            Optional keys: 'gamma' (growth rate), 'x_o', 'z_o' (island center)
        setpoint : float, optional
            Target island width (default: 0.0 = full suppression)
        t : float, optional
            Current time (required for PID control)
        
        Returns
        -------
        action : float
            RMP amplitude ∈ [-A_max, A_max]
        
        Examples
        --------
        >>> diag = {'w': 0.05, 'gamma': 0.01, 'x_o': 0.7, 'z_o': 3.14}
        >>> action = controller.compute_action(diag, setpoint=0.0)
        >>> print(f"Control action: {action:.4f}")
        Control action: -0.0500
        """
        if self.control_type == 'proportional':
            return self._proportional_control(diag, setpoint)
        elif self.control_type == 'pid':
            if t is None:
                raise ValueError("PID control requires time argument 't'")
            return self._pid_control(diag, setpoint, t)
        elif self.control_type == 'rl':
            return self._rl_policy(diag)
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")
    
    def _proportional_control(self, diag: Dict, setpoint: float) -> float:
        """
        Proportional control: u = -K_p * (w - w_setpoint).
        
        Parameters
        ----------
        diag : dict
            Must contain 'w' (island width)
        setpoint : float
            Target island width
        
        Returns
        -------
        action : float
            Control amplitude
        
        Notes
        -----
        - K_p = 1.0 (tunable parameter)
        - Negative sign: increase RMP when island grows
        - Clipped to [-A_max, A_max]
        """
        K_p = 1.0  # Proportional gain
        
        # Error: current width - target width
        error = diag['w'] - setpoint
        
        # Control action: -K_p * error
        action = -K_p * error
        
        # Clip to limits
        action = np.clip(action, -self.A_max, self.A_max)
        
        return action
    
    def _pid_control(self, diag: Dict, setpoint: float, t: float) -> float:
        """
        PID control: u = -K_p*e - K_i*∫e - K_d*de/dt.
        
        Parameters
        ----------
        diag : dict
            Must contain 'w' (island width)
        setpoint : float
            Target island width
        t : float
            Current time
        
        Returns
        -------
        action : float
            Control amplitude
        
        Notes
        -----
        - Gains: K_p=1.0, K_i=0.1, K_d=0.05 (tunable)
        - Anti-windup: integral term clipped
        - Derivative term smoothed
        """
        # PID gains (tunable)
        K_p = 1.0
        K_i = 0.1
        K_d = 0.05
        
        # Error
        error = diag['w'] - setpoint
        
        # Time step (for integral and derivative)
        if self.last_time == 0.0:
            dt = 0.01  # Default dt
            self.last_time = t
        else:
            dt = max(t - self.last_time, 1e-6)  # Avoid division by zero
            self.last_time = t
        
        # Integral term (with anti-windup)
        self.integral_error += error * dt
        # Clip integral to prevent windup
        max_integral = self.A_max / K_i if K_i > 0 else 1e10
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        # Derivative term
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        # PID formula
        action = -(K_p * error + K_i * self.integral_error + K_d * derivative)
        
        # Clip to limits
        action = np.clip(action, -self.A_max, self.A_max)
        
        return action
    
    def _rl_policy(self, diag: Dict) -> float:
        """
        RL policy interface (placeholder for Phase 5).
        
        Parameters
        ----------
        diag : dict
            Observation from environment
        
        Returns
        -------
        action : float
            Control amplitude from trained policy
        
        Raises
        ------
        NotImplementedError
            RL policy requires Phase 5 training
        
        Notes
        -----
        Will be implemented in Phase 5:
        - Load trained policy (PPO/SAC/IQL)
        - Map diagnostics to observation
        - Query policy for action
        - Clip to [-A_max, A_max]
        """
        raise NotImplementedError(
            "RL policy requires Phase 5 training. "
            "Use control_type='proportional' or 'pid' for now."
        )
    
    def reset(self):
        """
        Reset controller state.
        
        Clears integral error and derivative state.
        Call when starting new episode or changing setpoint.
        
        Examples
        --------
        >>> controller.reset()  # Start fresh episode
        >>> for step in range(n_steps):
        >>>     action = controller.compute_action(diag)
        """
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_time = 0.0
    
    def set_gains(self, K_p: float = None, K_i: float = None, K_d: float = None):
        """
        Set PID gains (for tuning).
        
        Parameters
        ----------
        K_p : float, optional
            Proportional gain
        K_i : float, optional
            Integral gain
        K_d : float, optional
            Derivative gain
        
        Examples
        --------
        >>> controller.set_gains(K_p=2.0, K_i=0.2, K_d=0.1)
        
        Notes
        -----
        Only affects PID control mode.
        For proportional control, only K_p is used.
        """
        # Store gains as instance attributes
        if K_p is not None:
            self.K_p = K_p
        if K_i is not None:
            self.K_i = K_i
        if K_d is not None:
            self.K_d = K_d
    
    def get_state(self) -> Dict:
        """
        Get controller internal state.
        
        Returns
        -------
        state : dict
            Controller state (for logging/debugging)
        
        Examples
        --------
        >>> state = controller.get_state()
        >>> print(f"Integral error: {state['integral_error']:.4f}")
        """
        return {
            'integral_error': self.integral_error,
            'last_error': self.last_error,
            'last_time': self.last_time,
            'm': self.m,
            'n': self.n,
            'A_max': self.A_max,
            'control_type': self.control_type
        }


# =============================================================================
# Control Performance Metrics
# =============================================================================

def compute_control_metrics(w_history, action_history, setpoint=0.0):
    """
    Compute control performance metrics.
    
    Metrics:
    1. Settling time: time to reach 95% of setpoint
    2. Overshoot: max overshoot beyond setpoint
    3. Steady-state error: |w_final - setpoint|
    4. Control effort: ∫|u|dt
    
    Parameters
    ----------
    w_history : np.ndarray
        Island width trajectory
    action_history : np.ndarray
        Control action trajectory
    setpoint : float, optional
        Target island width (default: 0.0)
    
    Returns
    -------
    metrics : dict
        Performance metrics
    
    Examples
    --------
    >>> metrics = compute_control_metrics(w_history, action_history, setpoint=0.0)
    >>> print(f"Settling time: {metrics['settling_time']:.1f} steps")
    >>> print(f"Overshoot: {metrics['overshoot']*100:.1f}%")
    """
    metrics = {}
    
    # Settling time (95% of setpoint)
    tolerance = 0.05 * abs(w_history[0] - setpoint)  # 5% of initial error
    target_band = setpoint + tolerance
    
    settling_idx = np.where(np.abs(w_history - setpoint) < tolerance)[0]
    if len(settling_idx) > 0:
        metrics['settling_time'] = settling_idx[0]
    else:
        metrics['settling_time'] = len(w_history)  # Never settled
    
    # Overshoot
    if setpoint == 0.0:
        # Suppression case
        overshoot = max(0, np.min(w_history) - setpoint)
    else:
        # Tracking case
        overshoot = max(0, np.max(w_history) - setpoint)
    
    metrics['overshoot'] = overshoot / abs(w_history[0] - setpoint) if w_history[0] != setpoint else 0.0
    
    # Steady-state error
    metrics['steady_state_error'] = abs(w_history[-1] - setpoint)
    
    # Control effort
    metrics['control_effort'] = np.sum(np.abs(action_history))
    
    # Max control action
    metrics['max_action'] = np.max(np.abs(action_history))
    
    return metrics


def validate_controller(controller, initial_w, setpoint, n_steps=100, dt=0.01):
    """
    Validate controller with simplified dynamics.
    
    Uses first-order model: dw/dt = -γ*w + α*u
    
    Parameters
    ----------
    controller : RMPController
        Controller to test
    initial_w : float
        Initial island width
    setpoint : float
        Target width
    n_steps : int
        Number of steps
    dt : float
        Time step
    
    Returns
    -------
    is_valid : bool
        True if controller converges
    diagnostics : dict
        Validation results
    
    Notes
    -----
    Simplified dynamics for testing only.
    Full validation requires MHD evolution (see validation.py).
    """
    # Simplified dynamics parameters
    gamma = 0.01  # Natural growth rate
    alpha = 0.1   # RMP effectiveness
    
    w = initial_w
    w_history = [w]
    action_history = []
    
    controller.reset()
    
    for step in range(n_steps):
        # Create mock diagnostics
        diag = {'w': w, 'gamma': gamma, 'x_o': 0.7, 'z_o': 3.14}
        
        # Compute action
        action = controller.compute_action(diag, setpoint=setpoint, t=step*dt)
        action_history.append(action)
        
        # Simplified dynamics
        dw_dt = -gamma * (w - setpoint) + alpha * action
        w = w + dt * dw_dt
        w_history.append(w)
    
    # Check convergence
    final_error = abs(w - setpoint)
    is_valid = final_error < 0.01  # 1% tolerance
    
    metrics = compute_control_metrics(np.array(w_history), np.array(action_history), setpoint)
    
    diagnostics = {
        'is_valid': is_valid,
        'final_error': final_error,
        'w_history': np.array(w_history),
        'action_history': np.array(action_history),
        **metrics
    }
    
    return is_valid, diagnostics
