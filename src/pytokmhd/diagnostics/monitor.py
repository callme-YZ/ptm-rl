"""
Real-time Tearing Mode Monitor

Track tearing mode diagnostics during MHD evolution.
"""

import numpy as np
from .magnetic_island import compute_island_width
from .growth_rate import compute_growth_rate


class TearingModeMonitor:
    """
    Real-time tearing mode diagnostics
    
    Parameters
    ----------
    m : int, optional
        Poloidal mode number (default: 2)
    n : int, optional
        Toroidal mode number (default: 1)
    track_every : int, optional
        Track diagnostics every N steps (default: 10)
    gamma_window : int, optional
        Number of points for growth rate calculation (default: 50)
    
    Attributes
    ----------
    w_history : list
        Island width time series
    r_s_history : list
        Rational surface radius time series
    phase_history : list
        Island phase time series
    gamma_history : list
        Growth rate time series
    t_history : list
        Time stamps
    
    Examples
    --------
    >>> monitor = TearingModeMonitor(m=2, n=1, track_every=5)
    >>> 
    >>> for step in range(n_steps):
    >>>     psi, omega = mhd_step(psi, omega, dt)
    >>>     
    >>>     diag = monitor.update(psi, omega, step*dt, r, z, q_profile)
    >>>     
    >>>     if diag is not None and diag['w'] > w_threshold:
    >>>         print(f"Warning: Island width {diag['w']:.4f} exceeded threshold")
    """
    
    def __init__(self, m=2, n=1, track_every=10, gamma_window=50):
        self.m = m
        self.n = n
        self.track_every = track_every
        self.gamma_window = gamma_window
        
        # History arrays
        self.w_history = []
        self.r_s_history = []
        self.phase_history = []
        self.gamma_history = []
        self.sigma_history = []
        self.t_history = []
        
        # Internal counter
        self._step_count = 0
    
    def update(self, psi, omega, t, r, z, q_profile):
        """
        Update diagnostics at current timestep
        
        Parameters
        ----------
        psi : ndarray (Nr, Nz)
            Poloidal flux
        omega : ndarray (Nr, Nz)
            Vorticity
        t : float
            Current time
        r : ndarray (Nr,)
            Radial coordinates
        z : ndarray (Nz,)
            Vertical coordinates
        q_profile : ndarray (Nr,)
            Safety factor profile
        
        Returns
        -------
        diagnostics : dict or None
            Dictionary with keys 'w', 'r_s', 'phase', 'gamma', 't'
            Returns None if not tracking this step
        """
        self._step_count += 1
        
        # Only track every N steps
        if self._step_count % self.track_every != 0:
            return None
        
        # Compute island width
        w, r_s, phase = compute_island_width(psi, r, z, q_profile, 
                                             self.m, self.n)
        
        # Store in history
        self.w_history.append(w)
        self.r_s_history.append(r_s)
        self.phase_history.append(phase)
        self.t_history.append(t)
        
        # Compute growth rate if enough history
        gamma = None
        sigma = None
        
        if len(self.w_history) >= self.gamma_window:
            gamma, sigma = compute_growth_rate(
                self.w_history[-self.gamma_window:],
                self.t_history[-self.gamma_window:],
                transient_fraction=0.0
            )
            
            if not np.isnan(gamma):
                self.gamma_history.append(gamma)
                self.sigma_history.append(sigma)
            else:
                gamma = None
                sigma = None
        
        # Return diagnostics
        return {
            'w': w,
            'r_s': r_s,
            'phase': phase,
            'gamma': gamma,
            'sigma': sigma,
            't': t,
            'step': self._step_count
        }
    
    def get_latest_gamma(self, n_avg=10):
        """
        Get average growth rate over last n_avg measurements
        
        Parameters
        ----------
        n_avg : int, optional
            Number of recent measurements to average (default: 10)
        
        Returns
        -------
        gamma_avg : float
            Average growth rate
        sigma_avg : float
            Average uncertainty
        """
        if len(self.gamma_history) == 0:
            return np.nan, np.nan
        
        recent_gamma = self.gamma_history[-n_avg:]
        recent_sigma = self.sigma_history[-n_avg:]
        
        gamma_avg = np.mean(recent_gamma)
        sigma_avg = np.sqrt(np.sum(np.array(recent_sigma)**2)) / len(recent_sigma)
        
        return gamma_avg, sigma_avg
    
    def is_unstable(self, threshold=0.0):
        """
        Check if mode is unstable (γ > threshold)
        
        Parameters
        ----------
        threshold : float, optional
            Growth rate threshold (default: 0.0)
        
        Returns
        -------
        unstable : bool
            True if mode is unstable
        """
        gamma_avg, sigma = self.get_latest_gamma()
        
        if np.isnan(gamma_avg):
            return False
        
        return gamma_avg > threshold
    
    def reset(self):
        """Reset all history"""
        self.w_history = []
        self.r_s_history = []
        self.phase_history = []
        self.gamma_history = []
        self.sigma_history = []
        self.t_history = []
        self._step_count = 0
    
    def get_summary(self):
        """
        Get summary statistics
        
        Returns
        -------
        summary : dict
            Summary statistics
        """
        if len(self.w_history) == 0:
            return {
                'n_samples': 0,
                'w_current': np.nan,
                'w_max': np.nan,
                'gamma_avg': np.nan,
                't_final': np.nan
            }
        
        gamma_avg, sigma = self.get_latest_gamma()
        
        return {
            'n_samples': len(self.w_history),
            'w_current': self.w_history[-1],
            'w_max': max(self.w_history),
            'gamma_avg': gamma_avg,
            'gamma_sigma': sigma,
            't_final': self.t_history[-1],
            'mode': f'm={self.m}, n={self.n}',
            'r_s': self.r_s_history[-1] if len(self.r_s_history) > 0 else np.nan
        }
