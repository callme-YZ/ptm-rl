"""
Rational Surface Locator

Find q = m/n rational surfaces in tokamak equilibria.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def find_rational_surface(q_profile, r_grid, q_target, method='spline'):
    """
    Find radius where q(r) = q_target
    
    Parameters
    ----------
    q_profile : array_like
        Safety factor profile q(r)
    r_grid : array_like
        Radial grid points
    q_target : float
        Target q value (e.g., 2.0 for m=2, n=1)
    method : str, optional
        Interpolation method: 'linear', 'spline', or 'newton'
        Default: 'spline'
    
    Returns
    -------
    r_s : float
        Rational surface radius
    accuracy : float
        Relative accuracy estimate |q(r_s) - q_target| / q_target
    
    Notes
    -----
    For monotonic q-profiles only. Returns NaN if q_target is out of range.
    """
    q_profile = np.asarray(q_profile)
    r_grid = np.asarray(r_grid)
    
    # Check if q_target is in range
    q_min, q_max = q_profile.min(), q_profile.max()
    if q_target < q_min or q_target > q_max:
        return np.nan, np.nan
    
    # Find bracketing interval
    if q_profile[0] < q_profile[-1]:  # q increasing
        idx = np.searchsorted(q_profile, q_target)
    else:  # q decreasing
        idx = len(q_profile) - np.searchsorted(q_profile[::-1], q_target)
    
    # Ensure valid index
    if idx == 0:
        idx = 1
    elif idx >= len(q_profile):
        idx = len(q_profile) - 1
    
    if method == 'linear':
        # Linear interpolation
        r1, r2 = r_grid[idx-1], r_grid[idx]
        q1, q2 = q_profile[idx-1], q_profile[idx]
        
        r_s = r1 + (q_target - q1) * (r2 - r1) / (q2 - q1)
        
    elif method == 'spline':
        # Cubic spline interpolation
        cs = CubicSpline(r_grid, q_profile)
        
        # Root finding
        try:
            r_s = brentq(lambda r: cs(r) - q_target, 
                        r_grid[idx-1], r_grid[idx])
        except ValueError:
            # Fallback to linear if brentq fails
            r1, r2 = r_grid[idx-1], r_grid[idx]
            q1, q2 = q_profile[idx-1], q_profile[idx]
            r_s = r1 + (q_target - q1) * (r2 - r1) / (q2 - q1)
    
    elif method == 'newton':
        # Newton iteration for high accuracy
        cs = CubicSpline(r_grid, q_profile)
        
        # Initial guess from linear interpolation
        r1, r2 = r_grid[idx-1], r_grid[idx]
        q1, q2 = q_profile[idx-1], q_profile[idx]
        r_s = r1 + (q_target - q1) * (r2 - r1) / (q2 - q1)
        
        # Newton iteration
        for _ in range(10):
            f = cs(r_s) - q_target
            df = cs(r_s, 1)  # First derivative
            
            if abs(df) < 1e-10:
                break
            
            r_s_new = r_s - f / df
            
            if abs(r_s_new - r_s) < 1e-10:
                break
            
            r_s = r_s_new
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Verify accuracy
    cs = CubicSpline(r_grid, q_profile)
    q_found = cs(r_s)
    accuracy = abs(q_found - q_target) / abs(q_target)
    
    return r_s, accuracy


def find_all_rational_surfaces(q_profile, r_grid, m_max=5, n_max=3):
    """
    Find all rational surfaces q = m/n in the domain
    
    Parameters
    ----------
    q_profile : array_like
        Safety factor profile
    r_grid : array_like
        Radial grid
    m_max : int, optional
        Maximum poloidal mode number (default: 5)
    n_max : int, optional
        Maximum toroidal mode number (default: 3)
    
    Returns
    -------
    surfaces : list of dict
        Each entry: {'m': int, 'n': int, 'q': float, 'r_s': float, 'accuracy': float}
    """
    surfaces = []
    
    for n in range(1, n_max + 1):
        for m in range(1, m_max + 1):
            q_target = m / n
            r_s, acc = find_rational_surface(q_profile, r_grid, q_target)
            
            if not np.isnan(r_s):
                surfaces.append({
                    'm': m,
                    'n': n,
                    'q': q_target,
                    'r_s': r_s,
                    'accuracy': acc
                })
    
    return surfaces
