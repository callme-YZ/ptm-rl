"""
Solovev Equilibrium Interface

Provides interface to PyTokEq Solovev analytical equilibrium solution
for force balance verification and benchmarking.

Solovev Solution
----------------
The Solovev equilibrium is an exact analytical solution to the Grad-Shafranov equation:
    Δ*ψ = -μ₀R²(dP/dψ + F·dF/dψ)

For circular cross-section, the poloidal flux is:
    ψ(r,θ) = ψ₀ + c₁r² + c₂r⁴ + c₃r⁴cos(2θ)

This satisfies force balance J×B = ∇P exactly (to machine precision).

Usage as Benchmark
------------------
1. Load Solovev equilibrium from PyTokEq
2. Compute J×B using our force_balance module
3. Compute ∇P using our pressure module
4. Verify residual |J×B - ∇P| < 1e-6

This validates our gradient operators and force balance implementation.

Requirements
------------
PyTokEq must be installed:
    pip install pytokeq

References
----------
- Solov'ev (1968): "The Theory of Hydromagnetic Stability of Toroidal Plasma Configurations"
- PyTokEq documentation: https://pytokeq.readthedocs.io/
- Freidberg (2014): "Ideal MHD", Section 6.4

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
from typing import Optional, Dict
from ..geometry import ToroidalGrid

try:
    import pytokeq
    PYTOKEQ_AVAILABLE = True
except ImportError:
    PYTOKEQ_AVAILABLE = False


def load_solovev_equilibrium(
    grid: ToroidalGrid,
    P0: float = 1e5,
    B0: float = 2.0,
    beta_p: float = 0.5,
    q0: float = 1.0,
    qa: float = 3.0
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load Solovev equilibrium from PyTokEq.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Toroidal grid to interpolate equilibrium onto
    P0 : float, optional
        Central pressure [Pa] (default: 1e5 = 1 bar)
    B0 : float, optional
        Toroidal field on axis [T] (default: 2.0)
    beta_p : float, optional
        Poloidal beta (default: 0.5)
    q0 : float, optional
        Safety factor on axis (default: 1.0)
    qa : float, optional
        Safety factor at edge (default: 3.0)
    
    Returns
    -------
    equilibrium : dict or None
        {
            'psi': np.ndarray (nr, ntheta),     # Poloidal flux [Wb]
            'P': np.ndarray (nr, ntheta),       # Pressure [Pa]
            'J_phi': np.ndarray (nr, ntheta),   # Toroidal current [A/m²]
            'B_pol': np.ndarray (nr, ntheta),   # Poloidal field [T]
            'q': np.ndarray (nr,),              # Safety factor
            'psi_edge': float,                  # Edge flux value [Wb]
        }
        
        Returns None if PyTokEq is not installed.
    
    Notes
    -----
    - Solovev solution satisfies force balance exactly
    - Use for benchmarking force_balance module
    - Requires PyTokEq: pip install pytokeq
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> eq = load_solovev_equilibrium(grid, P0=1e5, B0=2.0)
    >>> if eq is not None:
    ...     psi = eq['psi']
    ...     # Verify force balance
    ...     from pytokmhd.physics import force_balance_residual
    ...     result = force_balance_residual(psi, P0=1e5, psi_edge=eq['psi_edge'], grid=grid)
    ...     print(f"Max residual: {result['max_residual']:.2e}")
    """
    if not PYTOKEQ_AVAILABLE:
        print("Warning: PyTokEq not installed. Install with: pip install pytokeq")
        return None
    
    # TODO: Implement PyTokEq Solovev loading
    # This requires understanding PyTokEq API
    # For now, return placeholder
    
    # Placeholder implementation:
    # 1. Create PyTokEq Solovev equilibrium
    # 2. Interpolate onto our grid
    # 3. Extract ψ, P, J, B, q profiles
    
    raise NotImplementedError(
        "PyTokEq Solovev interface not yet implemented. "
        "This will be completed when PyTokEq is installed and API is studied."
    )


def verify_solovev_force_balance(
    grid: ToroidalGrid,
    P0: float = 1e5,
    B0: float = 2.0,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Verify force balance for Solovev equilibrium.
    
    Loads Solovev solution and checks |J×B - ∇P| < tolerance.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Toroidal grid
    P0 : float, optional
        Central pressure [Pa]
    B0 : float, optional
        Toroidal field [T]
    tolerance : float, optional
        Maximum allowed residual (default: 1e-6)
    
    Returns
    -------
    result : dict
        {
            'passed': bool,               # True if residual < tolerance
            'max_residual': float,        # Maximum |J×B - ∇P|
            'relative_error': float,      # Normalized error
        }
    
    Raises
    ------
    AssertionError
        If force balance is not satisfied within tolerance
    
    Examples
    --------
    >>> from pytokmhd.geometry import ToroidalGrid
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> result = verify_solovev_force_balance(grid, P0=1e5, B0=2.0, tolerance=1e-6)
    >>> assert result['passed'], f"Force balance failed: {result['max_residual']:.2e}"
    """
    if not PYTOKEQ_AVAILABLE:
        return {
            'passed': False,
            'max_residual': np.inf,
            'relative_error': np.inf,
            'message': 'PyTokEq not installed'
        }
    
    # Load equilibrium
    eq = load_solovev_equilibrium(grid, P0=P0, B0=B0)
    
    if eq is None:
        return {
            'passed': False,
            'max_residual': np.inf,
            'relative_error': np.inf,
            'message': 'Failed to load Solovev equilibrium'
        }
    
    # Verify force balance
    from ..physics import force_balance_residual
    fb_result = force_balance_residual(
        eq['psi'], P0, eq['psi_edge'], grid
    )
    
    passed = fb_result['max_residual'] < tolerance
    
    return {
        'passed': passed,
        'max_residual': fb_result['max_residual'],
        'relative_error': fb_result['relative_error'],
    }
