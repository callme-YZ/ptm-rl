"""
Physics Module

Fundamental physics calculations for MHD:
- Force balance J×B = ∇P
- Current density and magnetic field relations
- Equilibrium verification
- Pressure force term for vorticity equation

Author: 小P ⚛️
Created: 2026-03-19
"""

from .force_balance import (
    compute_current_density,
    compute_lorentz_force,
    force_balance_residual,
    pressure_force_term,
)

__all__ = [
    'compute_current_density',
    'compute_lorentz_force',
    'force_balance_residual',
    'pressure_force_term',
]
