"""
MHD Solvers

Toroidal reduced MHD solver combining geometry and time integration.
"""

from .toroidal_mhd import ToroidalMHDSolver
from .poisson_toroidal import (
    solve_poisson_toroidal,
    compute_residual,
    check_boundary_conditions,
)

__all__ = [
    'ToroidalMHDSolver',
    'solve_poisson_toroidal',
    'compute_residual',
    'check_boundary_conditions',
]
