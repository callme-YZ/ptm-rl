"""
Differential Operators Module

Provides differential operators (gradient, divergence, Laplacian, Poisson bracket)
in toroidal geometry.

Note: Poisson solver has been moved to pytokmhd.solvers.
      operators.poisson_solver is DEPRECATED (broken implementation).

Author: 小P ⚛️
Created: 2026-03-17
Updated: 2026-03-24 (deprecated poisson_solver)
"""

from .toroidal_operators import (
    gradient_toroidal,
    divergence_toroidal,
    laplacian_toroidal,
    divergence_B_toroidal
)
from .poisson_bracket import (
    poisson_bracket,
    jacobi_identity_residual,
    advection_bracket,
)
from .utils import B_poloidal_from_psi

# Deprecated imports (kept for backward compatibility, emit warnings)
# DO NOT USE - import from pytokmhd.solvers instead
# from .poisson_solver import solve_poisson_toroidal, laplacian_toroidal_check

__all__ = [
    # Active operators
    'gradient_toroidal',
    'divergence_toroidal',
    'laplacian_toroidal',
    'divergence_B_toroidal',
    'poisson_bracket',
    'jacobi_identity_residual',
    'advection_bracket',
    'B_poloidal_from_psi',
    
    # Removed (deprecated):
    # 'solve_poisson_toroidal',  # Use pytokmhd.solvers.solve_poisson_toroidal
    # 'laplacian_toroidal_check',  # Use laplacian_toroidal
]
