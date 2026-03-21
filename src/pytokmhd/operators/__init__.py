"""
Differential Operators Module

Provides differential operators (gradient, divergence, Laplacian, Poisson bracket)
in toroidal geometry.

Author: 小P ⚛️
Created: 2026-03-17
Updated: 2026-03-19 (added Poisson bracket)
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
from .poisson_solver import (
    solve_poisson_toroidal,
    laplacian_toroidal_check,
)
from .utils import B_poloidal_from_psi

__all__ = [
    'gradient_toroidal',
    'divergence_toroidal',
    'laplacian_toroidal',
    'divergence_B_toroidal',
    'poisson_bracket',
    'jacobi_identity_residual',
    'advection_bracket',
    'solve_poisson_toroidal',
    'laplacian_toroidal_check',
    'B_poloidal_from_psi'
]
