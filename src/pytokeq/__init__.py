"""
PyTokEq - Tokamak Equilibrium Solver

Physics-validated Grad-Shafranov solver for Layer 1 of PTM-RL.
"""

__version__ = "1.1.0"
__author__ = "小P ⚛️"

from .equilibrium.solver.picard_gs_solver import (
    Grid,
    CoilSet,
    Constraints,
    solve_picard_free_boundary
)

from .equilibrium.profiles.m3dc1_profile import M3DC1Profile
from .equilibrium.diagnostics.q_profile import QCalculator

__all__ = [
    'Grid',
    'CoilSet', 
    'Constraints',
    'solve_picard_free_boundary',
    'M3DC1Profile',
    'QCalculator',
]
