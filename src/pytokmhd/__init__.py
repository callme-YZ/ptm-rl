"""
PyTokMHD - Tokamak MHD Evolution Solver

A cylindrical reduced-MHD solver for tokamak plasma dynamics.
Implements Model-A formulation with RK4 time integration.

Author: 小P ⚛️
Phase 1: Core Solver (2026-03-16)
"""

__version__ = "0.1.0"
__author__ = "小P ⚛️"

from .solver import mhd_equations, time_integrator, boundary, poisson_solver

__all__ = [
    "mhd_equations",
    "time_integrator",
    "boundary",
    "poisson_solver",
]
