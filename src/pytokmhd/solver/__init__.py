"""
PyTokMHD Solver Module

Core MHD evolution components:
- mhd_equations: Cylindrical operators and RHS functions
- time_integrator: RK4 time stepping
- boundary: Boundary condition handlers
- poisson_solver: FFT-based Poisson solver
- equilibrium_loader: PyTokEq equilibrium loading (Phase 2)
- equilibrium_cache: Fast equilibrium caching (Phase 2)
- initial_conditions: Equilibrium initialization (Phase 1 & 2 & 4)
"""

from . import mhd_equations
from . import time_integrator
from . import boundary
from . import poisson_solver
from . import equilibrium_loader
from . import equilibrium_cache
from . import initial_conditions

# Export commonly used functions
from .initial_conditions import (
    harris_sheet_initial,
    pytokeq_initial,
    setup_tearing_mode  # Phase 4
)

__all__ = [
    "mhd_equations",
    "time_integrator", 
    "boundary",
    "poisson_solver",
    "equilibrium_loader",
    "equilibrium_cache",
    "initial_conditions",
    # Functions
    "harris_sheet_initial",
    "pytokeq_initial",
    "setup_tearing_mode",
]
