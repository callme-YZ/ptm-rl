"""
Equilibrium Solvers Package

Provides fixed-boundary and free-boundary Grad-Shafranov solvers.
"""

# Fixed-boundary solver
from .solver.picard_gs_solver import solve_picard_free_boundary

# Free-boundary components
from .diagnostics.xpoint_finder import (
    XPoint, 
    find_xpoints, 
    select_primary_xpoint, 
    is_xpoint_valid
)

from .boundary.free_boundary_constraints import (
    CoilSet,
    IsofluxPair,
    build_constraint_matrix,
    build_target_vector,
    optimize_coil_currents
)

from .boundary.free_boundary_picard import (
    solve_free_boundary_picard,
    FreeBoundaryResult
)

from .utils.greens_function import (
    greens_psi,
    greens_psi_gradient_R,
    greens_psi_gradient_Z
)

from .profiles.solovev_solution import SolovevSolution

# q-profile calculation (PHYS-01 fix)
from .diagnostics.flux_surface_tracer import FluxSurfaceTracer
from .diagnostics.q_profile import QCalculator, integrate_along_surface, surface_average

__all__ = [
    # Fixed-boundary
    'solve_picard_free_boundary',
    
    # Free-boundary
    'solve_free_boundary_picard',
    'FreeBoundaryResult',
    
    # X-point
    'XPoint',
    'find_xpoints',
    'select_primary_xpoint',
    'is_xpoint_valid',
    
    # Constraints
    'CoilSet',
    'IsofluxPair',
    'build_constraint_matrix',
    'build_target_vector',
    'optimize_coil_currents',
    
    # Green's function
    'greens_psi',
    'greens_psi_gradient_R',
    'greens_psi_gradient_Z',
    
    # Analytical solutions
    'SolovevSolution',
    
    # q-profile (PHYS-01 fix)
    'FluxSurfaceTracer',
    'QCalculator',
    'integrate_along_surface',
    'surface_average',
]
