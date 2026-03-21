"""
Diagnostics module for MHD simulations.

Provides analysis tools for:
- Fourier decomposition
- Energy diagnostics
- Constraint verification
- Mode analysis
"""

from .fourier import fourier_decompose, reconstruct_from_modes


def find_rational_surface(*args, **kwargs):
    """Stub for Phase 4 rational surface finder."""
    raise NotImplementedError("find_rational_surface deferred to Phase 4")


__all__ = [
    'fourier_decompose',
    'reconstruct_from_modes',
    'find_rational_surface',
]
