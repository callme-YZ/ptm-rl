"""
PyTokMHD Diagnostics Module

Tearing mode diagnostics tools for MHD simulations.

Main Components
---------------
- magnetic_island: Island width and structure detection
- growth_rate: Growth rate measurement
- rational_surface: Rational surface locator
- monitor: Real-time monitoring during evolution
- visualization: Plotting tools

Quick Start
-----------
>>> from pytokmhd.diagnostics import TearingModeMonitor
>>> 
>>> monitor = TearingModeMonitor(m=2, n=1, track_every=10)
>>> 
>>> for step in range(n_steps):
>>>     psi, omega = mhd_step(psi, omega, dt)
>>>     diag = monitor.update(psi, omega, t, r, z, q_profile)
>>>     if diag and diag['w'] > threshold:
>>>         print(f"Warning: Island width = {diag['w']:.4f}")
>>> 
>>> from pytokmhd.diagnostics.visualization import plot_island_evolution
>>> plot_island_evolution(monitor, save_path='island_evolution.png')
"""

from .magnetic_island import (
    compute_island_width,
    compute_helical_flux,
    extract_flux_surface,
    find_extrema
)

from .growth_rate import (
    compute_growth_rate,
    energy_growth_rate,
    compute_growth_rate_sliding_window
)

from .rational_surface import (
    find_rational_surface,
    find_all_rational_surfaces
)

from .monitor import TearingModeMonitor

from . import visualization


__all__ = [
    # Magnetic island
    'compute_island_width',
    'compute_helical_flux',
    'extract_flux_surface',
    'find_extrema',
    
    # Growth rate
    'compute_growth_rate',
    'energy_growth_rate',
    'compute_growth_rate_sliding_window',
    
    # Rational surface
    'find_rational_surface',
    'find_all_rational_surfaces',
    
    # Monitor
    'TearingModeMonitor',
    
    # Visualization
    'visualization'
]


__version__ = '0.1.0'
