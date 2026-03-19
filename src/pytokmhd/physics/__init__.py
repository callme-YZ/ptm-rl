"""
Physics Module

Core physics formulations for reduced MHD:
- Hamiltonian energy functional
- Conservation laws
- Energy transport

Author: 小P ⚛️
Created: 2026-03-19
"""

from .hamiltonian import (
    compute_hamiltonian,
    hamiltonian_density,
    kinetic_energy,
    magnetic_energy,
    energy_partition,
)

__all__ = [
    'compute_hamiltonian',
    'hamiltonian_density',
    'kinetic_energy',
    'magnetic_energy',
    'energy_partition',
]
