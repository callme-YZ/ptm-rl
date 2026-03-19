"""
Equilibrium Module

Provides equilibrium profiles and force balance calculations
for tokamak MHD simulations.

Key Components
--------------
- Pressure profiles P(ψ) for Grad-Shafranov equilibria
- Pressure gradient ∇P in toroidal geometry
- Force balance verification J×B = ∇P
- Integration with PyTokEq Solovev equilibrium

Author: 小P ⚛️
Created: 2026-03-19
"""

from .pressure import (
    pressure_profile,
    pressure_gradient,
    pressure_gradient_psi,
)

__all__ = [
    'pressure_profile',
    'pressure_gradient',
    'pressure_gradient_psi',
]

from .solovev import (
    load_solovev_equilibrium,
    verify_solovev_force_balance,
    PYTOKEQ_AVAILABLE,
)

__all__ += [
    'load_solovev_equilibrium',
    'verify_solovev_force_balance',
    'PYTOKEQ_AVAILABLE',
]
