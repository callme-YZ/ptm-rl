"""
Initial Conditions for 3D MHD

Provides equilibrium and perturbation initial conditions for 3D reduced MHD simulations.

Available ICs:
- Equilibrium: Axisymmetric equilibrium ψ₀(r, θ) with safety factor q(r)
- Ballooning modes: 3D perturbations ψ₁(r, θ, ζ) localized at bad curvature regions

Author: 小P ⚛️
Created: 2026-03-19
Phase: 2.2 (3D Initial Conditions)
"""

from .ballooning_mode import (
    Grid3D,
    create_q_profile,
    create_equilibrium_ic,
    create_ballooning_mode_ic,
)

__all__ = [
    "Grid3D",
    "create_q_profile",
    "create_equilibrium_ic",
    "create_ballooning_mode_ic",
]
