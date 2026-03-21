"""
Time integrators for MHD evolution.

This module provides time integration schemes for reduced MHD equations:
    - SymplecticIntegrator: Störmer-Verlet (2nd-order, baseline)
    - (Future) Wu adaptive time transformation
    
Author: 小P ⚛️
Created: 2026-03-17
"""

from .symplectic import SymplecticIntegrator

__all__ = ['SymplecticIntegrator']
