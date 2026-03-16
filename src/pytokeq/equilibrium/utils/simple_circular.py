"""
Simple Circular Equilibrium for M3D-C1 Benchmark

Based on paper observation: "r = √ψ_n"
→ ψ_n = (r/a)²

This is M3D-C1's approach:
  - Prescribe simple ψ ~ r²
  - Get q(r) from q(ψ_n) formula
  - Accept force imbalance (small for zero-β)
  - Linear analysis doesn't need perfect equilibrium

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Tuple


def create_simple_circular_equilibrium(
    R: np.ndarray,
    Z: np.ndarray,
    R0: float = 10.0,
    a: float = 1.0,
    psi0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple circular equilibrium: ψ = ψ0 × (r/a)²
    
    This is the simplest flux function for circular geometry.
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid arrays
    R0 : float
        Major radius
    a : float
        Minor radius
    psi0 : float
        Flux normalization
    
    Returns
    -------
    psi : np.ndarray, shape (Nr, Nz)
        Poloidal flux
    psi_n : np.ndarray, shape (Nr, Nz)
        Normalized flux ∈ [0,1]
    """
    Nr, Nz = len(R), len(Z)
    
    # 2D grids
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Poloidal radius from magnetic axis
    r = np.sqrt((R_2d - R0)**2 + Z_2d**2)
    
    # Simple prescription: ψ ~ r²
    psi = psi0 * (r / a)**2
    
    # Normalized flux
    psi_n = (r / a)**2
    
    # Apply boundary conditions (optional, already satisfied)
    # psi at plasma edge (r=a) = psi0
    
    return psi, psi_n


def compute_m3dc1_q_profile(
    psi_n: np.ndarray,
    q0: float = 1.75,
    qe: float = 2.5,
    alpha: float = 2.0
) -> np.ndarray:
    """
    Compute M3D-C1 q-profile
    
    q(ψ_n) = q0 × [1 + (ψ_n/ql)^α]^(1/α)
    
    Parameters
    ----------
    psi_n : np.ndarray
        Normalized flux [0,1]
    q0 : float
        Central safety factor
    qe : float
        Edge safety factor
    alpha : float
        Profile shaping parameter
    
    Returns
    -------
    q : np.ndarray
        Safety factor
    """
    # Compute ql from edge condition
    ql = ((qe / q0)**alpha - 1)**(-1.0/alpha)
    
    # M3D-C1 formula
    q = q0 * (1 + (psi_n / ql)**alpha)**(1.0/alpha)
    
    return q


def create_m3dc1_equilibrium(
    R: np.ndarray,
    Z: np.ndarray,
    R0: float = 10.0,
    a: float = 1.0,
    q0: float = 1.75,
    qe: float = 2.5,
    alpha: float = 2.0,
    psi0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create M3D-C1 benchmark equilibrium
    
    Simple approach:
      1. ψ_n = (r/a)²
      2. q from M3D-C1 formula
      3. No force balance enforcement (accept imbalance)
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid arrays
    R0 : float
        Major radius
    a : float
        Minor radius
    q0, qe, alpha : float
        M3D-C1 q-profile parameters
    psi0 : float
        Flux normalization
    
    Returns
    -------
    psi : np.ndarray
        Poloidal flux
    psi_n : np.ndarray
        Normalized flux
    q : np.ndarray
        Safety factor from M3D-C1 formula
    """
    # Create simple circular equilibrium
    psi, psi_n = create_simple_circular_equilibrium(R, Z, R0, a, psi0)
    
    # Compute q from M3D-C1 formula
    q = compute_m3dc1_q_profile(psi_n, q0, qe, alpha)
    
    return psi, psi_n, q


if __name__ == "__main__":
    # Test
    R = np.linspace(9.0, 11.0, 64)
    Z = np.linspace(-1.0, 1.0, 64)
    
    psi, psi_n, q = create_m3dc1_equilibrium(R, Z)
    
    print("M3D-C1 Simple Circular Equilibrium")
    print("=" * 70)
    print(f"ψ range: [{psi.min():.3f}, {psi.max():.3f}]")
    print(f"ψ_n range: [{psi_n.min():.3f}, {psi_n.max():.3f}]")
    print(f"q range: [{q.min():.3f}, {q.max():.3f}]")
    print()
    print(f"q at axis: {q[32,32]:.3f} (target: 1.75)")
    print(f"q at edge: {q[0,32]:.3f} or {q[-1,32]:.3f} (target: 2.5)")
    print()
    
    # Find q=2 surface
    r_2d = np.sqrt((np.meshgrid(R, Z, indexing='ij')[0] - 10.0)**2 + 
                   (np.meshgrid(R, Z, indexing='ij')[1])**2)
    mask = np.abs(q - 2.0) < 0.01
    if mask.any():
        r_q2 = r_2d[mask].mean()
        print(f"q=2 surface at r ≈ {r_q2:.3f} (r/a = {r_q2:.3f})")
    
    print("=" * 70)
    print("✅ Simple circular equilibrium created!")
    print("   No force balance enforcement needed for linear analysis")
