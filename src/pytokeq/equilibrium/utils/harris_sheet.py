"""
Harris Sheet Equilibrium

Classic current sheet equilibrium for tearing mode studies.

B_z ~ tanh(x/a) → current sheet at x=0
Known to be tearing-unstable

Author: 小P ⚛️
Date: 2026-03-11
"""

import numpy as np
from typing import Tuple


def create_harris_sheet_equilibrium(
    R: np.ndarray,
    Z: np.ndarray,
    R0: float = 10.0,
    sheet_width: float = 0.1,
    B0: float = 1.0,
    psi0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Harris sheet equilibrium in 2D.
    
    Magnetic field:
      B_Z ~ B0 × tanh((R-R0)/a)
      B_R ~ 0
      B_φ ~ const
    
    From ∇×B, this gives current sheet:
      J_φ ~ -dB_Z/dR ~ -B0/(a cosh²((R-R0)/a))
    
    Poloidal flux ψ:
      B_Z = (1/R) ∂ψ/∂R
      → ψ ~ ∫ R B_Z dR
    
    Parameters
    ----------
    R, Z : np.ndarray
        Grid arrays
    R0 : float
        Sheet center (major radius)
    sheet_width : float
        Current sheet width parameter
    B0 : float
        Asymptotic field strength
    psi0 : float
        Flux normalization
    
    Returns
    -------
    psi : np.ndarray, shape (Nr, Nz)
        Poloidal flux
    J_phi : np.ndarray, shape (Nr, Nz)
        Toroidal current density
    """
    Nr, Nz = len(R), len(Z)
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Distance from sheet center
    x = R_2d - R0
    a = sheet_width
    
    # Harris sheet field
    B_Z = B0 * np.tanh(x / a)
    
    # Current density (from ∇×B)
    # J_φ = -∂B_Z/∂R = -B0/(a cosh²(x/a))
    cosh_term = np.cosh(x / a)
    J_phi = -B0 / (a * cosh_term**2)
    
    # Poloidal flux
    # ψ ~ ∫ R B_Z dR
    # For tanh profile:
    # ∫ R tanh(x/a) dx ≈ a R ln(cosh(x/a)) + const
    
    # Simplified form (keeping it manageable):
    # ψ ≈ psi0 × x × tanh(x/a)
    psi = psi0 * x * np.tanh(x / a)
    
    # Apply boundary conditions
    psi[0, :] = 0
    psi[-1, :] = 0
    psi[:, 0] = 0
    psi[:, -1] = 0
    
    return psi, J_phi


def add_tearing_perturbation_to_harris(
    psi_eq: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    R0: float = 10.0,
    m: int = 2,
    n: int = 1,
    amplitude: float = 0.01,
    width: float = 0.3
) -> np.ndarray:
    """
    Add tearing perturbation to Harris sheet.
    
    Perturbation localized at sheet (R=R0) with mode structure m, n.
    
    Parameters
    ----------
    psi_eq : np.ndarray
        Equilibrium flux
    R, Z : np.ndarray
        Grids
    R0 : float
        Sheet location
    m, n : int
        Poloidal/toroidal mode numbers
    amplitude : float
        Perturbation amplitude
    width : float
        Radial extent of perturbation
    
    Returns
    -------
    psi_total : np.ndarray
        Equilibrium + perturbation
    """
    R_2d, Z_2d = np.meshgrid(R, Z, indexing='ij')
    
    # Radial coordinate from sheet
    x = R_2d - R0
    
    # Poloidal angle
    theta = np.arctan2(Z_2d, x)
    
    # Perturbation: Gaussian in x, sinusoidal in θ
    envelope = np.exp(-(x / width)**2)
    delta_psi = amplitude * envelope * np.sin(m * theta)
    
    # Boundary conditions
    delta_psi[0, :] = 0
    delta_psi[-1, :] = 0
    delta_psi[:, 0] = 0
    delta_psi[:, -1] = 0
    
    return psi_eq + delta_psi


if __name__ == "__main__":
    # Test
    R = np.linspace(9.0, 11.0, 64)
    Z = np.linspace(-1.0, 1.0, 64)
    
    psi_eq, J_phi = create_harris_sheet_equilibrium(R, Z, R0=10.0, sheet_width=0.1)
    
    print("Harris Sheet Equilibrium")
    print("=" * 70)
    print(f"ψ range: [{psi_eq.min():.3f}, {psi_eq.max():.3f}]")
    print(f"J_φ range: [{J_phi.min():.3f}, {J_phi.max():.3f}]")
    print(f"J_φ at sheet center: {J_phi[32, 32]:.3f}")
    print()
    print("Current sheet at R=10.0")
    print("Should be tearing-unstable!")
    print("=" * 70)
