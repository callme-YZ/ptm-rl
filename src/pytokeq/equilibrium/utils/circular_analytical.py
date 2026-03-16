"""
Circular Analytical Solution for G-S Equation

Simple analytical solution with zero boundary condition:
    Δ*ψ = -α (constant)
    ψ = 0 at circular boundary

Solution:
    ψ(R,Z) = -α/4 · (R² + Z² - r_b²)

This provides clean analytical benchmark with realistic BC.
"""

import numpy as np


class CircularAnalytical:
    """
    Analytical solution for constant-source G-S equation.
    
    Assumes:
    - p'(ψ) = p1 (constant pressure gradient)
    - f·f'(ψ) = f1 (constant toroidal field term)
    - Circular boundary at r = √(R² + Z²) = r_boundary
    
    G-S equation becomes:
        Δ*ψ = -μ₀R²p1 - f1 ≡ -α
    
    For constant α, the solution with ψ=0 at r=r_b is:
        ψ(R,Z) = -α/4 · (R² + Z² - r_b²)
    
    Parameters
    ----------
    R_center : float
        Radial center of circular boundary
    r_boundary : float
        Radius of circular boundary
    p1 : float
        Constant pressure gradient
    f1 : float
        Constant f·f' term
    mu0 : float
        Magnetic permeability (default: 4π×10⁻⁷)
    """
    
    def __init__(self, R_center=4.5, r_boundary=2.0, 
                 p1=1e4, f1=0.5, mu0=4*np.pi*1e-7):
        self.R_center = R_center
        self.r_boundary = r_boundary
        self.p1 = p1
        self.f1 = f1
        self.mu0 = mu0
        
        # Compute effective source term α
        # Note: for Δ*ψ, need to use R² term carefully
        # Here we use approximate constant source
        self.alpha = self.mu0 * R_center**2 * p1 + f1
    
    def psi(self, R, Z):
        """
        Poloidal flux function.
        
        ψ(R,Z) = -α/4 · (R² + Z² - r_b²)
        
        Parameters
        ----------
        R, Z : float or ndarray
            Coordinates (m)
        
        Returns
        -------
        psi : float or ndarray
            Poloidal flux (Wb/rad)
        """
        return -self.alpha / 4.0 * (R**2 + Z**2 - self.r_boundary**2)
    
    def inside_plasma(self, R, Z):
        """
        Check if point is inside plasma (r < r_boundary).
        
        Returns
        -------
        inside : bool or ndarray
            True if inside plasma
        """
        r = np.sqrt((R - self.R_center)**2 + Z**2)
        return r < self.r_boundary
    
    def verify_gs_equation(self, R, Z):
        """
        Verify solution satisfies G-S equation.
        
        For this simple case:
            Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
        
        Analytical:
            ∂ψ/∂R = -α/2 · R
            ∂²ψ/∂R² = -α/2
            ∂²ψ/∂Z² = -α/2
            Δ*ψ = -α/2 - (1/R)·(-α/2·R) + (-α/2)
                 = -α/2 + α/2 - α/2 = -α/2
        
        Wait, this is wrong! Let me recalculate...
        
        Actually, for Δ* = R·∂/∂R(1/R·∂ψ/∂R) + ∂²ψ/∂Z²:
            ∂ψ/∂R = -α/2·R
            1/R·∂ψ/∂R = -α/2
            ∂/∂R(1/R·∂ψ/∂R) = 0
            R·∂/∂R(...) = 0
            ∂²ψ/∂Z² = -α/2
            → Δ*ψ = -α/2
        
        But RHS = -μ₀R²p1 - f1 = -α (not -α/2!)
        
        So this simple solution DOESN'T work for general R!
        
        Need better analytical solution...
        """
        # Numerical check
        h = 1e-5
        
        psi_0 = self.psi(R, Z)
        psi_pR = self.psi(R + h, Z)
        psi_mR = self.psi(R - h, Z)
        psi_pZ = self.psi(R, Z + h)
        psi_mZ = self.psi(R, Z - h)
        
        dpsi_dR = (psi_pR - psi_mR) / (2*h)
        d2psi_dR2 = (psi_pR - 2*psi_0 + psi_mR) / h**2
        d2psi_dZ2 = (psi_pZ - 2*psi_0 + psi_mZ) / h**2
        
        delta_star = d2psi_dR2 - dpsi_dR / R + d2psi_dZ2
        
        rhs = -self.mu0 * R**2 * self.p1 - self.f1
        
        residual = delta_star - rhs
        
        return residual


# This simple form doesn't work! Need to account for R dependence properly.
# 
# The problem: Δ* has R-dependent terms, but RHS also has R².
# For exact analytical solution with ψ=0 BC, we need more complex form.
#
# BETTER APPROACH: Use existing Solov'ev with proper understanding!


if __name__ == '__main__':
    print("Circular analytical test:")
    print("WARNING: Simple ψ~(R²+Z²) doesn't satisfy Δ*ψ=-μ₀R²p1-f1")
    print("Need more sophisticated analytical form!")
    print()
    print("Recommendation: Use Solov'ev with homogeneous BC instead.")
