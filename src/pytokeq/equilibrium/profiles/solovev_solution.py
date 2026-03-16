"""
Solov'ev Analytical Solution for Grad-Shafranov Equation

Provides exact analytical solution for testing numerical solvers.

Reference: L.E. Zakharov, V.D. Shafranov (1986)
"""

import numpy as np


class SolovevSolution:
    """
    Solov'ev analytical equilibrium solution.
    
    Exact solution to G-S equation with specific assumptions:
    - Constant pressure gradient
    - Constant toroidal field function
    - Up-down symmetric
    
    Parameters
    ----------
    R0 : float
        Major radius (m)
    eps : float
        Inverse aspect ratio (a/R0)
    kappa : float
        Elongation
    delta : float
        Triangularity
    A : float
        Shafranov shift parameter
    """
    
    def __init__(self, R0, eps, kappa, delta, A):
        self.R0 = R0
        self.eps = eps
        self.kappa = kappa
        self.delta = delta
        self.A = A
        
        # Derived parameters
        self.a = eps * R0  # Minor radius
        
        # Solov'ev coefficients (from geometry)
        self._compute_coefficients()
        
        # Physical constants
        self.mu0 = 4 * np.pi * 1e-7
    
    def _compute_coefficients(self):
        """
        Compute Solov'ev coefficients c1-c7.
        
        From boundary conditions at separatrix.
        """
        R0 = self.R0
        eps = self.eps
        kappa = self.kappa
        delta = self.delta
        A = self.A
        
        # Normalized coordinates at X-point (approximation)
        # For up-down symmetric: Z_x = -kappa, R_x = 1 + delta
        
        # Coefficients (simplified model)
        # Full derivation in Shafranov (1986)
        
        # c1: controls overall scale
        self.c = np.zeros(8)  # c[0] unused, c[1]-c[7]
        
        self.c[1] = -1.0  # Normalization
        self.c[2] = A     # Shafranov shift
        self.c[3] = -0.5  # Pressure gradient (normalized)
        self.c[4] = -0.5  # FF' term (normalized)
        self.c[5] = delta / (eps * kappa)  # Triangularity
        self.c[6] = 1.0 / (eps**2 * kappa**2)  # Elongation
        self.c[7] = A     # Cross term
    
    def psi(self, R, Z):
        """
        Compute flux function ψ(R, Z).
        
        Solov'ev solution:
        ψ(x,y) = c₁x⁴/8 + Ax²ln(x) - c₂x² + c₃(x⁴/8 - x²ln(x))
               + c₄x² + c₅x²y² + c₆y⁴/8 + c₇y²
        
        where x = R/R₀, y = Z/R₀
        
        Parameters
        ----------
        R, Z : float or ndarray
            Coordinates (m)
        
        Returns
        -------
        psi : float or ndarray
            Poloidal flux (Wb/rad)
        """
        x = R / self.R0
        y = Z / self.R0
        
        c = self.c
        A = self.A
        
        # Solov'ev formula (polynomial in x, y)
        psi = (
            c[1] * x**4 / 8.0
            + A * x**2 * np.log(x)
            - c[2] * x**2
            + c[3] * (x**4 / 8.0 - x**2 * np.log(x))
            + c[4] * x**2
            + c[5] * x**2 * y**2
            + c[6] * y**4 / 8.0
            + c[7] * y**2
        )
        
        return psi
    
    def pressure(self, psi):
        """
        Pressure profile p(ψ).
        
        For Solov'ev: p'(ψ) = constant
        
        Returns
        -------
        p : float or ndarray
            Pressure (Pa)
        """
        # Constant gradient model
        p0 = 1e5  # Edge pressure (Pa)
        p_prime = -self.c[3] / (self.mu0 * self.R0**2)
        
        return p0 + p_prime * psi
    
    def f_function(self, psi):
        """
        Toroidal field function f = R·B_toroidal.
        
        For Solov'ev: f·f'(ψ) = constant
        
        Returns
        -------
        f : float or ndarray
            f(ψ) (T·m)
        """
        # Constant f·f' model
        f0 = 5.0  # Edge value (T·m, typical for tokamak)
        ff_prime = -self.c[4]
        
        # f² = f0² + 2·ff'·(ψ - ψ_edge)
        # For simplicity, use linearization
        return f0 + ff_prime * psi / (2 * f0)
    
    def check_gs_equation(self, R, Z):
        """
        Verify that solution satisfies G-S equation.
        
        Check: Δ*ψ + μ₀R²p'(ψ) + f·f'(ψ) = 0
        
        Returns
        -------
        residual : float or ndarray
            LHS of G-S equation (should be ~0)
        """
        # Numerical derivatives
        h = 1e-5
        
        psi_0 = self.psi(R, Z)
        
        psi_pR = self.psi(R + h, Z)
        psi_mR = self.psi(R - h, Z)
        psi_pZ = self.psi(R, Z + h)
        psi_mZ = self.psi(R, Z - h)
        
        # Second derivatives
        d2psi_dR2 = (psi_pR - 2*psi_0 + psi_mR) / h**2
        d2psi_dZ2 = (psi_pZ - 2*psi_0 + psi_mZ) / h**2
        
        # First derivative in R
        dpsi_dR = (psi_pR - psi_mR) / (2*h)
        
        # Δ* operator
        delta_star = d2psi_dR2 - dpsi_dR / R + d2psi_dZ2
        
        # Source terms (constant for Solov'ev)
        rhs = self.mu0 * R**2 * (-self.c[3] / (self.mu0 * self.R0**2)) + (-self.c[4])
        
        residual = delta_star + rhs
        
        return residual


# =============================================================================
# Validation Test
# =============================================================================

def test_solovev_analytical():
    """Test Solov'ev solution satisfies G-S equation."""
    print("Solov'ev Analytical Solution Validation")
    print("=" * 70)
    print()
    
    # Standard tokamak parameters
    R0 = 3.0   # Major radius (m)
    eps = 0.32  # Inverse aspect ratio
    kappa = 1.7  # Elongation
    delta = 0.33  # Triangularity
    A = -0.155   # Shafranov shift
    
    solovev = SolovevSolution(R0, eps, kappa, delta, A)
    
    print(f"Parameters:")
    print(f"  R₀ = {R0:.2f} m")
    print(f"  ε = {eps:.2f} (a = {solovev.a:.2f} m)")
    print(f"  κ = {kappa:.2f}")
    print(f"  δ = {delta:.2f}")
    print(f"  A = {A:.3f}")
    print()
    
    # Test points
    test_points = [
        (R0, 0.0),              # Magnetic axis
        (R0 + 0.5, 0.0),        # Outboard midplane
        (R0 - 0.5, 0.0),        # Inboard midplane
        (R0, 0.5),              # Top
        (R0, -0.5),             # Bottom
    ]
    
    print("G-S Equation Residual Check:")
    print("=" * 70)
    print(f"{'R (m)':<10} {'Z (m)':<10} {'ψ (Wb/rad)':<15} {'Residual':<15}")
    print("-" * 70)
    
    max_residual = 0
    
    for R, Z in test_points:
        psi = solovev.psi(R, Z)
        residual = solovev.check_gs_equation(R, Z)
        
        max_residual = max(max_residual, abs(residual))
        
        print(f"{R:<10.2f} {Z:<10.2f} {psi:<15.6e} {residual:<15.6e}")
    
    print()
    print(f"Max residual: {max_residual:.6e}")
    print()
    
    if max_residual < 1e-6:
        print("✅ Solov'ev solution satisfies G-S equation (residual < 1e-6)")
        return True
    else:
        print("❌ Large residual (numerical derivative error or wrong formula)")
        return False


if __name__ == '__main__':
    success = test_solovev_analytical()
    exit(0 if success else 1)
