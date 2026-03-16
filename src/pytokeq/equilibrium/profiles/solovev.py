"""
Solov'ev Analytical Equilibrium Solution

Reference: L.E. Solov'ev, Sov. Phys. JETP 26, 400 (1968)

Analytical solution to G-S equation with linear profiles:
    p(ψ) = p0 + p1·ψ
    f²(ψ) = f0² + f1·ψ

Solution:
    ψ(R,Z) = A·(R²·Z² + C·R⁴)

where A, C are constants determined by boundary conditions.
"""

import numpy as np


class SolovevEquilibrium:
    """
    Analytical Solov'ev equilibrium for validation.
    
    Parameters
    ----------
    A : float
        Amplitude parameter
    C : float
        R^4 coefficient (controls ellipticity)
    p0, p1 : float
        Pressure profile coefficients
    f0, f1 : float
        Toroidal field profile coefficients
    mu0 : float
        Magnetic permeability (default: 4π×10^-7)
    """
    
    def __init__(self, A=1.0, C=-0.1, p0=1e5, p1=1e4, 
                 f0=1.0, f1=0.1, mu0=4*np.pi*1e-7):
        self.A = A
        self.C = C
        self.p0 = p0
        self.p1 = p1
        self.f0 = f0
        self.f1 = f1
        self.mu0 = mu0
        
        # Verify consistency (G-S equation should be satisfied)
        self._verify_consistency()
    
    def _verify_consistency(self):
        """
        Verify that parameters satisfy G-S equation.
        
        For Solov'ev solution:
            Δ*ψ = -μ0·R²·p' - f·f'
        
        With linear profiles:
            p' = p1 (constant)
            f·f' = f1/2 (constant)
        
        And ψ = A(R²Z² + CR⁴):
            Δ*ψ = 2A(1 + 4C)R²
        
        Consistency requires:
            2A(1 + 4C) = -μ0·p1 - f1/2
        """
        lhs = 2 * self.A * (1 + 4*self.C)
        rhs = -self.mu0 * self.p1 - self.f1 / 2
        
        if not np.isclose(lhs, rhs, rtol=1e-6):
            raise ValueError(
                f"Solov'ev parameters inconsistent!\n"
                f"  Δ*ψ = {lhs:.6e}\n"
                f"  RHS  = {rhs:.6e}\n"
                f"  Must have: 2A(1+4C) = -μ0·p1 - f1/2"
            )
    
    def psi(self, R, Z):
        """Poloidal flux."""
        return self.A * (R**2 * Z**2 + self.C * R**4)
    
    def psi_R(self, R, Z):
        """∂ψ/∂R"""
        return self.A * (2*R*Z**2 + 4*self.C*R**3)
    
    def psi_Z(self, R, Z):
        """∂ψ/∂Z"""
        return self.A * (2*R**2*Z)
    
    def psi_RR(self, R, Z):
        """∂²ψ/∂R²"""
        return self.A * (2*Z**2 + 12*self.C*R**2)
    
    def psi_ZZ(self, R, Z):
        """∂²ψ/∂Z²"""
        return self.A * (2*R**2)
    
    def delta_star_psi(self, R, Z):
        """
        Δ*ψ = R·∂/∂R(1/R·∂ψ/∂R) + ∂²ψ/∂Z²
        """
        # Δ*ψ = ∂²ψ/∂R² - (1/R)·∂ψ/∂R + ∂²ψ/∂Z²
        return (
            self.psi_RR(R, Z) 
            - self.psi_R(R, Z) / R 
            + self.psi_ZZ(R, Z)
        )
    
    def pressure(self, psi):
        """p(ψ) = p0 + p1·ψ"""
        return self.p0 + self.p1 * psi
    
    def pressure_prime(self, psi):
        """dp/dψ = p1"""
        return self.p1
    
    def f_function(self, psi):
        """f(ψ) = sqrt(f0² + f1·ψ)"""
        return np.sqrt(self.f0**2 + self.f1 * psi)
    
    def ff_prime(self, psi):
        """f·df/dψ = f1/2"""
        return self.f1 / 2
    
    def rhs_gs(self, R, Z):
        """
        RHS of G-S equation: -μ0·R²·p' - f·f'
        
        For Solov'ev (linear profiles), this is constant!
        """
        return -self.mu0 * R**2 * self.p1 - self.f1 / 2
    
    def verify_solution(self, R, Z, rtol=1e-6):
        """
        Verify that ψ(R,Z) satisfies G-S equation.
        
        Returns
        -------
        satisfied : bool
        error : float
            Relative error ||Δ*ψ - RHS|| / ||RHS||
        """
        lhs = self.delta_star_psi(R, Z)
        rhs = self.rhs_gs(R, Z)
        
        error = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
        satisfied = error < rtol
        
        return satisfied, error
    
    def magnetic_field(self, R, Z):
        """
        Compute magnetic field components.
        
        Returns
        -------
        B_R, B_Z, B_phi : ndarray
            Magnetic field in cylindrical coordinates
        """
        psi_val = self.psi(R, Z)
        
        # Poloidal field
        B_R = -self.psi_Z(R, Z) / R
        B_Z = self.psi_R(R, Z) / R
        
        # Toroidal field
        B_phi = self.f_function(psi_val) / R
        
        return B_R, B_Z, B_phi
    
    def current_density(self, R, Z):
        """
        Toroidal current density.
        
        J_phi = R·p' + (1/μ0R)·f·f'
        """
        psi_val = self.psi(R, Z)
        
        J_phi = (
            R * self.pressure_prime(psi_val)
            + self.ff_prime(psi_val) / (self.mu0 * R)
        )
        
        return J_phi
    
    def plasma_current(self, R_min, R_max, Z_min, Z_max, nr=100, nz=100):
        """
        Compute total plasma current in rectangular region.
        
        I_p = ∫∫ J_phi dR dZ
        """
        R = np.linspace(R_min, R_max, nr)
        Z = np.linspace(Z_min, Z_max, nz)
        RR, ZZ = np.meshgrid(R, Z, indexing='ij')
        
        J = self.current_density(RR, ZZ)
        
        # Integrate
        dR = (R_max - R_min) / (nr - 1)
        dZ = (Z_max - Z_min) / (nz - 1)
        I_p = np.sum(J) * dR * dZ
        
        return I_p


def make_standard_solovev():
    """
    Create standard Solov'ev equilibrium for testing.
    
    Parameters chosen to give reasonable tokamak-like profiles.
    """
    # Choose A, C to give plasma in roughly [R=3-7, Z=-2 to 2]
    A = -0.01
    C = -0.05
    
    # Pressure profile
    p0 = 1e5  # 1 bar
    p1 = -1e4  # Decreasing with ψ
    
    # Toroidal field
    f0 = 5.0   # ~5 T at R=1m
    f1 = 0.0   # Constant f for simplicity
    
    # Verify consistency: 2A(1+4C) = -μ0·p1 - f1/2
    mu0 = 4*np.pi*1e-7
    # With f1=0:
    # 2A(1+4C) = -μ0·p1
    # p1 = -2A(1+4C)/μ0
    p1 = -2 * A * (1 + 4*C) / mu0
    
    return SolovevEquilibrium(
        A=A, C=C, p0=p0, p1=p1, f0=f0, f1=f1, mu0=mu0
    )


if __name__ == '__main__':
    # Test Solov'ev solution
    sol = make_standard_solovev()
    
    # Create grid
    R = np.linspace(1, 8, 100)
    Z = np.linspace(-3, 3, 100)
    RR, ZZ = np.meshgrid(R, Z)
    
    # Verify solution
    satisfied, error = sol.verify_solution(RR, ZZ)
    
    print("Solov'ev Equilibrium Test")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  A = {sol.A:.6f}")
    print(f"  C = {sol.C:.6f}")
    print(f"  p1 = {sol.p1:.6e}")
    print(f"  f0 = {sol.f0:.6f}")
    print(f"  f1 = {sol.f1:.6f}")
    print()
    print(f"G-S equation satisfied: {satisfied}")
    print(f"Relative error: {error:.6e}")
    print()
    
    # Compute at center
    R0, Z0 = 4.5, 0.0
    psi0 = sol.psi(R0, Z0)
    B_R, B_Z, B_phi = sol.magnetic_field(R0, Z0)
    J_phi = sol.current_density(R0, Z0)
    
    print(f"At (R,Z) = ({R0}, {Z0}):")
    print(f"  ψ = {psi0:.6e}")
    print(f"  B_R = {B_R:.6e} T")
    print(f"  B_Z = {B_Z:.6e} T")
    print(f"  B_φ = {B_phi:.6e} T")
    print(f"  J_φ = {J_phi:.6e} A/m²")
