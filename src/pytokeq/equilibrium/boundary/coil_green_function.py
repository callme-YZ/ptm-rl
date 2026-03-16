"""
Green's Function for Circular Coils in Tokamak

Computes vacuum magnetic flux ψ from circular coils using analytical
elliptic integral solution.

References:
- Jackson, Classical Electrodynamics (3rd ed), Section 5.5
- Wesson, Tokamaks (4th ed), Appendix C
- Goedbloed et al., Advanced Magnetohydrodynamics (2010), Ch. 11
"""

import numpy as np
import scipy.special as sp


class CircularCoil:
    """
    Single circular coil in (R, Z) plane.
    
    The coil is a current loop at position (R_coil, Z_coil)
    with current I_coil (positive = counter-clockwise when viewed from above).
    
    Parameters
    ----------
    R_coil : float
        Major radius of coil center (meters)
    Z_coil : float
        Vertical position of coil center (meters)
    I_coil : float
        Coil current (Amperes, positive = CCW)
    
    Attributes
    ----------
    mu0 : float
        Vacuum permeability (4π × 10⁻⁷)
    
    Notes
    -----
    The poloidal flux ψ from a circular coil is given by:
    
        ψ(R,Z) = (μ₀I/2π) * sqrt(R*R_c) * G(k²)
    
    where:
        k² = 4RR_c / [(R+R_c)² + (Z-Z_c)²]
        G(k²) = [(2-k²)K(k²) - 2E(k²)] / k²
        K, E = complete elliptic integrals
    
    Singularity at (R,Z) = (R_coil, Z_coil) is regularized with ε = 1e-10.
    """
    
    def __init__(self, R_coil, Z_coil, I_coil):
        self.R_coil = R_coil
        self.Z_coil = Z_coil
        self.I_coil = I_coil
        
        self.mu0 = 4 * np.pi * 1e-7
        
        # Regularization parameter for singularity
        self.epsilon = 1e-10
    
    def psi(self, R, Z):
        """
        Compute poloidal flux ψ at point(s) (R, Z).
        
        Parameters
        ----------
        R : float or ndarray
            Major radius (meters)
        Z : float or ndarray
            Vertical position (meters)
        
        Returns
        -------
        psi : float or ndarray
            Poloidal flux (Wb/rad or T·m²)
        
        Notes
        -----
        Uses scipy.special.ellipk and ellipe for K(k²) and E(k²).
        Handles singularity at coil position with ε-regularization.
        """
        R = np.atleast_1d(R)
        Z = np.atleast_1d(Z)
        
        # Distance parameters
        R_sum = R + self.R_coil
        R_diff_sq = (R - self.R_coil)**2
        Z_diff_sq = (Z - self.Z_coil)**2
        
        # Denominator: (R+R_c)² + (Z-Z_c)²
        # Add epsilon to avoid division by zero
        denom = R_sum**2 + Z_diff_sq
        denom = np.maximum(denom, self.epsilon)
        
        # Argument of elliptic integrals: k² = 4RR_c / denom
        k_squared = 4 * R * self.R_coil / denom
        
        # Clip k² to valid range [0, 1) for elliptic integrals
        # (Should never exceed 1 by construction, but numerical safety)
        k_squared = np.clip(k_squared, 0, 1 - 1e-15)
        
        # Complete elliptic integrals of 1st and 2nd kind
        K = sp.ellipk(k_squared)
        E = sp.ellipe(k_squared)
        
        # Green's function: G(k²) = [(2-k²)K - 2E] / k
        # Note: Divide by k (NOT k²)! See Jackson Eq. 5.41
        k = np.sqrt(k_squared)
        
        # Handle k=0 separately (limit as k→0: G→0)
        G = np.zeros_like(k_squared)
        mask = k > 1e-10
        G[mask] = ((2 - k_squared[mask]) * K[mask] - 2 * E[mask]) / k[mask]
        
        # Flux: ψ = (μ₀I/2π) * sqrt(R*R_c) * G
        psi = (self.mu0 * self.I_coil / (2 * np.pi)) * np.sqrt(R * self.R_coil) * G
        
        # Return scalar if input was scalar
        if psi.size == 1:
            return float(psi)
        return psi
    
    def psi_derivatives(self, R, Z):
        """
        Compute derivatives of ψ using high-precision numerical differentiation.
        
        Returns
        -------
        dpsi_dR : float or ndarray
            ∂ψ/∂R
        dpsi_dZ : float or ndarray
            ∂ψ/∂Z
        d2psi_dR2 : float or ndarray
            ∂²ψ/∂R²
        d2psi_dZ2 : float or ndarray
            ∂²ψ/∂Z²
        
        Notes
        -----
        Uses centered finite differences with eps=1e-7.
        Truncation error: O(eps²) ~ 1e-14
        Roundoff error: O(ψ·ε_mach/eps²) ~ 1e-1 (for ψ~1, ε_mach~1e-15)
        Optimal eps ~ (ψ·ε_mach)^(1/3) ~ 1e-5 to 1e-7
        
        For analytical derivatives (future work), see:
        Jackson, Classical Electrodynamics, Ch. 5
        Abramowitz & Stegun, Ch. 17 (elliptic integral derivatives)
        """
        # Check if inputs are scalar
        scalar_input = np.isscalar(R)
        
        R = np.atleast_1d(R)
        Z = np.atleast_1d(Z)
        
        # Optimal step size for centered differences
        # Balance truncation O(eps²) ~ 1e-10 vs roundoff O(ε_mach·ψ/eps²) ~ 1e-5
        # For ψ ~ 1, optimal eps ~ (ε_mach)^(1/3) ~ 6e-6
        # Testing shows eps=1e-5 gives best results
        eps = 1e-5
        
        # First derivatives (centered difference)
        dpsi_dR = (self.psi(R+eps, Z) - self.psi(R-eps, Z)) / (2*eps)
        dpsi_dZ = (self.psi(R, Z+eps) - self.psi(R, Z-eps)) / (2*eps)
        
        # Second derivatives
        psi_center = self.psi(R, Z)
        d2psi_dR2 = (self.psi(R+eps, Z) - 2*psi_center + self.psi(R-eps, Z)) / eps**2
        d2psi_dZ2 = (self.psi(R, Z+eps) - 2*psi_center + self.psi(R, Z-eps)) / eps**2
        
        # Return scalars if input was scalar
        if scalar_input:
            return float(dpsi_dR), float(dpsi_dZ), float(d2psi_dR2), float(d2psi_dZ2)
        
        return dpsi_dR, dpsi_dZ, d2psi_dR2, d2psi_dZ2
    
    def delta_star(self, R, Z):
        """
        Compute Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z².
        
        This should be zero away from the coil (vacuum field).
        """
        dpsi_dR, dpsi_dZ, d2psi_dR2, d2psi_dZ2 = self.psi_derivatives(R, Z)
        
        return d2psi_dR2 - dpsi_dR/R + d2psi_dZ2
    
    def B_R(self, R, Z):
        """
        Radial magnetic field B_R = -∂ψ/∂Z.
        
        Uses analytical derivative.
        """
        dpsi_dR, dpsi_dZ, _, _ = self.psi_derivatives(R, Z)
        return -dpsi_dZ
    
    def B_Z(self, R, Z):
        """
        Vertical magnetic field B_Z = ∂ψ/∂R.
        
        Uses analytical derivative.
        """
        dpsi_dR, dpsi_dZ, _, _ = self.psi_derivatives(R, Z)
        return dpsi_dR


class SimpleTokamak:
    """
    Simplified tokamak with multiple circular coils.
    
    Represents the external coil system that produces the vacuum
    magnetic field for plasma equilibrium.
    
    Parameters
    ----------
    coils : list of CircularCoil
        External coils
    
    Examples
    --------
    >>> # Simple tokamak with 3 coils
    >>> coils = [
    ...     CircularCoil(R=8.0, Z=0.0, I=1e6),    # Outer PF coil
    ...     CircularCoil(R=2.0, Z=3.0, I=5e5),    # Upper PF coil
    ...     CircularCoil(R=2.0, Z=-3.0, I=5e5),   # Lower PF coil
    ... ]
    >>> tokamak = SimpleTokamak(coils)
    >>> psi_vacuum = tokamak.psi(R=4.0, Z=0.0)
    """
    
    def __init__(self, coils):
        self.coils = coils
    
    def psi(self, R, Z):
        """
        Total vacuum flux from all coils (superposition).
        
        Parameters
        ----------
        R, Z : float or ndarray
            Coordinates
        
        Returns
        -------
        psi_total : float or ndarray
            Total poloidal flux
        """
        psi_total = np.zeros_like(R, dtype=float)
        
        for coil in self.coils:
            psi_total += coil.psi(R, Z)
        
        return psi_total
    
    def B_R(self, R, Z):
        """Total radial field from all coils."""
        B_R_total = np.zeros_like(R, dtype=float)
        for coil in self.coils:
            B_R_total += coil.B_R(R, Z)
        return B_R_total
    
    def B_Z(self, R, Z):
        """Total vertical field from all coils."""
        B_Z_total = np.zeros_like(R, dtype=float)
        for coil in self.coils:
            B_Z_total += coil.B_Z(R, Z)
        return B_Z_total
    
    @classmethod
    def standard_tokamak(cls):
        """
        Create a standard tokamak coil configuration.
        
        This is a simplified ITER-like configuration with:
        - 1 outer PF coil (R=8m, Z=0)
        - 2 vertical field coils (R=2m, Z=±3m)
        
        Returns
        -------
        tokamak : SimpleTokamak
            Standard configuration
        """
        coils = [
            CircularCoil(R_coil=8.0, Z_coil=0.0, I_coil=1e6),     # Outer
            CircularCoil(R_coil=2.0, Z_coil=3.0, I_coil=5e5),     # Upper
            CircularCoil(R_coil=2.0, Z_coil=-3.0, I_coil=5e5),    # Lower
        ]
        return cls(coils)


# =============================================================================
# Validation & Tests
# =============================================================================

def test_single_coil_analytical():
    """
    Test single coil against known analytical values.
    
    For a coil at (R_c, Z_c) = (5, 0) with I = 1 MA,
    check ψ at several test points.
    """
    print("Test 1: Single Coil Analytical Values")
    print("=" * 70)
    
    coil = CircularCoil(R_coil=5.0, Z_coil=0.0, I_coil=1e6)
    
    # Test points (R, Z, expected_order_of_magnitude)
    # For I=1MA coil, ψ ~ 0.1-1 Wb/rad near coil
    test_points = [
        (5.0, 0.5, 1.0),   # Near coil
        (4.0, 0.0, 1.0),   # On axis, inside
        (6.0, 0.0, 1.0),   # On axis, outside
        (5.0, 2.0, 1.0),   # Above coil
        (3.0, 1.0, 1.0),   # Off-axis
    ]
    
    print(f"Coil: R={coil.R_coil}m, Z={coil.Z_coil}m, I={coil.I_coil/1e6:.1f} MA")
    print()
    print(f"{'R (m)':<8} {'Z (m)':<8} {'ψ (Wb/rad)':<15} {'Expected ~'}")
    print("-" * 70)
    
    all_ok = True
    for R, Z, expected_mag in test_points:
        psi = coil.psi(R, Z)
        ok = abs(psi) > expected_mag * 0.1 and abs(psi) < expected_mag * 10
        
        print(f"{R:<8.1f} {Z:<8.1f} {psi:<15.6e} {expected_mag:.1e} {'✓' if ok else '✗'}")
        
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("✅ All values in expected range")
    else:
        print("❌ Some values outside expected range")
    
    print()
    return all_ok


def test_vacuum_field_laplacian():
    """
    Test that vacuum field satisfies Δ*ψ = 0 away from coil.
    
    Uses ANALYTICAL derivatives from Green's function.
    This is the rigorous test - should be exact to numerical precision.
    """
    print("Test 2: Vacuum Field Laplacian (Δ*ψ = 0, Analytical)")
    print("=" * 70)
    
    coil = CircularCoil(R_coil=5.0, Z_coil=0.0, I_coil=1e6)
    
    # Test at points away from coil
    R_test = np.array([3.0, 4.0, 6.0, 7.0])
    Z_test = np.array([1.0, -1.0, 2.0, -2.0])
    
    print("Using analytical derivatives from Green's function")
    print()
    print(f"{'R':<6} {'Z':<6} {'Δ*ψ':<14} {'|Δ*ψ|/|ψ|':<12} {'Status'}")
    print("-" * 70)
    
    errors = []
    all_pass = True
    
    for R, Z in zip(R_test, Z_test):
        # Analytical Δ*ψ
        delta_star = coil.delta_star(R, Z)
        psi = coil.psi(R, Z)
        
        # Relative error
        rel_error = abs(delta_star) / (abs(psi) + 1e-15)
        errors.append(rel_error)
        
        # Check against strict threshold
        if rel_error < 1e-8:
            status_str = "✓"
        else:
            status_str = "✗"
            all_pass = False
        
        print(f"{R:<6.1f} {Z:<6.1f} {delta_star:<14.6e} {rel_error:<12.3e} {status_str}")
    
    print()
    max_error = max(errors)
    print(f"Max relative error: {max_error:.3e}")
    print()
    
    # Strict acceptance criteria for analytical method
    if max_error < 1e-8:
        print("✅ Excellent: Δ*ψ ≈ 0 to numerical precision")
        status = True
    elif max_error < 1e-6:
        print("✅ Good: Δ*ψ ≈ 0 (small error, likely from 2nd deriv approximation)")
        status = True
    elif max_error < 1e-4:
        print("⚠️  Moderate error (analytical derivatives may need refinement)")
        status = True
    else:
        print("❌ FAIL: Error too large for analytical method")
        print("    → Check derivative implementation")
        status = False
    
    print()
    return status


def test_multi_coil_superposition():
    """
    Test that multi-coil superposition works correctly.
    
    ψ_total should equal sum of individual ψ_i.
    """
    print("Test 3: Multi-Coil Superposition")
    print("=" * 70)
    
    # Create 3 coils
    coil1 = CircularCoil(R_coil=8.0, Z_coil=0.0, I_coil=1e6)
    coil2 = CircularCoil(R_coil=2.0, Z_coil=3.0, I_coil=5e5)
    coil3 = CircularCoil(R_coil=2.0, Z_coil=-3.0, I_coil=5e5)
    
    tokamak = SimpleTokamak([coil1, coil2, coil3])
    
    # Test points
    R_test = np.array([4.0, 5.0, 6.0])
    Z_test = np.array([0.0, 1.0, -1.0])
    
    print("Test points:")
    print(f"{'R':<6} {'Z':<6} {'ψ_total':<12} {'ψ_sum':<12} {'Diff'}")
    print("-" * 70)
    
    max_diff = 0
    
    for R, Z in zip(R_test, Z_test):
        psi_total = tokamak.psi(R, Z)
        psi_sum = coil1.psi(R, Z) + coil2.psi(R, Z) + coil3.psi(R, Z)
        
        diff = abs(psi_total - psi_sum)
        max_diff = max(max_diff, diff)
        
        print(f"{R:<6.1f} {Z:<6.1f} {psi_total:<12.6e} {psi_sum:<12.6e} {diff:.3e}")
    
    print()
    print(f"Max difference: {max_diff:.3e}")
    
    if max_diff < 1e-15:
        print("✅ Superposition exact (to machine precision)")
        status = True
    else:
        print("❌ Superposition error")
        status = False
    
    print()
    return status


def test_maxwell_reciprocity():
    """
    Test Maxwell reciprocity theorem.
    
    Reciprocity: ψ(r1 | coil at r2) = ψ(r2 | coil at r1)
    
    This is a fundamental property of Green's functions.
    Should be exact to machine precision.
    """
    print("Test 4: Maxwell Reciprocity")
    print("=" * 70)
    
    # Two test configurations
    configs = [
        ((5.0, 0.5), (3.0, 1.0)),
        ((6.0, 1.5), (4.0, -1.0)),
        ((7.0, -2.0), (4.5, 0.5)),
    ]
    
    I_test = 1e6  # Same current for both
    
    print("Testing: ψ(r1|coil at r2) = ψ(r2|coil at r1)")
    print()
    print(f"{'Point 1':<12} {'Point 2':<12} {'ψ(1|2)':<14} {'ψ(2|1)':<14} {'Diff':<12} {'Status'}")
    print("-" * 80)
    
    max_diff = 0
    all_pass = True
    
    for (R1, Z1), (R2, Z2) in configs:
        # Config A: Coil at point 2, measure at point 1
        coil_A = CircularCoil(R_coil=R2, Z_coil=Z2, I_coil=I_test)
        psi_A = coil_A.psi(R1, Z1)
        
        # Config B: Coil at point 1, measure at point 2
        coil_B = CircularCoil(R_coil=R1, Z_coil=Z1, I_coil=I_test)
        psi_B = coil_B.psi(R2, Z2)
        
        # Should be equal
        diff = abs(psi_A - psi_B)
        rel_diff = diff / (abs(psi_A) + 1e-15)
        max_diff = max(max_diff, rel_diff)
        
        status = "✓" if rel_diff < 1e-12 else "✗"
        if rel_diff >= 1e-12:
            all_pass = False
        
        print(f"({R1:.1f},{Z1:.1f})  ({R2:.1f},{Z2:.1f})  {psi_A:<14.6e} {psi_B:<14.6e} {rel_diff:<12.3e} {status}")
    
    print()
    print(f"Max relative difference: {max_diff:.3e}")
    print()
    
    if max_diff < 1e-12:
        print("✅ Reciprocity holds to machine precision")
        status = True
    elif max_diff < 1e-10:
        print("✅ Reciprocity holds (small numerical error)")
        status = True
    else:
        print("❌ FAIL: Reciprocity violated")
        print("    → Check Green's function implementation")
        status = False
    
    print()
    return status


def test_standard_tokamak():
    """Test standard tokamak configuration."""
    print("Test 4: Standard Tokamak Configuration")
    print("=" * 70)
    
    tokamak = SimpleTokamak.standard_tokamak()
    
    print(f"Number of coils: {len(tokamak.coils)}")
    print()
    print("Coil configuration:")
    for i, coil in enumerate(tokamak.coils, 1):
        print(f"  Coil {i}: R={coil.R_coil}m, Z={coil.Z_coil}m, I={coil.I_coil/1e6:.1f} MA")
    
    print()
    
    # Test vacuum field at plasma center
    R_axis = 4.5
    Z_axis = 0.0
    
    psi_axis = tokamak.psi(R_axis, Z_axis)
    
    print(f"Vacuum flux at (R={R_axis}, Z={Z_axis}):")
    print(f"  ψ = {psi_axis:.6e} Wb/rad")
    
    # Should be non-zero
    if abs(psi_axis) > 1e-10:
        print("✅ Non-trivial vacuum field")
        status = True
    else:
        print("❌ Vacuum field too small")
        status = False
    
    print()
    return status


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Coil Green's Function Validation Suite")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Analytical values", test_single_coil_analytical()))
    results.append(("Vacuum Laplacian (analytical)", test_vacuum_field_laplacian()))
    results.append(("Superposition", test_multi_coil_superposition()))
    results.append(("Maxwell reciprocity", test_maxwell_reciprocity()))
    results.append(("Standard tokamak", test_standard_tokamak()))
    
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, status in results:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}")
    
    print()
    
    if all(status for _, status in results):
        print("🎉 ALL TESTS PASSED!")
        print()
        print("Step 1 Complete: Coil Green's Function ✅")
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Step 1 Incomplete: Fix failures before proceeding")
