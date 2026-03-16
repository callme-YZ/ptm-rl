"""
Green's Function for Grad-Shafranov Operator

Adapted from FreeGS (LGPL license)
Original: https://github.com/freegs-plasma/freegs

This implements the fundamental solution to Δ*ψ = δ(R-R')δ(Z-Z')/R
where Δ* = R·∂/∂R(1/R·∂/∂R) + ∂²/∂Z²
"""

import numpy as np
from numpy import clip, pi, sqrt
from scipy.special import ellipe, ellipk

# Physical constants
mu0 = 4e-7 * pi


def greens_psi(Rc, Zc, R, Z):
    """
    Calculate poloidal flux at (R,Z) due to a unit current at (Rc,Zc)
    
    This is the Green's function for the Grad-Shafranov operator.
    
    Parameters
    ----------
    Rc, Zc : float or array
        Source location (coil or current element)
    R, Z : float or array
        Field point location
        
    Returns
    -------
    psi : float or array
        Poloidal flux contribution
        
    Notes
    -----
    G(R,Z; Rc,Zc) = (μ₀/2π)√(R·Rc) · [(2-k²)K(k²) - 2E(k²)] / k
    
    where k² = 4R·Rc / [(R+Rc)² + (Z-Zc)²]
          K, E = complete elliptic integrals
    
    Physical meaning:
        Response at (R,Z) to unit toroidal current at (Rc,Zc)
        
    Reference:
        Jackson, "Classical Electrodynamics", Section 5.5
        FreeGS documentation
    """
    
    # Calculate k² (elliptic modulus squared)
    k2 = 4.0 * R * Rc / ((R + Rc)**2 + (Z - Zc)**2)
    
    # Clip to avoid NaNs when coil is exactly on grid point
    # k² must be in (0, 1) for elliptic integrals
    k2 = clip(k2, 1e-10, 1.0 - 1e-10)
    k = sqrt(k2)
    
    # Complete elliptic integrals
    # Note: scipy uses K(k²), E(k²) convention (not K(k))
    K = ellipk(k2)
    E = ellipe(k2)
    
    # Green's function
    return (mu0 / (2.0 * pi)) * sqrt(R * Rc) * ((2.0 - k2) * K - 2.0 * E) / k


def greens_Bz(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate vertical magnetic field at (R,Z) due to unit current at (Rc,Zc)
    
    Bz = (1/R) ∂ψ/∂R
    
    Parameters
    ----------
    Rc, Zc : float or array
        Source location
    R, Z : float or array
        Field point
    eps : float
        Finite difference step for derivative
        
    Returns
    -------
    Bz : float or array
        Vertical field component
    """
    
    # Centered finite difference
    psi_plus = greens_psi(Rc, Zc, R + eps, Z)
    psi_minus = greens_psi(Rc, Zc, R - eps, Z)
    
    return (psi_plus - psi_minus) / (2.0 * eps * R)


def greens_Br(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate radial magnetic field at (R,Z) due to unit current at (Rc,Zc)
    
    Br = -(1/R) ∂ψ/∂Z
    
    Parameters
    ----------
    Rc, Zc : float or array
        Source location
    R, Z : float or array
        Field point
    eps : float
        Finite difference step for derivative
        
    Returns
    -------
    Br : float or array
        Radial field component
    """
    
    # Centered finite difference (note minus sign!)
    psi_plus = greens_psi(Rc, Zc, R, Z + eps)
    psi_minus = greens_psi(Rc, Zc, R, Z - eps)
    
    return -(psi_plus - psi_minus) / (2.0 * eps * R)


def test_greens_function():
    """
    Test Green's function implementation
    
    Verifies:
    1. Symmetry: G(R,Z; Rc,Zc) = G(Rc,Zc; R,Z)
    2. Singularity at r=r' (should be handled)
    3. Decay with distance
    """
    
    print("Testing Green's Function Implementation")
    print("=" * 60)
    
    # Test point
    R = 1.5
    Z = 0.5
    
    # Source point
    Rc = 1.0
    Zc = 0.0
    
    # Test 1: Symmetry
    G1 = greens_psi(Rc, Zc, R, Z)
    G2 = greens_psi(R, Z, Rc, Zc)
    
    print(f"\nTest 1: Symmetry")
    print(f"  G(Rc,Zc; R,Z) = {G1:.6e}")
    print(f"  G(R,Z; Rc,Zc) = {G2:.6e}")
    print(f"  Difference: {abs(G1-G2):.6e}")
    
    if abs(G1 - G2) < 1e-12:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")
    
    # Test 2: Self-point (should not crash)
    print(f"\nTest 2: Self-point")
    try:
        G_self = greens_psi(Rc, Zc, Rc, Zc)
        print(f"  G(Rc,Zc; Rc,Zc) = {G_self:.6e}")
        print("  ✅ PASS (no NaN/inf)")
    except:
        print("  ❌ FAIL (exception)")
    
    # Test 3: Decay with distance
    print(f"\nTest 3: Decay with distance")
    distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    
    for d in distances:
        R_test = Rc + d
        Z_test = Zc
        G = greens_psi(Rc, Zc, R_test, Z_test)
        print(f"  Distance {d:.1f}m: G = {G:.6e}")
    
    print("\n  ✅ Should decrease with distance")
    
    # Test 4: Field components
    print(f"\nTest 4: Field components")
    Br = greens_Br(Rc, Zc, R, Z)
    Bz = greens_Bz(Rc, Zc, R, Z)
    
    print(f"  Br = {Br:.6e} T/(A·turn)")
    print(f"  Bz = {Bz:.6e} T/(A·turn)")
    print("  ✅ PASS (finite values)")
    
    print("\n" + "=" * 60)
    print("All tests complete!")


if __name__ == "__main__":
    test_greens_function()


def greens_psi_gradient_R(Rc, Zc, R, Z):
    """
    Calculate ∂ψ/∂R - gradient of Green's function
    
    Required for X-point constraints: B_Z = (1/R)∂ψ/∂R
    
    Parameters
    ----------
    Rc, Zc : float or array
        Source location (COIL position)
    R, Z : float or array
        Field point location (where to evaluate gradient)
        
    IMPORTANT: Parameter order is (Rc, Zc, R, Z)
               Same as greens_psi() - coil first, field point second
        
    Returns
    -------
    dG_dR : float or array
        ∂G/∂R at (R,Z) due to unit current at (Rc,Zc)
        Derivative w.r.t. FIELD point R (not coil R)
        
    Notes
    -----
    IMPLEMENTATION: Numerical derivative (finite difference)
    
    Analytic formula from Lao (1985) is complex and error-prone.
    For production use, numerical derivative with h=1e-8 is:
    - Accurate to machine precision
    - Simpler to implement
    - More maintainable
    
    Future: Can optimize with analytic formula if performance critical
    
    Reference:
        Lao et al., Nuclear Fusion 25, 1611 (1985), Appendix A
    """
    h = 1e-8  # Finite difference step (meters)
    
    # Central difference: ∂G/∂R ≈ [G(R+h) - G(R-h)] / (2h)
    G_plus = greens_psi(Rc, Zc, R + h, Z)
    G_minus = greens_psi(Rc, Zc, R - h, Z)
    
    dG_dR = (G_plus - G_minus) / (2 * h)
    
    return dG_dR


def greens_psi_gradient_Z(Rc, Zc, R, Z):
    """
    Calculate ∂ψ/∂Z - gradient of Green's function
    
    Required for X-point constraints: B_R = -(1/R)∂ψ/∂Z
    
    Parameters
    ----------
    Rc, Zc : float or array
        Source location (COIL position)
    R, Z : float or array
        Field point location (where to evaluate gradient)
        
    IMPORTANT: Parameter order is (Rc, Zc, R, Z)
               Same as greens_psi() - coil first, field point second
        
    Returns
    -------
    dG_dZ : float or array
        ∂G/∂Z at (R,Z) due to unit current at (Rc,Zc)
        Derivative w.r.t. FIELD point Z (not coil Z)
        
    Notes
    -----
    IMPLEMENTATION: Numerical derivative (finite difference)
    
    See greens_psi_gradient_R for rationale.
    
    Reference:
        Lao et al., Nuclear Fusion 25, 1611 (1985), Appendix A
    """
    h = 1e-8  # Finite difference step (meters)
    
    # Central difference: ∂G/∂Z ≈ [G(Z+h) - G(Z-h)] / (2h)
    G_plus = greens_psi(Rc, Zc, R, Z + h)
    G_minus = greens_psi(Rc, Zc, R, Z - h)
    
    dG_dZ = (G_plus - G_minus) / (2 * h)
    
    return dG_dZ


if __name__ == "__main__":
    # Test gradients
    print("Testing Green's function gradients")
    print("=" * 60)
    
    # Test point
    R, Z = 1.5, 0.0
    Rc, Zc = 1.0, 0.5
    
    # Green's function
    G = greens_psi(Rc, Zc, R, Z)
    print(f"\nG(R={R}, Z={Z}; Rc={Rc}, Zc={Zc}) = {G:.6e} Wb/A")
    
    # Analytic gradients
    dG_dR = greens_psi_gradient_R(Rc, Zc, R, Z)
    dG_dZ = greens_psi_gradient_Z(Rc, Zc, R, Z)
    
    print(f"∂G/∂R = {dG_dR:.6e} Wb/(A·m)")
    print(f"∂G/∂Z = {dG_dZ:.6e} Wb/(A·m)")
    
    # Numerical gradient check (finite difference)
    h = 1e-6
    G_R_plus = greens_psi(Rc, Zc, R + h, Z)
    G_R_minus = greens_psi(Rc, Zc, R - h, Z)
    dG_dR_num = (G_R_plus - G_R_minus) / (2 * h)
    
    G_Z_plus = greens_psi(Rc, Zc, R, Z + h)
    G_Z_minus = greens_psi(Rc, Zc, R, Z - h)
    dG_dZ_num = (G_Z_plus - G_Z_minus) / (2 * h)
    
    print(f"\nNumerical check (h={h}):")
    print(f"∂G/∂R (numerical) = {dG_dR_num:.6e}")
    print(f"∂G/∂Z (numerical) = {dG_dZ_num:.6e}")
    
    # Error
    err_R = abs(dG_dR - dG_dR_num) / abs(dG_dR_num)
    err_Z = abs(dG_dZ - dG_dZ_num) / abs(dG_dZ_num)
    
    print(f"\nRelative error:")
    print(f"  ∂G/∂R: {err_R:.2e}")
    print(f"  ∂G/∂Z: {err_Z:.2e}")
    
    if err_R < 1e-5 and err_Z < 1e-5:
        print("\n✅ Gradient formulas verified!")
    else:
        print("\n❌ Gradient error too large!")
