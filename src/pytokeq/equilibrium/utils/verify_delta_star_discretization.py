"""
Verify Δ* Discretization - Test on Analytical Solution

Following小A's systematic approach:
  Test linear solver on case with known analytical solution
  Identify if stencil is correct or not
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from equilibrium.picard_linear_solver_fixed_v2 import solve_gs_one_sweep_v2, MU0


def analytical_delta_star(psi_func, R, Z, dR, dZ):
    """
    Compute Δ*ψ analytically for given function
    
    Δ* = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²
       = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
    """
    # Compute derivatives using finite differences
    dpsi_dR = (psi_func(R + dR, Z) - psi_func(R - dR, Z)) / (2*dR)
    d2psi_dR2 = (psi_func(R + dR, Z) - 2*psi_func(R, Z) + psi_func(R - dR, Z)) / dR**2
    d2psi_dZ2 = (psi_func(R, Z + dZ) - 2*psi_func(R, Z) + psi_func(R, Z - dZ)) / dZ**2
    
    delta_star = d2psi_dR2 - dpsi_dR/R + d2psi_dZ2
    
    return delta_star


def test_case_1_quadratic():
    """
    Test Case 1: ψ = R²
    
    Analytical:
      ∂ψ/∂R = 2R
      ∂²ψ/∂R² = 2
      ∂²ψ/∂Z² = 0
      
      Δ*ψ = 2 - (1/R)×2R + 0 = 2 - 2 = 0
      
    So source should be 0 everywhere
    """
    print("\n" + "="*70)
    print("TEST CASE 1: ψ = R²")
    print("="*70)
    
    # Grid
    R_1d = np.linspace(1.0, 2.0, 65)
    Z_1d = np.linspace(-0.5, 0.5, 65)
    RR, ZZ = np.meshgrid(R_1d, Z_1d, indexing='ij')
    dR = R_1d[1] - R_1d[0]
    dZ = Z_1d[1] - Z_1d[0]
    
    # Analytical solution
    psi_analytical = RR**2
    
    # Analytical Δ*ψ
    # For ψ = R²: Δ*ψ = 2 - 2R/R = 0
    delta_star_analytical = np.zeros_like(RR)
    
    # Source term (should be zero)
    Jtor = np.zeros_like(RR)  # No current
    
    print(f"\nSetup:")
    print(f"  Grid: {RR.shape[0]}×{RR.shape[1]}")
    print(f"  ψ_analytical = R²")
    print(f"  Δ*ψ_analytical = 0 (everywhere)")
    print(f"  Expected: Solver should preserve ψ=R²")
    
    # Run solver
    psi_computed = solve_gs_one_sweep_v2(psi_analytical, RR, ZZ, Jtor, omega=1.0)
    
    # Check error
    error = psi_computed - psi_analytical
    error_norm = np.linalg.norm(error)
    error_max = np.abs(error).max()
    
    print(f"\nResults:")
    print(f"  ||ψ_computed - ψ_analytical|| = {error_norm:.3e}")
    print(f"  max|ψ_computed - ψ_analytical| = {error_max:.3e}")
    
    # Analysis
    print(f"\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    if error_max < 1e-10:
        print(f"  ✅ PASS: Stencil preserves ψ=R² (error negligible)")
        print(f"     Discretization likely correct for this case")
    else:
        print(f"  ❌ FAIL: Stencil does NOT preserve ψ=R²")
        print(f"     Error = {error_max:.3e} (should be ~0)")
        print(f"     → Boundary handling or omega issue?")
    
    return error_max


def test_case_2_parabolic():
    """
    Test Case 2: ψ = -(R² + Z²)
    
    Analytical:
      ∂ψ/∂R = -2R
      ∂²ψ/∂R² = -2
      ∂²ψ/∂Z² = -2
      
      Δ*ψ = -2 - (1/R)×(-2R) + (-2)
          = -2 + 2 - 2
          = -2
      
    So Jtor should be constant: -Δ*ψ/(μ₀R) = 2/(μ₀R)
    """
    print("\n" + "="*70)
    print("TEST CASE 2: ψ = -(R² + Z²)")
    print("="*70)
    
    # Grid
    R_1d = np.linspace(1.0, 2.0, 65)
    Z_1d = np.linspace(-0.5, 0.5, 65)
    RR, ZZ = np.meshgrid(R_1d, Z_1d, indexing='ij')
    dR = R_1d[1] - R_1d[0]
    dZ = Z_1d[1] - Z_1d[0]
    
    # Analytical solution
    psi_analytical = -(RR**2 + ZZ**2)
    
    # Analytical Δ*ψ = -2
    delta_star_analytical = -2.0 * np.ones_like(RR)
    
    # Current density: Jtor = -Δ*ψ/(μ₀R) = 2/(μ₀R)
    Jtor = 2.0 / (MU0 * RR)
    
    print(f"\nSetup:")
    print(f"  Grid: {RR.shape[0]}×{RR.shape[1]}")
    print(f"  ψ_analytical = -(R² + Z²)")
    print(f"  Δ*ψ_analytical = -2")
    print(f"  J_φ = 2/(μ₀R)")
    
    # Initial guess (zeros)
    psi_guess = np.zeros_like(RR)
    psi_guess[0,:] = psi_analytical[0,:]  # BC at R=R_min
    psi_guess[-1,:] = psi_analytical[-1,:]  # BC at R=R_max
    psi_guess[:,0] = psi_analytical[:,0]  # BC at Z=Z_min
    psi_guess[:,-1] = psi_analytical[:,-1]  # BC at Z=Z_max
    
    print(f"\nRunning solver (20 sweeps)...")
    
    # Run multiple sweeps
    psi_computed = psi_guess.copy()
    for sweep in range(20):
        psi_computed = solve_gs_one_sweep_v2(psi_computed, RR, ZZ, Jtor, omega=1.5)
        
        if sweep % 5 == 4:
            error = np.linalg.norm(psi_computed - psi_analytical)
            print(f"  Sweep {sweep+1}: ||error|| = {error:.3e}")
    
    # Final error
    error = psi_computed - psi_analytical
    error_interior = error[1:-1, 1:-1]  # Exclude boundary
    error_norm = np.linalg.norm(error_interior)
    error_max = np.abs(error_interior).max()
    
    print(f"\nResults:")
    print(f"  ||ψ_computed - ψ_analytical|| (interior) = {error_norm:.3e}")
    print(f"  max|ψ_computed - ψ_analytical| (interior) = {error_max:.3e}")
    
    # Analysis
    print(f"\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    if error_max < 0.01:
        print(f"  ✅ PASS: Converges to analytical solution")
        print(f"     Discretization correct!")
    elif error_max < 0.1:
        print(f"  ⚠️  ACCEPTABLE: Converges but slowly")
        print(f"     May need more sweeps or better BC")
    else:
        print(f"  ❌ FAIL: Does NOT converge to analytical")
        print(f"     Error = {error_max:.3e}")
        print(f"     → Stencil likely incorrect!")
    
    return error_max


def verify_stencil_coefficients():
    """
    Verify the actual stencil coefficients used in solver
    
    Compare with analytical formula
    """
    print("\n" + "="*70)
    print("STENCIL COEFFICIENT VERIFICATION")
    print("="*70)
    
    # Test point
    R_test = 1.5
    dR = 0.01
    dZ = 0.01
    
    print(f"\nTest point: R = {R_test} m")
    print(f"Grid spacing: dR = {dR} m, dZ = {dZ} m")
    
    # Analytical coefficients for Δ*
    coeff_im_correct = 1/dR**2 - 1/(2*R_test*dR)  # ψ_{i-1,j}
    coeff_ip_correct = 1/dR**2 + 1/(2*R_test*dR)  # ψ_{i+1,j}
    coeff_jm_correct = 1/dZ**2                     # ψ_{i,j-1}
    coeff_jp_correct = 1/dZ**2                     # ψ_{i,j+1}
    coeff_ij_correct = -(2/dR**2 + 2/dZ**2)       # ψ_{i,j}
    
    print(f"\nCorrect coefficients (5-point stencil):")
    print(f"  ψ_{{i-1,j}}: {coeff_im_correct:+.6e}")
    print(f"  ψ_{{i+1,j}}: {coeff_ip_correct:+.6e}")
    print(f"  ψ_{{i,j-1}}: {coeff_jm_correct:+.6e}")
    print(f"  ψ_{{i,j+1}}: {coeff_jp_correct:+.6e}")
    print(f"  ψ_{{i,j}}:   {coeff_ij_correct:+.6e}")
    
    # Sum should be zero (consistency)
    coeff_sum = (coeff_im_correct + coeff_ip_correct + 
                 coeff_jm_correct + coeff_jp_correct + coeff_ij_correct)
    
    print(f"\n  Sum of coefficients: {coeff_sum:.3e}")
    
    if abs(coeff_sum) < 1e-10:
        print(f"  ✅ Coefficients consistent (sum ≈ 0)")
    else:
        print(f"  ⚠️  Coefficients may be inconsistent")
    
    print(f"\n" + "="*70)
    print("Compare with solver implementation:")
    print("="*70)
    print(f"\nRead picard_linear_solver_fixed_v2.py lines 30-38")
    print(f"Check if coefficients match above")


def main():
    """
    Run all verification tests
    """
    print("\n" + "="*70)
    print("Δ* DISCRETIZATION VERIFICATION")
    print("="*70)
    print("\nSystematic verification following小A's approach:")
    print("  1. Test Case 1: ψ = R² (should preserve)")
    print("  2. Test Case 2: ψ = -(R²+Z²) (should converge)")
    print("  3. Verify stencil coefficients")
    
    # Test 1
    error1 = test_case_1_quadratic()
    
    # Test 2
    error2 = test_case_2_parabolic()
    
    # Verify coefficients
    verify_stencil_coefficients()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATION")
    print("="*70)
    
    if error1 < 1e-10 and error2 < 0.01:
        print(f"\n✅ DISCRETIZATION CORRECT")
        print(f"   Test 1 (ψ=R²): PASS (error {error1:.2e})")
        print(f"   Test 2 (parabolic): PASS (error {error2:.2e})")
        print(f"\n   → Stencil is likely correct!")
        print(f"   → Problem may be elsewhere:")
        print(f"       - Boundary conditions?")
        print(f"       - Initial guess?")
        print(f"       - Profile consistency?")
    elif error1 > 1e-6:
        print(f"\n⚠️  TEST 1 FAILED")
        print(f"   ψ=R² not preserved (error {error1:.2e})")
        print(f"   → Likely boundary handling issue")
        print(f"   → Check BC implementation")
    elif error2 > 0.1:
        print(f"\n❌ DISCRETIZATION INCORRECT")
        print(f"   Test 2 (parabolic): error {error2:.2e} too large")
        print(f"   → Stencil coefficients likely wrong")
        print(f"\n   FIX OPTIONS:")
        print(f"     A. Fix 5-point stencil (15 min)")
        print(f"     B. Use scipy sparse solver (10 min)")
        print(f"\n   Recommend: A (understand root cause)")
    else:
        print(f"\n✓  DISCRETIZATION ACCEPTABLE")
        print(f"   Test 1: {error1:.2e}")
        print(f"   Test 2: {error2:.2e}")
        print(f"\n   May need:")
        print(f"     - More SOR sweeps per Picard iteration")
        print(f"     - Better initial guess")
        print(f"     - Profile parameter tuning")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
