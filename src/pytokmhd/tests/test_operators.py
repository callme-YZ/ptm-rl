"""
Unit Tests for MHD Operators

Tests:
1. Laplacian accuracy: ∇²(r²) = 4
2. Poisson bracket: [r, z] = 1
3. Gradient accuracy: ∂r/∂r = 1
4. 2nd order convergence

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

from pytokmhd.solver import mhd_equations


def test_laplacian_r2():
    """Test: ∇²(r²) = 4 in cylindrical coordinates."""
    print("\n=== Test: Laplacian of r² ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Test function: f = r²
    f = R**2
    
    # Compute Laplacian
    lap_f = mhd_equations.laplacian_cylindrical(f, dr, dz, R)
    
    # Expected: ∇²(r²) = 4 (analytically)
    # Check interior (avoid boundaries)
    interior = lap_f[10:-10, 10:-10]
    expected = 4.0
    
    error = np.abs(interior - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    
    # Pass criterion
    assert max_error < 1e-6, f"Laplacian test failed: max error {max_error:.2e} > 1e-6"
    print("✅ PASSED")
    
    # Return removed for pytest compliance


def test_poisson_bracket():
    """Test: [r, z] = 1."""
    print("\n=== Test: Poisson Bracket [r, z] ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Compute [r, z]
    pb = mhd_equations.poisson_bracket(R, Z, dr, dz)
    
    # Expected: [r, z] = ∂r/∂r * ∂z/∂z - ∂r/∂z * ∂z/∂r = 1*1 - 0*0 = 1
    interior = pb[10:-10, 10:-10]
    expected = 1.0
    
    error = np.abs(interior - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    
    assert max_error < 1e-6, f"Poisson bracket test failed: max error {max_error:.2e} > 1e-6"
    print("✅ PASSED")
    
    # Return removed for pytest compliance


def test_gradient_r():
    """Test: ∂r/∂r = 1."""
    print("\n=== Test: Gradient ∂/∂r ===")
    
    # Grid
    Nr, Nz = 64, 128
    Lr, Lz = 1.0, 6.0
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Compute ∂r/∂r
    df_dr = mhd_equations.gradient_r(R, dr)
    
    # Expected: ∂r/∂r = 1
    interior = df_dr[10:-10, 10:-10]
    expected = 1.0
    
    error = np.abs(interior - expected)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    
    assert max_error < 1e-10, f"Gradient test failed: max error {max_error:.2e} > 1e-10"
    print("✅ PASSED")
    
    # Return removed for pytest compliance


def test_2nd_order_convergence():
    """Test: Operators have 2nd order accuracy."""
    print("\n=== Test: 2nd Order Convergence ===")
    
    # Test on three grid sizes
    resolutions = [32, 64, 128]
    errors = []
    
    for Nr in resolutions:
        Nz = 2 * Nr
        Lr, Lz = 1.0, 6.0
        r = np.linspace(0, Lr, Nr)
        z = np.linspace(0, Lz, Nz)
        dr, dz = r[1] - r[0], z[1] - z[0]
        R, Z = np.meshgrid(r, z, indexing='ij')
        
        # Test function: f = r² + sin(2πz/Lz)
        f = R**2 + np.sin(2*np.pi*Z/Lz)
        
        # Analytical Laplacian: ∇²f = 4 - (2π/Lz)²sin(2πz/Lz)
        lap_analytical = 4.0 - (2*np.pi/Lz)**2 * np.sin(2*np.pi*Z/Lz)
        
        # Numerical Laplacian
        lap_numerical = mhd_equations.laplacian_cylindrical(f, dr, dz, R)
        
        # Error (interior)
        error = np.max(np.abs(lap_numerical[5:-5, 5:-5] - lap_analytical[5:-5, 5:-5]))
        errors.append(error)
        
        print(f"Nr={Nr:3d}: Error = {error:.2e}")
    
    # Check convergence rate
    # error ∝ h² => log(error) ∝ 2*log(h)
    # ratio = error_coarse / error_fine ≈ 4 for 2nd order
    
    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    
    print(f"\nConvergence ratios:")
    print(f"  32→64:  {ratio_1:.2f} (expect ≈4)")
    print(f"  64→128: {ratio_2:.2f} (expect ≈4)")
    
    # Tolerance: ratio should be between 3 and 5 (2nd order)
    assert 3.0 < ratio_1 < 5.0, f"Convergence test failed: ratio {ratio_1:.2f} not ≈4"
    assert 3.0 < ratio_2 < 5.0, f"Convergence test failed: ratio {ratio_2:.2f} not ≈4"
    
    print("✅ PASSED: 2nd order convergence confirmed")
    
    # Return removed for pytest compliance


def run_all_tests():
    """Run all operator tests."""
    print("="*60)
    print("PyTokMHD Operator Tests")
    print("="*60)
    
    results = {}
    
    try:
        results['laplacian'] = test_laplacian_r2()
        results['poisson_bracket'] = test_poisson_bracket()
        results['gradient'] = test_gradient_r()
        results['convergence'] = test_2nd_order_convergence()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print("\nSummary:")
        print(f"  Laplacian accuracy:    {results['laplacian']:.2e}")
        print(f"  Poisson bracket error: {results['poisson_bracket']:.2e}")
        print(f"  Gradient error:        {results['gradient']:.2e}")
        print(f"  Convergence confirmed: 2nd order")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
