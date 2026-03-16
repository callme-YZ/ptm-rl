"""
Test q-profile Calculation

Validate flux surface tracer and q calculator.
"""

import sys
sys.path.insert(0, '..')

import numpy as np

from pytokeq.equilibrium.diagnostics.flux_surface_tracer import FluxSurfaceTracer
from pytokeq.equilibrium.diagnostics.q_profile import QCalculator


class TestFluxSurfaceTracer:
    """Test flux surface locator"""
    
    def test_circular_surfaces(self):
        """Test on circular flux surfaces"""
        # Create circular psi: psi = psi_0 * (1 - r^2/a^2)
        nr, nz = 65, 65
        R_1d = np.linspace(0.5, 1.5, nr)
        Z_1d = np.linspace(-0.5, 0.5, nz)
        
        R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')
        
        R0 = 1.0
        a = 0.4
        
        r = np.sqrt((R - R0)**2 + Z**2)
        psi = 1.0 - (r / a)**2
        psi = np.maximum(psi, 0)  # Zero outside a
        
        # Create tracer
        tracer = FluxSurfaceTracer(psi, R_1d, Z_1d)
        
        # Check axis location
        assert abs(tracer.R_axis - R0) < 0.01, f"Axis R: {tracer.R_axis} vs {R0}"
        assert abs(tracer.Z_axis - 0.0) < 0.01, f"Axis Z: {tracer.Z_axis}"
        
        # Find surface at r = 0.2m (half-radius)
        psi_target = 1.0 - (0.2 / a)**2
        
        R_surf, Z_surf = tracer.find_surface_points(psi_target, ntheta=64)
        
        assert len(R_surf) > 0, "No surface points found"
        
        # Check that points are approximately circular
        r_surf = np.sqrt((R_surf - R0)**2 + Z_surf**2)
        
        r_mean = np.mean(r_surf)
        r_std = np.std(r_surf)
        
        # Should be close to r = 0.2m
        assert abs(r_mean - 0.2) < 0.02, f"Mean radius: {r_mean} vs 0.2"
        assert r_std < 0.01, f"Radius std: {r_std} (should be nearly constant)"
        
        print(f"✓ Circular surface test passed")
        print(f"  Target r: 0.2m")
        print(f"  Found r: {r_mean:.4f} ± {r_std:.4f}m")
    
    def test_axis_finding(self):
        """Test magnetic axis detection"""
        # Simple parabolic psi (set boundary to zero)
        nr, nz = 65, 65
        R_1d = np.linspace(0.5, 1.5, nr)
        Z_1d = np.linspace(-0.5, 0.5, nz)
        
        R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')
        
        # Axis at R=1.0, Z=0.0
        # Make sure maximum is in interior
        psi = 1.0 - ((R - 1.0)**2 + Z**2) / 0.4**2
        psi = np.maximum(psi, 0)  # Zero outside
        
        tracer = FluxSurfaceTracer(psi, R_1d, Z_1d)
        
        # Should find axis within ~2 grid spacings (parabolic fit refinement)
        dR = R_1d[1] - R_1d[0]
        dZ = Z_1d[1] - Z_1d[0]
        
        assert abs(tracer.R_axis - 1.0) < 2*dR, f"Axis R: {tracer.R_axis} vs 1.0 (tol={2*dR:.4f})"
        assert abs(tracer.Z_axis - 0.0) < 2*dZ, f"Axis Z: {tracer.Z_axis} vs 0.0 (tol={2*dZ:.4f})"
        
        print(f"✓ Axis finding test passed")
        print(f"  Found axis at R={tracer.R_axis:.4f}, Z={tracer.Z_axis:.4f}")


class TestQCalculator:
    """Test q-profile calculator"""
    
    def test_circular_q_constant(self):
        """
        Test q on circular equilibrium with constant q
        
        For circular cross-section with B_phi ~ 1/R and uniform j_phi:
        q should be approximately constant
        """
        # Create equilibrium
        nr, nz = 65, 65
        R_1d = np.linspace(0.6, 1.4, nr)
        Z_1d = np.linspace(-0.4, 0.4, nz)
        
        R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')
        
        R0 = 1.0
        a = 0.3
        
        # Circular psi
        r = np.sqrt((R - R0)**2 + Z**2)
        psi = 1.0 - (r / a)**2
        psi = np.maximum(psi, 0)
        
        # Field functions
        # For simple test: F = const (toroidal field ~ 1/R)
        F0 = 2.0  # RB_phi = const
        
        def fpol(psi_norm):
            return F0
        
        # Poloidal field from psi
        def Br_func(R, Z):
            # Br = -(1/R) * dpsi/dZ
            dpsi_dZ = -2 * Z / a**2  # Gradient of circular psi
            return -dpsi_dZ / R
        
        def Bz_func(R, Z):
            # Bz = (1/R) * dpsi/dR
            dpsi_dR = -2 * (R - R0) / a**2
            return dpsi_dR / R
        
        # Create calculator
        calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)
        
        # Compute q at several radii
        psi_norm_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        q_values = []
        for pn in psi_norm_test:
            q = calc.compute_q_single(pn, ntheta=64)
            q_values.append(q)
        
        q_values = np.array(q_values)
        
        # For this simple case, q should be roughly constant
        # (Not exactly due to R variation, but should be within factor ~2)
        q_mean = np.mean(q_values)
        q_std = np.std(q_values)
        
        print(f"✓ Constant q test")
        print(f"  q values: {q_values}")
        print(f"  Mean: {q_mean:.2f}, Std: {q_std:.2f}")
        
        # Check all values are positive and reasonable
        assert np.all(q_values > 0), "q should be positive"
        assert np.all(q_values < 100), "q should be reasonable (<100)"
        
        # Check not too much variation (circular case)
        assert q_std / q_mean < 0.5, f"q variation too large: std/mean = {q_std/q_mean:.2f}"
    
    def test_q_profile_with_extrapolation(self):
        """Test q-profile calculation with axis extrapolation"""
        # Setup simple case
        nr, nz = 65, 65
        R_1d = np.linspace(0.6, 1.4, nr)
        Z_1d = np.linspace(-0.4, 0.4, nz)
        
        R, Z = np.meshgrid(R_1d, Z_1d, indexing='ij')
        
        R0 = 1.0
        a = 0.3
        
        r = np.sqrt((R - R0)**2 + Z**2)
        psi = 1.0 - (r / a)**2
        psi = np.maximum(psi, 0)
        
        # Monotonic q-profile: q increases from axis to edge
        def fpol(psi_norm):
            return 2.0 * (1 + 0.5 * psi_norm)
        
        def Br_func(R, Z):
            dpsi_dZ = -2 * Z / a**2
            return -dpsi_dZ / R
        
        def Bz_func(R, Z):
            dpsi_dR = -2 * (R - R0) / a**2
            return dpsi_dR / R
        
        calc = QCalculator(psi, R_1d, Z_1d, fpol, Br_func, Bz_func)
        
        # Request q at axis (psi_norm=0) - requires extrapolation
        psi_norm, q = calc.compute_q_profile(npsi=50, extrapolate=True)
        
        # Manually get q at axis with extrapolation
        q_axis = calc.compute_q_profile(np.array([0.0]), extrapolate=True)
        
        assert not np.isnan(q_axis), "q(axis) should not be NaN with extrapolation"
        assert q_axis > 0, f"q(axis) should be positive: {q_axis}"
        
        # Check q is monotonically increasing (for this profile)
        # (Allow small violations due to numerical noise)
        dq = np.diff(q)
        decreasing_count = np.sum(dq < -0.01)
        
        assert decreasing_count / len(dq) < 0.1, \
            f"q should be mostly increasing: {decreasing_count}/{len(dq)} decreasing"
        
        print(f"✓ Extrapolation test passed")
        print(f"  q(axis) = {q_axis:.3f}")
        print(f"  q(edge) = {q[-1]:.3f}")
        print(f"  Monotonic: {decreasing_count}/{len(dq)} violations")


def test_integration_utilities():
    """Test surface integration utilities"""
    from pytokeq.equilibrium.diagnostics.q_profile import integrate_along_surface, surface_average
    
    # Create circular path
    ntheta = 100
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    
    R0, Z0 = 1.0, 0.0
    radius = 0.2
    
    R_surf = R0 + radius * np.cos(theta)
    Z_surf = Z0 + radius * np.sin(theta)
    
    # Test 1: Integrate constant (should give circumference)
    f = np.ones(ntheta)
    
    integral = integrate_along_surface(f, R_surf, Z_surf)
    expected = 2 * np.pi * radius
    
    rel_error = abs(integral - expected) / expected
    
    assert rel_error < 0.01, \
        f"Integral of 1: {integral:.4f} vs {expected:.4f} (error: {rel_error:.2%})"
    
    # Test 2: Average of constant (should give that constant)
    f = 3.14 * np.ones(ntheta)
    
    avg = surface_average(f, R_surf, Z_surf)
    
    assert abs(avg - 3.14) < 1e-10, f"Average of constant: {avg} vs 3.14"
    
    # Test 3: Average of cos(theta) (should be ~0 by symmetry)
    f = np.cos(theta)
    
    avg = surface_average(f, R_surf, Z_surf)
    
    assert abs(avg) < 0.01, f"Average of cos: {avg} (should be ~0)"
    
    print(f"✓ Integration utilities test passed")


if __name__ == '__main__':
    print("="*70)
    print("Testing q-profile calculation (PHYS-01 fix)")
    print("="*70)
    
    # Run tests
    print("\n1. Flux Surface Tracer Tests")
    print("-" * 70)
    
    tracer_tests = TestFluxSurfaceTracer()
    tracer_tests.test_circular_surfaces()
    tracer_tests.test_axis_finding()
    
    print("\n2. Q Calculator Tests")
    print("-" * 70)
    
    q_tests = TestQCalculator()
    q_tests.test_circular_q_constant()
    q_tests.test_q_profile_with_extrapolation()
    
    print("\n3. Integration Utilities")
    print("-" * 70)
    
    test_integration_utilities()
    
    print("\n" + "="*70)
    print("All tests passed ✅")
    print("="*70)
