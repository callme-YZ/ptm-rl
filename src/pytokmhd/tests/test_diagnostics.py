"""
Unit tests for PyTokMHD diagnostics module
"""

import numpy as np
import pytest
from pytokmhd.diagnostics import (
    find_rational_surface,
    compute_island_width,
    compute_growth_rate,
    TearingModeMonitor
)


# ============================================================================
# Test 1: Rational Surface Finder
# ============================================================================

def test_rational_surface_solovev():
    """Test rational surface finder on Solovev q-profile"""
    # Solovev equilibrium: q(r) = q0 * (1 + r²)
    q0 = 1.0
    r = np.linspace(0, 1, 100)
    q = q0 * (1 + r**2)
    
    # Find q=2 surface
    r_s, acc = find_rational_surface(q, r, q_target=2.0, method='spline')
    
    # Analytical solution: q = q0(1 + r²) = 2
    # r² = 2/q0 - 1 = 1
    # r_s = 1.0
    r_s_expected = np.sqrt(2.0/q0 - 1)
    
    error = abs(r_s - r_s_expected)
    print(f"Rational surface test: r_s = {r_s:.6f}, expected = {r_s_expected:.6f}")
    print(f"Error: {error:.2e}, accuracy: {acc:.2e}")
    
    assert error < 1e-4, f"Rational surface error {error:.2e} > 1e-4"
    assert acc < 1e-6, f"Accuracy {acc:.2e} > 1e-6"


def test_rational_surface_linear_q():
    """Test on linear q-profile"""
    r = np.linspace(0, 1, 50)
    q = 1.0 + 2.0 * r  # q = 1 + 2r
    
    # Find q = 2
    r_s, acc = find_rational_surface(q, r, q_target=2.0, method='linear')
    
    # Analytical: 1 + 2r = 2 → r = 0.5
    r_s_expected = 0.5
    
    error = abs(r_s - r_s_expected)
    assert error < 1e-3, f"Linear q error {error:.2e} > 1e-3"


def test_rational_surface_out_of_range():
    """Test handling of out-of-range q values"""
    r = np.linspace(0, 1, 50)
    q = 1.0 + r  # q ∈ [1, 2]
    
    # q=3 is out of range
    r_s, acc = find_rational_surface(q, r, q_target=3.0)
    
    assert np.isnan(r_s), "Should return NaN for out-of-range q"


# ============================================================================
# Test 2: Island Width Measurement
# ============================================================================

def create_perturbed_solovev(Nr=64, Nz=64, delta=0.1, m=2):
    """
    Create Solovev equilibrium with m-mode perturbation
    
    ψ = ψ_0(r,z) + δ r^m cos(mθ)
    """
    r = np.linspace(0.1, 1.0, Nr)
    z = np.linspace(-0.5, 0.5, Nz)
    
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Solovev flux: ψ ~ r² + z²/2
    psi_0 = R**2 + 0.5 * Z**2
    
    # Poloidal angle (cylindrical geometry)
    theta = np.arctan2(Z, R)
    
    # Add helical perturbation: δψ ~ r^m cos(mθ)
    delta_psi = delta * R**m * np.cos(m * theta)
    
    psi = psi_0 + delta_psi
    
    # q-profile (Solovev)
    q0 = 1.0
    q = q0 * (1 + r**2)
    
    return psi, r, z, q


def test_island_width_perturbed_solovev():
    """Test island width on perturbed Solovev equilibrium"""
    delta_perturbed = 0.2
    delta_baseline = 0.0
    m = 2
    
    # Test with perturbation
    psi_pert, r, z, q = create_perturbed_solovev(Nr=64, Nz=64, delta=delta_perturbed, m=m)
    w_pert, r_s, phase = compute_island_width(psi_pert, r, z, q, m=m, n=1)
    
    # Test without perturbation
    psi_base, r, z, q = create_perturbed_solovev(Nr=64, Nz=64, delta=delta_baseline, m=m)
    w_base, _, _ = compute_island_width(psi_base, r, z, q, m=m, n=1)
    
    print(f"Island width test:")
    print(f"  With perturbation (δ={delta_perturbed}): w = {w_pert:.4f}")
    print(f"  Without perturbation (δ={delta_baseline}): w = {w_base:.4f}")
    print(f"  Ratio: {w_pert / w_base if w_base > 0 else float('inf'):.2f}")
    
    # Key test: perturbation should increase island width
    assert w_pert > w_base, \
        f"Perturbation should increase width: w_pert={w_pert:.4f} vs w_base={w_base:.4f}"
    
    # Rational surface should be found
    assert not np.isnan(r_s), "Rational surface not found"
    
    # Island width should be positive
    assert w_pert > 0, f"Island width {w_pert} should be positive"
    
    # Sanity check: width shouldn't be absurdly large
    assert w_pert < 10.0, f"Width {w_pert} unreasonably large"


def test_island_width_scaling():
    """Test that island width scales with perturbation amplitude"""
    m = 2
    deltas = [0.05, 0.10, 0.20]
    widths = []
    
    for delta in deltas:
        psi, r, z, q = create_perturbed_solovev(Nr=64, Nz=64, delta=delta, m=m)
        w, _, _ = compute_island_width(psi, r, z, q, m=m, n=1)
        widths.append(w)
    
    print(f"Scaling test:")
    for delta, w in zip(deltas, widths):
        print(f"  δ={delta:.2f} → w={w:.4f}")
    
    # Check monotonic increase
    for i in range(len(widths) - 1):
        assert widths[i+1] > widths[i], \
            f"Width should increase with perturbation: w[{deltas[i+1]}]={widths[i+1]:.4f} not > w[{deltas[i]}]={widths[i]:.4f}"


# ============================================================================
# Test 3: Growth Rate Measurement
# ============================================================================

def test_growth_rate_exponential():
    """Test growth rate on synthetic exponential data"""
    gamma_true = 0.05
    t = np.linspace(0, 10, 100)
    w0 = 0.01
    
    # Perfect exponential growth
    w = w0 * np.exp(gamma_true * t)
    
    gamma_measured, sigma = compute_growth_rate(w, t, transient_fraction=0.0)
    
    error = abs(gamma_measured - gamma_true)
    print(f"Growth rate test: γ = {gamma_measured:.6f} ± {sigma:.6f}")
    print(f"True: {gamma_true:.6f}, error: {error:.2e}")
    
    # Should be very accurate on perfect data
    assert error < 3*sigma, f"Growth rate error {error:.6f} > 3σ = {3*sigma:.6f}"
    assert error < 1e-4, f"Growth rate error {error:.2e} > 1e-4"


def test_growth_rate_with_noise():
    """Test growth rate with noisy data"""
    gamma_true = 0.03
    t = np.linspace(0, 10, 100)
    w0 = 0.01
    
    # Add 10% noise
    w = w0 * np.exp(gamma_true * t) * (1 + 0.1 * np.random.randn(len(t)))
    
    gamma_measured, sigma = compute_growth_rate(w, t, transient_fraction=0.1)
    
    error = abs(gamma_measured - gamma_true)
    print(f"Noisy data test: γ = {gamma_measured:.6f} ± {sigma:.6f}")
    print(f"Error: {error:.6f}, tolerance: {5*sigma:.6f}")
    
    # More tolerant with noise
    assert error < 5*sigma, f"Growth rate error {error:.6f} > 5σ"


def test_growth_rate_negative():
    """Test negative growth rate (decay)"""
    gamma_true = -0.02
    t = np.linspace(0, 10, 100)
    w0 = 1.0
    
    w = w0 * np.exp(gamma_true * t)
    
    gamma_measured, sigma = compute_growth_rate(w, t, transient_fraction=0.0)
    
    error = abs(gamma_measured - gamma_true)
    print(f"Negative growth test: γ = {gamma_measured:.6f} (expected {gamma_true:.6f})")
    
    assert error < 1e-4, f"Decay rate error {error:.2e} > 1e-4"


# ============================================================================
# Test 4: TearingModeMonitor Integration
# ============================================================================

def simple_mhd_step(psi, omega, dt=0.01):
    """
    Dummy MHD step for testing
    
    Just adds a small perturbation to simulate evolution
    """
    # Random small perturbation
    dpsi = 0.01 * dt * np.random.randn(*psi.shape)
    domega = 0.01 * dt * np.random.randn(*omega.shape)
    
    return psi + dpsi, omega + domega


def test_monitor_integration():
    """Test TearingModeMonitor in evolution loop"""
    # Initialize
    monitor = TearingModeMonitor(m=2, n=1, track_every=5)
    
    # Create initial state
    psi, r, z, q = create_perturbed_solovev(delta=0.1)
    omega = np.zeros_like(psi)
    
    dt = 0.01
    n_steps = 100
    
    # Run evolution
    for step in range(n_steps):
        psi, omega = simple_mhd_step(psi, omega, dt)
        t = step * dt
        
        diag = monitor.update(psi, omega, t, r, z, q)
        
        # Check diagnostics are returned every track_every steps
        if (step + 1) % 5 == 0:
            assert diag is not None, f"Diagnostics should be returned at step {step+1}"
            assert 'w' in diag, "Missing island width in diagnostics"
            assert 't' in diag, "Missing time in diagnostics"
            
            # After enough steps, gamma should be computed
            if step > 20:
                # gamma might be None if not enough valid data yet
                if diag['gamma'] is not None:
                    assert 'sigma' in diag, "Missing uncertainty in diagnostics"
    
    # Check history
    n_tracked = n_steps // 5
    assert len(monitor.w_history) == n_tracked, \
        f"History length {len(monitor.w_history)} != {n_tracked}"
    
    print(f"Monitor test: Tracked {len(monitor.w_history)} steps")
    print(f"Growth rate history: {len(monitor.gamma_history)} points")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"Summary: {summary}")
    
    assert summary['n_samples'] == n_tracked
    assert not np.isnan(summary['w_current'])


def test_monitor_reset():
    """Test monitor reset functionality"""
    monitor = TearingModeMonitor(m=2, n=1)
    
    # Add some fake data
    monitor.w_history = [1, 2, 3]
    monitor.t_history = [0, 1, 2]
    monitor._step_count = 10
    
    # Reset
    monitor.reset()
    
    assert len(monitor.w_history) == 0, "History not cleared"
    assert len(monitor.t_history) == 0, "Time history not cleared"
    assert monitor._step_count == 0, "Step count not reset"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Running PyTokMHD Diagnostics Tests")
    print("=" * 70)
    
    # Rational surface tests
    print("\n--- Test 1: Rational Surface ---")
    test_rational_surface_solovev()
    test_rational_surface_linear_q()
    test_rational_surface_out_of_range()
    print("✓ All rational surface tests passed")
    
    # Island width tests
    print("\n--- Test 2: Island Width ---")
    test_island_width_perturbed_solovev()
    test_island_width_no_perturbation()
    print("✓ All island width tests passed")
    
    # Growth rate tests
    print("\n--- Test 3: Growth Rate ---")
    test_growth_rate_exponential()
    test_growth_rate_with_noise()
    test_growth_rate_negative()
    print("✓ All growth rate tests passed")
    
    # Monitor integration tests
    print("\n--- Test 4: Monitor Integration ---")
    test_monitor_integration()
    test_monitor_reset()
    print("✓ All monitor tests passed")
    
    print("\n" + "=" * 70)
    print("All diagnostics tests passed! ✓")
    print("=" * 70)
