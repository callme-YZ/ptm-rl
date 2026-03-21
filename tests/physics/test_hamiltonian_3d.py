"""
Unit tests for 3D Hamiltonian implementation.

Test coverage:
1. Smooth field (analytical gradient verification)
2. Zero field (H = 0)
3. Uniform field (∂/∂θ = 0, ∂/∂ζ = 0)
4. Radial field (∂/∂θ = 0, ∂/∂ζ = 0, check metric)
5. Energy partition (U + K = H)

Conservation tests (Phase 2.3):
- Ideal MHD: dH/dt < 1e-10
- Resistive MHD: dH/dt < 0

Author: 小P ⚛️
Created: 2026-03-19
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')

from pytokmhd.physics.hamiltonian_3d import (
    compute_gradient_3d,
    compute_energy_density,
    compute_hamiltonian_3d,
    compute_magnetic_energy,
    compute_kinetic_energy,
)


class Grid3D:
    """Simple 3D grid for testing."""
    def __init__(self, nr=16, nθ=16, nζ=32, r_min=0.1, r_max=1.0):
        self.nr = nr
        self.nθ = nθ
        self.nζ = nζ
        self.r_min = r_min
        self.r_max = r_max
        
        self.dr = (r_max - r_min) / (nr - 1) if nr > 1 else 0.0
        self.dθ = 2 * np.pi / nθ
        self.dζ = 2 * np.pi / nζ
        
        self.r = np.linspace(r_min, r_max, nr)
        self.θ = np.linspace(0, 2*np.pi, nθ, endpoint=False)
        self.ζ = np.linspace(0, 2*np.pi, nζ, endpoint=False)
    
    def meshgrid(self):
        """Return 3D meshgrid (r, θ, ζ)."""
        r_3d, θ_3d, ζ_3d = np.meshgrid(self.r, self.θ, self.ζ, indexing='ij')
        return r_3d, θ_3d, ζ_3d


# ============================================================================
# Test 1: Zero Field
# ============================================================================

def test_zero_field():
    """Test: ψ=0, ω=0 → H=0."""
    print("\n" + "="*60)
    print("Test 1: Zero Field")
    print("="*60)
    
    grid = Grid3D(nr=16, nθ=32, nζ=64)
    
    psi = np.zeros((grid.nr, grid.nθ, grid.nζ))
    omega = np.zeros_like(psi)
    
    H = compute_hamiltonian_3d(psi, omega, grid)
    
    print(f"H = {H:.2e}")
    print(f"Expected: 0.0")
    
    assert np.abs(H) < 1e-14, f"Zero field should give H=0, got {H}"
    print("✅ PASSED")


# ============================================================================
# Test 2: Uniform Field (Constant ψ)
# ============================================================================

def test_uniform_field():
    """Test: ψ=const → |∇ψ|²=0, H = ∫ (1/2)ω² r dV."""
    print("\n" + "="*60)
    print("Test 2: Uniform Field (ψ = const)")
    print("="*60)
    
    grid = Grid3D(nr=16, nθ=32, nζ=64)
    
    # Constant stream function
    psi = 3.0 * np.ones((grid.nr, grid.nθ, grid.nζ))
    
    # ∇²(const) = 0
    omega = np.zeros_like(psi)
    
    H = compute_hamiltonian_3d(psi, omega, grid)
    
    print(f"H = {H:.2e}")
    print(f"Expected: 0.0 (since ∇ψ=0 and ω=0)")
    
    # Small numerical error from finite differences at boundaries
    assert np.abs(H) < 1e-10, f"Uniform field should give H≈0, got {H}"
    print("✅ PASSED")


# ============================================================================
# Test 3: Pure Radial Field (Analytical)
# ============================================================================

def test_radial_field():
    """Test: ψ = r² → |∇ψ|² = 4r², ω = ∇²ψ = 4."""
    print("\n" + "="*60)
    print("Test 3: Pure Radial Field (ψ = r²)")
    print("="*60)
    
    # Use higher resolution for better accuracy
    grid = Grid3D(nr=64, nθ=32, nζ=64, r_min=0.2, r_max=1.0)
    r_3d, _, _ = grid.meshgrid()
    
    # Stream function: ψ = r²
    psi = r_3d**2
    
    # Analytical Laplacian: ∇²(r²) = 4
    omega = 4.0 * np.ones_like(psi)
    
    # Compute gradient
    dpsi_dr, dpsi_dtheta, dpsi_dzeta = compute_gradient_3d(psi, grid)
    
    # Expected: ∂(r²)/∂r = 2r
    expected_dpsi_dr = 2 * r_3d
    
    # Interior error (exclude boundaries where FD is 1st-order)
    error_dr = np.max(np.abs(dpsi_dr[2:-2, :, :] - expected_dpsi_dr[2:-2, :, :]))
    
    print(f"Max |∂ψ/∂r - 2r| (interior): {error_dr:.2e}")
    assert error_dr < 1e-3, f"Radial derivative error too large: {error_dr}"
    
    # ∂ψ/∂θ and ∂ψ/∂ζ should be ~0 (axisymmetric field)
    error_theta = np.max(np.abs(dpsi_dtheta))
    error_zeta = np.max(np.abs(dpsi_dzeta))
    
    print(f"Max |∂ψ/∂θ|: {error_theta:.2e}")
    print(f"Max |∂ψ/∂ζ|: {error_zeta:.2e}")
    
    assert error_theta < 1e-12, f"θ derivative should be 0: {error_theta}"
    assert error_zeta < 1e-12, f"ζ derivative should be 0: {error_zeta}"
    
    # Compute energy
    H = compute_hamiltonian_3d(psi, omega, grid)
    
    # Expected (analytical):
    # H = ∫∫∫ [(1/2)|∇ψ|² + (1/2)ω²] r dr dθ dζ
    #   = ∫∫∫ [(1/2)(2r)² + (1/2)16] r dr dθ dζ
    #   = ∫∫∫ [2r² + 8] r dr dθ dζ
    #   = (2π)(2π) ∫_{r_min}^{r_max} [2r³ + 8r] dr
    #   = 4π² [ (1/2)r⁴ + 4r² ] |_{r_min}^{r_max}
    
    r_min, r_max = grid.r_min, grid.r_max
    H_exact = 4 * np.pi**2 * (
        0.5 * (r_max**4 - r_min**4) + 4 * (r_max**2 - r_min**2)
    )
    
    rel_error = np.abs(H - H_exact) / H_exact
    
    print(f"H (numerical): {H:.6f}")
    print(f"H (analytical): {H_exact:.6f}")
    print(f"Relative error: {rel_error:.2e}")
    
    # Note: Rectangle rule integration has O(h²) error
    # For nr=64, expect ~2% error
    assert rel_error < 0.02, f"Energy error too large: {rel_error}"
    print("✅ PASSED")


# ============================================================================
# Test 4: Smooth Field (Trigonometric)
# ============================================================================

def test_smooth_trigonometric_field():
    """Test: ψ = sin(kζ) → ∂ψ/∂ζ = k*cos(kζ) (spectral accuracy)."""
    print("\n" + "="*60)
    print("Test 4: Smooth Trigonometric Field")
    print("="*60)
    
    grid = Grid3D(nr=16, nθ=32, nζ=128, r_min=0.2, r_max=1.0)
    r_3d, θ_3d, ζ_3d = grid.meshgrid()
    
    # Field: ψ = sin(k*ζ)
    k = 3.0
    psi = np.sin(k * ζ_3d)
    
    # Compute gradient
    dpsi_dr, dpsi_dtheta, dpsi_dzeta = compute_gradient_3d(psi, grid)
    
    # Expected: ∂ψ/∂ζ = k*cos(k*ζ)
    expected_dzeta = k * np.cos(k * ζ_3d)
    
    error_zeta = np.max(np.abs(dpsi_dzeta - expected_dzeta))
    
    print(f"Max |∂ψ/∂ζ - k*cos(kζ)|: {error_zeta:.2e}")
    print(f"Expected: < 1e-10 (spectral accuracy)")
    
    assert error_zeta < 1e-10, f"FFT derivative error: {error_zeta}"
    print("✅ PASSED")


# ============================================================================
# Test 5: Energy Partition (U + K = H)
# ============================================================================

def test_energy_partition():
    """Test: H = U + K (magnetic + kinetic energy)."""
    print("\n" + "="*60)
    print("Test 5: Energy Partition")
    print("="*60)
    
    grid = Grid3D(nr=32, nθ=64, nζ=128)
    r_3d, θ_3d, ζ_3d = grid.meshgrid()
    
    # Random field
    np.random.seed(42)
    psi = np.random.randn(grid.nr, grid.nθ, grid.nζ) * 0.1
    omega = np.random.randn(grid.nr, grid.nθ, grid.nζ) * 0.1
    
    # Total energy
    H = compute_hamiltonian_3d(psi, omega, grid)
    
    # Magnetic energy (only |∇ψ|² term)
    U = compute_magnetic_energy(psi, grid)
    
    # Kinetic energy (only ω² term)
    K = compute_kinetic_energy(omega, grid)
    
    # Check partition
    error = np.abs(H - (U + K))
    rel_error = error / H if H > 0 else error
    
    print(f"H (total): {H:.6e}")
    print(f"U (magnetic): {U:.6e}")
    print(f"K (kinetic): {K:.6e}")
    print(f"U + K: {U + K:.6e}")
    print(f"Error |H - (U+K)|: {error:.2e}")
    print(f"Relative error: {rel_error:.2e}")
    
    assert rel_error < 1e-12, f"Energy partition error: {rel_error}"
    print("✅ PASSED")


# ============================================================================
# Test 6: r=0 Singularity Handling
# ============================================================================

def test_r_zero_singularity():
    """Test: r=0 singularity handled (no NaN/Inf)."""
    print("\n" + "="*60)
    print("Test 6: r=0 Singularity Handling")
    print("="*60)
    
    # Grid starting at r=0
    grid = Grid3D(nr=32, nθ=64, nζ=128, r_min=0.0, r_max=1.0)
    r_3d, θ_3d, ζ_3d = grid.meshgrid()
    
    # Field with θ dependence
    psi = r_3d * np.sin(θ_3d)
    omega = np.zeros_like(psi)
    
    # Compute energy
    H = compute_hamiltonian_3d(psi, omega, grid)
    
    print(f"H = {H:.6e}")
    
    # Check for NaN/Inf
    assert np.isfinite(H), f"Energy is not finite: {H}"
    assert H >= 0, f"Energy is negative: {H}"
    
    # Compute energy density
    E = compute_energy_density(psi, omega, grid)
    
    # Check no NaN/Inf in energy density
    assert np.all(np.isfinite(E)), "Energy density contains NaN/Inf"
    assert np.all(E >= 0), "Energy density is negative"
    
    print("✅ PASSED (no NaN/Inf)")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("3D Hamiltonian Unit Tests")
    print("="*60)
    
    test_zero_field()
    test_uniform_field()
    test_radial_field()
    test_smooth_trigonometric_field()
    test_energy_partition()
    test_r_zero_singularity()
    
    print("\n" + "="*60)
    print("All tests PASSED ✅")
    print("="*60)
