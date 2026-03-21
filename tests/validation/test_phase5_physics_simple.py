"""
Phase 5A: Physics Validation (Simplified ICs)

Use simpler, more stable initial conditions to validate physics correctness.

Author: 小P ⚛️ (subagent)
Date: 2026-03-20
Phase: 5A (Physics Validation)
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pytokmhd.solvers.imex_3d import evolve_3d_imex
from pytokmhd.ic.ballooning_mode import Grid3D
from pytokmhd.physics.hamiltonian_3d import compute_hamiltonian_3d

PLOT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'validation' / 'phase5a'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def create_simple_ic(grid, amplitude=0.01):
    """
    Create simple IC: Single toroidal mode with small amplitude.
    
    ψ(r,θ,ζ) = A·r·(1-r)·cos(m·θ + n·ζ)
    ω = ∇²ψ (computed numerically)
    
    This is much more stable than full ballooning mode.
    """
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    r_grid, theta_grid, zeta_grid = np.meshgrid(
        grid.r, grid.theta, grid.zeta, indexing='ij'
    )
    
    # Simple mode: m=1, n=1
    m, n = 1, 1
    psi = amplitude * r_grid * (1.0 - r_grid / grid.r_max) * np.cos(m * theta_grid + n * zeta_grid)
    
    # Compute omega = ∇²ψ using finite differences
    dr, dtheta, dzeta = grid.dr, grid.dtheta, grid.dzeta
    
    # ∇²ψ = ∂²ψ/∂r² + (1/r)∂ψ/∂r + (1/r²)∂²ψ/∂θ² + ∂²ψ/∂ζ²
    d2psi_dr2 = np.gradient(np.gradient(psi, dr, axis=0), dr, axis=0)
    dpsi_dr = np.gradient(psi, dr, axis=0)
    d2psi_dtheta2 = np.gradient(np.gradient(psi, dtheta, axis=1), dtheta, axis=1)
    d2psi_dzeta2 = np.gradient(np.gradient(psi, dzeta, axis=2), dzeta, axis=2)
    
    r_safe = np.where(r_grid > 1e-6, r_grid, 1e-6)
    omega = (
        d2psi_dr2 +
        dpsi_dr / r_safe +
        d2psi_dtheta2 / r_safe**2 +
        d2psi_dzeta2
    )
    
    return psi, omega


@pytest.fixture
def standard_grid():
    """Standard grid (32×64×128)."""
    return Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)

@pytest.fixture
def coarse_grid():
    """Coarse grid for quick tests (16×32×64)."""
    return Grid3D(nr=16, ntheta=32, nzeta=64, r_max=1.0)

@pytest.fixture
def simple_ic(standard_grid):
    """Simple IC for stability."""
    psi, omega = create_simple_ic(standard_grid, amplitude=0.01)
    return psi, omega, standard_grid


# =============================================================================
# Test 1: Energy Conservation (P0 - Critical)
# =============================================================================

class TestEnergyConservation:
    """P0: Ideal MHD must conserve energy."""
    
    def test_1_1_ideal_mhd_no_j_ext(self, simple_ic):
        """Test 1.1: Ideal MHD (η=0, J_ext=None)"""
        psi0, omega0, grid = simple_ic
        
        print(f"\n{'='*60}")
        print(f"IC Diagnostics:")
        print(f"  psi:   max={np.max(np.abs(psi0)):.4e}")
        print(f"  omega: max={np.max(np.abs(omega0)):.4e}")
        
        # Evolve (ideal MHD)
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.005,  # Smaller dt for stability
            n_steps=200,  # Shorter for speed
            J_ext=None,
            store_interval=10
        )
        
        energy = diag['energy']
        drift = abs((energy[-1] - energy[0]) / energy[0])
        
        print(f"\n{'='*60}")
        print(f"Test 1.1: Energy Conservation (Ideal MHD)")
        print(f"{'='*60}")
        print(f"H(0)     = {energy[0]:.8e}")
        print(f"H(t=1.0) = {energy[-1]:.8e}")
        print(f"|ΔH/H₀|  = {drift:.2e}")
        print(f"Target   : < 1e-6")
        print(f"Status   : {'✅ PASS' if drift < 1e-6 else '❌ FAIL'}")
        print(f"Max CFL  : {max(diag['cfl_number']):.4f}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        time = np.array(diag['time'])
        ax1.plot(time, (energy - energy[0]) / energy[0], 'b-', linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('$(H - H_0) / H_0$', fontsize=12)
        ax1.set_title('Energy Conservation', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time, diag['cfl_number'], 'r-', linewidth=2)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('CFL Number', fontsize=12)
        ax2.set_title('CFL Number Evolution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_1_1_simple_energy_conservation.png', dpi=150)
        plt.close()
        
        assert drift < 1e-6, f"Energy drift {drift:.2e} exceeds tolerance"
    
    def test_1_2_ideal_mhd_with_zero_j_ext(self, simple_ic):
        """Test 1.2: J_ext=0 should be same as J_ext=None."""
        psi0, omega0, grid = simple_ic
        J_ext = np.zeros_like(psi0)
        
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.005,
            n_steps=200,
            J_ext=J_ext,
            store_interval=10
        )
        
        energy = diag['energy']
        drift = abs((energy[-1] - energy[0]) / energy[0])
        
        print(f"\n{'='*60}")
        print(f"Test 1.2: J_ext=0 explicit")
        print(f"|ΔH/H₀| = {drift:.2e}")
        print(f"Status  : {'✅ PASS' if drift < 1e-6 else '❌ FAIL'}")
        
        assert drift < 1e-6


# =============================================================================
# Test 2: Energy Dissipation (P0 - Critical)
# =============================================================================

class TestEnergyDissipation:
    """P0: Resistive MHD must dissipate energy."""
    
    def test_2_1_resistive_mhd(self, simple_ic):
        """Test 2.1: Resistive MHD (η > 0)"""
        psi0, omega0, grid = simple_ic
        
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=1e-4,
            dt=0.005,
            n_steps=200,
            J_ext=None,
            store_interval=10
        )
        
        energy = diag['energy']
        dE = np.diff(energy)
        is_monotonic = np.all(dE <= 1e-12)
        dissipation_rate = (energy[-1] - energy[0]) / (energy[0] * 200 * 0.005)
        
        print(f"\n{'='*60}")
        print(f"Test 2.1: Energy Dissipation")
        print(f"{'='*60}")
        print(f"H(0)              = {energy[0]:.8e}")
        print(f"H(t=1.0)          = {energy[-1]:.8e}")
        print(f"ΔH/H₀             = {(energy[-1] - energy[0]) / energy[0]:.4f}")
        print(f"Dissipation rate  = {dissipation_rate:.4e}")
        print(f"Monotonic         : {'✅ YES' if is_monotonic else '❌ NO'}")
        print(f"Status            : {'✅ PASS' if energy[-1] < energy[0] else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        time = np.array(diag['time'])
        ax.plot(time, energy / energy[0], 'r-', linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$H / H_0$', fontsize=12)
        ax.set_title('Test 2.1: Energy Dissipation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_2_1_simple_dissipation.png', dpi=150)
        plt.close()
        
        assert energy[-1] < energy[0], "Energy should decrease"
        assert -1.0 < dissipation_rate < 0.0, f"Dissipation rate {dissipation_rate:.2e} unreasonable"


# =============================================================================
# Test 4: J_ext Energy Injection (P1)
# =============================================================================

class TestExternalCurrent:
    """P1: J_ext should inject energy."""
    
    def test_4_1_constant_j_ext(self, simple_ic):
        """Test 4.1: Constant J_ext in ideal MHD."""
        psi0, omega0, grid = simple_ic
        J_ext = 0.01 * np.ones_like(psi0)  # Small J_ext
        
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.005,
            n_steps=100,
            J_ext=J_ext,
            store_interval=10
        )
        
        energy = diag['energy']
        energy_increase = energy[-1] - energy[0]
        
        print(f"\n{'='*60}")
        print(f"Test 4.1: J_ext Energy Injection")
        print(f"{'='*60}")
        print(f"H(0)     = {energy[0]:.8e}")
        print(f"H(t=0.5) = {energy[-1]:.8e}")
        print(f"ΔH       = {energy_increase:.4e}")
        print(f"Status   : {'✅ PASS' if energy_increase > 0 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        time = np.array(diag['time'])
        ax.plot(time, energy, 'g-', linewidth=2, label='J_ext=0.01')
        ax.axhline(energy[0], color='k', linestyle='--', alpha=0.5, label='H(0)')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy H', fontsize=12)
        ax.set_title('Test 4.1: Energy Injection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_4_1_simple_j_ext.png', dpi=150)
        plt.close()
        
        assert energy_increase > 0, "J_ext should increase energy"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
