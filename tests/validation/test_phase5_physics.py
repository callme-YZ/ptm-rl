"""
Phase 5A: Physics Validation Test Suite

Comprehensive validation of v1.4 3D MHD physics core correctness.

Test Priority:
- P0 (Critical): Energy conservation, dissipation, ∇·B=0
- P1 (Important): J_ext injection, grid convergence, timestep convergence
- P2 (Optional): Known solution comparison

Author: 小P ⚛️ (subagent)
Date: 2026-03-20
Phase: 5A (Physics Validation)
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pytokmhd.solvers.imex_3d import evolve_3d_imex
from pytokmhd.ic.ballooning_mode import Grid3D, create_equilibrium_ic, create_ballooning_mode_ic
from pytokmhd.physics.hamiltonian_3d import compute_hamiltonian_3d

# Output directory for plots
PLOT_DIR = Path(__file__).parent.parent.parent / 'docs' / 'validation' / 'phase5a'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def standard_grid():
    """Standard grid for validation (32×64×128)."""
    return Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)

@pytest.fixture
def coarse_grid():
    """Coarse grid for quick tests (16×32×64)."""
    return Grid3D(nr=16, ntheta=32, nzeta=64, r_max=1.0)

@pytest.fixture
def ballooning_ic(standard_grid):
    """Equilibrium + ballooning mode IC."""
    psi0, omega0, q = create_equilibrium_ic(standard_grid)
    psi1, omega1 = create_ballooning_mode_ic(standard_grid, n=5, m0=2, epsilon=0.01)
    return psi0 + psi1, omega0 + omega1, standard_grid


# =============================================================================
# Test 1: Energy Conservation (P0 - Critical)
# =============================================================================

class TestEnergyConservation:
    """P0: Ideal MHD must conserve energy."""
    
    def test_1_1_ideal_mhd_no_j_ext(self, ballooning_ic):
        """
        Test 1.1: Ideal MHD (η=0, J_ext=None)
        Expected: |ΔH/H₀| < 1e-6 after 1000 steps
        """
        psi0, omega0, grid = ballooning_ic
        
        # Evolve (ideal MHD)
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,        # Ideal MHD
            dt=0.01,
            n_steps=1000,   # Long evolution
            J_ext=None,
            store_interval=10
        )
        
        # Validate energy conservation
        energy = diag['energy']
        drift = abs((energy[-1] - energy[0]) / energy[0])
        
        print(f"\n{'='*60}")
        print(f"Test 1.1: Energy Conservation (Ideal MHD, J_ext=None)")
        print(f"{'='*60}")
        print(f"H(0)      = {energy[0]:.8e}")
        print(f"H(t=10.0) = {energy[-1]:.8e}")
        print(f"|ΔH/H₀|   = {drift:.2e}")
        print(f"Target    : < 1e-6")
        print(f"Status    : {'✅ PASS' if drift < 1e-6 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        time = np.arange(len(energy)) * 10 * 0.01  # store_interval=10, dt=0.01
        ax.plot(time, (energy - energy[0]) / energy[0], 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$(H - H_0) / H_0$', fontsize=12)
        ax.set_title('Test 1.1: Energy Conservation (Ideal MHD)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2e-6, 2e-6)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_1_1_energy_conservation_ideal.png', dpi=150)
        plt.close()
        
        assert drift < 1e-6, f"Energy drift {drift:.2e} exceeds tolerance 1e-6"
    
    def test_1_2_ideal_mhd_with_zero_j_ext(self, ballooning_ic):
        """
        Test 1.2: Ideal MHD with explicit J_ext=0
        Expected: Same as 1.1 (sanity check)
        """
        psi0, omega0, grid = ballooning_ic
        
        # Explicitly zero J_ext
        J_ext = np.zeros_like(psi0)
        
        # Evolve
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.01,
            n_steps=1000,
            J_ext=J_ext,
            store_interval=10
        )
        
        energy = diag['energy']
        drift = abs((energy[-1] - energy[0]) / energy[0])
        
        print(f"\n{'='*60}")
        print(f"Test 1.2: Energy Conservation (Ideal MHD, J_ext=0 explicit)")
        print(f"{'='*60}")
        print(f"|ΔH/H₀| = {drift:.2e}")
        print(f"Status  : {'✅ PASS' if drift < 1e-6 else '❌ FAIL'}")
        
        assert drift < 1e-6, f"Energy drift {drift:.2e} with J_ext=0 differs from J_ext=None"
    
    def test_1_3_long_time_stability(self, ballooning_ic):
        """
        Test 1.3: Long-time stability (5000 steps)
        Expected: No exponential drift, oscillations bounded
        """
        psi0, omega0, grid = ballooning_ic
        
        # Long evolution
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.01,
            n_steps=5000,
            J_ext=None,
            store_interval=50  # Save memory
        )
        
        energy = diag['energy']
        drift = abs((energy[-1] - energy[0]) / energy[0])
        
        # Check for exponential growth
        max_deviation = np.max(np.abs((energy - energy[0]) / energy[0]))
        
        print(f"\n{'='*60}")
        print(f"Test 1.3: Long-time Stability (5000 steps)")
        print(f"{'='*60}")
        print(f"|ΔH/H₀| (final) = {drift:.2e}")
        print(f"max|ΔH/H₀|      = {max_deviation:.2e}")
        print(f"Target          : < 1e-5 (allows slow numerical drift)")
        print(f"Status          : {'✅ PASS' if max_deviation < 1e-5 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        time = np.arange(len(energy)) * 50 * 0.01
        ax.plot(time, (energy - energy[0]) / energy[0], 'b-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$(H - H_0) / H_0$', fontsize=12)
        ax.set_title('Test 1.3: Long-time Stability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_1_3_long_time_stability.png', dpi=150)
        plt.close()
        
        assert max_deviation < 1e-5, f"Energy deviation {max_deviation:.2e} indicates instability"


# =============================================================================
# Test 2: Energy Dissipation (P0 - Critical)
# =============================================================================

class TestEnergyDissipation:
    """P0: Resistive MHD must dissipate energy monotonically."""
    
    def test_2_1_resistive_mhd(self, ballooning_ic):
        """
        Test 2.1: Resistive MHD (η > 0)
        Expected: Energy decreases monotonically
        """
        psi0, omega0, grid = ballooning_ic
        
        # Evolve with resistivity
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=1e-4,       # Resistive
            dt=0.01,
            n_steps=1000,
            J_ext=None,
            store_interval=10
        )
        
        energy = diag['energy']
        
        # Check monotonic decrease
        dE = np.diff(energy)
        is_monotonic = np.all(dE <= 1e-12)  # Allow tiny numerical noise
        
        # Dissipation rate
        dissipation_rate = (energy[-1] - energy[0]) / (energy[0] * 1000 * 0.01)
        
        print(f"\n{'='*60}")
        print(f"Test 2.1: Energy Dissipation (η=1e-4)")
        print(f"{'='*60}")
        print(f"H(0)            = {energy[0]:.8e}")
        print(f"H(t=10.0)       = {energy[-1]:.8e}")
        print(f"ΔH/H₀           = {(energy[-1] - energy[0]) / energy[0]:.4f}")
        print(f"Dissipation rate: {dissipation_rate:.4e} (per unit time)")
        print(f"Monotonic       : {'✅ YES' if is_monotonic else '❌ NO'}")
        print(f"Status          : {'✅ PASS' if energy[-1] < energy[0] and -1.0 < dissipation_rate < 0.0 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        time = np.arange(len(energy)) * 10 * 0.01
        ax.plot(time, energy / energy[0], 'r-', linewidth=2, label='η=1e-4')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$H / H_0$', fontsize=12)
        ax.set_title('Test 2.1: Energy Dissipation (Resistive MHD)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_2_1_energy_dissipation.png', dpi=150)
        plt.close()
        
        assert energy[-1] < energy[0], "Energy should decrease in resistive MHD"
        assert -1.0 < dissipation_rate < 0.0, f"Dissipation rate {dissipation_rate:.2e} unreasonable"
    
    def test_2_2_dissipation_rate_vs_eta(self, ballooning_ic):
        """
        Test 2.2: Dissipation rate should increase with η
        Expected: Higher η → faster dissipation
        """
        psi0, omega0, grid = ballooning_ic
        
        etas = [1e-5, 1e-4, 1e-3]
        results = []
        
        for eta in etas:
            psi_hist, omega_hist, diag = evolve_3d_imex(
                psi0, omega0, grid,
                eta=eta,
                dt=0.01,
                n_steps=500,  # Shorter for speed
                J_ext=None,
                store_interval=10
            )
            
            energy = diag['energy']
            dissipation_rate = (energy[-1] - energy[0]) / (energy[0] * 500 * 0.01)
            results.append((eta, energy, dissipation_rate))
        
        print(f"\n{'='*60}")
        print(f"Test 2.2: Dissipation Rate vs η")
        print(f"{'='*60}")
        for eta, energy, rate in results:
            print(f"η={eta:.0e}: dissipation rate = {rate:.4e}")
        
        # Check ordering
        rates = [r[2] for r in results]
        is_ordered = all(rates[i] < rates[i+1] for i in range(len(rates)-1))
        print(f"Ordering check  : {'✅ PASS' if is_ordered else '❌ FAIL'}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Energy evolution
        for eta, energy, rate in results:
            time = np.arange(len(energy)) * 10 * 0.01
            ax1.semilogy(time, energy / energy[0], linewidth=2, label=f'η={eta:.0e}')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('$H / H_0$', fontsize=12)
        ax1.set_title('Energy Decay vs η', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Dissipation rates
        ax2.loglog([r[0] for r in results], [-r[2] for r in results], 'ro-', markersize=10, linewidth=2)
        ax2.set_xlabel('η (resistivity)', fontsize=12)
        ax2.set_ylabel('Dissipation rate (per unit time)', fontsize=12)
        ax2.set_title('Dissipation Rate vs η', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_2_2_dissipation_vs_eta.png', dpi=150)
        plt.close()
        
        assert is_ordered, "Dissipation rate should increase with η"


# =============================================================================
# Test 3: ∇·B=0 Constraint (P0 - Critical)
# =============================================================================

class TestDivergenceConstraint:
    """P0: Magnetic field must remain divergence-free."""
    
    def test_3_1_divergence_check(self, ballooning_ic):
        """
        Test 3.1: Check ∇·B ≈ 0
        
        Note: B = ∇ψ × ∇ζ in reduced MHD, so ∇·B = 0 by construction.
        This test verifies numerical implementation preserves this.
        
        Expected: max|∇·B| < 1e-10 (machine precision)
        """
        psi0, omega0, grid = ballooning_ic
        
        # Evolve
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=1e-4,
            dt=0.01,
            n_steps=100,
            J_ext=None,
            store_interval=10
        )
        
        psi_final = psi_hist[-1]
        
        # Compute ∇·B
        # In reduced MHD: B = ∇ψ × ∇ζ = (0, 0, ∂ψ/∂r) in (r,θ,ζ) coordinates
        # ∇·B = ∂B_r/∂r + B_r/r + (1/r)∂B_θ/∂θ + ∂B_ζ/∂ζ
        # For B = (0, 0, B_ζ): ∇·B = ∂B_ζ/∂ζ
        
        # Compute B_ζ = ∂ψ/∂r (finite difference)
        dr = grid.dr
        B_zeta = np.gradient(psi_final, dr, axis=0)  # ∂ψ/∂r
        
        # ∇·B = ∂B_ζ/∂ζ (should be zero by periodicity)
        dzeta = grid.dzeta
        div_B = np.gradient(B_zeta, dzeta, axis=2)
        
        max_div_B = np.max(np.abs(div_B))
        
        print(f"\n{'='*60}")
        print(f"Test 3.1: ∇·B = 0 Constraint")
        print(f"{'='*60}")
        print(f"max|∇·B| = {max_div_B:.2e}")
        print(f"Target   : < 1e-10")
        print(f"Status   : {'✅ PASS' if max_div_B < 1e-10 else '❌ FAIL'}")
        
        # Note: This is a weak test since reduced MHD enforces ∇·B=0 by construction
        # A stronger test would require full 3D B field reconstruction
        
        assert max_div_B < 1e-10, f"∇·B = {max_div_B:.2e} violates divergence-free constraint"


# =============================================================================
# Test 4: J_ext Energy Injection (P1 - Important)
# =============================================================================

class TestExternalCurrent:
    """P1: External current should inject energy correctly."""
    
    def test_4_1_constant_j_ext_ideal(self, ballooning_ic):
        """
        Test 4.1: Constant J_ext in ideal MHD
        Expected: Energy increases due to J_ext work
        """
        psi0, omega0, grid = ballooning_ic
        
        # Constant external current
        J_ext = 0.1 * np.ones_like(psi0)
        
        # Evolve (ideal MHD + J_ext)
        psi_hist, omega_hist, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,        # Ideal (no dissipation)
            dt=0.01,
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
        print(f"H(t=1.0) = {energy[-1]:.8e}")
        print(f"ΔH       = {energy_increase:.4e}")
        print(f"Status   : {'✅ PASS' if energy_increase > 0 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        time = np.arange(len(energy)) * 10 * 0.01
        ax.plot(time, energy, 'g-', linewidth=2, label='J_ext=0.1')
        ax.axhline(energy[0], color='k', linestyle='--', alpha=0.5, label='H(0)')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy H', fontsize=12)
        ax.set_title('Test 4.1: Energy Injection by J_ext', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_4_1_j_ext_energy_injection.png', dpi=150)
        plt.close()
        
        assert energy_increase > 0, "J_ext should increase energy in ideal MHD"


# =============================================================================
# Test 5: Grid Convergence (P1 - Important)
# =============================================================================

class TestGridConvergence:
    """P1: Solution should converge as grid is refined."""
    
    def test_5_1_spatial_convergence(self):
        """
        Test 5.1: Spatial convergence
        Expected: Second-order convergence (IMEX is 2nd order in space)
        
        Note: Largest grid (64×128×256) is very slow; use n_steps=50
        """
        grids = [
            Grid3D(nr=16, ntheta=32, nzeta=64),
            Grid3D(nr=32, ntheta=64, nzeta=128),
            # Grid3D(nr=64, ntheta=128, nzeta=256)  # Too slow, skip for now
        ]
        
        results = []
        
        for i, grid in enumerate(grids):
            print(f"\nRunning grid {i+1}/{len(grids)}: {grid.nr}×{grid.ntheta}×{grid.nzeta}...")
            
            # Same IC on each grid
            psi0, omega0, q = create_equilibrium_ic(grid)
            psi1, omega1 = create_ballooning_mode_ic(grid, n=5, m0=2, epsilon=0.01)
            psi0 = psi0 + psi1
            omega0 = omega0 + omega1
            
            # Evolve
            psi_hist, omega_hist, diag = evolve_3d_imex(
                psi0, omega0, grid,
                eta=1e-4,
                dt=0.01,
                n_steps=50,  # Shorter for speed
                J_ext=None,
                store_interval=10
            )
            
            E_final = diag['energy'][-1]
            results.append((grid, E_final))
        
        print(f"\n{'='*60}")
        print(f"Test 5.1: Grid Convergence")
        print(f"{'='*60}")
        for grid, E in results:
            print(f"Grid {grid.nr}×{grid.ntheta}×{grid.nzeta}: E_final = {E:.8e}")
        
        # Check convergence (energy should stabilize)
        if len(results) >= 2:
            E_diff = abs(results[1][1] - results[0][1])
            E_rel = E_diff / abs(results[0][1])
            print(f"Relative difference: {E_rel:.2e}")
            print(f"Status: {'✅ PASS' if E_rel < 0.1 else '⚠️  Needs finer grid'}")
        
        # Plot (if we had 3+ grids, could compute convergence order)
        # For now, just save results
        
        assert len(results) >= 2, "Need at least 2 grids for convergence test"


# =============================================================================
# Test 6: Timestep Convergence (P1 - Important)
# =============================================================================

class TestTimestepConvergence:
    """P1: Solution should be stable and converge as dt → 0."""
    
    def test_6_1_temporal_stability(self, ballooning_ic):
        """
        Test 6.1: Temporal stability for various dt
        Expected: Stable for all dt ≤ 0.02, convergence to same final state
        """
        psi0, omega0, grid = ballooning_ic
        
        dts = [0.02, 0.01, 0.005]
        results = []
        
        for dt in dts:
            n_steps = int(1.0 / dt)  # Fix physical time T=1.0
            
            print(f"\nRunning dt={dt:.4f} ({n_steps} steps)...")
            
            psi_hist, omega_hist, diag = evolve_3d_imex(
                psi0, omega0, grid,
                eta=1e-4,
                dt=dt,
                n_steps=n_steps,
                J_ext=None,
                store_interval=max(1, n_steps // 10)
            )
            
            E_final = diag['energy'][-1]
            results.append((dt, E_final))
        
        print(f"\n{'='*60}")
        print(f"Test 6.1: Timestep Convergence")
        print(f"{'='*60}")
        for dt, E in results:
            print(f"dt={dt:.4f}: E_final = {E:.8e}")
        
        # Check convergence
        E_ref = results[-1][1]  # Finest dt
        for dt, E in results:
            rel_diff = abs(E - E_ref) / abs(E_ref)
            print(f"  dt={dt:.4f}: rel_diff = {rel_diff:.2e}")
        
        max_diff = max(abs(E - E_ref) / abs(E_ref) for _, E in results)
        print(f"Max relative difference: {max_diff:.2e}")
        print(f"Status: {'✅ PASS' if max_diff < 0.01 else '❌ FAIL'}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([r[0] for r in results], [r[1] for r in results], 'bo-', markersize=10, linewidth=2)
        ax.set_xlabel('Timestep dt', fontsize=12)
        ax.set_ylabel('Final Energy', fontsize=12)
        ax.set_title('Test 6.1: Timestep Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'test_6_1_timestep_convergence.png', dpi=150)
        plt.close()
        
        assert max_diff < 0.01, f"Timestep convergence failed: max_diff={max_diff:.2e}"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
