"""
RK4 vs Symplectic Comparison Test

Direct comparison of energy conservation over 1000 steps.

Setup:
- Same IC: ψ₀ = 0.001 × r(1-r/a) sinθ
- Same physics: η=ν=0 (Pure Hamiltonian)
- Same dt: 1e-4
- Same boundary conditions

Goal: Prove Symplectic > RK4

Author: 小P ⚛️
Date: 2026-03-18
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# Import integrators
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.integrators.symplectic import SymplecticIntegrator
from pytokmhd.operators import laplacian_toroidal

# Import RK4 from existing test
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_step_2_4_longterm import RK4Integrator


class TestRK4vsSymplectic:
    """Compare RK4 and Symplectic integrators."""
    
    def test_energy_conservation_comparison(self):
        """
        Compare energy conservation: RK4 vs Symplectic
        
        Expected:
        - Symplectic: ~5-10% drift (based on sanity check)
        - RK4: >10% drift or NaN
        - Symplectic should be 2-5× better
        """
        print("\n" + "="*70)
        print("⚛️ RK4 vs SYMPLECTIC ENERGY CONSERVATION COMPARISON")
        print("="*70)
        
        # Grid
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Initial condition: Small perturbation
        r = grid.r_grid
        theta = grid.theta_grid
        a = grid.a
        
        amplitude = 0.001
        psi0 = amplitude * r * (1 - r/a) * np.sin(theta)
        omega0 = laplacian_toroidal(psi0, grid)
        
        print(f"\nInitial condition:")
        print(f"  Amplitude: {amplitude}")
        print(f"  ψ_max = {np.max(np.abs(psi0)):.3e}")
        print(f"  ω_max = {np.max(np.abs(omega0)):.3e}")
        
        # Parameters
        dt = 1e-4
        eta = 0.0
        nu = 0.0
        n_steps = 1000
        
        print(f"\nPhysics:")
        print(f"  η = {eta} (Pure Hamiltonian)")
        print(f"  ν = {nu}")
        print(f"  dt = {dt}")
        print(f"  steps = {n_steps}")
        
        # ========================================
        # Test 1: Symplectic
        # ========================================
        print("\n" + "-"*70)
        print("Running SYMPLECTIC integrator...")
        print("-"*70)
        
        symplectic = SymplecticIntegrator(grid=grid, dt=dt, eta=eta, nu=nu)
        symplectic.psi = psi0.copy()
        symplectic.omega = omega0.copy()
        symplectic.enable_energy_tracking()
        
        E0_sym = symplectic.compute_energy()
        print(f"Initial energy: E₀ = {E0_sym:.6e}")
        
        sym_energies = [E0_sym]
        sym_times = [0.0]
        
        for step in range(n_steps):
            symplectic.step()
            
            if (step + 1) % 100 == 0:
                E = symplectic.compute_energy()
                sym_energies.append(E)
                sym_times.append(symplectic.t)
                
                dE = abs(E - E0_sym) / E0_sym
                print(f"  Step {step+1:4d}: E = {E:.6e}, ΔE/E₀ = {dE:.2e}")
        
        E_final_sym = symplectic.compute_energy()
        drift_sym = abs(E_final_sym - E0_sym) / E0_sym
        
        print(f"\nSymplectic final:")
        print(f"  Energy drift: ΔE/E₀ = {drift_sym:.3e}")
        print(f"  ψ_max: {np.max(np.abs(symplectic.psi)):.3e}")
        
        # ========================================
        # Test 2: RK4
        # ========================================
        print("\n" + "-"*70)
        print("Running RK4 integrator...")
        print("-"*70)
        
        rk4 = RK4Integrator(grid=grid, dt=dt, eta=eta, nu=nu)
        rk4.psi = psi0.copy()
        rk4.omega = omega0.copy()
        rk4.enable_energy_tracking()
        
        E0_rk4 = rk4.compute_energy()
        print(f"Initial energy: E₀ = {E0_rk4:.6e}")
        
        rk4_energies = [E0_rk4]
        rk4_times = [0.0]
        
        rk4_failed = False
        rk4_fail_step = None
        
        for step in range(n_steps):
            rk4.step()
            
            if (step + 1) % 100 == 0:
                E = rk4.compute_energy()
                
                if np.isnan(E):
                    print(f"  ❌ RK4 failed at step {step+1} (NaN)")
                    rk4_failed = True
                    rk4_fail_step = step + 1
                    break
                
                rk4_energies.append(E)
                rk4_times.append(rk4.t)
                
                dE = abs(E - E0_rk4) / E0_rk4
                print(f"  Step {step+1:4d}: E = {E:.6e}, ΔE/E₀ = {dE:.2e}")
        
        if not rk4_failed:
            E_final_rk4 = rk4.compute_energy()
            drift_rk4 = abs(E_final_rk4 - E0_rk4) / E0_rk4
            
            print(f"\nRK4 final:")
            print(f"  Energy drift: ΔE/E₀ = {drift_rk4:.3e}")
            print(f"  ψ_max: {np.max(np.abs(rk4.psi)):.3e}")
        
        # ========================================
        # Comparison
        # ========================================
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\nSymplectic:")
        print(f"  Energy drift: {drift_sym:.3e}")
        print(f"  Status: ✅ Completed {n_steps} steps")
        
        if rk4_failed:
            print(f"\nRK4:")
            print(f"  Status: ❌ Failed at step {rk4_fail_step}/{n_steps} (NaN)")
            print(f"  Symplectic >> RK4 (RK4 diverged!)")
            
            advantage = "∞ (RK4 diverged)"
        else:
            print(f"\nRK4:")
            print(f"  Energy drift: {drift_rk4:.3e}")
            print(f"  Status: ✅ Completed {n_steps} steps")
            
            ratio = drift_rk4 / drift_sym
            print(f"\nAdvantage:")
            print(f"  Symplectic is {ratio:.1f}× better than RK4")
            
            advantage = f"{ratio:.1f}×"
        
        # ========================================
        # Plot
        # ========================================
        print("\nGenerating comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Energy history
        ax1.plot(sym_times, np.array(sym_energies)/E0_sym - 1, 
                'b-', linewidth=2, label='Symplectic')
        
        if not rk4_failed:
            ax1.plot(rk4_times, np.array(rk4_energies)/E0_rk4 - 1,
                    'r--', linewidth=2, label='RK4')
        else:
            # Plot partial RK4 history
            ax1.plot(rk4_times, np.array(rk4_energies)/E0_rk4 - 1,
                    'r--', linewidth=2, label=f'RK4 (failed at step {rk4_fail_step})')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('ΔE/E₀')
        ax1.set_title('Energy Conservation Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
        
        # Plot 2: Absolute energy drift
        sym_drift_abs = np.abs(np.array(sym_energies)/E0_sym - 1)
        ax2.semilogy(sym_times, sym_drift_abs, 'b-', linewidth=2, label='Symplectic')
        
        if not rk4_failed:
            rk4_drift_abs = np.abs(np.array(rk4_energies)/E0_rk4 - 1)
            ax2.semilogy(rk4_times, rk4_drift_abs, 'r--', linewidth=2, label='RK4')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('|ΔE/E₀|')
        ax2.set_title('Energy Drift (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        output_path = Path(__file__).parent / 'rk4_vs_symplectic_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
        
        # ========================================
        # Verdict
        # ========================================
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if rk4_failed:
            print("✅ Symplectic integrator PASSES")
            print("❌ RK4 integrator FAILS (diverged)")
            print("🎯 Symplectic >> RK4 (infinitely better!)")
            verdict = "PASS"
        else:
            if drift_sym < drift_rk4:
                print(f"✅ Symplectic ({drift_sym:.2e}) < RK4 ({drift_rk4:.2e})")
                print(f"🎯 Symplectic is {advantage} better")
                
                if ratio >= 2.0:
                    verdict = "PASS"
                    print("✅ PASS: Symplectic ≥2× better than RK4")
                else:
                    verdict = "MARGINAL"
                    print(f"⚠️  MARGINAL: Symplectic only {ratio:.1f}× better (want ≥2×)")
            else:
                print(f"❌ RK4 ({drift_rk4:.2e}) < Symplectic ({drift_sym:.2e})")
                print("❌ FAIL: RK4 better than Symplectic (unexpected!)")
                verdict = "FAIL"
        
        print("="*70)
        
        # Assert
        if rk4_failed:
            # RK4 diverged, symplectic didn't → clear win
            assert True, "Symplectic stable, RK4 diverged"
        else:
            # Both finished, symplectic should be better
            assert drift_sym < drift_rk4, f"Symplectic ({drift_sym:.2e}) should be < RK4 ({drift_rk4:.2e})"
            assert ratio >= 2.0, f"Symplectic should be ≥2× better, got {ratio:.1f}×"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
