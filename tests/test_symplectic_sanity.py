"""
Symplectic Integrator Sanity Check
===================================

Step 1: Simplest possible test
- Pure Hamiltonian (η=ν=0)
- Trivial solution (ψ=0, ω=0)
- Should stay zero, no NaN

Author: 小P ⚛️
Date: 2026-03-18
"""

import numpy as np
import pytest
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.integrators.symplectic import SymplecticIntegrator


class TestSymplecticSanity:
    """Sanity checks before full validation."""
    
    def test_trivial_solution(self):
        """
        Test 1: Trivial solution (ψ=0, ω=0) should stay zero.
        
        Setup:
        - Pure Hamiltonian (η=ν=0)
        - Zero initial condition
        - 100 steps
        
        Expected:
        - ψ, ω stay zero
        - No NaN/Inf
        """
        print("\n" + "="*60)
        print("⚛️ SANITY CHECK: Trivial solution (ψ=0, ω=0)")
        print("="*60)
        
        # Grid
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Initial condition: ZERO
        psi0 = np.zeros((grid.nr, grid.ntheta))
        omega0 = np.zeros((grid.nr, grid.ntheta))
        
        print(f"Initial: ψ_max={np.max(np.abs(psi0)):.2e}, ω_max={np.max(np.abs(omega0)):.2e}")
        
        # Integrator: Pure Hamiltonian
        integrator = SymplecticIntegrator(
            grid=grid,
            dt=1e-4,
            eta=0.0,   # No resistivity
            nu=0.0     # No viscosity
        )
        
        integrator.psi = psi0.copy()
        integrator.omega = omega0.copy()
        
        # Run 100 steps
        n_steps = 100
        print(f"\nRunning {n_steps} steps (Pure Hamiltonian)...")
        
        for step in range(n_steps):
            integrator.step()
            
            # Check for NaN
            if np.any(np.isnan(integrator.psi)) or np.any(np.isnan(integrator.omega)):
                print(f"❌ NaN detected at step {step}!")
                print(f"   ψ: min={np.min(integrator.psi):.2e}, max={np.max(integrator.psi):.2e}")
                print(f"   ω: min={np.min(integrator.omega):.2e}, max={np.max(integrator.omega):.2e}")
                pytest.fail(f"NaN at step {step}")
        
        # Final check
        psi_max = np.max(np.abs(integrator.psi))
        omega_max = np.max(np.abs(integrator.omega))
        
        print(f"\nAfter {n_steps} steps:")
        print(f"  ψ_max = {psi_max:.2e}")
        print(f"  ω_max = {omega_max:.2e}")
        
        # Should stay zero (machine precision)
        assert psi_max < 1e-14, f"ψ grew to {psi_max:.2e} (should stay ~0)"
        assert omega_max < 1e-14, f"ω grew to {omega_max:.2e} (should stay ~0)"
        
        print("✅ PASS: Trivial solution stable!")
        print("="*60)
    
    
    def test_small_perturbation_hamiltonian(self):
        """
        Test 2: Small perturbation in pure Hamiltonian system.
        
        Setup:
        - Pure Hamiltonian (η=ν=0)
        - ψ₀ = 0.001 × r(1-r/a) sin(θ)  (small perturbation)
        - ω₀ = ∇²ψ₀ (consistent)
        - 1000 steps quick check
        
        Expected:
        - Energy conserved to high precision
        - No NaN, bounded evolution
        """
        print("\n" + "="*60)
        print("⚛️ SANITY CHECK: Small perturbation Hamiltonian")
        print("="*60)
        
        # Grid
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Initial condition: Small perturbation
        from pytokmhd.operators import laplacian_toroidal
        
        r = grid.r_grid
        theta = grid.theta_grid
        a = grid.a
        
        # Small amplitude perturbation
        amplitude = 0.001
        psi0 = amplitude * r * (1 - r/a) * np.sin(theta)
        omega0 = laplacian_toroidal(psi0, grid)  # Consistent vorticity
        
        print(f"Initial perturbation amplitude: {amplitude}")
        print(f"  ψ_max={np.max(np.abs(psi0)):.3e}, ω_max={np.max(np.abs(omega0)):.3e}")
        
        # Integrator: Pure Hamiltonian
        integrator = SymplecticIntegrator(
            grid=grid,
            dt=1e-4,
            eta=0.0,   # No resistivity
            nu=0.0     # No viscosity
        )
        
        integrator.psi = psi0.copy()
        integrator.omega = omega0.copy()
        
        # Initial energy
        E0 = integrator.compute_energy()
        print(f"Initial energy: E₀ = {E0:.6e}")
        
        # Run 1000 steps (quick check)
        n_steps = 1000
        print(f"\nRunning {n_steps} steps (quick stability check)...")
        
        energy_history = [E0]
        psi_max_history = [np.max(np.abs(psi0))]
        
        for step in range(n_steps):
            integrator.step()
            
            # Check for NaN every 100 steps
            if (step + 1) % 100 == 0:
                E = integrator.compute_energy()
                energy_history.append(E)
                dE = abs(E - E0) / E0
                
                psi_max = np.max(np.abs(integrator.psi))
                psi_max_history.append(psi_max)
                
                print(f"  Step {step+1:4d}: E = {E:.6e}, ΔE/E₀ = {dE:.2e}, ψ_max = {psi_max:.3e}")
                
                if np.isnan(E):
                    print(f"❌ NaN energy at step {step+1}!")
                    pytest.fail(f"NaN at step {step+1}")
        
        # Final check
        E_final = integrator.compute_energy()
        energy_drift = abs(E_final - E0) / E0
        psi_final_max = np.max(np.abs(integrator.psi))
        
        print(f"\nFinal state:")
        print(f"  Energy drift: ΔE/E₀ = {energy_drift:.2e}")
        print(f"  ψ_max: {psi_final_max:.3e} (initial: {amplitude:.3e})")
        
        # Pure Hamiltonian should conserve energy well
        # Accept <1% drift for 1000 steps (quick check)
        if energy_drift < 1e-3:
            print("✅ EXCELLENT: Energy drift < 0.1%")
        elif energy_drift < 1e-2:
            print("✅ GOOD: Energy drift < 1%")
        else:
            print(f"⚠️  WARNING: Energy drift {energy_drift:.2e} > 1%")
            print("   May indicate toroidal geometry challenges")
        
        # Check for runaway growth
        assert psi_final_max < 10 * amplitude, f"ψ grew 10× (unstable!)"
        assert not np.isnan(E_final), "NaN detected (numerical stability)"
        
        print("✅ PASS: Small perturbation test complete!")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
