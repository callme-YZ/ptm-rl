"""
Test 1: IMEX Stability Range
Test resistivity range from η=0 (ideal) to η=1e-3 (large).

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Validation
"""
import numpy as np
import pytest
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.physics import compute_current_density, compute_hamiltonian
from pytokmhd.equilibrium import load_solovev_equilibrium


def test_stability_range():
    """Test stability for resistivity range η ∈ [0, 1e-3]"""
    
    # Test parameters
    eta_values = [0, 1e-6, 1e-5, 1e-4, 1e-3]
    n_steps = 100
    dt = 1e-3
    
    # Grid setup
    nr, nth = 64, 64
    R0, a = 1.0, 0.3
    grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=nth)
    
    results = {}
    
    for eta in eta_values:
        print(f"\n{'='*60}")
        print(f"Testing η = {eta:.1e}")
        print(f"{'='*60}")
        
        # Initialize solver
        solver = HamiltonianMHDIMEX(
            grid,
            dt=dt,
            eta=eta,
            nu=0.0,  # No viscosity for this test
            use_imex=True if eta > 0 else False,
            verbose=False
        )
        
        # Initial condition: Solovev equilibrium + small perturbation
        eq = load_solovev_equilibrium(grid, P0=1e5, B0=2.0, beta_p=0.5, q0=1.5)
        if eq is None:
            # Fallback: simple parabolic profile
            psi0 = grid.r_grid**2 * (1 - grid.r_grid/a)**2
        else:
            psi0 = eq['psi']
        
        # Add perturbation
        pert = 0.01 * np.sin(2*grid.theta_grid) * grid.r_grid**2 * (1 - grid.r_grid/a)**2
        psi = psi0 + pert
        omega = -laplacian_toroidal(psi, grid)
        
        # Storage
        energy_history = []
        dissipation_rates = []
        max_psi_history = []
        
        # Initial energy
        H0 = compute_hamiltonian(psi, omega, grid)
        energy_history.append(H0)
        max_psi_history.append(np.max(np.abs(psi)))
        
        # Time evolution
        failed = False
        for step in range(n_steps):
            # Compute J before step
            J = compute_current_density(psi, grid)
            # Volume element: dV = R * dr * dθ
            dV = (R0 + grid.r_grid * np.cos(grid.theta_grid)) * grid.dr * grid.dtheta
            J2_int = np.sum(J**2 * dV)
            
            # Step
            psi_old, omega_old = psi, omega
            psi, omega = solver.step(psi, omega)
            
            # Check for NaN
            if np.any(np.isnan(psi)) or np.any(np.isnan(omega)):
                print(f"  ❌ NaN at step {step}")
                failed = True
                break
            
            # Check for explosion
            if np.max(np.abs(psi)) > 100 * np.max(np.abs(psi0)):
                print(f"  ❌ Explosion at step {step}")
                failed = True
                break
            
            # Energy
            H = compute_hamiltonian(psi, omega, grid)
            energy_history.append(H)
            max_psi_history.append(np.max(np.abs(psi)))
            
            # Dissipation rate
            if eta > 0:
                dH_dt_numerical = (H - energy_history[-2]) / dt
                dH_dt_theory = -eta * J2_int
                dissipation_rates.append({
                    'numerical': dH_dt_numerical,
                    'theory': dH_dt_theory,
                    'J2_int': J2_int
                })
            
            if step % 20 == 0:
                if eta > 0:
                    print(f"  Step {step:3d}: H = {H:.6e}, dH/dt = {dH_dt_numerical:.3e} (theory: {dH_dt_theory:.3e})")
                else:
                    print(f"  Step {step:3d}: H = {H:.6e}, ΔH/H0 = {(H-H0)/H0:.3e}")
        
        if not failed:
            print(f"  ✅ Stable for {n_steps} steps")
            
            # Check energy monotonicity
            energy_history = np.array(energy_history)
            if eta > 0:
                if np.all(np.diff(energy_history) <= 0):
                    print(f"  ✅ Energy monotonically decreasing")
                else:
                    increases = np.sum(np.diff(energy_history) > 0)
                    print(f"  ⚠️  Energy increased {increases} times")
            else:
                # Ideal case: should conserve energy
                rel_error = np.abs((energy_history[-1] - H0) / H0)
                print(f"  Energy conservation: ΔH/H0 = {rel_error:.3e}")
                if rel_error < 1e-6:
                    print(f"  ✅ Excellent conservation")
                elif rel_error < 1e-4:
                    print(f"  ✅ Good conservation")
                else:
                    print(f"  ⚠️  Conservation error")
            
            # Check dissipation rate vs theory
            if eta > 0 and len(dissipation_rates) > 0:
                errors = []
                for dr in dissipation_rates:
                    if dr['theory'] != 0:
                        rel_err = abs((dr['numerical'] - dr['theory']) / dr['theory'])
                        errors.append(rel_err)
                
                if len(errors) > 0:
                    mean_error = np.mean(errors)
                    print(f"  Dissipation rate error: {mean_error:.1%}")
                    if mean_error < 0.10:
                        print(f"  ✅ Theory match within 10%")
                    else:
                        print(f"  ⚠️  Theory error > 10%")
        
        results[eta] = {
            'success': not failed,
            'energy_history': energy_history if not failed else None,
            'dissipation_rates': dissipation_rates if not failed and eta > 0 else None
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Stability Range Test")
    print(f"{'='*60}")
    for eta, res in results.items():
        status = "✅ PASS" if res['success'] else "❌ FAIL"
        print(f"η = {eta:.1e}: {status}")
    
    # Assert all passed
    assert all(res['success'] for res in results.values()), "Some η values failed"
    print("\n🎉 All resistivity values stable!")


if __name__ == "__main__":
    test_stability_range()
