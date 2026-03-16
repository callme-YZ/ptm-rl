"""
Grid Convergence Study

Tests tearing mode island width convergence across three grid resolutions:
- 32×64 (coarse)
- 64×128 (baseline)
- 128×256 (fine)

Objective: Confirm 64×128 is sufficient for production use.

Author: 小P ⚛️
Created: 2026-03-16
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

from pytokmhd.solver import time_integrator, boundary


def initialize_tearing_mode(Nr, Nz, Lr=1.0, Lz=6.0, epsilon=0.01, m=2):
    """
    Initialize tearing mode perturbation.
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    Lr, Lz : float
        Domain size
    epsilon : float
        Perturbation amplitude
    m : int
        Mode number
    
    Returns
    -------
    psi0, omega0, r_grid, dr, dz
    """
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    dr, dz = r[1] - r[0], z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    # Equilibrium: ψ_eq = (1 - r²)
    psi_eq = (1 - R**2)
    
    # Perturbation: ψ' = ε sin(m·2πz/Lz) (1-r²)
    psi_pert = epsilon * np.sin(m * 2*np.pi*Z/Lz) * (1 - R**2)
    
    psi0 = psi_eq + psi_pert
    omega0 = np.zeros_like(psi0)
    
    return psi0, omega0, R, dr, dz, psi_eq


def measure_island_width(psi, psi_eq):
    """
    Measure island width as max perturbation amplitude.
    
    Simple metric: w = max|ψ - ψ_eq|
    """
    psi_pert = psi - psi_eq
    w = np.max(np.abs(psi_pert))
    return w


def run_convergence_study():
    """
    Run grid convergence study on three resolutions.
    """
    print("="*60)
    print("Grid Convergence Study")
    print("="*60)
    
    resolutions = [
        (32, 64, "coarse"),
        (64, 128, "baseline"),
        (128, 256, "fine")
    ]
    
    results = {}
    
    # Physics parameters
    eta = 1e-3
    dt = 0.001
    t_final = 0.1  # Short run for convergence test
    
    for Nr, Nz, label in resolutions:
        print(f"\n=== Running {label} grid: {Nr}×{Nz} ===")
        
        # Initialize
        psi0, omega0, R, dr, dz, psi_eq = initialize_tearing_mode(Nr, Nz)
        
        # Initial island width
        w_initial = measure_island_width(psi0, psi_eq)
        print(f"Initial island width: {w_initial:.6e}")
        
        # Evolve
        psi = psi0.copy()
        omega = omega0.copy()
        t = 0.0
        
        n_steps = int(t_final / dt)
        
        for step in range(n_steps):
            psi, omega = time_integrator.rk4_step(
                psi, omega, dt, dr, dz, R, eta, nu=0.0,
                apply_bc=boundary.apply_combined_bc
            )
            t += dt
            
            # Progress
            if (step + 1) % 20 == 0:
                w_current = measure_island_width(psi, psi_eq)
                print(f"  Step {step+1:3d}/{n_steps}: w = {w_current:.6e}")
        
        # Final island width
        w_final = measure_island_width(psi, psi_eq)
        
        print(f"Final island width: {w_final:.6e}")
        print(f"Growth: {(w_final/w_initial - 1)*100:.2f}%")
        
        results[label] = {
            'Nr': Nr,
            'Nz': Nz,
            'w_initial': w_initial,
            'w_final': w_final,
            'psi_final': psi,
            'omega_final': omega
        }
    
    # Analyze convergence
    print("\n" + "="*60)
    print("Convergence Analysis")
    print("="*60)
    
    w_coarse = results['coarse']['w_final']
    w_baseline = results['baseline']['w_final']
    w_fine = results['fine']['w_final']
    
    print(f"\nFinal island widths:")
    print(f"  Coarse   (32×64):   {w_coarse:.6e}")
    print(f"  Baseline (64×128):  {w_baseline:.6e}")
    print(f"  Fine     (128×256): {w_fine:.6e}")
    
    # Relative differences
    diff_coarse_baseline = abs(w_coarse - w_baseline) / w_baseline * 100
    diff_baseline_fine = abs(w_baseline - w_fine) / w_fine * 100
    
    print(f"\nRelative differences:")
    print(f"  Coarse vs Baseline:  {diff_coarse_baseline:.2f}%")
    print(f"  Baseline vs Fine:    {diff_baseline_fine:.2f}%")
    
    # Convergence criterion
    print(f"\n--- Convergence Assessment ---")
    
    if diff_baseline_fine < 5.0:
        print(f"✅ Baseline (64×128) converged: <5% difference from fine grid")
        print(f"✅ 64×128 is SUFFICIENT for production")
        converged = True
    else:
        print(f"⚠️  Baseline not fully converged: {diff_baseline_fine:.2f}% > 5%")
        print(f"⚠️  Consider using finer grid")
        converged = False
    
    # Richardson extrapolation (estimate true solution)
    # Assuming 2nd order: w_true ≈ w_fine + (w_fine - w_baseline)/3
    w_extrapolated = w_fine + (w_fine - w_baseline) / 3.0
    error_baseline = abs(w_baseline - w_extrapolated) / w_extrapolated * 100
    
    print(f"\nRichardson extrapolation:")
    print(f"  Extrapolated w_true: {w_extrapolated:.6e}")
    print(f"  Baseline error:      {error_baseline:.2f}%")
    
    return results, converged


if __name__ == "__main__":
    results, converged = run_convergence_study()
    
    print("\n" + "="*60)
    print("Grid Convergence Study Complete")
    print("="*60)
    
    sys.exit(0 if converged else 1)
