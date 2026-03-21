#!/usr/bin/env python3
"""
v1.3 Stable Demo: Ideal and Viscous MHD Evolution

Demonstrates stable time evolution with v1.3 Hamiltonian MHD solver.
Uses ideal (η=ν=0) or viscous (η=0, ν>0) configurations to avoid
resistive instability.

Usage:
    python scripts/v1_3_demo_stable.py --mode ideal --steps 100
    python scripts/v1_3_demo_stable.py --mode viscous --steps 100

Author: 小P ⚛️
Date: 2026-03-19
"""

import numpy as np
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.solvers import HamiltonianMHD
from pytokmhd.physics import kinetic_energy, magnetic_energy


def run_evolution(mode: str, n_steps: int = 100, verbose: bool = True):
    """Run stable MHD evolution."""
    
    if verbose:
        print("="*70)
        print(f"v1.3 Hamiltonian MHD Evolution: {mode.upper()} mode")
        print("="*70)
        print()
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Initial condition: cylindrical equilibrium
    r_mesh = grid.r_grid
    psi = r_mesh**2 * (1 - r_mesh / grid.a)
    omega = -laplacian_toroidal(psi, grid)
    
    # Solver parameters
    if mode == "ideal":
        solver = HamiltonianMHD(grid, dt=1e-5, eta=0.0, nu=0.0)
    elif mode == "viscous":
        solver = HamiltonianMHD(grid, dt=1e-5, eta=0.0, nu=1e-4)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if verbose:
        print(f"Grid: {grid.nr} × {grid.ntheta}")
        print(f"Solver: dt={solver.dt}, η={solver.eta}, ν={solver.nu}")
        print(f"Steps: {n_steps} (total time: {n_steps * solver.dt:.3e} s)")
        print()
    
    # Initial energy
    phi = solver.compute_phi(omega)
    E_kin_0 = kinetic_energy(phi, grid)
    E_mag_0 = magnetic_energy(psi, grid)
    E_tot_0 = E_kin_0 + E_mag_0
    
    if verbose:
        print(f"Initial energy:")
        print(f"  E_kin = {E_kin_0:.6e}")
        print(f"  E_mag = {E_mag_0:.6e}")
        print(f"  E_tot = {E_tot_0:.6e}")
        print()
        print("Evolving...")
        print()
    
    # Evolution
    for step in range(n_steps):
        psi, omega = solver.step(psi, omega)
        
        # Monitor energy
        if (step + 1) % (n_steps // 10) == 0 or step == n_steps - 1:
            phi = solver.compute_phi(omega)
            E_kin = kinetic_energy(phi, grid)
            E_mag = magnetic_energy(psi, grid)
            E_tot = E_kin + E_mag
            
            dE_rel = (E_tot - E_tot_0) / E_tot_0 * 100
            
            if verbose:
                print(f"  Step {step+1:4d}: "
                      f"E_tot = {E_tot:.6e}, "
                      f"ΔE/E₀ = {dE_rel:+.4f}%, "
                      f"max|ψ| = {np.abs(psi).max():.3e}")
    
    # Final report
    phi_final = solver.compute_phi(omega)
    E_kin_final = kinetic_energy(phi_final, grid)
    E_mag_final = magnetic_energy(psi, grid)
    E_tot_final = E_kin_final + E_mag_final
    
    energy_drift = abs(E_tot_final - E_tot_0) / E_tot_0 * 100
    
    if verbose:
        print()
        print("="*70)
        print("Results:")
        print("="*70)
        print(f"Final energy:")
        print(f"  E_kin = {E_kin_final:.6e}")
        print(f"  E_mag = {E_mag_final:.6e}")
        print(f"  E_tot = {E_tot_final:.6e}")
        print(f"Energy drift: {energy_drift:.4f}%")
        print()
        
        if mode == "ideal":
            if energy_drift < 0.01:
                print("✅ PASS: Energy conserved (drift < 0.01%)")
            else:
                print(f"⚠️  WARNING: Energy drift {energy_drift:.4f}% (expected < 0.01%)")
        elif mode == "viscous":
            if E_tot_final < E_tot_0:
                dissipation = (E_tot_0 - E_tot_final) / E_tot_0 * 100
                print(f"✅ PASS: Energy dissipated {dissipation:.3f}% (expected for viscous)")
            else:
                print("⚠️  WARNING: Energy increased (unexpected for viscous)")
    
    return {
        "E_tot_0": E_tot_0,
        "E_tot_final": E_tot_final,
        "energy_drift": energy_drift,
        "psi_final": psi,
        "omega_final": omega,
    }


def main():
    parser = argparse.ArgumentParser(description="v1.3 stable MHD evolution demo")
    parser.add_argument("--mode", choices=["ideal", "viscous"], default="ideal",
                        help="Evolution mode (ideal or viscous)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of time steps")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    results = run_evolution(args.mode, args.steps, verbose=not args.quiet)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
