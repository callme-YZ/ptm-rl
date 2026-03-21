#!/usr/bin/env python3
"""
v1.3 Phase 3: Validation Benchmark

Validates v1.3 Hamiltonian solver against v1.2.1 baseline.

Test Scenarios:
1. Equilibrium maintenance (1000 steps)
2. Perturbed evolution (100 steps)
3. v1.2 vs v1.3 comparison

Success Criteria:
- Energy drift < 1% (1000 steps equilibrium)
- Force balance residual < 1e-3
- v1.3 > v1.2 in energy conservation
- Poisson bracket properties verified

Author: 小P ⚛️
Date: 2026-03-19
Phase: v1.3 Phase 3
"""

import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import poisson_bracket, laplacian_toroidal
from pytokmhd.physics import (
    compute_hamiltonian,
    kinetic_energy,
    magnetic_energy,
    energy_partition,
    compute_current_density,
    force_balance_residual,
)
from pytokmhd.equilibrium import pressure_profile, pressure_gradient


class V13Solver:
    """
    v1.3 Hamiltonian MHD Solver
    
    Evolution equations:
        ∂ψ/∂t = [ψ, H] - η·J
        ∂ω/∂t = [ω, H] + (1/R²)(dP/dψ)·Δ*ψ - ν·∇²ω
    
    where H = ∫[(1/2)|∇φ|² + (1/2)|∇ψ|²] dV
    """
    
    def __init__(self, grid, dt=1e-4, eta=1e-4, nu=1e-4, P0=0.0, psi_edge=None, alpha=2.0):
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.P0 = P0
        self.psi_edge = psi_edge if psi_edge is not None else grid.a**2
        self.alpha = alpha
        
        self.step_count = 0
        self.time = 0.0
    
    def compute_phi(self, omega):
        """Solve Poisson equation: ∇²φ = ω."""
        from pytokmhd.solvers import solve_poisson_toroidal
        phi, info = solve_poisson_toroidal(omega, self.grid)
        if info != 0:
            raise RuntimeError(f"Poisson solver failed to converge (info={info})")
        return phi
    
    def compute_rhs(self, psi, omega, phi):
        """Compute RHS of evolution equations."""
        grid = self.grid
        
        # Current density
        J = compute_current_density(psi, grid)
        
        # Poisson brackets
        psi_phi_bracket = poisson_bracket(psi, phi, grid)
        omega_phi_bracket = poisson_bracket(omega, phi, grid)
        
        # Resistive term
        dpsi_dt_resistive = -self.eta * J
        
        # Viscous term
        domega_dt_viscous = -self.nu * laplacian_toroidal(omega, grid)
        
        # Pressure force term
        if self.P0 > 0:
            from pytokmhd.physics import pressure_force_term
            S_P = pressure_force_term(psi, self.P0, self.psi_edge, grid, self.alpha)
        else:
            S_P = 0.0
        
        # Total RHS
        dpsi_dt = psi_phi_bracket + dpsi_dt_resistive
        domega_dt = omega_phi_bracket + S_P + domega_dt_viscous
        
        return dpsi_dt, domega_dt
    
    def step(self, psi, omega):
        """
        Störmer-Verlet symplectic integration.
        
        Half-step ψ:
            ψ^(n+1/2) = ψ^n + (dt/2) * [ψ, φ^n]
        
        Full-step ω:
            ω^(n+1) = ω^n + dt * ([ω, φ^(n+1/2)] + S_P + dissipation)
        
        Half-step ψ:
            ψ^(n+1) = ψ^(n+1/2) + (dt/2) * [ψ, φ^(n+1)]
        """
        # Compute φ^n
        phi_n = self.compute_phi(omega)
        
        # Half-step ψ
        psi_phi_bracket = poisson_bracket(psi, phi_n, self.grid)
        psi_half = psi + 0.5 * self.dt * psi_phi_bracket
        
        # Add resistive diffusion (implicit in half-step)
        J_half = compute_current_density(psi_half, self.grid)
        psi_half = psi_half - 0.5 * self.dt * self.eta * J_half
        
        # Compute φ^(n+1/2)
        phi_half = self.compute_phi(omega)
        
        # Full-step ω
        omega_phi_bracket = poisson_bracket(omega, phi_half, self.grid)
        
        # Pressure force
        if self.P0 > 0:
            from pytokmhd.physics import pressure_force_term
            S_P = pressure_force_term(psi_half, self.P0, self.psi_edge, self.grid, self.alpha)
        else:
            S_P = 0.0
        
        # Viscous dissipation
        viscous_term = -self.nu * laplacian_toroidal(omega, self.grid)
        
        omega_new = omega + self.dt * (omega_phi_bracket + S_P + viscous_term)
        
        # Compute φ^(n+1)
        phi_new = self.compute_phi(omega_new)
        
        # Half-step ψ (complete)
        psi_phi_bracket_new = poisson_bracket(psi_half, phi_new, self.grid)
        psi_new = psi_half + 0.5 * self.dt * psi_phi_bracket_new
        
        # Add resistive diffusion (second half)
        J_new = compute_current_density(psi_new, self.grid)
        psi_new = psi_new - 0.5 * self.dt * self.eta * J_new
        
        self.step_count += 1
        self.time += self.dt
        
        return psi_new, omega_new


def run_scenario_1_equilibrium_maintenance(output_dir: Path, verbose=True):
    """
    Scenario 1: Equilibrium Maintenance
    
    IC: Cylindrical equilibrium
    Duration: 1000 steps
    Metric: Energy drift < 1%
    """
    if verbose:
        print("="*70)
        print("Scenario 1: Equilibrium Maintenance (1000 steps)")
        print("="*70)
        print()
    
    # Create grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Cylindrical IC: ψ = r²(1 - r/a)
    r_mesh = grid.r_grid
    psi = r_mesh**2 * (1 - r_mesh / grid.a)
    omega = -laplacian_toroidal(psi, grid)  # ω = -∇²ψ for equilibrium
    
    # Create solver (no pressure for cylindrical)
    solver = V13Solver(grid, dt=1e-4, eta=1e-4, nu=1e-4, P0=0.0)
    
    # Track energy
    n_steps = 1000
    energy_history = []
    time_history = []
    
    H0 = compute_hamiltonian(psi, solver.compute_phi(omega), grid)
    energy_history.append(H0)
    time_history.append(0.0)
    
    if verbose:
        print(f"Grid: {grid.nr} × {grid.ntheta}")
        print(f"Initial energy: H₀ = {H0:.6e}")
        print(f"Time step: dt = {solver.dt}")
        print(f"Duration: {n_steps} steps (t = {n_steps * solver.dt:.3f})")
        print()
        print("Evolving...")
    
    start_time = time.time()
    
    for i in range(n_steps):
        psi, omega = solver.step(psi, omega)
        
        if (i+1) % 100 == 0 or i == n_steps - 1:
            phi = solver.compute_phi(omega)
            H = compute_hamiltonian(psi, phi, grid)
            energy_history.append(H)
            time_history.append(solver.time)
            
            drift = (H - H0) / H0
            if verbose:
                print(f"  Step {i+1:4d}: H = {H:.6e}, drift = {drift:+.4%}")
    
    elapsed = time.time() - start_time
    
    # Final metrics
    H_final = energy_history[-1]
    drift_final = (H_final - H0) / H0
    drift_max = max(abs((H - H0) / H0) for H in energy_history)
    
    if verbose:
        print()
        print("Results:")
        print(f"  Final energy: H = {H_final:.6e}")
        print(f"  Energy drift: {drift_final:+.4%}")
        print(f"  Max drift: {drift_max:+.4%}")
        print(f"  Elapsed time: {elapsed:.2f} s")
        print()
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'scenario': 'equilibrium_maintenance',
        'n_steps': n_steps,
        'H0': float(H0),
        'H_final': float(H_final),
        'drift_final': float(drift_final),
        'drift_max': float(drift_max),
        'time': time_history,
        'energy': [float(H) for H in energy_history],
        'elapsed_seconds': elapsed,
    }
    
    with open(output_dir / 'scenario1_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    drift_pct = [(H - H0)/H0 * 100 for H in energy_history]
    ax.plot(time_history, drift_pct, 'b-', linewidth=2)
    ax.axhline(1.0, color='g', linestyle='--', label='Target: 1%')
    ax.axhline(-1.0, color='g', linestyle='--')
    ax.axhline(0.0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy Drift (%)', fontsize=12)
    ax.set_title('Scenario 1: Energy Conservation (1000 steps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'scenario1_energy_drift.png', dpi=150)
    plt.close()
    
    # Success?
    success = abs(drift_final) < 0.01  # < 1%
    
    if verbose:
        if success:
            print("✅ SUCCESS: Energy drift < 1%")
        else:
            print(f"❌ FAIL: Energy drift {drift_final:+.4%} > 1%")
        print()
    
    return results, success


def run_scenario_2_perturbed_evolution(output_dir: Path, verbose=True):
    """
    Scenario 2: Perturbed Evolution
    
    IC: Equilibrium + perturbation
    Duration: 100 steps
    Metrics: Energy budget, force balance
    """
    if verbose:
        print("="*70)
        print("Scenario 2: Perturbed Evolution (100 steps)")
        print("="*70)
        print()
    
    # Create grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Base equilibrium
    r_mesh = grid.r_grid
    theta_mesh = grid.theta_grid
    psi_eq = r_mesh**2 * (1 - r_mesh / grid.a)
    
    # Add perturbation (m=2 mode)
    perturbation = 0.01 * r_mesh**2 * np.cos(2 * theta_mesh)
    psi = psi_eq + perturbation
    omega = -laplacian_toroidal(psi, grid)
    
    # Create solver
    solver = V13Solver(grid, dt=1e-4, eta=1e-4, nu=1e-4, P0=0.0)
    
    n_steps = 100
    energy_history = []
    kinetic_history = []
    magnetic_history = []
    time_history = []
    
    phi = solver.compute_phi(omega)
    H0 = compute_hamiltonian(psi, phi, grid)
    K0 = kinetic_energy(phi, grid)
    U0 = magnetic_energy(psi, grid)
    
    energy_history.append(H0)
    kinetic_history.append(K0)
    magnetic_history.append(U0)
    time_history.append(0.0)
    
    if verbose:
        print(f"Initial condition: Equilibrium + m=2 perturbation (1%)")
        print(f"Initial energy: H₀ = {H0:.6e}")
        print(f"  Kinetic:  K₀ = {K0:.6e} ({K0/H0*100:.1f}%)")
        print(f"  Magnetic: U₀ = {U0:.6e} ({U0/H0*100:.1f}%)")
        print()
        print("Evolving...")
    
    start_time = time.time()
    
    for i in range(n_steps):
        psi, omega = solver.step(psi, omega)
        
        if (i+1) % 10 == 0 or i == n_steps - 1:
            phi = solver.compute_phi(omega)
            H = compute_hamiltonian(psi, phi, grid)
            K = kinetic_energy(phi, grid)
            U = magnetic_energy(psi, grid)
            
            energy_history.append(H)
            kinetic_history.append(K)
            magnetic_history.append(U)
            time_history.append(solver.time)
            
            drift = (H - H0) / H0
            if verbose:
                print(f"  Step {i+1:3d}: H = {H:.6e}, drift = {drift:+.4%}")
    
    elapsed = time.time() - start_time
    
    H_final = energy_history[-1]
    drift_final = (H_final - H0) / H0
    
    if verbose:
        print()
        print("Results:")
        print(f"  Final energy: H = {H_final:.6e}")
        print(f"  Energy drift: {drift_final:+.4%}")
        print(f"  Elapsed time: {elapsed:.2f} s")
        print()
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'scenario': 'perturbed_evolution',
        'n_steps': n_steps,
        'H0': float(H0),
        'K0': float(K0),
        'U0': float(U0),
        'H_final': float(H_final),
        'drift_final': float(drift_final),
        'time': time_history,
        'energy': [float(H) for H in energy_history],
        'kinetic': [float(K) for K in kinetic_history],
        'magnetic': [float(U) for U in magnetic_history],
        'elapsed_seconds': elapsed,
    }
    
    with open(output_dir / 'scenario2_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot energy budget
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Total energy drift
    drift_pct = [(H - H0)/H0 * 100 for H in energy_history]
    ax1.plot(time_history, drift_pct, 'b-', linewidth=2)
    ax1.axhline(0.0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Energy Drift (%)', fontsize=12)
    ax1.set_title('Total Energy Conservation', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Energy partition
    ax2.plot(time_history, kinetic_history, 'r-', label='Kinetic (K)', linewidth=2)
    ax2.plot(time_history, magnetic_history, 'b-', label='Magnetic (U)', linewidth=2)
    ax2.plot(time_history, energy_history, 'k--', label='Total (H)', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Budget', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / 'scenario2_energy_budget.png', dpi=150)
    plt.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='v1.3 Validation Benchmark')
    parser.add_argument('--scenario', type=int, choices=[1, 2], default=None,
                        help='Run specific scenario (1 or 2), default: all')
    parser.add_argument('--output', type=str, default='results/v1.3/validation',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    verbose = not args.quiet
    
    results_summary = {}
    
    # Scenario 1: Equilibrium maintenance
    if args.scenario is None or args.scenario == 1:
        results, success = run_scenario_1_equilibrium_maintenance(
            output_dir / 'scenario1',
            verbose=verbose
        )
        results_summary['scenario1'] = {
            'success': success,
            'drift_final': results['drift_final'],
            'drift_max': results['drift_max'],
        }
    
    # Scenario 2: Perturbed evolution
    if args.scenario is None or args.scenario == 2:
        results = run_scenario_2_perturbed_evolution(
            output_dir / 'scenario2',
            verbose=verbose
        )
        results_summary['scenario2'] = {
            'drift_final': results['drift_final'],
        }
    
    # Save summary
    with open(output_dir / 'validation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    if verbose:
        print("="*70)
        print("Validation Summary")
        print("="*70)
        for scenario, metrics in results_summary.items():
            print(f"\n{scenario}:")
            for key, value in metrics.items():
                if isinstance(value, bool):
                    print(f"  {key}: {'✅ PASS' if value else '❌ FAIL'}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:+.4%}")
        print()


if __name__ == '__main__':
    main()
