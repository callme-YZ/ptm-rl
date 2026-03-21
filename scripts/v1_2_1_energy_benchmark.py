#!/usr/bin/env python3
"""
v1.2.1 Energy Drift Benchmark

Runs Config A (cylindrical) vs Config B (PyTokEq Solovev)
to verify PyTokEq equilibrium improves energy conservation.

Author: 小P ⚛️
Date: 2026-03-18
Phase: v1.2.1 Phase 2
"""

import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.toroidal_mhd import ToroidalMHDSolver
from pytokmhd.solver.initial_conditions import pytokeq_initial
from pytokmhd.solver.equilibrium_cache import EquilibriumCache
from pytokmhd.diagnostics.energy_conservation import (
    compute_total_energy,
    track_energy_drift,
    plot_energy_evolution
)
from pytokmhd.diagnostics.ic_validation import validate_pytokeq_ic


def run_config_a_cylindrical(
    output_dir: Path,
    n_steps: int = 1000,
    verbose: bool = True
):
    """
    Configuration A: Cylindrical Profile Baseline
    
    Simple analytical IC: ψ = r²(1 - r/a)
    Known baseline: ~15% energy drift
    """
    if verbose:
        print("="*60)
        print("Configuration A: Cylindrical Baseline")
        print("="*60)
        print()
    
    # Create grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Cylindrical IC
    r = grid.r_grid[:, 0]
    theta = grid.theta_grid[0, :]
    
    R_mesh = grid.R_grid
    Z_mesh = grid.Z_grid
    r_mesh = np.sqrt((R_mesh - 1.0)**2 + Z_mesh**2)
    
    # ψ = r²(1 - r/a)
    psi = r_mesh**2 * (1 - r_mesh / 0.3)
    omega = np.zeros_like(psi)
    
    if verbose:
        print(f"Grid: {grid.nr} × {grid.ntheta}")
        print(f"IC: Cylindrical profile")
        E0, _, _ = compute_total_energy(psi, omega, grid)
        print(f"Initial energy: E0 = {E0:.6e}")
        print()
    
    # Create solver
    solver = ToroidalMHDSolver(
        grid=grid,
        dt=1e-4,
        eta=1e-4,
        nu=1e-4
    )
    
    # Initialize with IC
    solver.initialize(psi, omega)
    
    # Track energy
    tracker = track_energy_drift(
        solver,
        n_steps=n_steps,
        monitor_components_every=50,
        verbose=verbose
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary = tracker.get_summary()
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Evolution curve
    times, E_norm = tracker.get_evolution_curve()
    np.savez(
        output_dir / 'energy_evolution.npz',
        times=times,
        energies=E_norm
    )
    
    # Plot
    plot_energy_evolution(
        tracker,
        save_path=str(output_dir / 'plot_energy.png'),
        show=False
    )
    
    if verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - metrics.json")
        print(f"  - energy_evolution.npz")
        print(f"  - plot_energy.png")
    
    return tracker


def run_config_b_pytokeq(
    output_dir: Path,
    n_steps: int = 1000,
    verbose: bool = True
):
    """
    Configuration B: PyTokEq Solovev Equilibrium
    
    Analytical Solovev solution from PyTokEq
    Target: < 1% energy drift (Tier 1) or < 5% (Tier 2)
    """
    if verbose:
        print("="*60)
        print("Configuration B: PyTokEq Solovev")
        print("="*60)
        print()
    
    # Create grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    # Load PyTokEq equilibrium
    if verbose:
        print("Loading PyTokEq Solovev equilibrium...")
    
    from pytokeq.equilibrium.profiles.solovev_solution import SolovevSolution
    
    # Create Solovev equilibrium
    # Parameters: R0, eps (a/R0), kappa, delta, A (Shafranov shift)
    solovev = SolovevSolution(
        R0=1.0,
        eps=0.3,      # a/R0 = 0.3/1.0
        kappa=1.0,    # circular
        delta=0.0,    # no triangularity
        A=-0.1        # small Shafranov shift
    )
    
    # Evaluate on grid
    psi = solovev.psi(grid.R_grid, grid.Z_grid)
    omega = np.zeros_like(psi)
    
    if verbose:
        print(f"✓ Solovev equilibrium loaded")
        E0, _, _ = compute_total_energy(psi, omega, grid)
        print(f"Initial energy: E0 = {E0:.6e}")
    
    # Validate IC quality (Phase 1.5)
    if verbose:
        print("\n" + "="*60)
        print("Phase 1.5: IC Quality Validation")
        print("="*60)
    
    ic_result = validate_pytokeq_ic(psi, grid, verbose=verbose)
    
    if ic_result.status == "NO-GO":
        print("\n❌ IC validation failed! Aborting Config B.")
        return None
    elif ic_result.status == "WARN":
        print("\n⚠️ IC validation warnings, but proceeding...")
    
    print()
    
    # Create solver
    solver = ToroidalMHDSolver(
        grid=grid,
        dt=1e-4,
        eta=1e-4,
        nu=1e-4
    )
    
    # Initialize with IC
    solver.initialize(psi, omega)
    
    # Track energy
    tracker = track_energy_drift(
        solver,
        n_steps=n_steps,
        monitor_components_every=50,
        verbose=verbose
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary = tracker.get_summary()
    summary['ic_validation'] = {
        'energy_error': ic_result.energy_error,
        'status': ic_result.status
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Evolution curve
    times, E_norm = tracker.get_evolution_curve()
    np.savez(
        output_dir / 'energy_evolution.npz',
        times=times,
        energies=E_norm
    )
    
    # Plot
    plot_energy_evolution(
        tracker,
        save_path=str(output_dir / 'plot_energy.png'),
        show=False
    )
    
    if verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - metrics.json")
        print(f"  - energy_evolution.npz")
        print(f"  - plot_energy.png")
    
    return tracker


def compare_results(
    tracker_A,
    tracker_B,
    output_dir: Path,
    verbose: bool = True
):
    """
    Compare Config A vs Config B
    
    Generate comparison report and plots
    """
    if verbose:
        print("\n" + "="*60)
        print("Comparison: Config A vs Config B")
        print("="*60)
        print()
    
    summary_A = tracker_A.get_summary()
    summary_B = tracker_B.get_summary() if tracker_B else None
    
    if summary_B is None:
        print("⚠️ Config B failed, cannot compare")
        return
    
    # Compute improvement
    drift_A = summary_A['drift_final']
    drift_B = summary_B['drift_final']
    
    improvement = (drift_A - drift_B) / drift_A * 100
    
    # Report
    report = {
        'config_A': {
            'name': 'Cylindrical Baseline',
            'drift_final': drift_A,
            'tier': summary_A['tier'],
            'status': summary_A['status']
        },
        'config_B': {
            'name': 'PyTokEq Solovev',
            'drift_final': drift_B,
            'tier': summary_B['tier'],
            'status': summary_B['status']
        },
        'comparison': {
            'improvement_percent': improvement,
            'absolute_reduction': drift_A - drift_B
        }
    }
    
    # Verdict
    if drift_B < 0.01:
        verdict = "✅ Tier 1 SUCCESS: < 1% drift achieved!"
    elif drift_B < 0.05:
        verdict = "✅ Tier 2 ACCEPTABLE: < 5% drift, improvement verified"
    else:
        verdict = "⚠️ Tier 3: > 5% drift, needs investigation"
    
    report['verdict'] = verdict
    
    # Save
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print
    if verbose:
        print(f"Config A (Cylindrical): drift = {drift_A:.2%} (Tier {summary_A['tier']})")
        print(f"Config B (PyTokEq):     drift = {drift_B:.2%} (Tier {summary_B['tier']})")
        print(f"\nImprovement: {improvement:.1f}%")
        print(f"Absolute reduction: {(drift_A - drift_B):.2%}")
        print(f"\n{verdict}")
        print(f"\nComparison saved to: {output_dir / 'comparison_report.json'}")


def main():
    parser = argparse.ArgumentParser(description='v1.2.1 Energy Drift Benchmark')
    parser.add_argument('--config', type=str, choices=['cylindrical', 'solovev', 'compare', 'all'],
                       default='all', help='Which configuration to run')
    parser.add_argument('--n-steps', type=int, default=1000, help='Number of evolution steps')
    parser.add_argument('--output-dir', type=str, default='results/v1.2.1/energy_benchmark',
                       help='Output directory')
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    
    # Run benchmarks
    if args.config in ['cylindrical', 'all']:
        print("\n" + "="*60)
        print("Running Configuration A...")
        print("="*60)
        tracker_A = run_config_a_cylindrical(
            base_dir / 'config_A_cylindrical',
            n_steps=args.n_steps,
            verbose=True
        )
    else:
        tracker_A = None
    
    if args.config in ['solovev', 'all']:
        print("\n" + "="*60)
        print("Running Configuration B...")
        print("="*60)
        tracker_B = run_config_b_pytokeq(
            base_dir / 'config_B_solovev',
            n_steps=args.n_steps,
            verbose=True
        )
    else:
        tracker_B = None
    
    # Compare
    if args.config in ['compare', 'all'] and tracker_A and tracker_B:
        compare_results(tracker_A, tracker_B, base_dir, verbose=True)
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
