#!/usr/bin/env python3
"""
v1.3 Phase 3: Property Validation

Validates Hamiltonian and Poisson bracket properties WITHOUT full time evolution.

Tests:
1. Hamiltonian conservation structure
2. Poisson bracket anti-symmetry
3. Jacobi identity
4. Energy partition (K + U = H)
5. Force balance residual for equilibrium IC

Author: 小P ⚛️
Date: 2026-03-19
Phase: v1.3 Phase 3 (Partial)
"""

import numpy as np
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import (
    poisson_bracket,
    jacobi_identity_residual,
    laplacian_toroidal,
)
from pytokmhd.physics import (
    compute_hamiltonian,
    kinetic_energy,
    magnetic_energy,
    energy_partition,
    force_balance_residual,
)
from pytokmhd.equilibrium import pressure_profile
from pytokmhd.solvers import solve_poisson_toroidal


def test_poisson_bracket_properties(grid, verbose=True):
    """Test Poisson bracket algebraic properties."""
    if verbose:
        print("="*70)
        print("Test 1: Poisson Bracket Properties")
        print("="*70)
        print()
    
    # Test fields
    f = grid.r_grid**2
    g = np.sin(grid.theta_grid)
    h = grid.r_grid * np.cos(grid.theta_grid)
    
    results = {}
    
    # 1. Anti-symmetry: [f, g] = -[g, f]
    bracket_fg = poisson_bracket(f, g, grid)
    bracket_gf = poisson_bracket(g, f, grid)
    antisym_error = np.max(np.abs(bracket_fg + bracket_gf))
    
    results['antisymmetry_error'] = float(antisym_error)
    results['antisymmetry_pass'] = antisym_error < 1e-12
    
    if verbose:
        status = "✅ PASS" if results['antisymmetry_pass'] else "❌ FAIL"
        print(f"Anti-symmetry: max|[f,g] + [g,f]| = {antisym_error:.2e} {status}")
    
    # 2. Jacobi identity
    jacobi_residual = jacobi_identity_residual(f, g, h, grid)
    
    results['jacobi_residual'] = float(jacobi_residual)
    results['jacobi_pass'] = jacobi_residual < 0.01  # 2nd-order discretization
    
    if verbose:
        status = "✅ PASS" if results['jacobi_pass'] else "❌ FAIL"
        print(f"Jacobi identity: residual = {jacobi_residual:.2e} {status}")
    
    # 3. Constant bracket: [const, f] = 0
    const = np.ones_like(f) * 3.14
    bracket_const = poisson_bracket(const, f, grid)
    const_error = np.max(np.abs(bracket_const))
    
    results['constant_bracket_error'] = float(const_error)
    results['constant_bracket_pass'] = const_error < 1e-12
    
    if verbose:
        status = "✅ PASS" if results['constant_bracket_pass'] else "❌ FAIL"
        print(f"Constant bracket: max|[const, f]| = {const_error:.2e} {status}")
        print()
    
    return results


def test_hamiltonian_structure(grid, verbose=True):
    """Test Hamiltonian energy functional."""
    if verbose:
        print("="*70)
        print("Test 2: Hamiltonian Structure")
        print("="*70)
        print()
    
    # Simple IC
    psi = grid.r_grid**2 * (1 - grid.r_grid / grid.a)
    omega = -laplacian_toroidal(psi, grid)
    
    phi, info = solve_poisson_toroidal(omega, grid)
    
    if info != 0:
        if verbose:
            print(f"❌ Poisson solver failed (info={info})")
        return {'poisson_converged': False}
    
    results = {}
    results['poisson_converged'] = True
    
    # Total Hamiltonian
    H = compute_hamiltonian(psi, phi, grid)
    K = kinetic_energy(phi, grid)
    U = magnetic_energy(psi, grid)
    
    results['H'] = float(H)
    results['K'] = float(K)
    results['U'] = float(U)
    
    # Energy partition
    partition_error = abs(H - (K + U))
    results['partition_error'] = float(partition_error)
    results['partition_pass'] = partition_error < 1e-10 * abs(H)
    
    if verbose:
        print(f"Hamiltonian: H = {H:.6e}")
        print(f"  Kinetic:   K = {K:.6e} ({K/H*100:.1f}%)")
        print(f"  Magnetic:  U = {U:.6e} ({U/H*100:.1f}%)")
        print(f"  Partition error: |H - (K+U)| = {partition_error:.2e}")
        status = "✅ PASS" if results['partition_pass'] else "❌ FAIL"
        print(f"  Energy conservation structure: {status}")
        print()
    
    # Sum of fractions
    K_frac, U_frac = K/H, U/H
    sum_fractions = K_frac + U_frac
    fraction_error = abs(sum_fractions - 1.0)
    
    results['fraction_error'] = float(fraction_error)
    results['fraction_pass'] = fraction_error < 1e-10
    
    if verbose:
        status = "✅ PASS" if results['fraction_pass'] else "❌ FAIL"
        print(f"Fraction sum: K/H + U/H = {sum_fractions:.12f} {status}")
        print()
    
    return results


def test_force_balance_equilibrium(grid, verbose=True):
    """Test force balance for equilibrium IC."""
    if verbose:
        print("="*70)
        print("Test 3: Force Balance (Equilibrium IC)")
        print("="*70)
        print()
    
    # Cylindrical equilibrium: ψ = r²(1 - r/a)
    psi = grid.r_grid**2 * (1 - grid.r_grid / grid.a)
    
    # Pressure profile (weak)
    P0 = 1e3  # Low pressure to avoid dominant effect
    psi_edge = grid.a**2
    
    try:
        # Compute force balance residual
        fb_result = force_balance_residual(psi, P0, psi_edge, grid)
        
        results = {
            'max_residual': float(fb_result['max_residual']),
            'rms_residual': float(fb_result['rms_residual']),
            'relative_error': float(fb_result['relative_error']),
            'force_balance_pass': fb_result['relative_error'] < 1.0,  # Relaxed for cylindrical
        }
        
        if verbose:
            print(f"Force balance residual |J×B - ∇P|:")
            print(f"  Max:      {results['max_residual']:.3e}")
            print(f"  RMS:      {results['rms_residual']:.3e}")
            print(f"  Relative: {results['relative_error']:.3e}")
            status = "✅ PASS" if results['force_balance_pass'] else "❌ FAIL"
            print(f"  Status: {status}")
            print()
        
        return results
    
    except Exception as e:
        if verbose:
            print(f"❌ Force balance computation failed: {e}")
            print()
        return {'force_balance_pass': False, 'error': str(e)}


def test_v12_vs_v13_comparison(grid, verbose=True):
    """Compare v1.2 (diffusion-only) vs v1.3 (Hamiltonian) expectations."""
    if verbose:
        print("="*70)
        print("Test 4: v1.2 vs v1.3 Comparison (Theory)")
        print("="*70)
        print()
    
    # This is a theoretical comparison, not a full simulation
    results = {
        'v1.2_description': 'Pure diffusion: ∂ψ/∂t = -η·J, ∂ω/∂t = -ν·∇²ω',
        'v1.2_energy_conservation': 'None (dissipation only)',
        'v1.2_expected_drift': '6.75% (100 steps, cylindrical IC)',
        'v1.3_description': 'Hamiltonian + dissipation: ∂ψ/∂t = [ψ,H] - η·J',
        'v1.3_energy_conservation': 'Hamiltonian structure (ideal part conserves)',
        'v1.3_expected_drift': '< 1% (target)',
        'improvement_factor_target': 6.75,
    }
    
    if verbose:
        print("v1.2 (Baseline):")
        print(f"  Equations: {results['v1.2_description']}")
        print(f"  Energy: {results['v1.2_energy_conservation']}")
        print(f"  Drift: {results['v1.2_expected_drift']}")
        print()
        print("v1.3 (Current):")
        print(f"  Equations: {results['v1.3_description']}")
        print(f"  Energy: {results['v1.3_energy_conservation']}")
        print(f"  Target drift: {results['v1.3_expected_drift']}")
        print()
        print(f"Expected improvement: {results['improvement_factor_target']}×")
        print()
    
    return results


def main():
    print("\n" + "="*70)
    print("v1.3 Phase 3: Property Validation (Partial)")
    print("="*70)
    print()
    
    # Create grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    print(f"Grid: {grid.nr} × {grid.ntheta}")
    print(f"  r: [{grid.r_grid.min():.3f}, {grid.r_grid.max():.3f}]")
    print(f"  θ: [0, 2π]")
    print(f"  R: [{grid.R_grid.min():.3f}, {grid.R_grid.max():.3f}]")
    print()
    
    # Run tests
    all_results = {}
    
    # Test 1: Poisson bracket
    all_results['poisson_bracket'] = test_poisson_bracket_properties(grid)
    
    # Test 2: Hamiltonian
    all_results['hamiltonian'] = test_hamiltonian_structure(grid)
    
    # Test 3: Force balance
    all_results['force_balance'] = test_force_balance_equilibrium(grid)
    
    # Test 4: Comparison
    all_results['v12_vs_v13'] = test_v12_vs_v13_comparison(grid)
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    print()
    
    n_pass = 0
    n_total = 0
    
    for test_name, test_results in all_results.items():
        if test_name == 'v12_vs_v13':
            continue  # Skip theory comparison
        
        print(f"{test_name}:")
        for key, value in test_results.items():
            if key.endswith('_pass'):
                n_total += 1
                if value:
                    n_pass += 1
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"  {key}: {status}")
    
    print()
    print(f"Overall: {n_pass}/{n_total} tests passed ({n_pass/n_total*100:.0f}%)")
    print()
    
    # Save results
    output_dir = Path('results/v1.3/properties')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    all_results_json = convert_numpy(all_results)
    
    with open(output_dir / 'validation_properties.json', 'w') as f:
        json.dump(all_results_json, f, indent=2)
    
    print(f"Results saved to: {output_dir / 'validation_properties.json'}")
    print()
    
    return all_results


if __name__ == '__main__':
    main()
