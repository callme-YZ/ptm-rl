"""
Stage 2 Verification: Task 2.2 - Poisson Bracket Properties

Tests Morrison bracket implementation satisfies:
1. Antisymmetry: {F, G} = -{G, F}
2. Jacobi identity: {{F,G},H} + cyclic = 0
3. Leibniz rule: {F, GH} = {F,G}H + G{F,H}

From Stage 1 theory: These properties guarantee Hamiltonian structure.

Author: 小P ⚛️
For: 小A 🤖 to execute
Date: 2026-03-23
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from pytokmhd.grid import ToroidalGrid
from pytokmhd.operators.poisson_bracket import poisson_bracket


def test_antisymmetry():
    """
    Test Property 1: {F, G} = -{G, F}
    
    Use test functions:
    - F = sin(r) * cos(θ)
    - G = r² + θ
    """
    print("\n" + "="*60)
    print("TEST 1: Antisymmetry {F,G} = -{G,F}")
    print("="*60)
    
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    r = grid.R[:, None]
    theta = grid.theta[None, :]
    
    # Test functions
    F = np.sin(r) * np.cos(theta)
    G = r**2 + theta
    
    # Compute brackets
    FG = poisson_bracket(F, G, grid)
    GF = poisson_bracket(G, F, grid)
    
    # Check antisymmetry
    diff = FG + GF
    max_error = np.max(np.abs(diff))
    rms_error = np.sqrt(np.mean(diff**2))
    
    print(f"max|{{F,G}} + {{G,F}}|: {max_error:.2e}")
    print(f"RMS error:              {rms_error:.2e}")
    
    threshold = 1e-12
    if max_error < threshold:
        print(f"\n✅ PASS: Antisymmetry verified (error < {threshold:.0e})")
        return True
    else:
        print(f"\n❌ FAIL: Antisymmetry violated (error {max_error:.2e})")
        return False


def test_jacobi_identity():
    """
    Test Property 2: {{F,G},H} + {{G,H},F} + {{H,F},G} = 0
    
    Use test functions:
    - F = r * sin(θ)
    - G = r² * cos(θ)
    - H = exp(-r) * sin(2θ)
    """
    print("\n" + "="*60)
    print("TEST 2: Jacobi Identity")
    print("="*60)
    
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    r = grid.R[:, None]
    theta = grid.theta[None, :]
    
    # Test functions
    F = r * np.sin(theta)
    G = r**2 * np.cos(theta)
    H = np.exp(-r) * np.sin(2*theta)
    
    # Compute nested brackets
    FG = poisson_bracket(F, G, grid)
    GH = poisson_bracket(G, H, grid)
    HF = poisson_bracket(H, F, grid)
    
    FG_H = poisson_bracket(FG, H, grid)
    GH_F = poisson_bracket(GH, F, grid)
    HF_G = poisson_bracket(HF, G, grid)
    
    # Jacobi sum
    jacobi_sum = FG_H + GH_F + HF_G
    
    max_error = np.max(np.abs(jacobi_sum))
    rms_error = np.sqrt(np.mean(jacobi_sum**2))
    
    print(f"max|{{{{F,G}},H}} + cyclic|: {max_error:.2e}")
    print(f"RMS error:                   {rms_error:.2e}")
    
    threshold = 1e-10  # Slightly relaxed (nested operations accumulate error)
    if max_error < threshold:
        print(f"\n✅ PASS: Jacobi identity verified (error < {threshold:.0e})")
        return True
    else:
        print(f"\n❌ FAIL: Jacobi identity violated (error {max_error:.2e})")
        return False


def test_leibniz_rule():
    """
    Test Property 3: {F, GH} = {F,G}H + G{F,H}
    
    Use test functions:
    - F = sin(r*θ)
    - G = r
    - H = cos(θ)
    """
    print("\n" + "="*60)
    print("TEST 3: Leibniz Rule {F,GH} = {F,G}H + G{F,H}")
    print("="*60)
    
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    
    r = grid.R[:, None]
    theta = grid.theta[None, :]
    
    # Test functions
    F = np.sin(r * theta)
    G = r
    H = np.cos(theta)
    
    GH = G * H
    
    # LHS: {F, GH}
    lhs = poisson_bracket(F, GH, grid)
    
    # RHS: {F,G}H + G{F,H}
    FG = poisson_bracket(F, G, grid)
    FH = poisson_bracket(F, H, grid)
    rhs = FG * H + G * FH
    
    diff = lhs - rhs
    max_error = np.max(np.abs(diff))
    rms_error = np.sqrt(np.mean(diff**2))
    
    print(f"max|{{F,GH}} - {{F,G}}H - G{{F,H}}|: {max_error:.2e}")
    print(f"RMS error:                           {rms_error:.2e}")
    
    threshold = 1e-12
    if max_error < threshold:
        print(f"\n✅ PASS: Leibniz rule verified (error < {threshold:.0e})")
        return True
    else:
        print(f"\n❌ FAIL: Leibniz rule violated (error {max_error:.2e})")
        return False


def test_energy_bracket():
    """
    Test Property 4: {H, H} = 0 (from Stage 1 proof)
    
    This is a consequence of antisymmetry, but important to verify
    for the actual Hamiltonian.
    """
    print("\n" + "="*60)
    print("TEST 4: Energy Self-Bracket {H,H} = 0")
    print("="*60)
    
    from pytokmhd.solvers.hamiltonian_mhd import HamiltonianMHD
    from pytokmhd.physics.initial_conditions import ballooning_ic
    
    grid = ToroidalGrid(nr=32, ntheta=32, r_min=0.1, r_max=1.0)
    solver = HamiltonianMHD(grid=grid, dt=1e-3, eta=0.0, nu=0.0)
    
    psi, omega = ballooning_ic(grid, beta=0.17, q_axis=1.2, shear=0.5)
    
    # Compute Hamiltonian (energy)
    phi = solver.poisson_solver.solve(omega)
    
    grad_phi_r, grad_phi_theta = solver.gradient(phi)
    grad_psi_r, grad_psi_theta = solver.gradient(psi)
    
    grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.R[:, None])**2
    grad_psi_sq = grad_psi_r**2 + (grad_psi_theta / grid.R[:, None])**2
    
    H = 0.5 * (grad_phi_sq + grad_psi_sq)
    
    # Compute {H, H}
    HH = poisson_bracket(H, H, grid)
    
    max_error = np.max(np.abs(HH))
    rms_error = np.sqrt(np.mean(HH**2))
    
    print(f"max|{{H,H}}|: {max_error:.2e}")
    print(f"RMS error:    {rms_error:.2e}")
    
    threshold = 1e-12
    if max_error < threshold:
        print(f"\n✅ PASS: {{{H,H}} = 0 verified (Stage 1 prediction)")
        return True
    else:
        print(f"\n❌ FAIL: {{H,H}} should be zero (error {max_error:.2e})")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Stage 2 Verification: Poisson Bracket Properties")
    print(" Issue #23 - Task 2.2")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['antisymmetry'] = test_antisymmetry()
    results['jacobi'] = test_jacobi_identity()
    results['leibniz'] = test_leibniz_rule()
    results['energy'] = test_energy_bracket()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Antisymmetry {{F,G}}=-{{G,F}}: {'✅ PASS' if results['antisymmetry'] else '❌ FAIL'}")
    print(f"Jacobi identity:              {'✅ PASS' if results['jacobi'] else '❌ FAIL'}")
    print(f"Leibniz rule:                 {'✅ PASS' if results['leibniz'] else '❌ FAIL'}")
    print(f"Energy bracket {{H,H}}=0:      {'✅ PASS' if results['energy'] else '❌ FAIL'}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n🎉 All tests PASSED - Poisson bracket structure verified! ✅")
        print("\nNext: Task 2.3 - Integrator comparison")
    else:
        print("\n⚠️  Some tests FAILED - Morrison bracket implementation may have issues")
    
    print("="*70)
