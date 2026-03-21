"""
Test exact Poisson solver vs cylindrical approximation

Compares:
1. poisson_sparse_exact (exact toroidal stencil)
2. poisson_hybrid (cylindrical approximation + refinement)

Metrics:
- Residual max|∇²φ - ω|
- Solution accuracy
- Computational cost

Author: 小P ⚛️
Created: 2026-03-18
"""

import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

import numpy as np
import time
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators.toroidal_operators import laplacian_toroidal
from pytokmhd.integrators.poisson_sparse_exact import solve_poisson_exact, build_laplacian_matrix
from pytokmhd.integrators.poisson_hybrid import solve_poisson_hybrid


def test_comparison(nr=32, ntheta=64):
    """Compare exact vs hybrid solver."""
    print("=" * 70)
    print("Poisson Solver Comparison: Exact vs Hybrid")
    print("=" * 70)
    
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=nr, ntheta=ntheta)
    print(f"\nGrid: {nr} x {ntheta} = {nr*ntheta} points")
    
    # Test case: ω = r²·sin(2θ)
    omega = grid.r_grid**2 * np.sin(2 * grid.theta_grid)
    print(f"Test case: ω = r²·sin(2θ)")
    print(f"Source norm: {np.linalg.norm(omega):.6e}")
    
    # -------------------------------------------------------------------
    # Exact solver
    # -------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. EXACT SOLVER (Phase 1 toroidal stencil)")
    print("-" * 70)
    
    # Build matrix (one-time cost)
    t0 = time.time()
    L_matrix = build_laplacian_matrix(grid)
    t_build = time.time() - t0
    print(f"Matrix build time: {t_build:.3f} s")
    
    # Solve
    t0 = time.time()
    phi_exact = solve_poisson_exact(omega, grid, L_matrix=L_matrix)
    t_solve_exact = time.time() - t0
    
    # Verify
    residual_exact = laplacian_toroidal(phi_exact, grid) - omega
    max_res_exact = np.max(np.abs(residual_exact))
    rms_res_exact = np.sqrt(np.mean(residual_exact**2))
    
    print(f"Solve time: {t_solve_exact:.3f} s")
    print(f"max|∇²φ - ω| = {max_res_exact:.6e}")
    print(f"RMS residual  = {rms_res_exact:.6e}")
    
    # -------------------------------------------------------------------
    # Hybrid solver
    # -------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. HYBRID SOLVER (cylindrical + refinement)")
    print("-" * 70)
    
    t0 = time.time()
    phi_hybrid = solve_poisson_hybrid(omega, grid)
    t_solve_hybrid = time.time() - t0
    
    residual_hybrid = laplacian_toroidal(phi_hybrid, grid) - omega
    max_res_hybrid = np.max(np.abs(residual_hybrid))
    rms_res_hybrid = np.sqrt(np.mean(residual_hybrid**2))
    
    print(f"Solve time: {t_solve_hybrid:.3f} s")
    print(f"max|∇²φ - ω| = {max_res_hybrid:.6e}")
    print(f"RMS residual  = {rms_res_hybrid:.6e}")
    
    # -------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\nAccuracy improvement:")
    print(f"  max residual: {max_res_hybrid:.2e} → {max_res_exact:.2e}  ({max_res_hybrid/max_res_exact:.1f}x better)")
    print(f"  RMS residual: {rms_res_hybrid:.2e} → {rms_res_exact:.2e}  ({rms_res_hybrid/rms_res_exact:.1f}x better)")
    
    print(f"\nSpeed (amortized cost, excluding matrix build):")
    print(f"  Hybrid: {t_solve_hybrid*1000:.2f} ms/solve")
    print(f"  Exact:  {t_solve_exact*1000:.2f} ms/solve  ({t_solve_exact/t_solve_hybrid:.2f}x)")
    
    print(f"\nOne-time setup:")
    print(f"  Matrix build: {t_build:.2f} s  (amortized over {int(t_build/t_solve_exact)} solves)")
    
    # Solution difference
    diff = phi_exact - phi_hybrid
    diff_norm = diff - np.mean(diff)  # Remove constant
    print(f"\nSolution difference:")
    print(f"  max|φ_exact - φ_hybrid| = {np.max(np.abs(diff_norm)):.6e}")
    
    # Pass/fail criteria
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    exact_pass = max_res_exact < 1e-9
    hybrid_pass = max_res_hybrid < 1e-3
    
    print(f"Exact solver  (< 1e-9):  {'✅ PASS' if exact_pass else '❌ FAIL'}")
    print(f"Hybrid solver (< 1e-3):  {'✅ PASS' if hybrid_pass else '❌ FAIL'}")
    
    return {
        'max_res_exact': max_res_exact,
        'max_res_hybrid': max_res_hybrid,
        't_solve_exact': t_solve_exact,
        't_solve_hybrid': t_solve_hybrid,
        't_build': t_build,
    }


if __name__ == "__main__":
    results = test_comparison(nr=32, ntheta=64)
    
    # Exit code
    success = results['max_res_exact'] < 1e-9
    sys.exit(0 if success else 1)
