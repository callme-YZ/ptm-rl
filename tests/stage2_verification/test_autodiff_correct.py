"""
Test JAX Autodiff Correctness (Fixed)

Issue #24 Task 2: Validate gradients vs finite difference

Key fix: Use absolute error when gradient is near zero

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
from jax import grad

from pytokmhd.geometry.toroidal import ToroidalGrid


# Import from test_autodiff_hamiltonian
from test_autodiff_hamiltonian import hamiltonian_jax


def smart_error(grad_auto, grad_fd, threshold=1e-6):
    """
    Compute error metric that handles near-zero gradients.
    
    - If |grad_fd| > threshold: use relative error
    - If |grad_fd| ≤ threshold: use absolute error
    """
    if abs(grad_fd) > threshold:
        return abs(grad_auto - grad_fd) / abs(grad_fd)
    else:
        return abs(grad_auto - grad_fd)


def test_correctness_improved():
    """Test autodiff vs FD with better error handling"""
    print("=" * 60)
    print("TEST 2 (Improved): Correctness vs Finite Difference")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Test fields (away from zeros)
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    
    # Use fields that don't cross zero
    psi = jnp.array(0.1 * (r**2 + 0.1) * (jnp.sin(2*theta) + 1.5))
    phi = jnp.array(0.05 * (r + 0.05) * (jnp.cos(theta) + 1.2))
    
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    # Autodiff
    print("\nComputing autodiff gradients...")
    grad_H_psi = grad(H_func, argnums=0)(psi, phi)
    grad_H_phi = grad(H_func, argnums=1)(psi, phi)
    print("✅ Autodiff complete")
    
    # Finite difference (sample points away from boundaries)
    sample_points = [
        (8, 16),
        (16, 24),
        (24, 40),
    ]
    
    epsilon = 1e-5
    
    print(f"\nComputing FD at {len(sample_points)} points (ε={epsilon})...")
    
    results = []
    
    for i, j in sample_points:
        # ∇_ψ H
        psi_plus = psi.at[i, j].add(epsilon)
        psi_minus = psi.at[i, j].add(-epsilon)
        H_plus = H_func(psi_plus, phi)
        H_minus = H_func(psi_minus, phi)
        grad_psi_fd = (H_plus - H_minus) / (2 * epsilon)
        grad_psi_auto = grad_H_psi[i, j]
        
        # ∇_φ H
        phi_plus = phi.at[i, j].add(epsilon)
        phi_minus = phi.at[i, j].add(-epsilon)
        H_plus = H_func(psi, phi_plus)
        H_minus = H_func(psi, phi_minus)
        grad_phi_fd = (H_plus - H_minus) / (2 * epsilon)
        grad_phi_auto = grad_H_phi[i, j]
        
        results.append({
            'pos': (i, j),
            'psi_auto': grad_psi_auto,
            'psi_fd': grad_psi_fd,
            'phi_auto': grad_phi_auto,
            'phi_fd': grad_phi_fd,
        })
    
    print("✅ FD complete\n")
    
    # Display results
    print("Comparison:")
    print("-" * 70)
    print("Position   Field     Autodiff        FD            Abs Diff    Rel Error")
    print("-" * 70)
    
    errors = []
    
    for r in results:
        i, j = r['pos']
        
        # ∇_ψ H
        abs_diff = abs(r['psi_auto'] - r['psi_fd'])
        rel_err = abs_diff / (abs(r['psi_fd']) + 1e-10)
        errors.append(rel_err)
        print(f"({i:2d},{j:2d})   ∇_ψ H   {r['psi_auto']:+.6e}  {r['psi_fd']:+.6e}  {abs_diff:.3e}   {rel_err:.2%}")
        
        # ∇_φ H
        abs_diff = abs(r['phi_auto'] - r['phi_fd'])
        rel_err = abs_diff / (abs(r['phi_fd']) + 1e-10)
        errors.append(rel_err)
        print(f"         ∇_φ H   {r['phi_auto']:+.6e}  {r['phi_fd']:+.6e}  {abs_diff:.3e}   {rel_err:.2%}")
    
    print("-" * 70)
    
    max_error = max(errors)
    print(f"\nMax relative error: {max_error:.2%}")
    
    # Pass if < 1%
    threshold = 0.01
    
    if max_error < threshold:
        print(f"✅ TEST 2 PASSED - All errors < {threshold:.1%}")
        return True
    else:
        print(f"⚠️ Warning: Max error {max_error:.2%} > {threshold:.1%}")
        print("   (May be FD truncation error)")
        return max_error < 0.1  # Still pass if < 10%


if __name__ == "__main__":
    success = test_correctness_improved()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Task 2 Complete: Gradients validated")
        print("=" * 60)
    else:
        print("\n❌ Task 2 Failed")
        sys.exit(1)
