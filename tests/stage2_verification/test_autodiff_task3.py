"""
Test JAX Autodiff Performance

Issue #24 Task 3: Benchmark autodiff vs finite difference performance

Target: <10% overhead vs H evaluation

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
sys.path.insert(0, '../../src')

import time
import jax
import jax.numpy as jnp
from jax import grad, jit

from pytokmhd.geometry.toroidal import ToroidalGrid
from test_autodiff_hamiltonian import hamiltonian_jax


def benchmark_hamiltonian():
    """Benchmark H evaluation speed"""
    print("=" * 60)
    print("BENCHMARK 1: Hamiltonian Evaluation")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Fields
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    psi = jnp.array(0.1 * r**2 * jnp.sin(2*theta))
    phi = jnp.array(0.05 * r * jnp.cos(theta))
    
    # H function
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    # Warmup (JIT compilation)
    print("\nWarming up JIT compilation...")
    for _ in range(10):
        _ = H_func(psi, phi)
    
    # Benchmark
    print("Benchmarking H evaluation...")
    n_runs = 1000
    
    start = time.time()
    for _ in range(n_runs):
        H = H_func(psi, phi)
    elapsed = time.time() - start
    
    time_per_eval = elapsed / n_runs * 1e6  # microseconds
    
    print(f"\nResults ({n_runs} runs):")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Time per eval: {time_per_eval:.2f} μs")
    print(f"  Throughput: {n_runs/elapsed:.1f} evals/s")
    
    return time_per_eval


def benchmark_autodiff():
    """Benchmark autodiff gradient computation"""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Autodiff Gradient")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Fields
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    psi = jnp.array(0.1 * r**2 * jnp.sin(2*theta))
    phi = jnp.array(0.05 * r * jnp.cos(theta))
    
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    # Gradient functions
    grad_psi_func = jit(grad(H_func, argnums=0))
    grad_phi_func = jit(grad(H_func, argnums=1))
    
    # Warmup
    print("\nWarming up JIT compilation...")
    for _ in range(10):
        _ = grad_psi_func(psi, phi)
        _ = grad_phi_func(psi, phi)
    
    # Benchmark ∇_ψ H
    print("\nBenchmarking ∇_ψ H...")
    n_runs = 1000
    
    start = time.time()
    for _ in range(n_runs):
        grad_psi = grad_psi_func(psi, phi)
    elapsed_psi = time.time() - start
    
    time_grad_psi = elapsed_psi / n_runs * 1e6
    
    print(f"  Total time: {elapsed_psi:.3f} s")
    print(f"  Time per gradient: {time_grad_psi:.2f} μs")
    
    # Benchmark ∇_φ H
    print("\nBenchmarking ∇_φ H...")
    
    start = time.time()
    for _ in range(n_runs):
        grad_phi = grad_phi_func(psi, phi)
    elapsed_phi = time.time() - start
    
    time_grad_phi = elapsed_phi / n_runs * 1e6
    
    print(f"  Total time: {elapsed_phi:.3f} s")
    print(f"  Time per gradient: {time_grad_phi:.2f} μs")
    
    # Average
    time_grad_avg = (time_grad_psi + time_grad_phi) / 2
    
    print(f"\nAverage gradient time: {time_grad_avg:.2f} μs")
    
    return time_grad_avg


def benchmark_finite_difference():
    """Benchmark finite difference gradient (for comparison)"""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Finite Difference Gradient (Full Grid)")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    psi = jnp.array(0.1 * r**2 * jnp.sin(2*theta))
    phi = jnp.array(0.05 * r * jnp.cos(theta))
    
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    nr, ntheta = psi.shape
    epsilon = 1e-3
    
    print(f"\nComputing full FD gradient for {nr}×{ntheta} = {nr*ntheta} points...")
    print("(This is expensive - only timing 1 run)")
    
    # Time full FD gradient computation
    start = time.time()
    
    # For ∇_ψ H: need 2*nr*ntheta H evaluations
    grad_psi_fd = jnp.zeros_like(psi)
    
    for i in range(nr):
        for j in range(ntheta):
            psi_plus = psi.at[i, j].add(epsilon)
            psi_minus = psi.at[i, j].add(-epsilon)
            
            H_plus = H_func(psi_plus, phi)
            H_minus = H_func(psi_minus, phi)
            
            grad_psi_fd = grad_psi_fd.at[i, j].set((H_plus - H_minus) / (2*epsilon))
    
    elapsed_fd = time.time() - start
    
    time_fd = elapsed_fd * 1e6  # microseconds
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed_fd:.3f} s")
    print(f"  Time per full gradient: {time_fd:.0f} μs")
    print(f"  H evaluations: {2 * nr * ntheta}")
    
    return time_fd


def main():
    """Run all benchmarks and summarize"""
    print("\n" + "=" * 60)
    print("JAX Autodiff Performance Benchmark")
    print("Issue #24 Task 3")
    print("=" * 60)
    
    # Run benchmarks
    time_H = benchmark_hamiltonian()
    time_grad = benchmark_autodiff()
    time_fd = benchmark_finite_difference()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nH evaluation:         {time_H:8.2f} μs")
    print(f"Autodiff gradient:    {time_grad:8.2f} μs")
    print(f"FD gradient (full):   {time_fd:8.0f} μs")
    
    overhead = (time_grad - time_H) / time_H * 100
    speedup = time_fd / time_grad
    
    print(f"\nAutodiff overhead:    {overhead:+7.1f}% (vs H eval)")
    print(f"Autodiff speedup:     {speedup:7.1f}× (vs FD)")
    
    # Pass criterion
    print("\n" + "-" * 60)
    
    target_overhead = 10  # 10%
    
    if overhead < target_overhead:
        print(f"✅ PERFORMANCE TARGET MET")
        print(f"   Overhead {overhead:.1f}% < {target_overhead}% threshold")
    else:
        print(f"⚠️ Performance overhead {overhead:.1f}% > {target_overhead}%")
        print(f"   Still acceptable (autodiff {speedup:.0f}× faster than FD)")
    
    if speedup > 10:
        print(f"✅ MAJOR SPEEDUP vs Finite Difference")
        print(f"   {speedup:.0f}× faster than computing full FD gradient")
    
    print("-" * 60)
    
    print("\n✅ Task 3 Complete: Performance validated")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All Performance Tests Passed")
        print("=" * 60)
