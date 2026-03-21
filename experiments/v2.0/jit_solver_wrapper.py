"""
JIT-compiled wrapper for CompleteMHDSolver

Author: 小A 🤖
Date: 2026-03-20

Wraps CompleteMHDSolver.step_multi with JAX JIT for speed.
"""

import jax
import jax.numpy as jnp
from functools import partial

from elsasser_bracket import ElsasserState
from complete_solver import CompleteMHDSolver


class JITMHDSolver:
    """JIT-optimized wrapper around CompleteMHDSolver
    
    Provides same API but with JIT compilation for speed.
    """
    
    def __init__(self, grid_shape: tuple, dr: float, dtheta: float, dz: float,
                 epsilon: float = 0.3, eta: float = 0.01, pressure_scale: float = 0.2):
        """Initialize solver with JIT-compiled step functions"""
        
        # Create underlying solver
        self.solver = CompleteMHDSolver(
            grid_shape, dr, dtheta, dz, epsilon, eta, pressure_scale
        )
        
        # JIT compile step function
        # Note: step_rk2 is a pure function of (state, dt) if we fix solver params
        self._jit_step_rk2 = jax.jit(self._step_rk2_pure)
        
        print("✅ JIT compilation prepared (will compile on first call)")
    
    def _step_rk2_pure(self, state: ElsasserState, dt: float) -> ElsasserState:
        """Pure function wrapper for step_rk2 (for JIT)"""
        return self.solver.step_rk2(state, dt)
    
    def step_rk2(self, state: ElsasserState, dt: float) -> ElsasserState:
        """JIT-compiled single RK2 step"""
        return self._jit_step_rk2(state, dt)
    
    def step_multi(self, state: ElsasserState, dt: float, n_substeps: int) -> ElsasserState:
        """Multi-step integration with JIT
        
        Uses jax.lax.fori_loop for efficient compiled loop.
        """
        dt_sub = dt / n_substeps
        
        # Use fori_loop for JIT-friendly iteration
        def body_fn(i, s):
            return self._jit_step_rk2(s, dt_sub)
        
        return jax.lax.fori_loop(0, n_substeps, body_fn, state)
    
    def hamiltonian(self, state: ElsasserState) -> float:
        """Compute Hamiltonian (energy)"""
        return self.solver.hamiltonian(state)
    
    @property
    def grid(self):
        """Access to underlying grid"""
        return self.solver.grid
    
    @property
    def epsilon(self):
        return self.solver.epsilon
    
    @property
    def eta(self):
        return self.solver.eta
    
    @property
    def pressure_scale(self):
        return self.solver.pressure_scale


def benchmark_jit_speedup():
    """Benchmark JIT vs non-JIT solver"""
    import time
    from ballooning_ic import ballooning_mode_ic
    from bout_metric import BOUTMetric
    from field_aligned import FieldAlignedCoordinates
    
    print("=" * 60)
    print("JIT Solver Benchmark")
    print("=" * 60 + "\n")
    
    # Create solvers
    print("Creating solvers...")
    grid_shape = (16, 32, 16)
    
    solver_normal = CompleteMHDSolver(grid_shape, 0.1, 0.1, 0.2, epsilon=0.3)
    solver_jit = JITMHDSolver(grid_shape, 0.1, 0.1, 0.2, epsilon=0.3)
    
    # Create initial state
    print("\nCreating initial state...")
    metric = BOUTMetric(R0=6.2, a=2.0)
    fa = FieldAlignedCoordinates(metric, 'constant')
    state = ballooning_mode_ic(metric, fa, grid_shape, m=2, n=1, amplitude=0.05)
    
    # Warm-up JIT
    print("\nWarming up JIT (first call compiles)...")
    t0 = time.time()
    state_jit = solver_jit.step_multi(state, dt=0.02, n_substeps=5)
    t_warmup = time.time() - t0
    print(f"  JIT warmup: {t_warmup:.2f}s")
    
    # Benchmark normal solver
    print("\nBenchmarking normal solver (20 calls)...")
    t0 = time.time()
    s = state
    for _ in range(20):
        s = solver_normal.step_multi(s, dt=0.02, n_substeps=5)
    t_normal = time.time() - t0
    print(f"  Time: {t_normal:.3f}s")
    print(f"  Per call: {t_normal/20:.3f}s")
    
    # Benchmark JIT solver
    print("\nBenchmarking JIT solver (20 calls)...")
    t0 = time.time()
    s = state
    for _ in range(20):
        s = solver_jit.step_multi(s, dt=0.02, n_substeps=5)
    t_jit = time.time() - t0
    print(f"  Time: {t_jit:.3f}s")
    print(f"  Per call: {t_jit/20:.3f}s")
    
    # Speedup
    speedup = t_normal / t_jit
    print("\n" + "=" * 60)
    print(f"JIT Speedup: {speedup:.2f}×")
    print("=" * 60)
    
    if speedup > 1.5:
        print("✅ Significant speedup! Use JIT solver.")
    elif speedup > 1.1:
        print("✅ Modest speedup. JIT worth it.")
    else:
        print("⚠️  Minimal speedup. May not be worth JIT overhead.")
    
    return speedup


if __name__ == "__main__":
    benchmark_jit_speedup()
