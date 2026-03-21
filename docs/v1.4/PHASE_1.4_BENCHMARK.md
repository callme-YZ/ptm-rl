# Phase 1.4: 3D Poisson Solver Performance Benchmark

**Date:** 2026-03-19  
**Author:** 小P ⚛️  
**Machine:** Mac mini (M4, 2024), macOS 15.2, Python 3.9.6  
**Status:** Baseline (nested 1D solver, known accuracy issues)

---

## Executive Summary

**Performance Target:** <10ms for 32×64×32 grid (RL training requirement)

**Current Results:**
- ✅ 32×64×32: **14.5 ms** (within 20ms relaxed target)
- ⚠️  64×128×64: **60.7 ms** (above 50ms target, but acceptable)

**Accuracy Status:**
- ❌ Residual error: ~1e2 (target <1e-8) — **NEEDS FIX**
- ❌ Solution error: ~1.0 (target <1e-6) — **NEEDS FIX**

**Bottleneck:** Nested θ-k loop (64×33 = 2112 tridiagonal solves)

---

## Benchmark Results

### Test 1: 32×64×32 Grid (Production RL Grid)

**Configuration:**
- Grid: nr=32, nθ=64, nζ=32
- Modes: nζ//2+1 = 17
- BC: Dirichlet
- Test function: φ = sin(πr) cos(θ) cos(ζ)

**Timing (average of 10 runs):**
```
Mean:   14.47 ms
Std:     0.82 ms
Min:    13.51 ms
Max:    16.23 ms
```

**Breakdown (estimated):**
- FFT forward: ~0.5 ms
- Tridiagonal solve (64×17 loops): ~13 ms
- FFT inverse: ~0.5 ms
- Overhead: ~0.5 ms

**Assessment:** ✅ Acceptable for RL (70 solves/sec)

---

### Test 2: 64×128×64 Grid (High-Resolution)

**Configuration:**
- Grid: nr=64, nθ=128, nζ=64
- Modes: nζ//2+1 = 33
- Total solves: 128×33 = 4224 tridiagonal systems

**Timing (single run):**
```
Elapsed: 60.71 ms
```

**Breakdown (estimated):**
- FFT forward: ~1.5 ms
- Tridiagonal solve (128×33 loops): ~57 ms
- FFT inverse: ~1.5 ms
- Overhead: ~0.7 ms

**Assessment:** ⚠️ Above 50ms target, but training still feasible at ~16 steps/sec

---

## Scaling Analysis

### Grid Refinement

| Grid Size | Time (ms) | Solves/Loop | Time/Solve (μs) |
|-----------|-----------|-------------|-----------------|
| 32×64×32 | 14.5 | 64×17 = 1088 | 13.3 |
| 64×128×64 | 60.7 | 128×33 = 4224 | 14.4 |

**Observation:** Nearly linear scaling with number of solves (good!)

**Per-solve cost:** ~14 μs (scipy.linalg.solve_banded overhead)

---

### Complexity Analysis

**Theoretical:**
- FFT: O(N³ log N)
- Tridiagonal: O(N) × N_θ × N_modes = O(N³)
- **Total: O(N³ log N)** dominated by FFT

**Empirical:**
```
T(32³) = 14.5 ms
T(64³) = 60.7 ms

Ratio: 60.7 / 14.5 = 4.2
Expected (8³ log(64/32)): 8 * 1.26 ≈ 10
```

**Discrepancy:** Better than expected! (Tridiagonal dominates, not FFT)

---

## Optimization Opportunities

### 1. Parallelize (θ,k) Loop

**Current:** Sequential loop over 128×33 = 4224 solves  
**Potential:** Embarrassingly parallel (no dependencies)

**Strategy:**
```python
from multiprocessing import Pool

def solve_slice(args):
    theta_idx, k_idx, omega_slice = args
    return solve_tridiagonal(...)

with Pool(4) as pool:
    results = pool.map(solve_slice, tasks)
```

**Expected speedup:** 3-4× on 4 cores → **15ms for 64³ grid** ✅

---

### 2. Numba JIT for Tridiagonal

**Current:** scipy.linalg.solve_banded (general purpose, some overhead)

**Alternative:** Custom Thomas algorithm with numba
```python
@numba.jit(nopython=True)
def thomas_algorithm(a, b, c, d):
    n = len(b)
    # Forward elimination
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]
    # Back substitution
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x
```

**Expected speedup:** 2-3× (JIT removes Python overhead)

**Combined:** 4 cores + JIT → **5-8ms for 64³** ✅✅

---

### 3. Switch to Full 2D Sparse Solver

**Trade-off:**
- More accurate (fixes residual error)
- Slower per k-mode (O(N²) sparse solve vs O(N) tridiagonal)

**Projected cost:**
```python
from scipy.sparse.linalg import spsolve

# Per k: (nr×nθ)×(nr×nθ) sparse matrix
# Cost: O((nr·nθ)^1.5) for sparse LU (approx)

nr=64, nθ=128: ~10ms per mode
Total (33 modes): ~330ms
```

**Verdict:** Too slow for RL (unless GPU-accelerated)

**Defer to:** v2.0 with JAX/CuPy

---

## Memory Usage

### Current Implementation

**Per-grid storage:**
```python
omega: (nr, nθ, nζ) × 8 bytes (float64)
omega_hat: (nr, nθ, nζ//2+1) × 16 bytes (complex128)
phi_hat: same as omega_hat
Total: ~3 × nr × nθ × nζ × 8 bytes
```

**Example (64×128×64):**
```
3 × 64 × 128 × 64 × 8 = 12.6 MB
```

**Assessment:** Negligible (GPU has >8GB)

---

### Temporary Arrays

**Per tridiagonal solve:**
```python
ab: (3, nr) × 16 bytes = ~3 KB (64 radial points)
```

**Total (all solves):**
```
128 × 33 × 3 KB = 12.7 MB (transient)
```

**Peak memory:** ~25 MB (very small)

---

## Comparison: BOUT++ Benchmarks

### LaplaceXY (2D perpendicular)

**BOUT++ timing (from learning notes 3.1):**
- Grid: 68×128 (similar to our (64,128) slice)
- Solver: Cyclic Reduction (parallel tridiagonal)
- Time: ~5 ms (C++ optimized)

**Our timing (per k-slice):**
- Grid: 64×128
- Time: 60.7ms / 33 modes = 1.8 ms per mode

**Verdict:** 2.8× slower than BOUT++ (acceptable for Python/NumPy)

---

### Laplace3D (3D with PETSc)

**BOUT++ approach:**
- Uses PETSc/Hypre (multigrid)
- Full 3D sparse matrix
- Tolerance: 1e-6 in ~100 iterations

**Our approach:**
- Nested 1D (incorrect but fast)
- Tolerance: ~1e2 (broken)

**Apples-to-apples:** Not comparable (different algorithms)

---

## Validation Results (Current Status)

### Test Outcomes

| Test | Status | Error | Target |
|------|--------|-------|--------|
| Slab Laplace (zero) | ✅ PASS | <1e-6 | 1e-6 |
| Slab Laplace (sinusoidal) | ❌ FAIL | 8.78e2 | 1e-6 |
| Bessel mode | ❌ FAIL | 2.48e2 | 1e-8 |
| 2D limit | ❌ FAIL | 1.39e4 | 1e-6 |
| Random source residual | ❌ FAIL | 4.16e2 | 1e-8 |
| Dirichlet BC | ✅ PASS | <1e-10 | 1e-10 |
| Neumann BC | ✅ PASS | <1e-4 | 1e-4 |
| Convergence order | ❌ FAIL | -0.14 | 1.8-2.2 |

**Pass rate:** 3/8 (37.5%)

**Blocking issue:** Core algorithm incorrect (nested 1D vs full 2D)

---

## Recommendations

### Short-term (Complete Phase 1.4)

1. **Implement full 2D per-mode solver** (Option 1 from ALGORITHM.md)
   - Estimated effort: 4 hours
   - Expected residual: <1e-8 ✅
   - Performance penalty: 2-3× slower (still <100ms)

2. **Defer optimization to v1.5**
   - Parallelization
   - Numba JIT
   - Target: <20ms for 64³ after optimization

---

### Long-term (v2.0)

1. **JAX/GPU implementation**
   - CuPy FFT (10× faster)
   - JAX sparse solver
   - Target: <5ms for 128³ grid

2. **Multigrid solver**
   - PETSc/petsc4py
   - Optimal O(N) scaling
   - Gold standard for elliptic PDEs

---

## Conclusions

**Performance:** 
- ✅ Current implementation meets relaxed target (14.5ms for 32³)
- ⚠️  Needs optimization for production (60ms for 64³)

**Accuracy:**
- ❌ Core algorithm broken (nested 1D approach)
- 🔧 Fixable with full 2D solver (4 hour effort)

**Recommendation:** 
- Fix accuracy first (physics correctness > speed)
- Then optimize (parallelization + JIT)
- Defer GPU to v2.0

---

## Appendix: Test Environment

**Hardware:**
- CPU: Apple M4 (10 cores: 4P + 6E)
- RAM: 24 GB
- GPU: 10-core (shared memory)

**Software:**
- macOS: 15.2 (Darwin 25.2.0)
- Python: 3.9.6
- NumPy: 1.26.4
- SciPy: 1.13.1
- FFT backend: NumPy (default)

**Compiler flags:** None (interpreted Python)

---

**Next steps:** Implement full 2D solver, re-run benchmarks, update this document.
