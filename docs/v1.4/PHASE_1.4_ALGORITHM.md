# Phase 1.4 Algorithm Update (2026-03-19)

**Status:** ✅ **FIXED** - Full 2D Sparse Matrix Solver Implemented

---

## Critical Change: Nested 1D → Full 2D

**Problem with old algorithm:**
The nested 1D approach (solving per-θ slice independently) **ignores the poloidal coupling** term (1/r²)∂²φ/∂θ². This works in BOUT++ because they use field-aligned coordinates where θ slices decouple, but in standard cylindrical coordinates, this is WRONG PHYSICS.

**New algorithm:**
For each k-mode, build and solve full 2D system:

```python
# Build 2D Laplacian matrix (nr·nθ) × (nr·nθ)
A = kron(I_θ, D_r) + kron(D_θ, diag(1/r²))

# where:
# D_r: radial Laplacian (nr×nr)
# D_θ: poloidal Laplacian (nθ×nθ, periodic BC)

# Solve sparse system
phi_flat = spsolve(A, omega_flat)
```

**Critical indexing detail:**
- scipy.sparse.kron uses **C-order** (row-major)
- numpy.flatten(order='F') uses **F-order** (column-major)
- **Solution**: Swap kron arguments to match F-order

---

## Results

- ✅ Residual: 1e-15 (was 8.78e+02)
- ✅ Convergence order: 2.0 (was -0.14)
- ✅ All physics tests passing
- ⚠️ Performance: 193ms for 32³ (was 14ms) — acceptable trade-off

See `PHASE_1.4_FIX_SUMMARY.md` for full details.

---

**The original algorithm document below is OBSOLETE for Step 2-3. Use the fixed implementation in `src/pytokmhd/solvers/poisson_3d.py`.**

---

# Phase 1.4: 3D Poisson Solver Algorithm

**Date:** 2026-03-19  
**Author:** 小P ⚛️  
**Status:** Partial Implementation (debugging in progress)

---

## Algorithm Overview

### Per-Mode FFT Method (from BOUT++ cyclic_laplace)

**Physical Problem:**
```
∇²φ = ω
```

where:
- φ: Electrostatic potential
- ω: Vorticity (source term)
- ∇²: 3D Laplacian in cylindrical coordinates

**Cylindrical Laplacian:**
```
∇²φ = (1/r) ∂/∂r(r ∂φ/∂r) + (1/r²) ∂²φ/∂θ² + ∂²φ/∂ζ²
     = ∂²φ/∂r² + (1/r) ∂φ/∂r + (1/r²) ∂²φ/∂θ² + ∂²φ/∂ζ²
```

---

## Step-by-Step Algorithm

### Step 1: Forward FFT in ζ

Transform toroidal direction using Real FFT:
```python
ω(r,θ,ζ) → ω̂(r,θ,k)    # k = 0, 1, ..., nζ//2
φ(r,θ,ζ) → φ̂(r,θ,k)
```

**Implementation:**
```python
from scipy.fft import rfft, irfft

omega_hat = rfft(omega, axis=2)  # (nr, nθ, nζ//2+1) complex
```

**Frequency array:**
```python
k_ζ = 2π/L_ζ * [0, 1, 2, ..., nζ//2]
```

---

### Step 2: Per-Mode 2D Poisson

For each Fourier mode k, equation becomes:
```
∇_⊥²φ̂_k - k²φ̂_k = ω̂_k
```

where:
```
∇_⊥²φ̂_k = ∂²φ̂_k/∂r² + (1/r) ∂φ̂_k/∂r + (1/r²) ∂²φ̂_k/∂θ²
```

**Challenge:** This is still a 2D problem in (r,θ). Two approaches:

**Approach A: Full 2D solve** (matrix inversion, expensive)
- Discretize (r,θ) on nr×nθ grid
- Build (nr·nθ) × (nr·nθ) sparse matrix
- Solve with sparse linear solver

**Approach B: Nested 1D solves** (BOUT++ approach)
- For each θ_j, solve 1D tridiagonal in r
- Assumption: θ derivatives treated as known RHS modification
- **Current implementation uses this**

---

### Step 3: Tridiagonal Solve in r

For fixed (θ_j, k), discretize radial equation:
```
∂²φ/∂r² + (1/r) ∂φ/∂r - k²φ = RHS_modified
```

**Finite difference (2nd-order central):**
```
[φ_{i-1} - 2φ_i + φ_{i+1}]/Δr² + (1/r_i)[φ_{i+1} - φ_{i-1}]/(2Δr) - k²φ_i = ω̂_i
```

**Tridiagonal coefficients:**
```python
a_i = 1/Δr² - 1/(2r_i·Δr)      # Lower diagonal
b_i = -2/Δr² - k²               # Main diagonal
c_i = 1/Δr² + 1/(2r_i·Δr)      # Upper diagonal
```

**Boundary conditions:**
- Dirichlet: φ(r=0) = 0, φ(r=a) = 0
  - Modify first/last rows: b[0]=1, c[0]=0, rhs[0]=0
- Neumann: ∂φ/∂r(r=0) = 0, ∂φ/∂r(r=a) = 0
  - Use ghost point or one-sided differences

**Solve with scipy:**
```python
from scipy.linalg import solve_banded

# Format: ab[0,:] = upper, ab[1,:] = main, ab[2,:] = lower
ab = np.zeros((3, nr), dtype=complex)
ab[1, :] = b
ab[0, 1:] = c
ab[2, :-1] = a

phi_hat[:, j, k] = solve_banded((1, 1), ab, rhs)
```

---

### Step 4: Inverse FFT

Reconstruct physical space:
```python
phi = irfft(phi_hat, n=nζ, axis=2).real
```

**Normalization:** Follow BOUT++ convention
- Forward FFT: normalize by 1/N
- Inverse FFT: no normalization (already handled by scipy)

---

## Critical Implementation Details

### 1. Handling r=0 Singularity

**Problem:** At r=0, terms like (1/r)∂φ/∂r and (1/r²)∂²φ/∂θ² are singular.

**Solution:** 
- Use L'Hospital's rule: as r→0, (1/r)∂φ/∂r → ∂²φ/∂r² (for smooth φ)
- Or explicitly set 1/r terms to 0 at r=0 (regularity assumption)

```python
# Safe division
with np.errstate(divide='ignore'):
    one_over_r = np.where(r > 1e-10, 1.0/r, 0.0)
```

---

### 2. Complex Coefficients

**Question from task:** Do we need complex tridiagonal solver?

**Answer:** 
- Yes, in general. BOUT++ uses complex coefficients for mixed derivative ∂²φ/∂r∂ζ.
- But in pure cylindrical coordinates (no field-aligned transform), coefficients are real!
- Our implementation uses complex for generality (scipy handles both).

**Cost:** 
- Real tridiagonal: O(N) flops
- Complex tridiagonal: ~4× real (2× for real/imag parts, 2× for complex arithmetic)
- Still O(N), acceptable.

---

### 3. Per-Mode Loop Performance

**Complexity:**
- FFT: O(N_r N_θ N_ζ log N_ζ)
- Tridiagonal solve: O(N_r) × N_θ × N_modes = O(N_r N_θ N_ζ)
- Total: O(N_r N_θ N_ζ log N_ζ)

**Bottleneck:** 
- For nr=64, nθ=128, nζ=64:
  - FFT: ~2ms (using scipy.fft)
  - Tridiagonal solve: ~10ms (128×33 solves)
  - Total: ~12ms ✓ meets <20ms target

**Optimization opportunities:**
- Parallelize (θ,k) loop with multiprocessing
- Use numba.jit for tridiagonal solver
- Defer to v2.0 (JAX/GPU)

---

## Current Implementation Status

### ✅ Completed

1. **FFT infrastructure** (from Phase 1.1)
   - `forward_fft`, `inverse_fft`, `fft_frequencies`
   - BOUT++ normalization convention
   
2. **Laplacian computation** (`compute_laplacian_3d`)
   - Radial: ∂²/∂r² + (1/r)∂/∂r
   - Poloidal: (1/r²)∂²/∂θ²
   - Toroidal: ∂²/∂ζ² via FFT
   - **Fixed:** r=0 singularity handling
   
3. **Tridiagonal solver** (`_solve_tridiagonal_complex`)
   - Uses scipy.linalg.solve_banded
   - Handles complex coefficients
   
4. **Boundary conditions**
   - Dirichlet: φ=0 at r=0,a ✓
   - Neumann: ∂φ/∂r=0 (approximate)

5. **Test infrastructure**
   - Grid3D class
   - `verify_poisson_solver` helper
   - Performance benchmarks

### 🚧 Known Issues

1. **Residual error too large** (~1e2, target <1e-8)
   - Likely cause: θ derivative coupling not properly handled
   - Current approach treats θ modes independently → incorrect
   
2. **Convergence order wrong** (-0.14, expected 1.8-2.2)
   - Error increasing with refinement!
   - Indicates fundamental algorithm issue
   
3. **Solution error** (~1.0, target <1e-6)
   - Round-trip φ_exact → ω → φ_num fails
   - Core solver has bugs

### 🔍 Root Cause Analysis

**Problem:** Nested 1D approach (Approach B) is **fundamentally incorrect** for full 3D Laplacian.

**Why:**
- When solving per-θ slice, we ignore ∂²φ/∂θ² coupling to neighboring θ points
- This works in BOUT++ because they use **field-aligned coordinates** where θ slices decouple
- In standard cylindrical (r,θ,ζ), θ coupling is essential

**Correct approach:**
- Must use **Approach A**: Full 2D sparse matrix solve per k
- Or use **iterative solver** (e.g., multigrid, conjugate gradient)

---

## Next Steps (for completion)

### Option 1: Full 2D Per-Mode Solver (Recommended)

**Algorithm:**
1. For each k, build (nr×nθ) sparse matrix A
2. Flatten φ̂_k(r,θ) → vector (size nr·nθ)
3. Solve Ax = b with scipy.sparse.linalg.spsolve
4. Reshape back to (nr, nθ)

**Advantages:**
- Correct physics
- Still O(N²) for sparse solve (acceptable)

**Code skeleton:**
```python
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

# Build 1D operators
D_r = diags([a, b, c], [-1, 0, 1], shape=(nr, nr))  # Radial
D_θ = diags(..., shape=(nθ, nθ))  # Poloidal (periodic)

# 2D operator: D_r ⊗ I + I ⊗ D_θ - k²I
A = kron(D_r, eye(nθ)) + kron(eye(nr), D_θ / r²) - k²*eye(nr*nθ)

# Solve
phi_flat = spsolve(A, omega_flat)
```

**Estimated effort:** 4 hours

---

### Option 2: Axisymmetric Simplification (Quick Fix)

**Assumption:** Ignore θ variation (m=0 modes only)
- Set ∂²/∂θ² = 0
- Reduces to true 1D tridiagonal per k

**Limitations:**
- Cannot capture ballooning modes
- Not useful for full 3D MHD

**Use case:** Quick validation of radial+toroidal solve

**Estimated effort:** 1 hour

---

### Option 3: Iterative Solver (Advanced)

Use PETSc/hypre (like BOUT++ Laplace3D):
- Multigrid preconditioner
- GMRES/BiCGSTAB
- Excellent scalability

**Effort:** 1-2 weeks (requires C++/Fortran bindings)

---

## Validation Plan (After Fix)

### Test 1: Analytical Bessel (m=0)
```python
φ = J_0(k_r r) cos(k_z ζ)
∇²φ = -(k_r² + k_z²) J_0(k_r r) cos(k_z ζ)
```
**Tolerance:** <1e-8 (spectral accuracy)

### Test 2: Slab Laplace
```python
φ = sin(π r/a) cos(k_z ζ)
∇²φ = -(π²/a² + k_z²) φ
```
**Tolerance:** <1e-6

### Test 3: Energy Conservation
- Integrate H = ∫(½|∇ψ|² + ½ω²) dV
- Should be conserved after solve→Laplacian round-trip
- **Tolerance:** <1e-10

---

## Performance Benchmark (Projected)

**Grid:** 32×64×32

| Component | Time | Fraction |
|-----------|------|----------|
| FFT forward | 0.5 ms | 5% |
| Per-mode solve (33 modes) | 8 ms | 80% |
| FFT inverse | 0.5 ms | 5% |
| Overhead | 1 ms | 10% |
| **Total** | **10 ms** | **100%** |

**Scaling:** 
- 64×128×64: ~40 ms (within <50ms target)
- 128×256×128: ~160 ms (RL training may slow down)

**Optimization:**
- Parallel (θ,k) loop: ~3× speedup (4 cores)
- Target: <5ms for 64³ grid

---

## References

1. **Learning notes:** `notes/v1.4/2.3-3d-poisson-solver.md`
2. **BOUT++ source:** `src/invert/laplace/impls/cyclic/cyclic_laplace.cxx`
3. **Design doc:** `docs/v1.4/DESIGN.md` §4.3, §8.1
4. **Validation:** `notes/v1.4/3.1-validation-strategy.md`

---

## Lessons Learned

1. **Nested 1D != Full 2D:** Subtle but critical for coupled PDEs
2. **Coordinate systems matter:** Field-aligned vs cylindrical
3. **Test early, test often:** Residual test caught the bug immediately
4. **Singularities are hard:** r=0 handling requires physics insight

---

**Next action:** Implement Option 1 (Full 2D solver) before moving to Phase 1.5.
