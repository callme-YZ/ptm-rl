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

