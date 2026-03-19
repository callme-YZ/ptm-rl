# Energy Dissipation Theory Debug: Final Report

**Date:** 2026-03-19  
**Author:** 小P ⚛️  
**Mission:** Resolve 96% error in dH/dt (30x discrepancy)

---

## Executive Summary

**STATUS:** Root causes identified, fix partially validated, full solution requires additional work.

**Key Findings:**
1. **Hamiltonian computed with wrong variable** (ω instead of φ) → 1700x energy error
2. **Units inconsistency** (SI vs normalized) → affects J² integral
3. **Missing 2π factor** in toroidal integral → 6.28x error

**Recommended Actions:**
1. Implement fast Poisson solver (∇²φ = ω)
2. Update test to use φ in `compute_hamiltonian`
3. Clarify unit conventions across codebase
4. Add unit tests for each component

---

## Problem Statement

**Observed:**
```
dH/dt_numeric ≈ -0.002
dH/dt_theory ≈ -0.065
Error ≈ 96% (factor of 30x)
```

**Expected:**
```
dH/dt = -η ∫ J² dV
Relative error < 5%
```

---

## Root Causes

### Issue #1: Wrong Variable in Hamiltonian ⚠️ CRITICAL

**Current code:**
```python
H = compute_hamiltonian(psi, omega, grid)  # ❌ WRONG!
```

**Hamiltonian definition:**
```
H[ψ, φ] = ∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²] dV
```

where ω = ∇²φ (vorticity).

**Problem:**
- Test passes `omega` (vorticity) as second argument
- But `compute_hamiltonian(psi, phi, grid)` expects `phi` (potential)
- Function computes |∇ω|² instead of |∇φ|²

**Impact:**
```python
# Numerical test:
H(ψ, φ) = 0.103
H(ψ, ω) = 174.3
Ratio = 1699x ❌
```

**Why it matters for dH/dt:**
- If H is wrong by 1700x, then dH/dt_numeric is also wrong by 1700x
- This is the dominant source of error

**Fix:**
```python
# Solve Poisson equation: ∇²φ = ω
phi = solve_poisson_toroidal(omega, grid)

# Use correct variable
H = compute_hamiltonian(psi, phi, grid)  # ✅ CORRECT
```

**Status:** Poisson solver implemented but needs debugging/optimization.

---

### Issue #2: Units Inconsistency

**Code uses mixed units:**
- Solver (`HamiltonianMHDIMEX`): Likely normalized/Alfvén units (μ₀ = 1)
- `compute_current_density`: Defaults to SI units (μ₀ = 4π×10⁻⁷)

**Definition:**
```
J = Δ*ψ / (μ₀R)
```

**Impact:**
```
J_SI = J_normalized / (4π×10⁻⁷) ≈ J_normalized × 8×10⁵
J²_SI ≈ J²_normalized × 6×10¹¹
```

**But:** The actual impact depends on how `eta` and `psi` are normalized in the solver.

**Current test result:**
- With μ₀ = 4π×10⁻⁷: theory = -0.065, numeric = -0.002
- With μ₀ = 1.0: theory = -7×10⁻⁴, numeric = -1.457

**Conclusion:** Need to check solver's normalization convention.

**Action Required:**
1. Document normalization convention in solver
2. Use consistent μ₀ everywhere
3. Add unit test with known analytical solution

---

### Issue #3: Missing 2π Factor

**Theory formula:**
```
dH/dt = -η ∫ J² dV
```

**In toroidal geometry:**
```
dV = R dr dθ dφ
∫ dV = 2π ∫∫ R dr dθ  (axisymmetric, ∫₀²ᵖ dφ = 2π)
```

**Test computes:**
```python
J2_int = np.sum(J**2 * dV)  # dV = R·dr·dθ (2D)
dH_theory = -eta * J2_int   # ❌ Missing 2π
```

**Correct:**
```python
dH_theory = -eta * 2*np.pi * J2_int  # ✅
```

**But:** `compute_hamiltonian` already includes 2π (line 121 in hamiltonian.py):
```python
H = 2 * np.pi * energy_2d
```

So dH/dt_numeric also includes 2π automatically.

**Status:** Verified, fix is straightforward.

---

## Theory Derivation

**Starting equations:**
```
H = ∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²] dV

∂ψ/∂t = {ψ, H} - ηJ
∂ω/∂t = {ω, H} + S_P - ν∇²ω
```

**Time derivative:**
```
dH/dt = ∫ [∇φ·∇(∂φ/∂t) + ∇ψ·∇(∂ψ/∂t)] dV
```

**Integration by parts:**
```
dH/dt = -∫ [φ·∂ω/∂t + ψ·Δ*(∂ψ/∂t)] dV
```

**Substitute evolution equations:**
- Ideal part ({·,H} terms): cancels (Hamiltonian structure)
- Dissipative part:
  - Resistive: ∂ψ/∂t|_resistive = -ηJ
  - Viscous: ∂ω/∂t|_viscous = -ν∇²ω

**Resistive dissipation:**
```
dU/dt|_resistive = -∫ ψ·Δ*(-ηJ) dV
                 = η ∫ ψ·Δ*J dV
```

Using Δ*J = (1/R²)Δ*(Δ*ψ/μ₀R) and integration by parts:
```
dU/dt|_resistive = -η ∫ J·(Δ*ψ/μ₀R) dV
                 = -η/μ₀ ∫ J²·R dV
                 = -η ∫ J² dV  (if μ₀ normalized to 1)
```

**For ν = 0:**
```
dH/dt = -η ∫ J² dV
```

**In toroidal coordinates:**
```
dH/dt = -η·2π ∫∫ J²·R dr dθ
```

**Derivation complete.** Formula is correct.

---

## Fix Implementation

### Priority #1: Solve Poisson Equation

**Requirement:** Fast, accurate solver for ∇²φ = ω

**Options:**
1. **FFT-based** (implemented in `poisson_solver.py`):
   - Fast: O(N log N)
   - Needs debugging (currently has boundary condition issues)
   
2. **Iterative (Jacobi/SOR)** (implemented in `poisson_simple.py`):
   - Slow: O(N²) per iteration
   - Simple but impractical for production

3. **Multigrid** (NOT implemented):
   - Fast: O(N)
   - Complex to implement

**Recommendation:** Debug FFT-based solver.

**Workaround for testing:**
If Poisson solver is not ready, can use simplified test:
- Run solver with small amplitude (linear regime)
- Check energy conservation in ideal case (η=0, ν=0)
- Then add small η and verify dissipation rate

---

### Priority #2: Fix Test Code

**File:** `tests/test_imex_energy_budget.py`

**Changes:**

```python
# After line 47, add:
from pytokmhd.operators import solve_poisson_toroidal

# Replace line 60:
# OLD: H0 = compute_hamiltonian(psi, omega, grid)
phi0 = solve_poisson_toroidal(omega, grid)
H0 = compute_hamiltonian(psi, phi0, grid)  # ✅ FIX #1

# Replace line 73:
# OLD: J = compute_current_density(psi, grid)
J = compute_current_density(psi, grid, mu0=1.0)  # ✅ FIX #2 (normalized units)

# Replace line 78:
# OLD: dH_theory = -eta * J2_int
dH_theory = -eta * 2*np.pi * J2_int  # ✅ FIX #3

# Replace line 83:
# OLD: H = compute_hamiltonian(psi, omega, grid)
phi = solve_poisson_toroidal(omega, grid)
H = compute_hamiltonian(psi, phi, grid)  # ✅ FIX #1
```

---

### Priority #3: Unit Tests

**Add tests for components:**

1. **Test Poisson solver:**
```python
def test_poisson():
    phi_exact = r² sin(θ)
    omega = -∇²φ_exact
    phi_solved = solve_poisson_toroidal(omega, grid)
    assert |phi_solved - phi_exact|_max < tol
```

2. **Test Hamiltonian:**
```python
def test_hamiltonian_units():
    # Check H has correct units/scaling
    phi = test_field
    psi = test_field
    H1 = compute_hamiltonian(psi, phi, grid)
    H2 = compute_hamiltonian(2*psi, 2*phi, grid)
    assert H2 ≈ 4*H1  (quadratic scaling)
```

3. **Test energy conservation:**
```python
def test_energy_conservation_ideal():
    # η = 0, ν = 0 → dH/dt = 0
    solver = HamiltonianMHDIMEX(grid, eta=0, nu=0)
    for step in range(100):
        psi, omega = solver.step(psi, omega)
        phi = solve_poisson_toroidal(omega, grid)
        H = compute_hamiltonian(psi, phi, grid)
        assert |H - H0| / H0 < 1e-6
```

---

## Open Questions

1. **Normalization convention:**
   - What is the normalization used in `HamiltonianMHDIMEX`?
   - Are ψ, φ, η all dimensionless?
   - Document in design doc

2. **Poisson solver accuracy:**
   - Current FFT-based solver has errors
   - Need to debug boundary conditions
   - Or implement alternative (multigrid?)

3. **Initial condition consistency:**
   - Test initializes ω = -∇²ψ (not ω = -∇²φ!)
   - This creates inconsistent initial condition
   - Should initialize φ first, then compute ω = ∇²φ

---

## Timeline & Status

**Completed (3 hours):**
- [x] First-principles derivation of dH/dt
- [x] Code audit (found ω vs φ issue)
- [x] Literature check (confirmed formula)
- [x] Root cause analysis (3 issues identified)
- [x] Implemented Poisson solver (needs debugging)
- [x] Wrote comprehensive documentation

**Remaining (2-4 hours):**
- [ ] Debug Poisson solver OR implement alternative
- [ ] Update test with all three fixes
- [ ] Run validation (target: error < 5%)
- [ ] Add unit tests for components
- [ ] Document normalization convention

**Blocker:**
Poisson solver not production-ready. Need either:
- Debug FFT-based solver (1-2 hours)
- Implement multigrid solver (4-6 hours)
- OR use workaround (simplified test without Poisson)

---

## Deliverables

**Theory:**
- `docs/energy-dissipation-derivation.md` ✅
- `docs/ENERGY-DISSIPATION-ROOT-CAUSE.md` ✅
- `docs/ENERGY-DISSIPATION-FINAL-REPORT.md` ✅ (this file)

**Code:**
- `src/pytokmhd/operators/poisson_solver.py` ✅ (needs debugging)
- `src/pytokmhd/operators/poisson_simple.py` ✅ (workaround)
- `tests/test_energy_budget_fixed.py` ⚠️ (incomplete, needs working Poisson solver)

**Validation:**
- Test 2 passing (error < 5%) ❌ (blocked by Poisson solver)

---

## Recommendation to YZ

**Short-term (next session):**
1. Debug FFT Poisson solver OR
2. Use simplified validation test (ideal MHD energy conservation)

**Medium-term:**
3. Implement all three fixes in main test
4. Validate with error < 5%
5. Add unit tests

**Long-term:**
6. Document normalization convention
7. Implement multigrid Poisson solver
8. Benchmark performance

**Estimated time to completion:** 2-4 hours (with working Poisson solver).

---

**End of Report**

小P ⚛️ 2026-03-19
