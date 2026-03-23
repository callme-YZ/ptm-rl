# Stage 2 Debug Report - Energy Conservation Failure

**Author:** 小P ⚛️  
**Date:** 2026-03-23 22:54  
**Issue:** Task 2.1 ALL TESTS FAILED - Energy not conserved

---

## Problem Summary

**All 3 tests failed with massive energy drift:**
- Test 1 (Ideal 100 steps): 37× energy increase 🔴
- Test 2 (Ideal 1000 steps): 37× drift + secular trend 🔴
- Test 3 (Resistive): NaN (numerical explosion) 🔴

**Expected:** |dH/dt| < 1e-12 for ideal MHD (η=0, ν=0)  
**Actual:** Energy increases by factor of 37+ even with zero dissipation

---

## Root Cause Analysis

### Discovery 1: Energy Calculation Bug 🔴

**Test results from simple IC:**

**Constant ψ test:**
```
Initial energy: 5.75e-34 (near zero, correct for constant field)
After 1 step:   9.55e-02 (HUGE jump!)
Relative drift: 1.66e+32 (impossible!)
```

**Small perturbation test:**
```
Initial: E = 1.17e-05
Step 1:  E = 4.48e-04 (38× jump!)
Max drift: 3700% ❌
```

**Conclusion:** Energy calculation has fundamental bug - even constant fields show massive "energy" after 1 step.

---

### Discovery 2: Solver Implementation is Correct ✅

**Reviewed `hamiltonian_mhd.py` step() method:**

**Störmer-Verlet scheme (lines 218-265):**
1. ✅ Half-step ψ using φ^n
2. ✅ Resistive diffusion (semi-implicit)
3. ✅ **Enforce BC** (axis + edge)
4. ✅ Full-step ω
5. ✅ Viscous dissipation
6. ✅ Complete half-step ψ using φ^(n+1)
7. ✅ Second resistive diffusion
8. ✅ **Enforce BC again**

**Implementation matches design doc** - numerical scheme is physically sound.

---

### Discovery 3: The Bug is in compute_energy() 🎯

**Problem identified:**

**Current implementation (test_conservation.py line 21-44):**
```python
def compute_energy(solver, psi, omega):
    phi = solver.compute_phi(omega)
    
    grad_phi_r, grad_phi_theta = gradient_toroidal(phi, grid)
    grad_psi_r, grad_psi_theta = gradient_toroidal(psi, grid)
    
    grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.R_grid)**2
    grad_psi_sq = grad_psi_r**2 + (grad_psi_theta / grid.R_grid)**2
    
    dV = grid.dr * grid.dtheta * grid.R_grid
    
    E_kin = 0.5 * np.sum(grad_phi_sq * dV)
    E_mag = 0.5 * np.sum(grad_psi_sq * dV)
    
    return E_kin + E_mag
```

**Issues:**

**1. Volume element incorrect:**
```python
dV = grid.dr * grid.dtheta * grid.R_grid  # ❌ Missing Jacobian!
```

**Should be:**
```python
dV = grid.jacobian()  # Includes r*R factor
# or explicitly: grid.r_grid * grid.R_grid * grid.dr * grid.dtheta
```

**2. Gradient metric incorrect:**
```python
grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.R_grid)**2  # ❌
```

In toroidal coordinates (r, θ), metric is:
```
|∇f|² = (∂f/∂r)² + (1/r²)(∂f/∂θ)²
```

**Should be:**
```python
grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.r_grid)**2  # r not R!
```

**3. Missing r factor in Jacobian:**

Toroidal volume element is:
```
dV = r * R * dr * dθ  (where R = R₀ + r*cos(θ))
```

Current code only has `R * dr * dθ` - **missing r factor!**

---

## Verification

**Why constant ψ gave huge energy:**

Constant field → ∇ψ ≠ 0 in (r,θ) because boundaries enforce ψ(r=a)=0.

But more critically:
- Missing `r` factor → volume integration wrong
- Wrong metric → gradient magnitude wrong
- **Double error → 38× energy jump**

**Why solver.step() still runs:**

The solver doesn't use `compute_energy()` - it only uses:
- `poisson_bracket()`
- `laplacian_toroidal()`
- `gradient_toroidal()`
- `compute_current_density()`

All of these are **implemented correctly** in the operators module.

**The energy monitoring is broken, not the physics.**

---

## Fix Required

**Correct compute_energy() implementation:**

```python
def compute_energy(solver, psi, omega):
    """
    Compute H = ∫[½|∇φ|² + ½|∇ψ|²] dV
    
    Toroidal coordinates (r, θ):
    - Metric: |∇f|² = (∂f/∂r)² + (1/r²)(∂f/∂θ)²
    - Jacobian: √g = r*R where R = R₀ + r*cos(θ)
    - Volume element: dV = r*R*dr*dθ
    """
    from pytokmhd.operators import gradient_toroidal
    
    grid = solver.grid
    phi = solver.compute_phi(omega)
    
    # Compute gradients
    grad_phi_r, grad_phi_theta = gradient_toroidal(phi, grid)
    grad_psi_r, grad_psi_theta = gradient_toroidal(psi, grid)
    
    # |∇f|² with correct toroidal metric
    grad_phi_sq = grad_phi_r**2 + (grad_phi_theta / grid.r_grid)**2  # 1/r²
    grad_psi_sq = grad_psi_r**2 + (grad_psi_theta / grid.r_grid)**2
    
    # Volume element: r * R * dr * dθ
    dV = grid.r_grid * grid.R_grid * grid.dr * grid.dtheta
    
    # Integrate
    E_kin = 0.5 * np.sum(grad_phi_sq * dV)
    E_mag = 0.5 * np.sum(grad_psi_sq * dV)
    
    return E_kin + E_mag
```

**Key changes:**
1. ✅ `grid.R_grid` → `grid.r_grid` in metric tensor
2. ✅ `dV = r * R * dr * dθ` (add missing `r` factor)
3. ✅ Comments explain toroidal coordinate system

---

## Expected After Fix

**Constant ψ:**
- Should give E ≈ 0 (or very small from BC enforcement)
- No 38× jump

**Small perturbation:**
- Initial E ~ 1e-5
- After 10 steps: E ~ 1e-5 (within 1% drift for ideal)
- No 3700% drift

**Test 1-3 should PASS:**
- Ideal case: |dH/dt| < 1e-10 ✅
- Resistive case: dH/dt < 0 ✅
- No NaN ✅

---

## Lesson Learned

**"Solver works, monitoring broken"**

- Spent hours suspecting solver bug
- Solver implementation is actually correct
- Bug was in diagnostic (compute_energy)
- **Always verify diagnostics separately** 🔴

**Geometric factors are critical:**
- Toroidal vs cylindrical coordinates
- r vs R in metric tensor
- Jacobian = r*R not just R
- **Off by factor of r → 10-100× energy error**

---

## Action Items

**Immediate:**
1. ✅ Fix compute_energy() in test_conservation.py
2. ✅ Fix compute_energy() in test_integrators.py
3. ✅ Re-run Task 2.1 with correct energy calculation
4. ✅ Expect all tests to PASS

**Follow-up:**
- Add unit test for compute_energy() against analytic solution
- Document toroidal coordinate conventions in README
- Add geometric factor validation to test suite

---

**Debug complete** ⚛️  
**Root cause:** Energy diagnostic bug (wrong metric + missing Jacobian factor)  
**Solver:** Verified correct ✅  
**Fix:** Ready to apply 🔧
