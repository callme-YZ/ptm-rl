# Toroidal MHD Solver Debug & Fix Plan

**Version:** v1.2  
**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Status:** Design Phase

---

## 1. Problem Summary

### 1.1 Observed Failure

**Timeline:**
- **v1.1 development (Week 2):** ToroidalMHDSolver implementation completed
- **Initial testing:** Numerical explosion within 10-50 time steps
- **Symptom:** Fields (B, velocity) grow exponentially → NaN/Inf
- **Severity:** Complete solver failure, unusable for RL environment

**Test Record:**

| Test Case | dt | Grid | Steps Before Explosion | Notes |
|-----------|--------|------|------------------------|-------|
| Uniform B₀ | 1e-5 | 32³ | ~15 steps | B grows ~10³ |
| Grad-Shafranov | 1e-4 | 64³ | ~8 steps | div(B) → 10² |
| Cylindrical approx | 1e-3 | 32³ | ~5 steps | Fastest failure |

**Key observations:**
- **Not** CFL violation (tested dt down to 1e-5, CFL ~ 0.01)
- **Not** resolution issue (64³ grid still fails)
- Explosion rate **increases** with smaller dt → suggests finite-difference error, not time integration

### 1.2 v1.1 Decision

**Workaround adopted:**
- Use `CylindricalMHDSolver` with large aspect ratio (R₀/a → ∞)
- Approximates toroidal geometry in straight-field limit
- **Limitation:** Cannot model toroidal effects (grad-B drift, trapped particles)

**Deferred to v1.2:**
- Root cause analysis of toroidal solver
- Systematic debug and fix
- Full toroidal geometry support

---

## 2. Root Cause Hypotheses

### H1: Toroidal Laplacian Finite-Difference Error

**Likelihood:** ⭐⭐⭐⭐⭐ (Very High)

**Theory:**  
Toroidal Laplacian in (r, θ, φ) coordinates:

$$
\nabla^2 f = \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial f}{\partial r}\right) + \frac{1}{r^2} \frac{\partial^2 f}{\partial \theta^2} + \frac{1}{(R_0 + r\cos\theta)^2} \frac{\partial^2 f}{\partial \phi^2}
$$

**Suspected bug:**
- Incorrect discretization of the $1/(R_0 + r\cos\theta)^2$ term
- Off-by-one index error in stencil
- Wrong sign in cross-derivative terms

**Test method:**
1. Apply Laplacian to known analytical functions
2. Compare numerical vs. analytical $\nabla^2 f$
3. Check convergence rate (should be O(dr²) for 2nd-order scheme)

**Expected evidence if true:**
- Large error localized in φ-direction terms
- Error grows with toroidicity (small R₀/a)
- Cylindrical limit (R₀ → ∞) should work

---

### H2: Boundary Condition Handling

**Likelihood:** ⭐⭐⭐⭐ (High)

**Theory:**  
Toroidal geometry requires:
- **Periodic in φ:** f(r, θ, 0) = f(r, θ, 2π)
- **Periodic in θ:** f(r, 0, φ) = f(r, 2π, φ)
- **Axis boundary (r=0):** Special treatment (regularity condition)

**Suspected bug:**
- Incorrect ghost cell filling at r=0
- Periodicity not enforced correctly
- Axis singularity not handled (1/r terms)

**Test method:**
1. Initialize uniform field → should remain constant
2. Check div(B) at boundaries
3. Test regularity: df/dr|_{r=0} finite?

**Expected evidence if true:**
- Explosion starts near r=0 or boundaries
- Boundary-value fields (e.g., conducting wall) fail faster

---

### H3: Metric Tensor Jacobian Error

**Likelihood:** ⭐⭐⭐ (Medium)

**Theory:**  
Volume element in toroidal coordinates:

$$
dV = J \, dr \, d\theta \, d\phi, \quad J = r(R_0 + r\cos\theta)
$$

Divergence includes Jacobian:

$$
\nabla \cdot \mathbf{B} = \frac{1}{J} \left[ \frac{\partial (J B^r)}{\partial r} + \frac{\partial (J B^\theta)}{\partial \theta} + \frac{\partial (J B^\phi)}{\partial \phi} \right]
$$

**Suspected bug:**
- Forgot to include J in divergence calculation
- Wrong Jacobian formula (missing r or R₀ term)

**Test method:**
1. Compute div(B) for solenoidal field (should be ~0)
2. Check if $\int \nabla \cdot \mathbf{B} \, dV = 0$ globally

**Expected evidence if true:**
- div(B) accumulates systematically (not random noise)
- Error proportional to 1/R₀ (toroidicity dependence)

---

### H4: Time Integration Scheme Mismatch

**Likelihood:** ⭐⭐ (Low)

**Theory:**  
MHD equations are stiff (fast Alfvén waves + slow diffusion). If using:
- **Explicit Euler:** Requires very small dt
- **RK4:** Better stability, but still limited
- **IMEX:** Implicit for diffusion, explicit for advection (ideal)

**Suspected issue:**
- Toroidal terms introduce faster timescales than cylindrical
- Current dt choice stable for cylindrical but not toroidal

**Test method:**
1. Reduce dt by 10× → if still explodes, not time integration
2. Switch to implicit diffusion solver
3. Compute max Alfvén speed: $v_A = B / \sqrt{\mu_0 \rho}$

**Expected evidence if true:**
- Explosion delayed (but not eliminated) with smaller dt
- CFL number near 1 at failure

---

## 3. Debug Strategy

### Phase 1: Unit Test Each Operator (Week 1, Days 1-2)

**Goal:** Isolate which differential operator is broken

**Tests:**
1. **Gradient:**
   ```python
   f = r**2 + (R0 + r*cos(theta))**2
   grad_f_numerical = solver.gradient(f)
   grad_f_analytical = [2*r, -2*r*(R0+r*cos(theta))*sin(theta), 0]
   assert max_error < 1e-6
   ```

2. **Divergence:**
   ```python
   B = [r, theta, phi]  # Not physical, but known div
   div_B = solver.divergence(B)
   div_B_analytical = 1/r + 1/r  # ∂(rBr)/r∂r + ∂Bθ/r∂θ
   ```

3. **Laplacian:**
   ```python
   f = r**2 * cos(theta)
   lap_f_numerical = solver.laplacian(f)
   lap_f_analytical = 2*cos(theta) - r**2*cos(theta)/...
   ```

**Pass criteria:** Relative error < 1e-4 for all operators

---

### Phase 2: Analytical Solution Validation (Week 1, Days 3-4)

**Test 1: Diffusion Equation**

Initial condition:
$$
B(r, \theta, \phi, t=0) = B_0 \sin(\pi r / a)
$$

Analytical solution (cylindrical limit):
$$
B(r, t) = B_0 \sin(\pi r / a) e^{-\eta (\pi/a)^2 t}
$$

**Implementation:**
```python
dt = 1e-4
for n in range(1000):
    B_new = B + dt * eta * solver.laplacian(B)
    B = B_new
    
    # Compare with analytical
    B_analytical = B0 * np.sin(np.pi * r / a) * np.exp(-eta * (np.pi/a)**2 * n * dt)
    error = np.max(np.abs(B - B_analytical))
    assert error < 1e-3
```

**Pass criteria:** Match analytical solution within 0.1% for 1000 steps

---

**Test 2: Constant Field (Trivial Equilibrium)**

$$
\mathbf{B} = B_0 \hat{\phi}, \quad \mathbf{v} = 0
$$

Should remain constant (no forces, no diffusion for uniform B).

**Pass criteria:** $\max(|B - B_0|) < 10^{-10}$ after 1000 steps

---

### Phase 3: Long-Time Stability (Week 2, Days 1-2)

**Goal:** Ensure no secular growth over 10,000 steps

**Test setup:**
- Grad-Shafranov equilibrium (exact MHD solution)
- Run for $t_{max} = 100 \tau_A$ (100 Alfvén times)
- Monitor:
  - Total energy: $E = \int (B^2 + \rho v^2) dV$
  - div(B) max violation
  - Velocity growth

**Pass criteria:**
- Energy drift < 1% over 100 τ_A
- $\nabla \cdot \mathbf{B} < 10^{-6}$ sustained
- No exponential growth in any field

---

### Phase 4: Convergence Study (Week 2, Days 3-4)

**Grid refinement:**
- Run same test at Nr = 16, 32, 64, 128
- Measure L2 error vs. analytical solution
- Plot log(error) vs. log(dr)

**Expected:** Slope = 2 (2nd-order accurate)

**dt refinement:**
- Fix grid, vary dt = 1e-3, 1e-4, 1e-5, 1e-6
- Error should plateau (spatial error dominates)
- If error keeps shrinking → time integration bug

---

## 4. Test Plan Details

### Test 1: Laplacian of Analytical Functions

**Function suite:**
1. $f_1 = r^2$ → $\nabla^2 f_1 = 4$ (constant)
2. $f_2 = \cos(m\theta)$ → $\nabla^2 f_2 = -m^2/r^2 \cos(m\theta)$
3. $f_3 = (R_0 + r\cos\theta)^2$ → test φ-direction term
4. $f_4 = \sin(k\phi)$ → pure toroidal variation

**Code:**
```python
def test_laplacian_analytical():
    r, theta, phi = solver.get_coordinates()
    
    # Test 1: r²
    f1 = r**2
    lap_f1 = solver.laplacian(f1)
    expected = 4 * np.ones_like(r)
    assert np.allclose(lap_f1, expected, rtol=1e-4)
    
    # Test 2: cos(mθ)
    m = 3
    f2 = np.cos(m * theta)
    lap_f2 = solver.laplacian(f2)
    expected = -m**2 / r**2 * np.cos(m * theta)
    assert np.allclose(lap_f2, expected, rtol=1e-3)
    
    # ... (similar for f3, f4)
```

**Pass/Fail:**
- ✅ Pass: All 4 functions within tolerance
- ❌ Fail: Any function error > 1%

---

### Test 2: Constant Field Preservation

**Setup:**
```python
B = np.zeros((3, Nr, Ntheta, Nphi))
B[2, :, :, :] = 1.0  # Uniform Bφ = 1

v = np.zeros_like(B)
rho = np.ones((Nr, Ntheta, Nphi))

dt = 1e-4
for step in range(1000):
    B, v = solver.step(B, v, rho, dt)
    
    max_deviation = np.max(np.abs(B[2] - 1.0))
    assert max_deviation < 1e-8, f"Step {step}: deviation = {max_deviation}"
```

**Pass/Fail:**
- ✅ Pass: $|B_\phi - 1| < 10^{-8}$ for all 1000 steps
- ❌ Fail: Any deviation > 1e-6

---

### Test 3: Energy Decay for Diffusion

**Physical setup:**
- Initial: Perturbed B field (e.g., $B_r = \epsilon \sin(kr)$)
- No flow: $\mathbf{v} = 0$
- Resistivity: $\eta > 0$

**Expected behavior:**
$$
\frac{dE}{dt} = -\int \eta |\nabla \times \mathbf{B}|^2 dV < 0
$$

**Test:**
```python
E_history = []
for step in range(5000):
    B, v = solver.step(B, v, rho, dt, eta=1e-4)
    E = np.sum(B**2) * dV
    E_history.append(E)
    
    # Energy must decrease
    if step > 0:
        assert E_history[-1] <= E_history[-2], "Energy increased!"
        
# Check monotonic decay
dE = np.diff(E_history)
assert np.all(dE <= 0), "Non-monotonic energy"
```

**Pass/Fail:**
- ✅ Pass: Monotonic energy decay for 5000 steps
- ❌ Fail: Any upward step in energy

---

### Test 4: Cylindrical Limit (R₀/a → ∞)

**Idea:** As R₀ → ∞, toroidal → cylindrical geometry

**Implementation:**
1. Run `ToroidalMHDSolver` with R₀/a = 100 (nearly cylindrical)
2. Run `CylindricalMHDSolver` with same r, θ grid
3. Compare solutions at t = 10 τ_A

**Metric:**
$$
\epsilon_{rel} = \frac{|| B_{toroidal} - B_{cylindrical} ||_2}{|| B_{cylindrical} ||_2}
$$

**Pass/Fail:**
- ✅ Pass: $\epsilon_{rel} < 1/R_0$ (expected scaling)
- ❌ Fail: $\epsilon_{rel} > 10\%$ even with R₀/a = 100

---


---

### Test 5: Curl-Divergence Consistency (小A建议)

**Identity:** $\nabla \cdot (\nabla \times \mathbf{A}) = 0$

For any vector potential $\mathbf{A}$, the magnetic field $\mathbf{B} = \nabla \times \mathbf{A}$ must be solenoidal.

**Test setup:**
```python
# Vector potential from flux function
A_r = 0
A_theta = 0
A_phi = psi(r, theta)  # Poloidal flux

# Compute B = curl(A)
B_r, B_theta, B_phi = curl_toroidal(A, grid)

# Check divergence
div_B = divergence_toroidal(B, grid)

# Pass criterion
assert np.max(np.abs(div_B)) < 1e-10  # Machine precision
```

**Why critical for MHD:**
- $\nabla \cdot \mathbf{B} = 0$ is **fundamental constraint**
- Violation → unphysical monopoles
- Must hold to **machine precision** for curl-derived fields

**小P认可:** 这是MHD最基本的测试,v1.1遗漏了 ⚛️

---

### Test 6: Performance vs Cylindrical Solver (小A建议)

**Goal:** Toroidal overhead should be acceptable (< 2× cylindrical)

**Benchmark setup:**
```python
import time

# Toroidal solver
grid_tor = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128, nphi=16)
solver_tor = ToroidalMHDSolver(grid_tor, dt=1e-4)
solver_tor.initialize(psi0, omega0)

t0 = time.time()
solver_tor.run(n_steps=1000)
t_tor = time.time() - t0

# Cylindrical baseline (v1.0)
grid_cyl = CylindricalGrid(nr=64, ntheta=128, nz=16)
solver_cyl = CylindricalMHDSolver(grid_cyl, dt=1e-4)
solver_cyl.initialize(psi0_cyl, omega0_cyl)

t0 = time.time()
solver_cyl.run(n_steps=1000)
t_cyl = time.time() - t0

# Performance ratio
ratio = t_tor / t_cyl
print(f"Toroidal/Cylindrical: {ratio:.2f}×")

# Pass criterion
assert ratio < 2.0  # Acceptable overhead
```

**Why important:**
- v1.2 must be practical for RL (10k+ env steps)
- If toroidal too slow → training bottleneck
- Acceptable: 1.5-2× slower (extra metric terms)
- Unacceptable: 5-10× slower → implementation bug

---


## 5. Acceptance Criteria

### Mandatory (Must Pass All)

- [ ] **Stability:** 100+ time steps without NaN/Inf for dt = 1e-4, grid 32³
- [ ] **Energy conservation:** Drift < 1% over 100 Alfvén times (ideal MHD)
- [ ] **Divergence-free:** $\max(|\nabla \cdot \mathbf{B}|) < 10^{-6}$ sustained
- [ ] **Grid convergence:** L2 error ∝ dr² (2nd-order scheme verified)
- [ ] **Cylindrical limit:** Match v1.0 cylindrical solver within 1% for R₀/a > 50

### Desirable (Nice to Have)

- [ ] Energy decay rate matches analytical prediction (diffusion test)
- [ ] Works for aspect ratio R₀/a down to 3 (tight tokamak)
- [ ] No explosion for dt up to CFL limit (dt_max ~ dr / v_A)

---

## 6. Implementation Plan

### Week 1: Testing & Diagnosis

**Day 1-2: Unit Tests**
- Implement Test 1 (Laplacian analytical)
- Implement Test 2 (Constant field)
- Run full operator test suite
- **Deliverable:** Test report identifying which operator(s) fail

**Day 3-4: Analytical Validation**
- Implement Test 3 (Energy decay)
- Implement Test 4 (Cylindrical limit)
- **Deliverable:** Hypothesis confirmed (which of H1-H4)

---

### Week 2: Bug Fix & Verification

**Day 1-2: Fix Implementation**
- Correct identified bug(s) in finite-difference stencils
- Update boundary condition handling if needed
- Re-run all Phase 1-2 tests
- **Deliverable:** All unit tests pass ✅

**Day 3: Long-Time Stability**
- Run Phase 3 tests (10,000 steps)
- Monitor energy, div(B), stability
- **Deliverable:** 100% acceptance criteria met

**Day 4: Convergence Study**
- Grid refinement: 16³ → 128³
- dt refinement: 1e-3 → 1e-6
- Verify O(dr²) convergence
- **Deliverable:** Convergence plot, error scaling confirmed

---

### Week 3: Documentation & Integration

**Day 1-2: Code Documentation**
- Add docstrings to all toroidal operators
- Write theory doc: "Finite Differences in Toroidal Geometry"
- **Deliverable:** `docs/theory/toroidal-numerics.md`

**Day 3: Integration Testing**
- Test with RL environment (from v1.2 main branch)
- Run 1000-step episodes
- Verify no crashes
- **Deliverable:** RL-ready solver

**Day 4: Buffer**
- Handle unexpected issues
- Code review
- Prepare v1.2 release notes

---

**Total timeline:** 3-4 weeks (conservative estimate with buffer)

---

## 7. Backup Plans

### Plan A: Unfixable Finite-Difference Bug

**Trigger:** If Week 1-2 fails to fix explosion

**Alternative:** Switch to **spectral method** for toroidal direction
- Fourier series in φ (periodic direction)
- Finite-difference in r, θ
- **Pros:** Spectral accuracy, natural periodicity
- **Cons:** More complex, FFT overhead

**Timeline:** +1 week to implement spectral solver

---

### Plan B: Too Complex for v1.2 Timeline

**Trigger:** If fix requires >4 weeks or major rewrite

**Decision:** **Defer full toroidal to v2.0**
- Use cylindrical solver in v1.2 (already working)
- Document limitations (no grad-B drift, etc.)
- Add to v2.0 roadmap

**Impact:** Reduced physics fidelity, but RL still functional

---

### Plan C: Fundamental Numerical Issue

**Trigger:** If toroidal MHD is inherently unstable with explicit methods

**Alternative:** Switch to **implicit time integration**
- Use PETSc or scipy.sparse solvers
- Implicit-Explicit (IMEX) for stiff terms
- **Pros:** Unconditionally stable
- **Cons:** Slower per step, more dependencies

**Timeline:** +2 weeks for implicit solver integration

---

## 8. Success Metrics

**Quantitative:**
- Solver stability: 1000+ steps without crash
- div(B) violation: < 1e-6 (machine precision limited)
- Energy conservation: < 0.1% drift per 100 τ_A
- Convergence rate: 1.9 < p < 2.1 (for O(dr²) scheme)

**Qualitative:**
- Physics team confident in results
- No workarounds or hacks in production code
- Clear documentation for future debugging

**RL Integration:**
- 1000 episodes without numerical issues
- Toroidal effects observable (e.g., banana orbits)
- Performance acceptable (< 1s per step on M1)

---

## References

1. **Numerical methods:**
   - Hirshman & Whitson (1983) - "Steepest-descent moment method for 3D MHD equilibria"
   - Jardin (2010) - "Computational Methods in Plasma Physics"

2. **Toroidal geometry:**
   - D'haeseleer et al. (1991) - "Flux Coordinates and Magnetic Field Structure"
   - Wesson (2011) - "Tokamaks" (4th ed.), Ch. 2

3. **v1.0 cylindrical solver:**
   - `/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src/solvers/cylindrical_mhd.py`
   - Known-good baseline for comparison

---

**Document Status:** Ready for review  
**Next Action:** Await approval → Begin Week 1 testing

---

_⚛️ Physics correctness is non-negotiable. We fix this properly._
