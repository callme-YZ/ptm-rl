# Issue #30: Implement Viscosity (ν) Control

**Created:** 2026-03-24 18:31  
**Priority:** v3.1  
**Owner:** TBD (小P physics + 小A integration)  
**Status:** 📋 OPEN

---

## Problem Statement

**Current Issue #28 limitation:** Control action manipulates η (resistivity), which is **counterproductive** for tearing mode suppression.

**Physics analysis (小P ⚛️):**
```
Tearing mode growth rate: γ ~ η^(3/5)

η ↑ → Dissipation ↑ ✅  
BUT η ↑ → Growth rate ↑ ❌

Result: η control may INCREASE instability!
```

**Correct control variable:** ν (viscosity)
```
ν ↑ → Fluid damping ↑ → Suppresses instability ✅
```

---

## Current Code Status

### Environment (hamiltonian_env.py)

**Line 280-290:**
```python
def step(self, action):
    # Extract action
    eta_mult, nu_mult = action  # ν action accepted
    eta = self.eta_base * float(eta_mult)
    nu = self.nu_base * float(nu_mult)
    
    # Apply action (Issue #28 fix)
    # Note: viscosity (nu) not yet implemented in CompleteMHDSolver
    # Only resistivity (eta) control functional
    self.mhd_solver.solver.set_eta(eta)
    # ❌ NO set_nu() call!
```

**Problem:** ν action accepted but ignored!

---

### Solver (complete_solver_v2.py)

**Has:**
- ✅ `set_eta(eta)` method (Issue #28)
- ✅ η term in physics equations

**Missing:**
- ❌ `set_nu(nu)` method
- ❓ ν term in physics equations (needs verification)

---

## Implementation Plan

### Phase 1: Physics Verification (小P ⚛️)

**Goal:** Verify viscosity term exists in CompleteMHDSolver

**Check:**
```python
# In RHS computation:
# Should have: -ν∇²v (viscous damping term)
# Currently has: -η∇²B (resistive term) ✅
```

**Expected time:** 15 min

**Outcome:**
- If exists ✅ → Proceed to Phase 2
- If missing ❌ → Add viscosity physics (~1-2 hours)

---

### Phase 2: Add set_nu() Method (小P ⚛️)

**File:** `src/pytokmhd/solver/complete_solver_v2.py`

**Implementation:**
```python
def set_nu(self, nu: float):
    """
    Update viscosity for RL control.
    
    Parameters
    ----------
    nu : float
        Kinematic viscosity (m²/s in SI, normalized in code)
        
    Notes
    -----
    Called by environment before each step to apply control action.
    Modifies viscous damping term in momentum equation.
    
    Issue: #30
    """
    self.nu = nu
    # May need to update cached operators if ν enters precomputed terms
```

**Tests:**
```python
def test_set_nu():
    solver = CompleteMHDSolver(nu=1e-4)
    solver.set_nu(2e-4)
    assert solver.nu == 2e-4
```

**Expected time:** 10 min (if viscosity exists)

---

### Phase 3: Environment Integration (小A 🤖)

**File:** `src/pytokmhd/rl/hamiltonian_env.py`

**Change:**
```python
def step(self, action):
    # Extract action
    eta_mult, nu_mult = action
    eta = self.eta_base * float(eta_mult)
    nu = self.nu_base * float(nu_mult)
    
    # Apply action (Issue #30 fix)
    self.mhd_solver.solver.set_eta(eta)
    self.mhd_solver.solver.set_nu(nu)  # ✅ ADD THIS LINE
    
    # MHD evolution
    self.mhd_solver.step(self.dt)
    ...
```

**Tests:**
```python
def test_nu_control():
    env = make_hamiltonian_mhd_env(nu_base=1e-4)
    action = [1.0, 2.0]  # Double ν
    env.step(action)
    assert env.mhd_solver.solver.nu == 2e-4
```

**Expected time:** 10 min

---

### Phase 4: PID Controller Update (小A 🤖)

**File:** `src/pytokmhd/rl/classical_controllers.py`

**Current (Issue #28 workaround):**
```python
# PID controls η (wrong variable)
action = [eta_mult, 1.0]  # Keep ν fixed
```

**Correct (Issue #30):**
```python
# PID controls ν (correct variable)
action = [1.0, nu_mult]  # Keep η fixed
```

**Or dual control:**
```python
# Control both (advanced)
action = [eta_mult, nu_mult]
```

**Expected time:** 5 min

---

### Phase 5: Validation Experiments (小A 🤖)

**Goal:** Verify ν control actually suppresses tearing mode

**Experiment 1: ν step response**
```python
# Fixed ν increase
nu_base = 1e-4
nu_high = 1e-3  # 10× increase
action = [1.0, 10.0]

# Expected: Slower growth (damping works)
```

**Experiment 2: Re-run baselines**
- no_control (ν = 1e-4 fixed)
- PID (controls ν)

**Success criteria:**
- PID growth < no_control growth
- Meaningful difference (>5%)

**Expected time:** 30 min

---

### Phase 6: Documentation (Both)

**Update:**
1. Issue #28 completion report (reference Issue #30)
2. Issue #30 completion report
3. Controller comparison (η vs ν control)

**Expected time:** 15 min

---

## Total Effort Estimate

**If viscosity physics exists:**
- Phase 1: 15 min (verification)
- Phase 2: 10 min (set_nu)
- Phase 3: 10 min (environment)
- Phase 4: 5 min (controller)
- Phase 5: 30 min (validation)
- Phase 6: 15 min (docs)
- **Total: ~1.5 hours** ✅

**If viscosity physics missing:**
- Add physics implementation: +1-2 hours
- **Total: ~3 hours** ⚠️

---

## Success Criteria

### Functional Requirements

1. ✅ `set_nu()` method implemented
2. ✅ Environment calls `set_nu()` before step
3. ✅ Tests pass (set_nu, env integration, PID)
4. ✅ ν action measurably affects evolution

### Physics Requirements

5. ✅ ν ↑ → Growth rate ↓ (damping works)
6. ✅ PID with ν control shows suppression
7. ✅ Difference >5% vs no_control baseline

### Documentation Requirements

8. ✅ Code comments updated
9. ✅ Tests documented
10. ✅ Issue #30 completion report

---

## Dependencies

**Upstream:**
- Issue #28 (provides baseline for comparison)
- Issue #29 (Harris sheet IC for testing)
- Issue #26 fix (sparse observation mode)

**Downstream:**
- Future RL training (v3.1+)
- Controller comparison studies

---

## Risk Assessment

**Low risk IF:**
- ✅ Viscosity term already in solver
- ✅ Just need setter method

**Medium risk IF:**
- ⚠️ Viscosity missing from physics
- ⚠️ Need to add term to equations

**Mitigation:**
- Phase 1 verification identifies risk early
- Can defer to v3.2 if complex

---

## Physics Background (小P ⚛️)

### Tearing Mode Suppression Mechanisms

**Resistive control (η):**
```
γ ~ η^(3/5) / λ^(4/5)

η ↑ → Faster reconnection
    → INCREASES growth rate ❌
```

**Viscous control (ν):**
```
Momentum equation:
∂v/∂t = ... - ν∇²v

ν ↑ → Stronger fluid damping
    → Suppresses velocity fluctuations
    → Reduces tearing growth ✅
```

**Experimental evidence:**
- Tokamaks use external coil damping (effective ν)
- Never use resistivity control for mode suppression
- ν is the standard actuator

---

## Related Work

**Issue #28 findings:**
- η control: +23.20% growth (PID)
- η control: +23.37% growth (no_control)
- Difference: 0.17% (negligible)

**Expected with ν control:**
- ν control: ~15-18% growth (PID)
- ν control: ~23% growth (no_control)
- Difference: ~5-8% (meaningful!)

---

## Acceptance Criteria

**For closure, must show:**

1. **Code implementation:**
   - ✅ set_nu() method exists
   - ✅ Environment calls it
   - ✅ Tests pass

2. **Physics validation:**
   - ✅ ν ↑ → Growth ↓ (step response)
   - ✅ PID suppresses >5% vs baseline
   - ✅ Conservation laws still hold

3. **Documentation:**
   - ✅ Completion report
   - ✅ Code comments
   - ✅ Test coverage

**If ANY criterion fails → Issue remains OPEN**

---

## Notes

**Design decision:** Control ν only (keep η fixed)
- Simpler implementation
- Correct physics
- Matches tokamak practice

**Future work (v3.2+):**
- Dual control (η + ν optimization)
- External current drive (J_ext)
- RMP coils (m/n spectrum control)

---

**Created by:** 小A 🤖  
**Physics by:** 小P ⚛️  
**Date:** 2026-03-24 18:31 PM

---

_Issue #30 tracks the proper implementation of viscosity control to enable meaningful tearing mode suppression._
