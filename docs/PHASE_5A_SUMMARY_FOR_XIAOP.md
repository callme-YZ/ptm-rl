# Phase 5A Validation Summary for 小P

**Date:** 2026-03-20  
**From:** 小P subagent (Phase 5A validation)  
**To:** 小P (main)  
**Subject:** v1.4.0 Physics Validation Results - BLOCKER FOUND

---

## TL;DR

❌ **v1.4.0 NO-GO:** Critical energy conservation bug in ideal MHD (1.4% drift vs <1e-6 target)

---

## What I Did

Executed Phase 5A comprehensive physics validation test suite as requested.

**Tests Completed:**
- ✅ Test 1.1: Energy conservation (ideal MHD) - **FAILED**
- ✅ Test 1.2: J_ext=0 sanity check - **FAILED** (same issue)
- ✅ Test 2.1: Energy dissipation (resistive MHD) - **PASSED**
- ✅ Test 4.1: J_ext energy injection - **PASSED**
- ⏸️ Tests 1.3, 2.2, 3.1, 5.1, 6.1: **SKIPPED** (blocked by Test 1.1 failure)

**Total Runtime:** ~40 minutes (including failed tests and plot generation)

---

## Critical Finding

### 🔴 BLOCKER: Energy Conservation Failure

**Test 1.1 Result:**
```
Setup:
- IC: Simple single-mode ψ(r,θ,ζ) = 0.01·r·(1-r)·cos(θ+ζ)
- Grid: 32×64×128
- dt=0.005, n_steps=200 (T=1.0)
- η=0 (ideal MHD), J_ext=None

Result:
  H(0)     = 5.00768e-03
  H(t=1.0) = 4.93786e-03
  |ΔH/H₀|  = 1.39e-02 (1.4% drift)
  
Expected: |ΔH/H₀| < 1e-6
Actual:   |ΔH/H₀| = 1.4e-02 (~10⁴× worse)

Status: ❌ FAIL
```

**Plot:** `docs/validation/phase5a/test_1_1_simple_energy_conservation.png`

**Analysis:**
- Energy decays monotonically (not oscillating) → consistent numerical dissipation
- CFL = 0.017 (safe) → not a stability issue
- Independent of J_ext handling (Test 1.2 confirms)
- **Root cause:** IMEX scheme or Poisson bracket discretization

---

## What Works ✅

1. **Resistive MHD (Test 2.1):** Energy dissipates monotonically as expected (53.7% over t=1.0 with η=1e-4)
2. **J_ext injection (Test 4.1):** Energy increases correctly (~20% with J_ext=0.01)

**Implication:** The problem is specific to **ideal MHD energy conservation**, not the resistive term or J_ext.

---

## Suspected Bugs

### Bug #1: Energy Conservation (BLOCKER)

**Likely Locations:**
1. `src/pytokmhd/operators/poisson_bracket_3d.py` - Arakawa scheme implementation
2. `src/pytokmhd/solvers/imex_3d.py` - IMEX time step logic
3. `src/pytokmhd/solvers/imex_3d.py` - Helmholtz solver boundary conditions

**Recommended Investigation:**
1. **Unit test Poisson bracket:** [ψ,ω] should conserve energy (antisymmetric property)
2. **Unit test IMEX with η=0:** (I - 0·∇²)φ = φ should be identity
3. **Unit test Helmholtz BC:** Check if Dirichlet BC at r=0, r=r_max introduces dissipation

---

### Bug #2: CFL Computation (Minor, Non-blocker)

**Location:** `src/pytokmhd/solvers/imex_3d.py:588`

**Issue:**
```python
v_max = np.max(np.abs(np.gradient(psi, axis=0))) / grid.dr  # WRONG
```

`np.gradient` already divides by `dr`, so this double-divides → wrong units.

**Fix:**
```python
v_max = np.max(np.abs(np.gradient(psi, grid.dr, axis=0)))  # CORRECT
```

**Impact:** Only affects diagnostic warning (not physics), but caused earlier test to overflow.

---

## Deliverables

**Created Files:**
1. `tests/validation/test_phase5_physics_simple.py` - Simplified test suite (stable ICs)
2. `docs/PHASE_5A_PHYSICS_VALIDATION_REPORT.md` - Full validation report
3. `docs/PHASE_5A_SUMMARY_FOR_XIAOP.md` - This summary
4. `docs/validation/phase5a/*.png` - 4 plots

**Logs:**
- `phase5a_test_output.log` - Full pytest output (original ballooning IC tests)
- `phase5a_simple_test.log` - Simplified IC test output

---

## Recommendation

### v1.4.0 Release Decision

**❌ NO-GO**

**Reasoning:**
- P0 critical test failed (energy conservation)
- This is **fundamental physics correctness**, not an edge case
- Shipping with this bug would:
  - Corrupt RL training (小A's work)
  - Produce scientifically invalid results
  - Undermine project credibility

**What You Need to Do:**

1. **Fix energy conservation bug** (estimated: 2-4 hours)
   - Debug Poisson bracket (priority #1)
   - Debug IMEX time step (priority #2)
   - Debug Helmholtz solver (priority #3)

2. **Patch CFL computation** (5 minutes)

3. **Re-run validation** (call this subagent again)
   - `sessions_spawn(task="Phase 5A validation (re-run after fix)")`
   - Target: All P0 tests PASS

4. **If all P0 pass → GO for v1.4.0**

---

## Physics Implications

**For RL Training (小A):**

If v1.4 ships with this bug:
- RL policy will learn **numerical dissipation**, not real physics
- Energy-based rewards will be **systematically biased**
- Trained policies will **not generalize** to real tokamaks

**Do NOT start Phase 5B (RL validation) until this is fixed.**

---

## What I Learned

### Test Design Insights

1. **Simplified ICs crucial:** Full ballooning mode IC (from `ic/ballooning_mode.py`) caused numerical overflow, masking the real bug. Simple single-mode IC isolated the energy conservation issue.

2. **P0 test prioritization saved time:** Running energy conservation first immediately identified the blocker, avoiding wasted effort on downstream tests.

3. **Unit tests should come first:** Should have unit-tested Poisson bracket and IMEX step before full 3D integration.

---

## Acceptance Criteria Status

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| **Test Coverage** | | | |
| P0 tests executed | All (3) | 3/3 ✅ | ✅ |
| P1 tests executed | ≥2 | 1/3 ⚠️ | ⚠️ (blocked) |
| Results documented | Yes | Yes ✅ | ✅ |
| **Pass Rate** | | | |
| P0 tests PASS | 100% | **33%** ❌ | ❌ **BLOCKER** |
| P1 tests PASS | ≥50% | 100% (of 1) ✅ | ✅ |
| **Documentation** | | | |
| Validation report | Complete | ✅ | ✅ |
| Test scripts committed | Yes | ✅ | ✅ |
| Plots generated | Yes | ✅ (4 plots) | ✅ |
| **Decision** | | | |
| Go/No-go | Clear | **NO-GO** ❌ | ✅ |

**Overall:** 3/4 criteria met, but **P0 pass rate failure is a blocker**.

---

## Next Steps

**For You (小P):**
1. Read full report: `docs/PHASE_5A_PHYSICS_VALIDATION_REPORT.md`
2. Debug energy conservation (start with Poisson bracket unit test)
3. Fix bugs
4. Re-run validation (spawn new subagent or run `pytest tests/validation/test_phase5_physics_simple.py -v`)
5. If PASS → update report → declare GO for v1.4.0

**For 小A:**
- **BLOCKED:** Do not start Phase 5B (RL validation) until physics layer passes Phase 5A

**For ∞:**
- **FYI:** v1.4.0 release delayed pending physics bug fix

---

**End of Summary**

---

**Subagent Status:** Task complete, results delivered.  
**Your call now.**
