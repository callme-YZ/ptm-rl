# Stage 2 Verification Test Suite

**Issue #23 - Numerical Verification**

Author: 小P ⚛️  
Executor: 小A 🤖  
Date: 2026-03-23

---

## Overview

Stage 2 验证 Stage 1 理论推导的数值正确性。

**Goal:** 证明 PyTokMHD 实现符合 Pseudo-Hamiltonian 理论。

---

## Test Files

### Task 2.1: Energy Conservation (`test_conservation.py`)

**Tests:**
1. **Short run (100 steps):** Quick validation, |dH/dt| < 1e-10
2. **Long run (1000 steps):** Stage 2 requirement, |dH/dt| < 1e-12
3. **Resistive dissipation:** η>0 case, verify dH/dt < 0

**Expected Results:**
- Ideal (η=0, ν=0): Energy conserved to machine precision
- Resistive (η>0): Energy decreases monotonically
- No secular drift over 1000 steps

**Run:**
```bash
cd tests/stage2_verification
python test_conservation.py
```

**Output:**
- Console: Test results and verdicts
- Plot: `stage2_energy_conservation.png`

---

### Task 2.2: Poisson Bracket Properties (`test_poisson_bracket.py`)

**Tests:**
1. **Antisymmetry:** {F,G} = -{G,F}
2. **Jacobi identity:** {{F,G},H} + cyclic = 0
3. **Leibniz rule:** {F,GH} = {F,G}H + G{F,H}
4. **Energy bracket:** {H,H} = 0 (from Stage 1 proof)

**Expected Results:**
- All properties satisfied to < 1e-12 (1e-10 for Jacobi)
- Confirms Morrison bracket implementation correct

**Run:**
```bash
python test_poisson_bracket.py
```

**Output:**
- Console: Property verification results

---

### Task 2.3: Integrator Comparison (`test_integrators.py`)

**Integrators tested:**
1. **RK2 (Baseline):** Current PyTokMHD implementation
2. **Symplectic Euler:** First-order symplectic
3. **Störmer-Verlet:** Second-order symplectic

**Expected Results:**
- Symplectic integrators: Better long-term energy conservation
- Quantify improvement vs RK2
- Inform Issue #26 (symplectic integrator implementation)

**Run:**
```bash
python test_integrators.py
```

**Output:**
- Console: Comparison table
- Plot: `stage2_integrator_comparison.png`

---

## Running All Tests

**Quick validation:**
```bash
python test_conservation.py  # ~1 min
python test_poisson_bracket.py  # ~10 sec
```

**Full suite:**
```bash
python test_integrators.py  # ~2-3 min
```

**Total runtime:** ~5 min

---

## Success Criteria

**Stage 2 PASS if:**
- ✅ Task 2.1: All 3 conservation tests pass
- ✅ Task 2.2: All 4 bracket properties verified
- ✅ Task 2.3: Integrator comparison complete + recommendation

**If any FAIL:**
- Investigate PyTokMHD implementation
- May indicate bug in Morrison bracket or integrator
- **Do not proceed to Stage 3 until resolved** 🔴

---

## Metrics Reference

**From Stage 1 Theory:**

| Metric | Ideal (η=0) | Resistive (η>0) |
|--------|-------------|-----------------|
| dH/dt | 0 (exact) | -∫[η\|∇J\|² + ν\|∇ω\|²]dV < 0 |
| {H,H} | 0 (antisymmetry) | 0 (still antisymmetric) |
| Volume | Preserved (symplectic) | Shrinks (dissipative) |

**Numerical Thresholds:**
- Energy conservation: < 1e-12 (long-term)
- Poisson bracket: < 1e-12 (antisymmetry, Leibniz), < 1e-10 (Jacobi)
- Integrator drift: Compare relative to RK2 baseline

---

## For 小A 🤖

**Execution Steps:**

1. **Setup:**
   ```bash
   cd /Users/yz/.openclaw/workspace-xiaop/pim-rl-v3.0
   git checkout v3.0-phase1
   ```

2. **Run tests sequentially:**
   - Start with `test_conservation.py` (validates basic setup)
   - Then `test_poisson_bracket.py` (checks structure)
   - Finally `test_integrators.py` (performance comparison)

3. **Review results:**
   - Check console for ✅/❌ verdicts
   - Inspect plots for visual confirmation
   - Note any failures → discuss with 小P before proceeding

4. **Report to GitHub:**
   - Update Issue #23 with Stage 2 results
   - Attach plots if tests pass
   - List any failures for 小P investigation

5. **If all pass:**
   - Confirm Stage 2 complete ✅
   - Ready for Stage 3 (dissipation characterization)

---

## Troubleshooting

**Common issues:**

**Import errors:**
```bash
# Ensure PyTokMHD is in path
export PYTHONPATH=/path/to/pim-rl-v3.0/src:$PYTHONPATH
```

**Numerical failures:**
- Check grid resolution (nr=32, ntheta=32 should work)
- Verify dt=1e-3 (too large → instability)
- Ensure η=0, ν=0 for ideal tests

**Performance:**
- 1000-step runs may take 1-2 min on laptop
- Use `n_steps=100` for quick debugging
- Full suite ~5 min total

---

## Next Steps

**After Stage 2:**

**If PASS:**
- Stage 3: Dissipation characterization (resistive case validation)
- Stage 4: Synthesis + RL integration recommendations
- Issue #23 → Close after Stage 4

**If FAIL:**
- Debug PyTokMHD implementation
- Re-run tests after fixes
- **Do not proceed until Stage 2 passes** 🔴

---

**小P prepared this suite for 小A** ⚛️🤖

**Tests reflect Stage 1 theoretical predictions**

**Numerical verification = Foundation for Phase 2 Hamiltonian RL** ✅
