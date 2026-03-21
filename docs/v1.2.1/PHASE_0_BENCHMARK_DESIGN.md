# Phase 0: Energy Drift Benchmark Design

**Date:** 2026-03-18  
**Author:** 小P ⚛️  
**Purpose:** Design benchmark protocol to verify PyTokEq equilibrium improves energy conservation

---

## Context

**Phase 0.1 Discovery:**
- PyTokEq integration already exists (Phase 2)
- Integration tests: 7/7 PASSED ✅
- Basic physics validated (div_B, q-profile)
- **Missing:** Energy drift benchmark

**v1.2.1 Goal:**
- Verify PyTokEq equilibrium → energy drift < 1% (ideal) or < 5% (acceptable)
- Compare against cylindrical profile baseline (~15% drift)

---

## Benchmark Protocol

### 1. Test Configurations

**Configuration A: Cylindrical Baseline (Control)**
```python
IC: cylindrical_profile(r, theta)
  - ψ = r²(1 - r/a)
  - Simple analytical profile
  - Known: ~15% energy drift over 1000 steps
```

**Configuration B: PyTokEq Solovev (Treatment)**
```python
IC: pytokeq_initial(r, theta, eq_cache)
  - Solovev analytical equilibrium
  - Force balance < 1e-6 (theoretical)
  - div(B) = 0 by construction
```

**Configuration C: PyTokEq M3DC1 (Optional)**
```python
IC: pytokeq_initial(r, theta, eq_cache, profile='M3DC1')
  - Realistic tokamak profile
  - Only if Solovev insufficient
```

---

### 2. Evolution Parameters

**Grid:**
- `nr = 32, ntheta = 64` (standard)
- `R0 = 1.0, a = 0.3`

**Integration:**
- Symplectic IMEX integrator
- `dt = 1e-4` (stable timestep from v1.2)
- `n_steps = 1000`
- Total time: `T = 0.1` (in normalized units)

**Physics:**
- `eta = 1e-4` (resistivity)
- `nu = 1e-4` (viscosity)
- No external forcing (conservation test)

---

### 3. Metrics

**Primary Metric: Relative Energy Drift**
```python
E(t) = E_magnetic(t) + E_kinetic(t)
E_magnetic = ∫ |∇ψ|² dV
E_kinetic = ∫ ω² dV

drift(t) = |E(t) - E(0)| / E(0)

# Report:
drift_max = max(drift(t)) over t ∈ [0, T]
drift_final = drift(T)
```

**Secondary Metrics:**
```python
1. Energy evolution curve: E(t)/E(0) vs t
2. Drift rate: d(drift)/dt
3. Monotonicity: Check if E(t) always decreases (dissipative)
```

**Physics Validation:**
```python
4. div(B) conservation: max|div(B)| over time
5. Enstrophy: ∫ ω² dV evolution
6. NaN/Inf check: Ensure stability
```

---

### 4. Success Criteria

**Tier 1 (Ideal): drift_final < 1%**
- PyTokEq significantly better than cylindrical
- v1.2.1 COMPLETE ✅
- Continue to v1.3

**Tier 2 (Acceptable): 1% ≤ drift_final < 5%**
- Improvement over cylindrical (~15%)
- Document limitation
- Continue to v1.3
- Revisit in v1.4 if needed

**Tier 3 (Needs Debug): drift_final ≥ 5%**
- Investigate causes:
  - Force balance degradation?
  - Grid interpolation error?
  - Integration timestep too large?
- Debug or defer to v1.4

---

### 5. Comparison Analysis

**Required Comparisons:**

**A vs B: PyTokEq Impact**
```python
improvement = (drift_A - drift_B) / drift_A × 100%

Expected: improvement > 50% (15% → < 7.5%)
Best case: improvement > 90% (15% → < 1.5%)
```

**B vs C (Optional): Profile Sensitivity**
```python
# Only if Solovev insufficient
# Check if M3DC1 profile gives better result
```

---

### 6. Diagnostic Output

**Per-run Output:**
```
results/v1.2.1/energy_benchmark/
  ├── config_A_cylindrical/
  │   ├── energy_evolution.npz
  │   ├── metrics.json
  │   └── plot_energy.png
  ├── config_B_solovev/
  │   ├── energy_evolution.npz
  │   ├── metrics.json
  │   └── plot_energy.png
  └── comparison_report.md
```

**metrics.json format:**
```json
{
  "drift_final": 0.0123,
  "drift_max": 0.0145,
  "drift_rate": 1.2e-5,
  "div_B_max": 1.3e-11,
  "runtime_seconds": 45.2,
  "success": true
}
```

---

## Implementation Plan

### Phase 1: Implement Energy Diagnostic

**Files to create/modify:**

**1. New diagnostic module**
```python
# src/pytokmhd/diagnostics/energy_conservation.py

def compute_total_energy(psi, omega, grid):
    """Compute E_mag + E_kin"""
    
def track_energy_drift(solver, n_steps):
    """Run evolution and track drift"""
    
def plot_energy_evolution(times, energies):
    """Visualization"""
```

**2. Benchmark script**
```python
# scripts/v1.2.1_energy_benchmark.py

def run_benchmark(config_name, initial_condition):
    """Run single configuration"""
    
def compare_results(results_A, results_B):
    """Generate comparison report"""
```

---

### Phase 2: Run Benchmark

**Execution:**
```bash
# Config A
python scripts/v1.2.1_energy_benchmark.py --config cylindrical

# Config B
python scripts/v1.2.1_energy_benchmark.py --config solovev

# Compare
python scripts/v1.2.1_energy_benchmark.py --compare
```

**Expected runtime:** ~2-5 minutes per config

---

### Phase 2.5: Cross-Validation by 小A

**小A独立验证:**
1. Re-run both configs
2. Verify metrics match (< 5% difference)
3. Independent visualization
4. Report any discrepancies

---

### Phase 3: Documentation

**Deliverables:**
1. `ENERGY_BENCHMARK_REPORT.md`
   - Results for all configs
   - Comparison analysis
   - Success tier achieved
   
2. Updated `v1.2.1_spec.md`
   - Energy drift verified
   - PyTokEq integration validated

---

## Risk Assessment

**Risk 1: drift_final > 5% even with PyTokEq**

**Possible causes:**
- Force balance degraded during interpolation
- Timestep too large for high-quality IC
- Numerical dissipation from operators

**Mitigation:**
1. Check force balance after interpolation
2. Reduce `dt` (e.g., 5e-5)
3. Verify operator accuracy on PyTokEq grid

**Escalation:** If no improvement, defer to v1.4

---

**Risk 2: Solovev insufficient, need M3DC1**

**Mitigation:**
- Config C already designed
- M3DC1 profile available in PyTokEq
- ~1h additional work

---

**Risk 3: Cross-validation finds discrepancy**

**Mitigation:**
- Small differences (< 5%): acceptable (numerical noise)
- Large differences (> 10%): investigate bugs
- Re-run with fixed seeds for reproducibility

---

## Expected Outcomes

**Best Case:**
- Config B: drift_final < 1% ✅
- Improvement > 90%
- Tier 1 achieved

**Likely Case:**
- Config B: drift_final 2-4% ✅
- Improvement 70-85%
- Tier 2 achieved

**Worst Case:**
- Config B: drift_final 6-10% ⚠️
- Improvement 30-60%
- Tier 3: debug required

---

## Next Steps

**After Phase 0:**
1. YZ approval of benchmark design
2. Phase 1: Implement diagnostics
3. Phase 2: Run benchmark
4. Assess results → determine v1.2.1 completion

---

**Phase 0 Complete** ✅

**Awaiting YZ approval to proceed to Phase 1**

---

**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Status:** Design Complete, Ready for Implementation

---

## Updates Based on 小A Review (2026-03-18 23:00)

### Section 2.5: IC Quality Validation (NEW - 小A建议1)

**Purpose:** Verify grid conversion preserves equilibrium quality

**Phase 0.0 Discovery:** Interpolation error ~12%  
**Risk:** Conversion may degrade physics quality

**Validation Steps:**

**Step 1: Energy Conservation**
```python
# Before conversion (PyTokEq grid)
E_before = compute_total_energy(psi_RZ, grid_pte)

# After conversion (ToroidalGrid)
psi_rtheta = convert_grid(psi_RZ, grid_pte, grid_ptk)
E_after = compute_total_energy(psi_rtheta, grid_ptk)

# Check
energy_error = abs(E_after - E_before) / E_before
assert energy_error < 0.05  # 5% tolerance
```

**Why 5% tolerance?**
- Grid geometry difference (R,Z) → (r,θ)
- Interpolation introduces O(h²) error
- 5% acceptable for IC, should not accumulate

---

**Step 2: Force Balance Preservation**
```python
# Critical: Force balance must remain < 1e-6
FB_before = max(abs(J×B - ∇P))  # On PyTokEq grid
FB_after = max(abs(J×B - ∇P))   # On ToroidalGrid

assert FB_after < 1e-6  # Physics requirement
assert FB_after / FB_before < 10  # Not much worse
```

**Why stricter than energy?**
- Force balance determines equilibrium quality
- If FB degrades → not a true equilibrium IC
- Energy can tolerate grid mismatch, physics cannot

---

**Step 3: div(B) Verification**
```python
# Theoretical: div(B) = 0 by flux function
# Numerical: Check discretization error

div_B = compute_divergence_B(psi_rtheta, grid_ptk)
assert max(abs(div_B)) < 1e-10  # Target
assert max(abs(div_B)) < 1e-8   # Acceptable
```

**Why < 1e-10?**
- Flux function → analytically zero
- Numerical error should be machine precision
- If > 1e-8 → bug in operators

---

**Deliverable:**
```
results/v1.2.1/ic_validation/
  ├── config_B_conversion_report.md
  │   ├── Energy: before/after/error
  │   ├── Force balance: before/after
  │   └── div(B): max value
  └── pass_fail_status.json
```

**Go/No-Go Criteria:**
- ✅ GO: All 3 checks pass
- ⚠️ WARN: Energy error 5-10% → investigate but continue
- ❌ NO-GO: FB > 1e-6 or div(B) > 1e-8 → fix converter

---

### Section 3.1: Monitoring Configuration (NEW - 小A建议2)

**Purpose:** Balance diagnostic completeness vs performance

**Monitoring Frequency:**

```python
monitoring_config = {
    # Primary Metrics (cheap, every step)
    'total_energy': {
        'frequency': 'every step',
        'cost': 'O(N)',  # Just integration
        'storage': 'array[n_steps]'
    },
    
    # Secondary Metrics (moderate, periodic)
    'energy_components': {
        'frequency': 'every 50 steps',
        'cost': 'O(N)',
        'components': ['E_mag', 'E_kin', 'E_dissipation']
    },
    
    'div_B': {
        'frequency': 'every 100 steps',  # Expensive!
        'cost': 'O(N) + gradient ops',
        'reason': 'Check constraint drift'
    },
    
    # Tertiary Metrics (expensive, sparse)
    'force_balance': {
        'frequency': 'step 0 only',  # Very expensive
        'cost': 'O(N) + curl ops',
        'reason': 'Validate IC only'
    },
    
    'enstrophy': {
        'frequency': 'every 100 steps',
        'cost': 'O(N)',
        'purpose': 'Alternative conservation check (小A建议)'
    }
}
```

**Alert Thresholds:**

```python
thresholds = {
    # Energy drift (primary)
    'energy_drift_tier1': 0.01,    # 1% → Tier 1 success
    'energy_drift_tier2': 0.05,    # 5% → Tier 2 acceptable
    'energy_drift_fail': 0.10,     # 10% → Investigation required
    
    # div(B) (secondary)
    'div_B_target': 1e-10,         # Target (analytical)
    'div_B_warn': 1e-8,            # Warning level
    'div_B_fail': 1e-6,            # Failure (physics broken)
    
    # Enstrophy (小A alternative metric)
    'enstrophy_increase': False,   # Should always decrease!
    'enstrophy_rate_warn': -1e-3   # If decaying too fast
}
```

**Performance Estimate:**

```python
# 1000 steps benchmark
baseline_time = 45s  # No diagnostics

overhead = {
    'energy_every_step': +5s,       # ~10%
    'div_B_every_100': +3s,         # ~7%
    'enstrophy_every_100': +2s,     # ~4%
    'total': +10s                   # ~22% overhead
}

expected_runtime = 55s per config
```

**Trade-off Decision:**
- div(B) every 100 steps: Good balance
- If > 1e-8 detected → run again with `every 10 steps` for diagnosis
- Don't monitor every step unless debugging

---

### Section 6: Phase 2.5 Cross-Validation Protocol (NEW - 小A建议3)

**Purpose:** Independent verification by 小A to ensure robustness

**小A's Validation Checklist:**

---

**Check 1: Independent Re-run (MANDATORY)**

```python
# 小A independently runs Config A and B
# Using same parameters, different code execution

# Success Criteria:
tolerance_drift = 0.001  # 0.1% absolute
tolerance_energy = 0.01  # 1% relative

小P_drift_A = 0.150
小A_drift_A = 0.148  # Difference: 0.002 → PASS (< 0.1%)

小P_drift_B = 0.035
小A_drift_B = 0.038  # Difference: 0.003 → PASS (< 0.1%)

# If差异 > 0.1%:
# → Check random seed
# → Check numerical precision
# → Report discrepancy
```

**Escalation Path:**
- < 0.1% difference: ✅ PASS
- 0.1-0.5% difference: ⚠️ Investigate (可能numerical noise)
- \> 0.5% difference: ❌ FAIL → 找bug

---

**Check 2: Energy Curve Monotonicity (MANDATORY)**

```python
# Physics: Symplectic + dissipative → E(t) should monotonically decrease

E_curve = [E(t) for t in times]
dE = np.diff(E_curve)

# Check 1: Monotonic decrease?
monotonic = all(dE <= 0)
if not monotonic:
    violations = np.where(dE > 0)[0]
    print(f"⚠️ Energy increased at steps: {violations}")
    # → Possible symplectic破坏 or bug

# Check 2: Smooth decay?
d2E = np.diff(dE)
oscillations = sum(np.sign(d2E[:-1]) != np.sign(d2E[1:]))
if oscillations > 10:
    print(f"⚠️ Energy curve has {oscillations} oscillations")
    # → Possible numerical instability
```

**小A's Plot:**
- Overlay 小P's E(t) and 小A's E(t)
- Visual check for agreement
- Highlight discrepancies

---

**Check 3: Alternative Metrics (RECOMMENDED)**

**小A独立计算:**

**Enstrophy Conservation:**
```python
# Ω(t) = ∫ ω² dV
# Should: dΩ/dt ≤ 0 (dissipative system)

Omega_curve = [compute_enstrophy(omega) for omega in states]
dOmega = np.diff(Omega_curve)

# Physics Check:
assert all(dOmega <= 0), "Enstrophy should decay!"

# Cross-check with energy:
# If E conserved但Ω不是 → 可能物理错误
```

**Magnetic Helicity (if applicable):**
```python
# H = ∫ A·B dV
# Ideal MHD: conserved
# Resistive MHD: slowly decaying

# 这个check validates physics consistency
# 如果E/Ω都对但H错 → 某个operator有问题
```

**Why important:** (小A的insight)
- Energy可能accidentally conserved (wrong reasons)
- Multiple conservation laws → robust validation
- Physics consistency check

---

**Check 4: Edge Case Test (RECOMMENDED)**

**小A设计Config D:**
```python
# Extreme parameters to test robustness

Config D options:
1. High beta: beta_p = 1.0 (vs 0.1 baseline)
   - Test: 是否仍然stable?
   
2. Low resistivity: eta = 1e-6 (vs 1e-4 baseline)
   - Test: 是否energy drift更小?
   
3. Large perturbation: amp = 0.1 (vs 0.01 baseline)
   - Test: Benchmark robust to IC perturbation?
```

**Purpose:**
- Verify benchmark不是偶然在特定参数work
- Find parameter sensitivity
- Discover potential failure modes

**Deliverable:**
```
results/v1.2.1/edge_cases/
  ├── config_D_high_beta.json
  ├── config_D_low_eta.json
  └── sensitivity_analysis.md
```

---

**Phase 2.5 Success Criteria:**

**小A给出verdict:**

**✅ APPROVED:**
- Check 1: Difference < 0.1%
- Check 2: Monotonic E(t)
- Check 3: Enstrophy/Helicity consistent
- Check 4: Edge cases stable

**⚠️ APPROVED with Notes:**
- Check 1: Difference 0.1-0.5% (numerical noise documented)
- Check 2: Minor oscillations (< 1% amplitude)
- Check 3: Alternative metrics略有差异 (documented)

**❌ REJECTED:**
- Check 1: Difference > 0.5% → Bug!
- Check 2: Energy increasing → Physics broken
- Check 3: Conservation violated → Operator bug

**Escalation:**
- If rejected → 小P修复 → 小A re-validate

---

## Summary of Updates

**小A Review采纳:**
1. ✅ Section 2.5: IC Quality Validation (建议1)
2. ✅ Section 3.1: Monitoring Configuration (建议2)
3. ✅ Section 6: Phase 2.5 Cross-Validation Protocol (建议3)

**Impact:**
- Design completeness: 8/10 → 10/10
- Robustness: Significantly improved
- Clarity: Validation criteria明确

**Time Investment:** +1h documentation  
**Time Saved:** Prevent Phase 2 debug时间 (估计2-4h)

**ROI:** Positive ✅

---

**Updated by:** 小P ⚛️  
**Review by:** 小A 🤖  
**Date:** 2026-03-18 23:03  
**Status:** Design Complete, Ready for Phase 1 Implementation

