# Phase 1: Hamiltonian Structure Verification for v3.0

**Author:** 小P ⚛️  
**Date:** 2026-03-23  
**Phase:** Phase 1 - Foundation & Verification  
**Issue:** #23 (P0-critical)  
**Status:** Design Document

---

## Executive Summary

**Purpose:** Verify Morrison bracket symplectic structure to gate Hamiltonian RL exploration in v3.0

**Critical Question:** Is PyTokMHD's Morrison bracket implementation suitable for Hamiltonian RL?

**Deliverables:**
1. **Classification:** True Hamiltonian vs Pseudo-Hamiltonian (with dissipation)
2. **Numerical verification:** Long-term energy conservation, symplectic structure tests
3. **RL Integration Guidance:** What's safe to use, what needs modification

---

## 1. Background & Motivation

### 1.1 Why This Matters for v3.0

**v3.0 Goal:** First competitive structure-preserving RL framework for MHD control

**Hamiltonian RL Foundation:**
- HNN (Hamiltonian Neural Networks) assumes symplectic dynamics
- Structure-preserving RL depends on verified conservation laws
- **Without verification → RL may learn wrong physics** 🔴

**Risk if we skip this:**
- Train HNN on non-Hamiltonian system → model learns inconsistent dynamics
- Policy exploits numerical artifacts instead of physics
- Debug RL failures that are actually unverified physics assumptions

### 1.2 小A's v2.1 Experience (Critical Lesson)

**Issue #6 from v2.1:** Hamiltonian RL failed

**What went wrong:**
- Attempted HNN without verifying symplectic structure
- Unclear if failures due to:
  - RL algorithm choice (SAC vs PPO)
  - HNN architecture
  - **OR underlying physics not Hamiltonian** ← This!

**小A's conclusion (2026-03-22):**
> "MHD is NOT a Hamiltonian system (has resistivity dissipation)"
> "Energy is NOT conserved"
> "H(z,a) should be **pseudo-Hamiltonian** or control Lyapunov function"

**This verification prevents v3.0 from repeating v2.1 mistake** ✅

### 1.3 Current Status

**What we have:**
- ✅ Morrison bracket implemented (`toroidal_bracket.py`)
- ✅ Energy conservation: 0.0000% drift over 100 steps (v2.0 validation)
- ✅ Basic Poisson bracket tests (`test_hamiltonian.py`)

**What's MISSING:**
- ❌ Formal proof of symplectic structure
- ❌ Long-term tests (1000+ steps) to verify conservation doesn't degrade
- ❌ Classification: True vs Pseudo-Hamiltonian (conservative vs dissipative decomposition)
- ❌ Guidance for RL: Which parts can use standard HNN, which need modification

**This is the gate for all Phase 2 Hamiltonian RL work** 🚪

---

## 2. Theoretical Framework

### 2.1 Reduced MHD Equations

**Full system (toroidal geometry):**

```
∂ψ/∂t = [φ, ψ] + η J_∥          (flux evolution)
∂ω/∂t = [φ, ω] + [J_∥, ψ] + ν ∇²ω   (vorticity evolution)
```

**Where:**
- **ψ**: Poloidal magnetic flux
- **ω = ∇²φ**: Vorticity
- **φ**: Electrostatic potential (stream function)
- **J_∥ = -∇²ψ**: Parallel current
- **[f,g]**: Poisson bracket = (1/R)(∂_R f ∂_Z g - ∂_Z f ∂_R g)
- **η**: Resistivity (dissipation)
- **ν**: Viscosity (dissipation)

### 2.2 Hamiltonian Energy Functional

**Total energy:**

```
H[ψ, ω] = ∫ [1/2 |∇φ|² + 1/2 |∇ψ|²] √g dV

where:
  - |∇φ|² = kinetic energy density (fluid motion)
  - |∇ψ|² = magnetic energy density
  - √g = R (toroidal Jacobian, simplified)
```

**Energy conservation (ideal case, η=ν=0):**

```
dH/dt = {H, H} = 0
```

**Energy dissipation (resistive case, η>0):**

```
dH/dt = -∫ [η |∇J_∥|² + ν |∇ω|²] dV < 0
```

**Key distinction:**
- **True Hamiltonian:** dH/dt = 0 exactly (ideal MHD)
- **Pseudo-Hamiltonian:** dH/dt < 0 (resistive MHD, energy decreases)

### 2.3 Morrison Bracket (Non-Canonical Poisson Structure)

**General Poisson bracket for functionals F, G:**

```
{F, G} = ∫∫ [δF/δψ · [δG/δω, ψ] + δF/δω · [ψ, δG/δψ]] √g dV
```

**This is Lie-Poisson bracket** (not canonical {q,p} form!)

**Key properties to verify:**

1. **Antisymmetry:** {F, G} = -{G, F}
2. **Jacobi identity:** {{F,G},H} + {{G,H},F} + {{H,F},G} = 0
3. **Leibniz rule:** {F, GH} = {F,G}H + G{F,H}

**Symplectic structure consequence:**

If Poisson bracket satisfies above 3 properties, the flow preserves:
- Phase space volume (Liouville theorem)
- Symplectic 2-form ω = dψ ∧ dω

**This is what enables HNN to work!**

### 2.4 Decomposition Strategy

**Following Chacón (2020) - Energy-Conserving Methods:**

```
Total dynamics = Conservative part + Dissipative part

∂ψ/∂t = [φ, ψ]       + η J_∥
        ↑ Hamiltonian   ↑ Dissipative

∂ω/∂t = [φ,ω] + [J_∥,ψ]  + ν ∇²ω
        ↑ Hamiltonian      ↑ Dissipative
```

**Implication for RL:**
- **Conservative part:** Can use standard HNN (symplectic-preserving)
- **Dissipative part:** Need modified architecture (e.g., port-Hamiltonian, PINN with dissipation)

**Verification goal:** Confirm conservative part IS truly Hamiltonian (symplectic)

---

## 3. Verification Plan

### 3.1 Classification: True vs Pseudo-Hamiltonian

**Task 1.1: Code Decomposition**

**Examine:** `PyTokMHD` resistive MHD solver

**Check:**
1. Does code split conservative vs dissipative terms?
2. η, ν parameters - can they be set to zero?
3. What's the default mode: ideal (η=0) or resistive (η>0)?

**Expected finding:**
- PyTokMHD likely has resistivity η > 0 by default
- → System is **pseudo-Hamiltonian** (not true Hamiltonian)

**Deliverable:** Classification report (`classification.md`)

---

**Task 1.2: Theoretical Decomposition**

**Derive:** Explicit split of equations

```
Ideal part (Hamiltonian):
  ∂ψ/∂t |_ideal = [φ, ψ]
  ∂ω/∂t |_ideal = [φ, ω] + [J_∥, ψ]

Dissipative part:
  ∂ψ/∂t |_diss = η J_∥
  ∂ω/∂t |_diss = ν ∇²ω
```

**Verify:**
- Energy conservation for ideal part: dH/dt |_ideal = 0
- Energy dissipation for dissipative part: dH/dt |_diss < 0

**Deliverable:** Mathematical derivation (`theory/decomposition.md`)

---

### 3.2 Numerical Verification of Symplectic Structure

**Task 2.1: Long-Term Energy Conservation (Ideal Case)**

**Test:** Run ideal MHD (η=0, ν=0) for 1000+ steps

**Metrics:**
```python
# Energy drift
E_drift = |H(t) - H(0)| / H(0)

# Target: E_drift < 1e-10 over 1000 steps
```

**Test cases:**
1. Linear waves (analytical solution available)
2. Ballooning mode (v2.0 validated scenario)
3. Random initial conditions (stress test)

**Acceptance criteria:**
- Energy conserved to machine precision (< 1e-10 relative error)
- No secular drift over 1000 steps

**Deliverable:** Test script (`tests/test_long_term_conservation.py`)

---

**Task 2.2: Poisson Bracket Properties**

**Extend existing tests** (`test_hamiltonian.py`)

**Tests:**

1. **Antisymmetry** (already exists)
```python
def test_antisymmetry():
    assert |{F,G} + {G,F}| < 1e-14
```

2. **Jacobi Identity** (already exists, but extend)
```python
def test_jacobi_identity():
    residual = {{F,G},H} + {{G,H},F} + {{H,F},G}
    assert |residual| < 1e-12
```

3. **Leibniz Rule** (NEW)
```python
def test_leibniz():
    lhs = {F, G*H}
    rhs = {F,G}*H + G*{F,H}
    assert |lhs - rhs| < 1e-12
```

4. **Volume Preservation** (NEW)
```python
def test_volume_preservation():
    # Liouville theorem: div(Hamiltonian flow) = 0
    # Numerical: sample phase space volume before/after step
    assert |V(t+dt) - V(t)| / V(t) < 1e-10
```

**Deliverable:** Enhanced test suite

---

**Task 2.3: Symplectic Integrator Comparison**

**Comparison:**
- Current RK2 (2nd order, NOT symplectic)
- Symplectic Euler (1st order, symplectic)
- Störmer-Verlet (2nd order, symplectic)

**Metric:** Energy conservation over 1000 steps

**Expected:**
- RK2: Energy drift ~ O(dt²) * N_steps
- Symplectic: Energy oscillates but NO secular drift

**If symplectic integrator shows better long-term conservation:**
→ Confirms system IS Hamiltonian (benefits from symplectic integration)

**If no difference:**
→ May indicate dissipation dominates or structure not symplectic

**Deliverable:** Comparison report (`reports/integrator_comparison.md`)

---

**Task 2.4: Shadow Hamiltonian Analysis (Optional, Phase 2)**

**Advanced verification:** Backward error analysis

**Idea:**
- Numerical integrator conserves "shadow Hamiltonian" H̃
- For symplectic integrator: H̃ close to true H

**Test:**
```python
# If H̃ exists and is conserved → symplectic structure verified
H_shadow = compute_shadow_hamiltonian(trajectory)
assert |H_shadow(t) - H_shadow(0)| < 1e-12
```

**Note:** Complex analysis, defer to Phase 2 if time limited

---

### 3.3 Dissipative Part Characterization

**Task 3.1: Energy Dissipation Rate**

**For resistive case (η > 0):**

**Measure:**
```python
dH_dt = (H(t+dt) - H(t)) / dt

# Expected: dH/dt < 0 (energy decreases)
# Theoretical: dH/dt = -∫ η|∇J|² dV
```

**Verify:**
1. Measured dH/dt matches theoretical prediction
2. dH/dt scales correctly with η (linear relationship)

**Deliverable:** Dissipation validation (`tests/test_dissipation_rate.py`)

---

**Task 3.2: Operator Splitting Verification**

**Test splitting algorithm:**

```python
# Full step
state_full = step_full(state, dt, eta=η, nu=ν)

# Split step
state_half1 = step_ideal(state, dt, eta=0, nu=0)  # Hamiltonian
state_split = step_dissipative(state_half1, dt, eta=η, nu=ν)

# Should be close (splitting error ~ O(dt²))
assert |state_full - state_split| < C * dt**2
```

**Deliverable:** Splitting validation test

---

## 4. Implementation Plan

### 4.1 Task Breakdown

**Phase 1 Scope: Classification & Basic Verification**

**Stage 1: Classification**
- Task 1.1: Code inspection
- Task 1.2: Theoretical decomposition
- **Deliverable:** `classification.md`

**Stage 2: Numerical Tests**
- Task 2.1: Long-term conservation tests
- Task 2.2: Extended Poisson bracket tests
- Task 2.3: Integrator comparison
- **Deliverable:** Test suite + results

**Stage 3: Dissipation Characterization**
- Task 3.1: Energy dissipation rate
- Task 3.2: Operator splitting
- **Deliverable:** Validation report

**Stage 4: Documentation & RL Guidance**
- Synthesize findings
- Write RL integration recommendations
- **Deliverable:** `hamiltonian_verification_report.md`

---

**Phase 2 (Optional Advanced Verification):**
- Shadow Hamiltonian analysis
- Poincaré sections
- Detailed Jacobi identity proof
- Comparison with other MHD codes

---

### 4.2 File Structure

```
docs/v3.0/
├── phase1-hamiltonian-verification.md  (this file)
├── classification.md                    (Task 1)
├── theory/
│   └── decomposition.md                (Task 1.2)
└── reports/
    ├── integrator_comparison.md        (Task 2.3)
    └── hamiltonian_verification_report.md  (Final)

tests/
├── test_long_term_conservation.py      (Task 2.1)
├── test_poisson_bracket_extended.py    (Task 2.2)
├── test_dissipation_rate.py            (Task 3.1)
└── test_operator_splitting.py          (Task 3.2)
```

---

### 4.3 Success Criteria

**Minimum (Phase 1):**
- ✅ Classification complete: True vs Pseudo documented
- ✅ Long-term tests pass (1000 steps, energy drift < 1e-10 for ideal case)
- ✅ Poisson bracket properties verified (antisymmetry, Jacobi, Leibniz)
- ✅ RL guidance document: What to use, what to modify

**Ideal (Phase 2 if time permits):**
- ✅ Symplectic integrator shows advantage → confirms symplectic structure
- ✅ Shadow Hamiltonian analysis confirms structure preservation
- ✅ Detailed mathematical proofs documented

**Failure modes:**
- ❌ Energy does not conserve for ideal case → Morrison bracket implementation bug
- ❌ Jacobi identity violated → Not a Poisson structure
- ❌ No advantage from symplectic integrator → Dissipation too strong or structure absent

**In case of failure:** Debug and fix before proceeding to Phase 2 RL

---

## 5. RL Integration Recommendations (Preliminary)

**Based on expected findings:**

### 5.1 If True Hamiltonian (Ideal MHD)

**Safe to use:**
- Standard HNN architecture (Greydanus 2019)
- Symplectic integrators (Störmer-Verlet, leapfrog)
- Energy conservation as RL reward component

**Architecture:**
```python
# Standard HNN
class HamiltonianNN:
    def __init__(self, latent_dim):
        self.H_net = MLP([latent_dim, 128, 64, 1])  # Scalar H
    
    def forward(self, z):
        return self.H_net(z)
    
    def dynamics(self, z):
        dH_dz = grad(self.H_net)(z)
        return symplectic_matrix @ dH_dz  # Canonical flow
```

---

### 5.2 If Pseudo-Hamiltonian (Resistive MHD) ← **Expected Case**

**Need modifications:**

**Option A: Port-Hamiltonian Framework**
```python
# Split conservative + dissipative
class PortHamiltonianNN:
    def __init__(self, latent_dim):
        self.H_net = MLP([latent_dim, 128, 1])      # Conservative part
        self.D_net = MLP([latent_dim, 128, 1])      # Dissipative part
    
    def dynamics(self, z):
        dH_dz = grad(self.H_net)(z)
        dD_dz = grad(self.D_net)(z)
        return symplectic_matrix @ dH_dz - damping_matrix @ dD_dz
```

**Option B: Learn Conservative Part Only**
```python
# Use HNN for conservative, treat dissipation as disturbance
class ConservativeHNN:
    # Train on ideal MHD trajectories (η=0)
    # Use for structure guidance
    # Handle dissipation separately in RL
```

**Option C: Physics-Informed Loss**
```python
# Standard HNN + physics penalty
loss = mse_loss + λ * energy_drift_penalty
# where energy_drift_penalty = |dH/dt - (-η|∇J|²)|²
```

**Recommendation:** Start with Option B (conservative part only), most robust

---

### 5.3 Observation Design

**Based on verification, expose to RL:**

**Essential:**
- `H`: Total energy (Hamiltonian value)
- `dH/dt`: Energy rate of change (should be ≤0 for resistive)

**Structure-aware:**
- `δH/δψ`: Functional derivative w.r.t. flux
- `δH/δω`: Functional derivative w.r.t. vorticity

**Conservation metrics:**
- `|{H,H}|`: Self-bracket (should be ~0)
- `∫ η|∇J|²`: Dissipation integral

**JAX autodiff ready:** All above should be differentiable (Issue #24)

---

## 6. Dependencies & Risks

### 6.1 Dependencies

**Upstream (must complete before):**
- None (this is Phase 1 foundation)

**Downstream (blocked until this completes):**
- Issue #24: JAX autodiff for ∇H (depends on H definition)
- Issue #25: Observation design (depends on what to expose)
- Issue #26: Symplectic integrator (depends on verification)
- Phase 2 Hamiltonian RL (all blocked)

**Critical path:** This IS the critical path for v3.0 RL exploration

---

### 6.2 Risks & Mitigations

**Risk 1: Morrison bracket implementation bug**
- **Probability:** Low (v2.0 showed 0.0000% energy drift)
- **Impact:** High (would invalidate all downstream work)
- **Mitigation:** Extensive testing, compare with analytical solutions

**Risk 2: Dissipation too strong for symplectic structure**
- **Probability:** Medium (resistive MHD has η > 0)
- **Impact:** Medium (can still decompose, use Option B)
- **Mitigation:** Test both ideal (η=0) and resistive (η>0) cases

**Risk 3: Scope creep**
- **Probability:** Medium (complex topic can expand)
- **Impact:** Medium (delays Phase 1 completion)
- **Mitigation:** 
  - Defer Phase 2 advanced analysis
  - Focus on minimum viable verification (classification + basic tests)
  - Parallel work on other Phase 1 issues (#17, #19, #13)

**Risk 4: Findings contradict v2.0 validation**
- **Probability:** Low
- **Impact:** High (would require v2.0 re-validation)
- **Mitigation:** Careful comparison, consult YZ if discrepancy found

---

## 7. References

### 7.1 Theory

**Primary:**
- Morrison, P. J. (2023) "Hamiltonian Description of Fluid and Plasma Systems" (review)
- Chacón, L. (2020) "Energy- and Helicity-Conserving Finite Element Schemes for the 3D MHD Equations"
- Morrison & Greene (1980) "Noncanonical Hamiltonian Density Formulation of Hydrodynamics"

**Symplectic Methods:**
- Hairer, Lubich, Wanner (2006) "Geometric Numerical Integration"
- Channell & Scovel (1990) "Symplectic Integration of Hamiltonian Systems"

**HNN for RL:**
- Greydanus et al. (2019) "Hamiltonian Neural Networks"
- Zhong et al. (2020) "Symplectic ODE-Net"
- Chen et al. (2020) "Symplectic Recurrent Neural Networks"

### 7.2 Implementation References

**PyTokMHD codebase:**
- `docs/v1.1/theory/hamiltonian-mhd-formulation.md`
- `src/pim_rl/physics/v2/toroidal_bracket.py`
- `tests/test_hamiltonian.py`

**小A's v2.1 experience:**
- `experiments/v2.1_hamiltonian/designs/hamiltonian_policy_v2.0_REVISED.md`
- Issue #6 lesson: pseudo-Hamiltonian vs true Hamiltonian

---

## 8. Approval & Sign-off

**Design by:** 小P ⚛️

**Review by:** ∞ (before execution)

**Approve by:** YZ (before starting implementation)

**Execution start:** After YZ approval

---

**Next Steps After Design Approval:**

1. Create GitHub Issue comment with approved plan
2. Create test files structure
3. Start Stage 1: Classification tasks
4. Regular progress updates in Issue #23

---

_This design document defines the foundation for all v3.0 Hamiltonian RL work. Thoroughness here prevents future debugging costs._

**小P committed to getting this right** ⚛️✅

---

## Appendix A: Detailed Metrics Specification

### A.1 Energy Conservation Metrics

**Absolute drift:**
```python
E_abs = |H(t) - H(0)|
```

**Relative drift:**
```python
E_rel = |H(t) - H(0)| / max(|H(0)|, 1e-10)
```

**Secular drift rate:**
```python
# Linear fit: H(t) ≈ H(0) + α*t
α = (H(t_end) - H(0)) / t_end
# Target: |α| < 1e-12 * H(0) / t_step
```

**Oscillation amplitude:**
```python
# H should oscillate around H(0) for symplectic integrator
A_osc = std(H(t) - mean(H(t)))
# Target: A_osc < 1e-10 * H(0)
```

### A.2 Poisson Bracket Accuracy

**Point-wise error:**
```python
err_point = |{F,G}(r,θ,z) - expected(r,θ,z)|
```

**L∞ norm:**
```python
err_Linf = max over domain |{F,G} - expected|
# Target: < 1e-12 for smooth test functions
```

**L² norm:**
```python
err_L2 = sqrt(∫ |{F,G} - expected|² dV)
# Target: < 1e-11
```

### A.3 Symplectic Volume Preservation

**Phase space volume element:**
```python
# Sample N points in phase space
V(t) = det(Jacobian of flow map)
# For symplectic flow: V(t) = V(0) exactly
```

**Volume ratio:**
```python
V_ratio = V(t) / V(0)
# Target: |V_ratio - 1| < 1e-10
```

---

## Appendix B: Connection to Issue #24 (JAX Autodiff)

### B.1 Required Differentiability

**For Hamiltonian RL, need:**

1. **H(ψ, ω) → scalar** (Hamiltonian value)
   - Must be JAX-differentiable w.r.t. state
   
2. **∂H/∂ψ, ∂H/∂ω** (functional derivatives)
   - Need efficient JAX autodiff
   - Current: finite differences? ← Check in Issue #24
   
3. **{F, H}** (Poisson bracket with H)
   - Must be differentiable for policy gradient
   - JAX implementation: use `jax.grad` on bracket

### B.2 JAX Implementation Requirements

**From this verification, Issue #24 needs:**

```python
import jax
import jax.numpy as jnp

# Requirement 1: H must be JAX function
@jax.jit
def hamiltonian(state):
    psi, omega = state
    # Compute H (kinetic + magnetic energy)
    return jnp.sum(...)  # Must be JAX operations

# Requirement 2: Functional derivatives via JAX
dH_dpsi = jax.grad(hamiltonian, argnums=0)  # ∂H/∂ψ
dH_domega = jax.grad(hamiltonian, argnums=1)  # ∂H/∂ω

# Requirement 3: Poisson bracket differentiable
@jax.jit
def poisson_bracket(F, G, state):
    # Morrison bracket implementation in JAX
    # Must use jnp operations (not np)
    return ...

# For policy gradient:
def policy_gradient(state, action):
    H_value = hamiltonian(state)
    dH_daction = jax.grad(H_value, action)  # Policy uses ∂H/∂a
    return dH_daction
```

**Verification impact on Issue #24:**

- If Hamiltonian IS conservative → standard JAX HNN
- If pseudo-Hamiltonian → need separate H_conservative, H_dissipative
  - Both must be JAX-differentiable
  - Dissipation term: `D = ∫ η|∇J|² dV` also needs JAX

### B.3 Performance Requirements

**From RL real-time needs (Issue #30):**

- Hamiltonian evaluation: < 0.1 ms
- Gradient ∂H/∂state: < 0.5 ms
- Poisson bracket {F,H}: < 1 ms

**JAX optimization needed:**
- JIT compilation
- Vectorization over batch
- GPU acceleration (if available)

**Issue #24 must verify these timings** ⏱️

---

## Appendix C: Comparison with Other MHD Codes (Optional)

**If time permits in Phase 2:**

**Benchmark against:**
1. **BOUT++** (open-source MHD)
   - Has symplectic bracket implementation?
   - Energy conservation quality?
   
2. **M3D-C1** (high-fidelity)
   - Uses implicit methods (not symplectic)
   - Compare energy drift rates
   
3. **FreeGS** (equilibrium solver)
   - Different scope (static vs dynamic)
   - But can verify Hamiltonian formulation

**Metrics:**
- Energy conservation over 1000 steps
- Computational cost (FLOPs per step)
- Symplectic integrator advantage (if any)

**Deliverable:** `reports/code_comparison.md`

**Note:** This is NOT critical path for Phase 1

---

_End of Appendices_

**Total document:** 700+ lines → 850+ lines (with appendices)

**小P additions focused on:**
- Precise metrics (小A's possible concern #1)
- JAX autodiff connection (小A's possible concern #2)  
- Optional benchmarks (completeness)

✅ Ready for final submission
