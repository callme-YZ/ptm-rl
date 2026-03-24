# Issue #28: Classical Control Baselines

**Owner:** 小A 🤖  
**Support:** 小P ⚛️ (physics review)  
**Status:** Design phase  
**Date:** 2026-03-24  

---

## Goal

Establish classical control baselines for comparing Hamiltonian-aware RL performance.

**Why needed:**
- RL needs comparison benchmarks
- Validate RL provides real improvement
- Understand what classical methods can/cannot do
- Establish performance metrics

---

## Scope

### Controllers to Implement

**1. No Control (Baseline 0)**
- No action, let system evolve naturally
- Establishes "do nothing" performance
- **Expectation:** Tearing mode grows exponentially

**2. Random Control (Baseline 1)**
- Random actions within valid range
- Tests if any control helps
- **Expectation:** Slightly better than no control, but unstable

**3. PID Controller (Baseline 2)**
- Classical feedback control
- Tune on tearing mode amplitude
- **Expectation:** Can stabilize if well-tuned, but may not be optimal

**4. LQR Controller (Baseline 3, optional)**
- Linear Quadratic Regulator
- Optimal for linearized system
- **Expectation:** Better than PID if MHD is weakly nonlinear

---

## Metrics

### Primary Metrics

**1. Success Rate**
- Episode reaches max_steps without divergence
- Divergence criterion: |ψ| > threshold or NaN/Inf

**2. Energy Efficiency**
- Total Hamiltonian dissipation: ∫ |dH/dt| dt
- Lower is better (less energy injected)

**3. Stability**
- Max tearing mode amplitude during episode
- Standard deviation of Hamiltonian

### Secondary Metrics

**4. Control Effort**
- Mean action magnitude
- Action smoothness (std of action changes)

**5. Convergence Time**
- Time to stabilize (if achieved)

---

## Experimental Design

### Test Case: Tearing Mode Control

**Initial condition:**
- m=1 tearing mode perturbation
- Small amplitude (ψ ~ 0.01)
- Zero flow initially (φ = 0)

**Episode:**
- Duration: 1000 steps (dt=1e-4 → 0.1s total)
- Success: Tearing mode amplitude < initial at end

**Parameters:**
- Grid: 32 × 64 (standard)
- η (resistivity): 1e-5
- ν (viscosity): 1e-4

**Trials:**
- 10 episodes per controller
- Different random seeds
- Report mean ± std

---

## PID Controller Design

### Control Variable

**Measure:** m=1 Fourier mode amplitude
```python
m1_amp = |ψ_{1,1}|  # (m=1, n=1) mode
```

**Control:** Resistivity multiplier η_mult

### PID Formula

```python
error = target - m1_amp
error_int += error * dt
error_der = (error - error_prev) / dt

eta_mult = Kp * error + Ki * error_int + Kd * error_der
eta_mult = clip(eta_mult, 0.5, 2.0)
```

**Target:** m1_amp = 0 (full suppression)

**Tuning (initial guess):**
- Kp = 10.0
- Ki = 1.0
- Kd = 0.1

(May need tuning based on results)

---

## LQR Controller Design (Optional)

### State Space

**State:** x = [ψ_modes, φ_modes] (linearized)

**Dynamics:** dx/dt = Ax + Bu
- A: Linearized MHD dynamics
- B: Control influence matrix

**Cost:** J = ∫ (x'Qx + u'Ru) dt
- Q: State penalty (diagonal, emphasize m=1 mode)
- R: Control penalty (small, allow active control)

**Challenge:** Linearization validity
- MHD is nonlinear
- LQR may fail for large perturbations

**Decision:** Implement if time permits, not blocking

---

## Implementation Plan

### Phase 1: No Control & Random (30 min)

**Files:**
- `classical_controllers.py` (NoControlAgent, RandomAgent)
- `test_baselines.py` (basic tests)

**Tests:**
- No control diverges ✓
- Random control marginally better ✓

---

### Phase 2: PID Controller (1 hour)

**Files:**
- Add `PIDController` to `classical_controllers.py`

**Implementation:**
1. Extract m=1 mode from observation
2. Compute PID terms
3. Output η_mult (keep ν_mult = 1.0)

**Tuning:**
- Run 10 episodes, adjust Kp/Ki/Kd if unstable

**Success criterion:**
- Stabilizes tearing mode (amplitude decreasing)

---

### Phase 3: Benchmark Experiments (1-2 hours)

**Script:** `run_baseline_experiments.py`

**Experiments:**
1. No control (10 episodes)
2. Random control (10 episodes)
3. PID control (10 episodes)
4. (Optional) LQR control (10 episodes)

**Output:**
- CSV: metrics for each episode
- Plots: amplitude evolution, Hamiltonian, actions
- Summary table: mean ± std for each metric

---

### Phase 4: Analysis & Documentation (30 min)

**Report:** `BASELINE_RESULTS.md`

**Contents:**
- Experiment setup
- Results table
- Plots
- Analysis
- Conclusions

---

## Success Criteria

**Deliverables:**
- ✅ Classical controllers implemented
- ✅ Baseline experiments completed
- ✅ Metrics defined and measured
- ✅ Results documented

**Performance targets:**
- No control: Diverges (expected)
- Random: Marginally stable (expected)
- PID: Stabilizes tearing mode (target)
- (If implemented) LQR: Better than PID (target)

---

## Open Questions for 小P ⚛️

**Q1: PID control variable**
- Use m=1 amplitude as feedback signal?
- Or use Hamiltonian directly?
- Or use enstrophy?

**Q2: Linearization for LQR**
- Is MHD weakly enough nonlinear for LQR to work?
- Or skip LQR and focus on PID?

**Q3: Success criterion**
- "Tearing mode amplitude < initial" sufficient?
- Or need complete suppression (< 1% of initial)?

---

## Timeline

**Total: ~3-4 hours**

- Phase 1: 30 min
- Phase 2: 1 hour
- Phase 3: 1-2 hours
- Phase 4: 30 min

**Target completion:** Today (2026-03-24)

---

## Next Steps

1. 小P review this design ⚛️
2. 小A implement Phase 1-2 🤖
3. Run experiments (Phase 3)
4. Document results (Phase 4)

---

**小A ready to start implementation after小P review** ⚛️🤖

Author: 小A 🤖  
Date: 2026-03-24  
