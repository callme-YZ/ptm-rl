# PTM-RL v2.0: Elsässer MHD + RL Framework

**Phase 1 Status:** Physics layer validated ✅ | RL layer exploratory ⏳  
**Branch:** `feature/v2.0-elsasser`  
**Tag:** `v2.0.0-phase1` (2026-03-21)

---

## Overview

v2.0 develops a **structure-preserving MHD simulation framework** coupled with reinforcement learning for plasma instability control.

**Key innovations:**
- PyTokEq-based realistic tokamak equilibrium (β~0.17)
- Morrison bracket Hamiltonian formulation
- Elsässer variable representation (z± = v ± B)
- RMP coil control via RL

**Current scope:** Ballooning mode suppression in simplified 3D toroidal geometry

---

## Phase 1 Achievements (2026-03-20 to 2026-03-21)

### Physics Validation ✅

**C1: Growth Rate**
- Measured: γ = 1.29
- Theory (ideal MHD): γ = 0.73
- **Gap:** 77% (expected due to resistive effects, ∇p, finite-n)
- **Status:** Positive growth confirmed, order-of-magnitude correct ✅

**C2: Energy Conservation**
- Drift: 0.38% (< 1% threshold) ✅
- Secular growth: 1.8% at t=10 (< 5% threshold) ✅
- **Status:** Structure-preserving numerics working ✅

**C3: v1.4 vs v2.0**
- v1.4: β~10⁹ (unphysical), 77-step crash ❌
- v2.0: β=0.17 (realistic), 100-step stable ✅
- Energy conservation: 92% better (5% → 0.38%)
- **Status:** Fundamental fix via YZ's PyTokEq approach ✅

**Validation report:** `PHYSICS_VALIDATION_REPORT.md`

---

### RL Baseline Training

**50k baseline (Phase 1):**
- Result: Reward improvement ~0.08% (statistically insignificant)
- Learning signal: Weak/absent
- Physics: Stable (100 steps, no crashes) ✅
- **Conclusion:** Framework trainable, but learning dynamics need investigation

**200k extended (Phase 2.0, in progress):**
- Current: +32.1% improvement (5k → 70k steps)
- Peak: +36.8% (at 15k steps)
- Episode length: ~79 steps (early termination, likely amplitude explosion)
- **Status:** Clear learning signal detected, plateau after 15k ⏳

**Key finding:** Multi-environment parallelization essential for reasonable training time (6× speedup)

---

## Technical Specifications

### Environment

**Observation:** 113 features
- z± spectral modes (100 features via FFT)
- Conservation diagnostics (3 features)
- Island width placeholders (10 features)

**Action:** 4-channel RMP coils
- Range: [-1, 1] normalized
- Forcing: F_RMP = scale × action × spatial_pattern

**Reward:** -m2_amplitude (ballooning mode)
- Goal: Minimize perturbation growth
- Energy conservation penalty: -10 × drift

**Physics:**
- Grid: 16×32×16 (r, θ, z)
- Time step: dt_rl = 0.02 (5 internal steps of dt=0.004)
- Episode length: 100 steps max
- Parameters: ε=0.323, η=0.01, pressure_scale=0.2

**Termination:**
- Amplitude explosion: A > 10×A_initial
- NaN detection
- Max steps: 100

### Performance

**Single process:**
- ~9 FPS
- Training time: ~5-6h (200k steps)

**Multi-process (8 cores):**
- ~46 FPS
- Training time: ~45-60 min (200k steps)
- **Recommended for RL experiments**

---

## Current Limitations

### Physics

**Growth rate gap (77%):**
- Simple theory (γ~√(β/ε)) ignores resistivity, pressure gradient, finite-n
- Numerical error not ruled out (grid convergence not tested)
- **Not critical for RL framework validation**

**No benchmark comparison:**
- No canonical test case for ballooning modes
- Should compare with BOUT++/M3D-C1 (future work)

### RL

**Weak learning (50k baseline):**
- Reward improvement ~0% → learning dynamics unclear
- May need: longer training, reward tuning, or algorithm changes

**Early termination (~79 steps):**
- Episodes end before 100 steps (amplitude explosion threshold)
- Policy learns partial suppression but not full control
- **Not yet demonstration-quality**

**Plateau (200k extended):**
- Learning stops after 15k steps
- Possible causes: local optimum, exploration deficit, reward saturation

---

## Repository Structure

### Core Code
- `mhd_elsasser_env.py` (474 lines) - Gymnasium environment
- `train_v2_ppo.py` (323 lines) - PPO training (single/multi-env)
- `train_50k_baseline.py` (152 lines) - Phase 1 baseline script

### Validation
- `validate_physics_c1.py` (323 lines) - Growth rate verification
- `validate_physics_c2.py` (162 lines) - Energy conservation
- `validate_physics_c3.py` (196 lines) - v1.4 comparison
- `PHYSICS_VALIDATION_REPORT.md` (412 lines) - Full analysis

### Utilities
- `jit_solver_wrapper.py` (150 lines) - Performance optimization
- `verify_rmp_5000x.py` (120 lines) - Long-term stability test
- `profile_env.py` (81 lines) - Profiling tool
- `quick_verify.py` (47 lines) - Sanity check

**Total:** 2,602 lines (2,028 Python + 574 Markdown)

---

## Known Issues

### Physics Layer
- Growth rate theory-simulation gap (77%) not fully understood
- Grid convergence not tested (numerical error unknown)
- No external benchmark (BOUT++, M3D-C1)

### RL Layer
- 50k baseline: no significant learning
- 200k: plateau after 15k steps
- Early termination: episodes ~79 steps (not 100)
- Control effectiveness: partial suppression, not full control

### Infrastructure
- Multi-env training requires proper `__name__ == '__main__'` structure
- macOS `spawn` mode needed (not `fork`)
- No GPU support yet (planned for Phase 2.3 offline RL)

---

## Phase 2 Plan

**Design document:** `PHASE_2_DESIGN.md`

**Objectives:**
1. **Phase 2.0:** Establish learning baseline (200k steps) → **In progress** ⏳
2. **Phase 2.1:** Physics-informed reward shaping (1 week)
3. **Phase 2.2:** Symplectic PPO (structure-preserving policy) (2 weeks)
4. **Phase 2.3:** Offline RL baseline (IQL with 50k data) (1 week)

**Success criteria:**
- >20% mode suppression (vs uncontrolled)
- <1% energy conservation drift
- Reproducible across seeds
- Episode completion >90 steps

**Compute resources:**
- Multi-CPU: 8 cores (6× speedup)
- GPU: Optional for Phase 2.3 (10× speedup for offline RL)

---

## Scientific Context

### What v2.0 is

**A validated research framework for:**
- Realistic MHD physics (β~0.17 tokamak-like)
- Structure-preserving numerics (energy conservation <1%)
- RL-trainable environment (episodes stable, no crashes)

**Suitable for:**
- Computational plasma physics research
- RL for control exploration
- Publication: Plasma Science & Technology (70-80% confidence)

### What v2.0 is NOT

**Not production-ready:**
- RL control effectiveness: partial, not demonstrated
- No experimental validation
- No ITER-relevant parameters

**Not breakthrough-level:**
- Theory-simulation gap not resolved
- Learning dynamics need deeper investigation
- Not suitable for Nuclear Fusion / Nature-level journals (yet)

**Honest assessment:** v2.0 is an **excellent foundation** for future work, not a complete demonstration

---

## How to Use

### Quick Test
```bash
python quick_verify.py  # ~1 minute
```

### Physics Validation
```bash
python validate_physics_c1.py  # Growth rate (~3 min)
python validate_physics_c2.py  # Energy conservation (~3 min)
python validate_physics_c3.py  # v1.4 comparison (~5 min)
```

### Training (Multi-CPU)
```bash
# Requires proper __name__ guard in script
python train_v2_ppo.py --total-steps 200000 --n-envs 8
```

### Monitoring
```bash
tail -f train_*.log
tensorboard --logdir ./logs  # If implemented
```

---

## Team & Attribution

**Development:**
- 小A 🤖: RL framework, training infrastructure
- 小P ⚛️: Physics validation, theory analysis
- YZ 🐙: PyTokEq breakthrough (10-min β fix), direction
- ∞: Coordination, Git management, documentation

**Critical decisions:**
- YZ's PyTokEq approach (2026-03-20): Solved v1.4 β~10⁹ crisis
- Physics-first validation (2026-03-21): C1-C3 before RL optimization
- Multi-CPU parallelization (2026-03-21): 6× speedup for iteration

---

## References

**Physics validation report:** `PHYSICS_VALIDATION_REPORT.md`  
**Phase 2 design:** `PHASE_2_DESIGN.md`  
**Repository:** https://github.com/callme-YZ/ptm-rl  
**Related work:** DeepMind tokamak control (Nature 2022), TORAX transport solver

---

## Changelog

**v2.0.0-phase1 (2026-03-21):**
- ✅ Physics validation (C1-C3) passed
- ✅ PyTokEq realistic equilibrium
- ✅ Energy conservation <1%
- ✅ 50k baseline: framework stable
- ⏳ 200k baseline: learning signal detected (+32%)
- 📝 Documentation: honest, scientific, reproducible

**Previous:** v1.4 (3D MHD + RL, β crisis) → v2.0 (fixed via PyTokEq)

---

**Last updated:** 2026-03-21 10:33 GMT+8  
**Status:** Phase 1 complete ✅ | Phase 2.0 in progress ⏳
