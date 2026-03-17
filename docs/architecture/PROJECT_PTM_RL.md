# PROJECT_PTM_RL.md

**Project:** PTM-RL (Plasma Tearing Mode RL Framework)  
**Version:** 1.0  
**Created:** 2026-03-16  
**Status:** Planning Phase

---

## 0. Executive Summary

### Mission

Develop a physics-based RL framework for tokamak tearing mode control by integrating:
1. **PyTokEq** — Real tokamak equilibrium solver (Layer 1)
2. **PyTearRL** — MHD dynamics simulation (Layer 2)
3. **RL Control** — RMP-based suppression (Layer 3)

### Key Objectives

- ✅ **Scientific Rigor:** First-principles physics, no shortcuts
- ✅ **Real-World Applicability:** Transferable to EAST/ITER
- ✅ **Dual Architecture:** CPU (accessible) & GPU (high-performance)
- ✅ **New Git Workflow:** Feature Branch + PR review standard

### Success Criteria

1. Physics validation: Conservation < 1%, tearing mode growth matches theory
2. RL performance: >50% suppression vs baseline
3. Clean codebase: All imports work, tests pass, documentation complete
4. Dual architecture: Both CPU and GPU versions validated

---

## 1. Background & Motivation

### Problem Statement

**Current PyTearRL limitation:**
- Uses simplified Harris sheet equilibrium (toy model)
- Not representative of real tokamak physics
- RL-learned control may not transfer to real devices

**Solution:**
- Integrate PyTokEq for realistic equilibrium (Layer 1)
- Build MHD evolution on top of real physics
- Train RL on physics-correct simulations

### Strategic Alignment

**Team values (SOUL.md):**
- ✅ **勇敢** — Tackle hard problem (real physics integration)
- ✅ **科学** — First principles, not shortcuts
- ✅ **行动** — Clean execution with new Git workflow

**Project goals:**
- Contribute to controllable fusion
- Develop transferable RL control strategies
- Build high-quality research tools

---

## 2. Technical Architecture

### 2.1 Three-Layer Design

```
┌─────────────────────────────────────┐
│  Layer 3: RL Control Framework      │
│  - Gymnasium environment            │
│  - PPO training (Stable-Baselines3) │
│  - RMP control actions              │
└──────────────┬──────────────────────┘
               │ Observations: w, γ, E
               │ Actions: RMP currents
┌──────────────▼──────────────────────┐
│  Layer 2: MHD Dynamics Solver       │
│  - Reduced MHD equations            │
│  - Tearing mode evolution           │
│  - RMP forcing integration          │
└──────────────┬──────────────────────┘
               │ Initial condition
               │ Boundary conditions
┌──────────────▼──────────────────────┐
│  Layer 1: PyTokEq Equilibrium       │
│  - Grad-Shafranov solver            │
│  - Real tokamak profiles            │
│  - Physics-validated equilibrium    │
└─────────────────────────────────────┘
```

### 2.2 Dual Architecture

#### CPU Version (NumPy + Ray)

**Stack:**
```python
PyTokEq (NumPy) → equilibrium solve (~1s)
    ↓
MHD Solver (NumPy) → dynamics evolution
    ↓
Ray.remote → 10-core parallelization
    ↓
Stable-Baselines3 → PPO training
```

**Performance:**
- ~10× speedup vs single-core
- Accessible on standard workstations
- Production-ready and stable

#### GPU Version (JAX)

**Stack:**
```python
PyTokEq (JAX port) → equilibrium solve (~0.01s)
    ↓
MHD Solver (JAX) → JIT-compiled dynamics
    ↓
GPU acceleration → batch parallelization
    ↓
JAX-optimized RL → custom training loop
```

**Performance:**
- ~100× speedup vs CPU
- Requires GPU hardware
- Cutting-edge performance

---

## 3. Work Breakdown Structure

### Phase 1: Project Setup (Week 1)

**Deliverables:**
- [x] Git feature branch created
- [x] Directory structure established
- [x] Core documentation (README, STATUS, this doc)
- [ ] Technical design reviewed by team
- [ ] PyTokEq quality assessment complete

**Acceptance:**
- ∞ verification: Clean clone works
- YZ approval: Technical plan sound

---

### Phase 2: Layer 1 - PyTokEq Integration (Week 2)

**Owner:** 小P ⚛️

**Tasks:**
1. Fix PyTokEq quality issues (Joy review items)
   - Correct TRL designation (Research Prototype)
   - Data consistency (precision claims)
   - Add missing validation (FreeGS comparison if time allows)

2. Define integration interface
   ```python
   class EquilibriumProvider:
       def solve(self, params) -> Equilibrium:
           """Returns psi, profiles, q"""
   ```

3. Validation suite
   - Analytical test cases
   - M3D-C1 benchmark (existing)
   - Conservation checks

**Deliverables:**
- PyTokEq quality fixes committed
- Integration API defined
- 5+ validation tests passing

**Acceptance:**
- 小A review: API usable for Layer 2
- ∞ verification: Tests pass, imports work
- YZ approval: Physics correct

---

### Phase 3: Layer 2 - MHD Dynamics (Week 3-4)

**Owner:** 小A 🤖 + 小P ⚛️ (collaborative)

**CPU Version (Week 3):**

1. MHD solver using PyTokEq equilibrium
   ```python
   def initialize_from_equilibrium(eq: Equilibrium):
       """Set up MHD state from real equilibrium"""
   ```

2. Tearing mode perturbation
   - (2,1) mode initialization
   - Physics-correct growth rate

3. RMP forcing integration
   - m=2 helical coils
   - Phase-optimized configuration

4. Ray parallelization
   ```python
   @ray.remote
   def run_episode(env_config, seed):
       # Parallel training
   ```

**Deliverables:**
- CPU version functional
- 10-core parallelization working
- Physics validation: γ matches theory

**GPU Version (Week 4):**

1. JAX port of PyTokEq (if needed)
2. JAX-native MHD solver
3. GPU performance optimization

**Deliverables:**
- GPU version functional
- 100× speedup demonstrated
- Physics parity with CPU version

**Acceptance:**
- 小P review: Physics equations correct
- 小A review: Code quality and performance
- ∞ verification: Tests pass, clean clone works
- YZ approval: Both versions validated

---

### Phase 4: Layer 3 - RL Framework (Week 5)

**Owner:** 小A 🤖

**Tasks:**
1. Gymnasium environment wrapper
   ```python
   class PTMRLEnv(gym.Env):
       observation_space = Box(...)  # [w, γ, E_kin, E_mag, ...]
       action_space = Box(...)       # RMP currents
   ```

2. PPO training pipeline (CPU version)
   - Stable-Baselines3 integration
   - Hyperparameter tuning

3. Baseline validation
   - Random policy performance
   - Deterministic control comparison

**Deliverables:**
- RL environment complete
- PPO training functional
- Baseline metrics established

**Acceptance:**
- 小P review: Reward function physics-motivated
- ∞ verification: Training reproducible
- YZ approval: Results make sense

---

### Phase 5: Validation & Documentation (Week 6)

**Owner:** ∞ + team

**Tasks:**
1. Integration testing
2. Documentation completion
3. Performance benchmarking (CPU vs GPU)
4. Clean-up and code review

**Deliverables:**
- Complete README with usage examples
- Technical report (design + results)
- GitHub ready for merge to main

**Acceptance:**
- All tests passing
- Documentation complete
- ∞ clean clone verification ✅
- YZ final approval

---

## 4. Team Responsibilities

### 小P ⚛️ (Physics Lead)

**Primary:**
- Layer 1: PyTokEq quality fixes + integration
- Physics validation across all layers
- Conservation / growth rate verification

**Secondary:**
- Review 小A's MHD implementation
- Learn Ray (CPU parallelization)
- Learn JAX (GPU implementation)

### 小A 🤖 (ML/RL Lead)

**Primary:**
- Layer 2: MHD solver implementation (CPU & GPU)
- Layer 3: RL framework and training
- Performance optimization (Ray & JAX)

**Secondary:**
- Review 小P's physics integration
- Git workflow execution
- Code quality standards

### ∞ (Project Manager)

**Primary:**
- Git workflow enforcement (new standards)
- Clean clone verification at each phase
- Documentation oversight
- Cross-team coordination

**Secondary:**
- Technical review (physics + ML)
- Timeline tracking
- YZ escalation when needed

### YZ 🐙 (Decision Maker)

**Responsibilities:**
- Technical direction approval
- Phase gate reviews
- Conflict resolution
- Strategic guidance

---

## 5. Git Workflow (New Standard)

### Branch Strategy

```
main
  ↑
  └─ feature/ptm-rl (development)
       ├─ (频繁 commit)
       ├─ (每天 push)
       └─ (完成后 PR review → merge)
```

### Commit Standards

**Frequency:**
- Every small feature (~1-2 hours of work)
- Minimum once per day

**Format:**
```
<type>: <subject>

<body>

type: feat, fix, docs, refactor, test
```

**Example:**
```
feat: Add PyTokEq equilibrium integration interface

- Define EquilibriumProvider abstract class
- Implement NumPy-based solver wrapper
- Add validation tests for Solov'ev analytical case

Physics review: 小P ✅
```

### Verification Checklist

**Before every push:**
- [ ] Code runs without errors
- [ ] Imports successful
- [ ] Relevant tests pass
- [ ] Documentation updated

**Before PR:**
- [ ] Clean clone test (∞执行)
- [ ] Cross-review (小P ↔ 小A)
- [ ] All acceptance criteria met

---

## 6. Technical Decisions

### 6.1 CPU Parallelization: Ray vs Multiprocessing

**Decision:** Ray ✅

**Rationale:**
- Multiprocessing seed handling broken (2026-03-15 实验失败)
- Ray专门设计for Python并行
- 更stable,支持distributed扩展

### 6.2 GPU Framework: JAX vs PyTorch

**Decision:** JAX ✅

**Rationale:**
- 自动微分 + JIT编译天然适合physics
- NumPy-like API,小P容易上手
- GPU性能优于PyTorch (physics benchmarks)

### 6.3 RL Library: Stable-Baselines3 vs Custom

**Decision:** Stable-Baselines3 for CPU, Custom for GPU

**Rationale:**
- SB3成熟稳定 (CPU版本)
- JAX版本需custom实现 (性能优化)

---

## 7. Risks & Mitigation

### Risk 1: PyTokEq Performance Bottleneck

**Risk:** ~1s per equilibrium solve → slow RL training

**Mitigation:**
- Equilibrium caching (fixed params)
- JAX port for GPU speedup (~0.01s)
- Offline equilibrium generation

**Probability:** High  
**Impact:** Medium  
**Owner:** 小P

---

### Risk 2: Physics Integration Complexity

**Risk:** Layer 1→2 interface复杂,调试困难

**Mitigation:**
- Well-defined API contract
- Incremental testing (analytical → simple → complex)
- 小P+小A紧密协作

**Probability:** Medium  
**Impact:** High  
**Owner:** 小A + 小P

---

### Risk 3: GPU Availability

**Risk:** 无GPU硬件 → GPU版本无法开发

**Mitigation:**
- CPU版本优先 (Week 3)
- Cloud GPU (Colab/AWS) if needed
- GPU版本作为Phase 2 (optional)

**Probability:** Low  
**Impact:** Low (CPU版本已足够)  
**Owner:** 小A

---

## 8. Success Metrics

### Technical Metrics

1. **Physics Correctness**
   - Energy conservation: < 1% drift over 1000 steps
   - Tearing growth rate: within 10% of FKR theory
   - Equilibrium validation: M3D-C1 benchmark match

2. **RL Performance**
   - Island width suppression: > 50% vs baseline
   - Training stability: Converge in < 100K steps
   - Reproducibility: Variance < 10% across 5 seeds

3. **Code Quality**
   - Clean clone: 100% success rate
   - Test coverage: > 80% (core modules)
   - Documentation: README + API docs complete

### Process Metrics

1. **Git Workflow**
   - Commit frequency: Daily
   - PR review: 100% coverage
   - Clean clone tests: Every phase gate

2. **Collaboration**
   - Cross-reviews: 小P ↔ 小A every major commit
   - ∞ verification: Every deliverable
   - YZ approval: Every phase gate

---

## 9. Timeline Summary

| Week | Phase | Owner | Deliverable |
|------|-------|-------|-------------|
| 1 | Project Setup | ∞ | Docs + Git branch ✅ |
| 2 | Layer 1: PyTokEq | 小P | Integration API + fixes |
| 3 | Layer 2: CPU MHD | 小A+小P | NumPy+Ray working |
| 4 | Layer 2: GPU MHD | 小A+小P | JAX version |
| 5 | Layer 3: RL | 小A | PPO training |
| 6 | Validation | Team | Tests + docs complete |

**Total:** 6 weeks (1.5 months)

---

## 10. Appendices

### A. References

- PyTokEq documentation (local)
- PyTearRL M3.3 design docs
- PROJECT_MANAGEMENT_STANDARDS.md (Git workflow)
- COLLABORATION_PROTOCOL.md (team boundaries)

### B. Related Projects

- **PyTokEq:** Tokamak equilibrium solver (Layer 1 source)
- **PyTearRL:** Simplified MHD tearing mode (to be replaced)
- **rl-framework:** Stage 0 baseline RL (reference implementation)

### C. Change Log

- 2026-03-16: Project initialization, v1.0 of this document

---

**Document Owner:** ∞  
**Reviewers:** 小A 🤖, 小P ⚛️  
**Approver:** YZ 🐙

**Status:** Draft → awaiting team review
