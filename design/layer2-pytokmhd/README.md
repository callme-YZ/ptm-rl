# PyTokMHD Layer 2 Design — Overview

**Date:** 2026-03-16  
**Author:** 小P ⚛️  
**Status:** Design Complete, Ready for Implementation

---

## What is PyTokMHD?

**PyTokMHD = Physics-realistic MHD evolution layer for PTM-RL project**

**Purpose:**
- Evolve MHD systems from PyTokEq equilibria
- Simulate tearing mode dynamics with real tokamak physics
- Provide validated physics environment for RL control experiments

**Key upgrade from PyTearRL:**
- PyTokEq真实平衡态 (not Harris sheet)
- Cylindrical geometry (tokamak-like)
- Complete reduced MHD equations (not simplified)
- Validated against FKR theory

---

## Design Documents

### 1. **PYTOKMHD_DESIGN.md** (18 KB)
**Architecture and technical design**

**Contents:**
- Layer 2 objectives and scope
- Module structure (solver/diagnostics/validation)
- Core API design (PyTokMHDSolver, Diagnostics)
- Physics model (Reduced MHD equations)
- Numerical methods (Finite Difference + RK4)
- PyTokEq integration scheme
- Validation strategy (FKR benchmark)
- Performance considerations
- Implementation milestones

**Key decisions:**
- ✅ Reduced MHD in cylindrical coordinates
- ✅ Finite Difference (复用PyTearRL)
- ✅ Equilibrium caching (avoid bottleneck)
- ✅ 6-week phased development

---

### 2. **PHYSICS_REQUIREMENTS.md** (13 KB)
**Physics accuracy standards**

**Contents:**
- MHD equations completeness (no simplifications)
- Geometry requirements (cylindrical, realistic boundaries)
- Physical parameter ranges (S, Re, η, ν)
- Conservation laws (energy, flux, ∇·B=0)
- Tearing mode physics (FKR growth, Rutherford saturation)
- External control requirements (RMP)
- Initial conditions (PyTokEq equilibrium + perturbation)
- Numerical precision (grid, timestep, integration)
- Diagnostics requirements (island width, growth rate)
- Performance requirements

**Key standards:**
- ✅ ∇·B < 1e-6
- ✅ Energy conservation < 1%
- ✅ FKR growth rate error < 20%
- ✅ RMP control effective (>30% suppression)

---

### 3. **MIGRATION_PLAN.md** (19 KB)
**PyTearRL → PyTokMHD migration strategy**

**Contents:**
- Migration objectives and rationale
- Key differences (PyTearRL vs PyTokMHD)
- Code audit (复用85%代码)
- Function-level mapping (Cartesian → Cylindrical)
- Step-by-step migration (7 steps, 3 weeks)
- Regression testing (compatibility checks)
- Risk mitigation (operators bugs, integration issues)
- Rollout plan (Alpha/Beta/Release)
- Success metrics
- Handoff to 小A (RL integration)

**Key points:**
- ✅ 85% code reuse from PyTearRL
- ✅ 完全替换: 初始化函数
- ✅ 改造: Operators (cylindrical)
- ✅ 保留: Time integrator (RK4)

---

### 4. **IMPLEMENTATION_ROADMAP.md** (本目录)
**6-week development plan**

**Contents:**
- Executive summary
- Key design decisions
- Architecture overview
- Implementation phases (Week 1-6)
  - Phase 1: Core solver
  - Phase 2: PyTokEq integration
  - Phase 3: Diagnostics + FKR benchmark
  - Phase 4: External control (RMP)
  - Phase 5: RL interface
- Validation strategy (unit/integration/benchmark tests)
- Deliverables (code/docs/validation)
- Resource requirements
- Risk analysis
- Success criteria

---

## Quick Reference

### Timeline:
```
Week 1-2: Core solver + cylindrical operators
Week 3:   PyTokEq integration
Week 4:   Diagnostics + FKR benchmark
Week 5:   External control (RMP)
Week 6:   RL environment + documentation

Total: 6 weeks + 2 weeks buffer = 8 weeks
```

### Key Modules:
```
pytokmhd/
├── solver/           ← MHD evolution (ψ, ω equations)
├── diagnostics/      ← Island width, growth rate
├── external_field/   ← RMP coils, PyTokEq interface
└── validation/       ← FKR benchmark, conservation tests
```

### Physics Model:
```
Reduced MHD in cylindrical (r,z):

∂ψ/∂t = -[φ, ψ] + η∇²ψ       (Induction)
∂ω/∂t = -[φ, ω] + ν∇²ω + J×B (Vorticity)
∇²φ = -ω                      (Poisson)
```

### Validation Targets:
- ∇·B < 1e-6 ✅
- Energy conservation < 1% ✅
- FKR benchmark error < 20% ✅
- RMP suppression > 30% ✅

---

## Status

**Design phase:** ✅ Complete (2026-03-16)

**Documents:**
- [x] PYTOKMHD_DESIGN.md — Architecture
- [x] PHYSICS_REQUIREMENTS.md — Physics standards
- [x] MIGRATION_PLAN.md — Migration strategy
- [x] IMPLEMENTATION_ROADMAP.md — Development plan

**Next:** YZ/小A review → Week 1 implementation

---

## For YZ Review

**Review points:**
1. **Physics model:** Reduced MHD足够吗?需要toroidal effects吗?
2. **Timeline:** 6周realistic吗?(小P估算基于PyTearRL经验)
3. **Validation标准:** FKR 20% error acceptable?
4. **Architecture:** 模块划分合理吗?

**Approval needed before:** Week 1 implementation starts

---

## For 小A Review

**Integration points:**
1. **RL Environment API:** 与PyTearRL保持兼容 (无需改RL代码)
2. **Observation/Action:** 沿用现有定义
3. **Performance:** NumPy版本~1s per episode (acceptable?)
4. **Handoff:** Week 6交付,小A接手Layer 3训练

**Questions for 小A:**
- Observation space需要调整吗?
- Action space (RMP coils)合理吗?
- Reward function设计OK吗?

---

**小P签字: 2026-03-16 ⚛️**

**Ready for review and implementation ✅**
