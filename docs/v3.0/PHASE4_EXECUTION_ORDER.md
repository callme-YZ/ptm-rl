# v3.0 Phase 4: Execution Order (Performance-First Strategy)

**Updated:** 2026-03-24 21:04  
**Decision:** Performance优先 (YZ指导)  
**Rationale:** 探索保结构+RL需要速度支撑 ⚡

---

## Core Insight (YZ教导)

**"更好的探索保结构+RL" = 需要大量实验**

- **Slow (5 Hz):** 1个实验200s → 一天几十个实验 → 探索浅
- **Fast (500 Hz):** 1个实验2s → 一天几千个实验 → 探索深
- **Performance直接决定探索深度!** ⚡

**→ Phase 4必须performance-first!**

---

## Revised Priority (Performance-First)

### P1-Critical (Must-have)

**#15: GPU/JIT Acceleration** ⚡  
**#21: Performance Profiling** 📊  
**#12: q-profile axis accuracy** ⚛️  
**#14: q-profile <5% error** ⚛️

### P2-Important (Should-have)

**#18: 3D Tearing mode** 🌍  
**#22: Physics documentation** 📋

### P3-Optional (Nice-to-have, defer if needed)

**#16: Batch processing** 🔧  
**#20: 2D Morrison limits** 📝

---

## Execution Order (3 Days)

### Day 1: Performance Sprint ⚡

**Goal:** Maximize simulation speed

**Morning (小A 🤖, 3h):**

**9:00-12:00: Issue #21 - Performance Profiling**
- Profile current v3.0 solver
- Identify bottlenecks:
  - Poisson conversion?
  - Morrison bracket?
  - Time integration?
  - Observation computation?
- Quantify each component
- **Deliverable:** `docs/v3.0/issue21/PROFILING_REPORT.md`

**Afternoon (小A + 小P, 4-6h):**

**13:00-19:00: Issue #15 - GPU/JIT Acceleration**

**小A responsibilities (JAX optimization):**
- Add `@jax.jit` to bottleneck functions (guided by profiling)
- Handle dataclass issues (pytree registration if needed)
- Test on GPU (if available) vs CPU
- Benchmark speedup

**小P responsibilities (physics validation):**
- Verify energy conservation unchanged
- Check ∇·B constraint maintained
- Validate Hamiltonian structure preserved
- Approve or reject optimization

**Go/No-Go Decision:**
- **If speedup >2×:** Commit and merge ✅
- **If speedup 1.5-2×:** Commit with note (marginal gain) ⚠️
- **If speedup <1.5×:** Document attempt, defer to v3.1 ❌

**Target:** 5-10× speedup (50 Hz → 250-500 Hz)

**End of Day 1:**
- Performance ceiling known ✅
- Optimization committed or documented ✅

---

### Day 2: Physics Quality ⚛️

**Goal:** Accurate equilibrium for exploration

**Morning (小P ⚛️, 5-6h):**

**9:00-11:30: Issue #12 - q-profile Axis Accuracy**
- Problem: q(r=0) boundary condition inaccurate
- Solution: L'Hôpital's rule for ψ''(0)/r
- Fix `src/pytokeq/equilibrium/solver.py`
- Test: q(0) vs Solovev analytical (error <1%)
- **Deliverable:** `tests/test_q_axis_accuracy.py`

**11:30-15:00: Issue #14 - q-profile <5% Error**
- Problem: Overall q(r) error >5%
- Solution:
  - Improve Newton iteration tolerance
  - Better initial guess
  - Possibly adaptive grid
- Benchmark vs FreeGS/M3D-C1
- **Deliverable:** `tests/test_q_profile_accuracy.py`

**Afternoon (小P ⚛️, 3h):**

**15:00-18:00: Issue #22 Part A - Physics Derivation**
- Morrison bracket formulation
- Elsasser variables derivation
- Toroidal coupling (ε terms)
- Resistive + pressure extensions
- Why 2D+Fourier for ballooning
- **Deliverable:** `docs/PHYSICS_DERIVATION.md` (publication supplement quality)

**End of Day 2:**
- PyTokEq accuracy <5% ✅
- Physics derivation documented ✅

---

### Day 3: Scenarios & Integration 🌍

**Goal:** More fusion scenarios (if time permits)

**Morning (小P ⚛️, 4h):**

**9:00-13:00: Issue #18 Phase 1 - 3D Tearing Research**
- Extend 2D Harris sheet (Issue #29) to 3D
- Add toroidal coupling
- Design 3D IC structure
- **Deliverable:** `docs/v3.0/issue18/3D_TEARING_DESIGN.md`

**Afternoon (小P ⚛️, 4h):**

**13:00-17:00: Issue #18 Phase 2 - 3D Tearing Implementation**
- `src/pim_rl/physics/v2/tearing_3d_ic.py`
- 3D current sheet
- m=1,2 perturbations with toroidal mode numbers
- **Deliverable:** `tests/test_tearing_3d.py`

**Contingency:**
- If Day 2 runs late: Defer #18 to v3.1 (2D tearing from #29 sufficient)
- If Day 1 GPU failed: Prioritize #18 (more scenarios > marginal speedup)

**Evening (All, 2h):**

**17:00-19:00: Integration & Documentation**
- Run full test suite
- Update CHANGELOG
- Phase 4 completion report
- Close all issues

**End of Day 3:**
- v3.0 Phase 4 complete ✅
- Ready for release prep ✅

---

## Deferred Issues (Explicit)

### #16: Multi-shot Batch Processing
- **Reason:** Utility, not critical for exploration
- **Defer to:** v3.1
- **Small A:** If有extra time在Day 3, can implement (2h)

### #20: 2D Morrison Bracket Limits
- **Reason:** Science honesty important但not blocking
- **Defer to:** v3.1 or publication appendix
- **Small P:** If Day 3 finishes early, write technical note (2h)

### #22 Part B: API Documentation
- **Reason:** User tutorial nice-to-have, physics docs more critical
- **Defer to:** v3.1
- **Small A:** If有time, generate Sphinx docs (3h)

---

## Success Metrics (Phase 4)

### Performance (P1)
- ✅ Profiling report complete (#21)
- ✅ Speedup measured and documented (#15)
- 🎯 Target: 2-10× faster (100-500 Hz)

### Physics Quality (P1)
- ✅ q-profile axis <1% error (#12)
- ✅ q-profile overall <5% error (#14)
- ✅ Physics derivation documented (#22 Part A)

### Scenarios (P2)
- ⚠️ 3D tearing mode working (#18, if time permits)
- ✅ 2D tearing already available (Issue #29 fallback)

### Process
- ✅ 4-6 issues closed (P1 issues必须, P2 optional)
- ✅ Full test suite passing
- ✅ CHANGELOG updated
- ✅ Ready for publication

---

## Daily Sync Protocol

**Every morning (9:00):**
- Review yesterday's progress
- Confirm today's tasks
- Adjust if blocked

**Every evening (19:00):**
- Demo completed work
- Identify blockers
- Plan next day

**Critical decisions:**
- GPU speedup threshold: YZ decides if <2×
- 3D tearing defer: YZ decides if Day 2 late
- Documentation depth: YZ preference

---

## Risk Management

### High-Risk: GPU Optimization (#15)

**Risk:** JAX JIT may not work with current code structure  
**Impact:** 失去主要speedup path  
**Mitigation:**
1. Profiling先行 (找准bottleneck)
2. 小A+小P pair (JIT + physics validation)
3. 1.5× threshold (marginal gain也接受)
4. If fail: Document limitation, defer to v3.1

**Contingency:** Focus on #12, #14, #18 (质量 > 速度)

### Medium-Risk: 3D Tearing (#18)

**Risk:** 8h estimate可能不够  
**Impact:** Day 3 overflow  
**Mitigation:**
1. Start Day 3 morning (8h available)
2. Phase design先行 (确认可行性)
3. If超时: 记录进度, defer完成到v3.1

**Contingency:** 2D tearing (Issue #29) already sufficient for v3.0

---

## Phase 4 vs v3.x Goals Alignment

**v3.x Goal 1: 更好的物理环境**
- ✅ #12, #14: Accurate equilibrium
- ✅ #22: Physics derivation (可验证)

**v3.x Goal 2: 保结构+RL探索**
- ✅ #15, #21: Performance → 探索深度 ⚡
- ✅ Physics quality → 结果可信

**v3.x Goal 3: 更多核聚变场景**
- ✅ Ballooning (Phase 1-3已有)
- ✅ Tearing 2D (Issue #29已有)
- ⚠️ Tearing 3D (Issue #18, if time permits)
- ✅ Kink, Interchange (Issue #27已有)

**All goals supported** ✅

---

## New Issues Needed? (检查)

**Current Phase 4 issues cover:**
- Performance (#15, #21) ✅
- Accuracy (#12, #14) ✅
- Scenarios (#18) ✅
- Docs (#20, #22) ✅
- Utils (#16) ✅

**Missing from discussion:**
- ❓ Multi-mode RL validation? (Not an issue yet)
- ❓ Hyperparameter sweep automation? (Not an issue yet)

**YZ: 需要新建issues吗?** 🤔

**小P猜测不需要 - Phase 4专注polish,新功能defer到v3.1** ✅

---

**Created by:** 小P ⚛️  
**Reviewed by:** 小A 🤖  
**Date:** 2026-03-24 21:04  
**Status:** Awaiting YZ approval

**YZ批准这个执行顺序?** 🎯⚡⚛️
