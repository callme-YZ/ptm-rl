# v3.0 Phase 4: Polish & Release - 计划和分工

**Milestone:** v3.0-phase4  
**Goal:** Polish PyTokEq/PyTokMHD, complete documentation, prepare for publication  
**Issues:** 8个 (6 technical + 2 documentation)  
**Target:** 2-3 days  

**Created:** 2026-03-24 20:54  
**Owner:** 小P ⚛️ + 小A 🤖 (协作)

---

## Issue Overview & Assignment

### Group A: PyTokEq Enhancement (4个)

**Owner: 主要小P ⚛️, #16小A 🤖**

| Issue | Title | Owner | Priority | Effort |
|-------|-------|-------|----------|--------|
| #12 | q-profile accuracy at axis | 小P ⚛️ | P1-high | 2-3h |
| #14 | q-profile <5% error | 小P ⚛️ | P1-high | 3-4h |
| #15 | GPU acceleration (JAX) | 小P+小A | P2-medium | 4-6h |
| #16 | Multi-shot batch processing | 小A 🤖 | P2-medium | 2-3h |

**Total (Group A):** ~15 hours

---

### Group B: PyTokMHD Polish (4个)

**Owner: 主要小P ⚛️, #21小A 🤖**

| Issue | Title | Owner | Priority | Effort |
|-------|-------|-------|----------|--------|
| #18 | Tearing mode 3D support | 小P ⚛️ | P2-medium | 6-8h |
| #20 | 2D Morrison bracket limits | 小P ⚛️ | P2-medium | 2-3h |
| #21 | Performance profiling | 小A 🤖 | P2-medium | 2-3h |
| #22 | API & physics documentation | 小P+小A | P1-high | 8-10h |

**Total (Group B):** ~20 hours

**Phase 4 Total:** ~35 hours → 2-3 days (parallel work)

---

## Detailed Execution Plan

### Day 1: Critical Path (High Priority Issues)

#### Morning (小P ⚛️)

**9:00-12:00: Issue #12 + #14 (PyTokEq Accuracy)**

**#12: q-profile accuracy at axis (2-3h)**
- Problem: q(r=0) 边界条件不准确
- Solution: L'Hôpital's rule for ψ''(0) / r
- Deliverables:
  - Fix axis boundary condition
  - Add test: q(0) vs analytical (Solovev)
  - Verify <1% error at axis
- Tests: `test_q_axis_accuracy.py`

**#14: q-profile <5% error (3-4h)**
- Problem: q(r) 整体误差 >5%
- Solution: 
  - Improve Newton iteration tolerance
  - Better initial guess
  - Adaptive grid refinement (optional)
- Deliverables:
  - Reduce q-profile error to <5% (r ∈ [0,1])
  - Benchmark vs FreeGS/M3D-C1
  - Document convergence parameters
- Tests: `test_q_profile_accuracy.py`

**Outcome:** PyTokEq accuracy meets publication standard ✅

---

#### Afternoon (小P ⚛️ + 小A 🤖 parallel)

**小P: 13:00-16:00: Issue #20 (Morrison Bracket Limits)**

**#20: 2D Morrison bracket realistic limits (2-3h)**
- Problem: When does 2D approximation break down?
- Solution:
  - Analyze 3D coupling terms (ε = a/R₀)
  - Identify ε threshold for 2D validity
  - Document ballooning vs tearing regimes
- Deliverables:
  - Technical note: "2D Morrison Bracket Validity"
  - ε threshold analysis (likely ε < 0.3)
  - Guidance for users
- Output: `docs/v3.0/2D_MORRISON_LIMITS.md`

**Outcome:** Physics limitations明确 ✅

---

**小A: 13:00-16:00: Issue #16 + #21 start**

**#16: Multi-shot batch processing (2-3h)**
- Problem: No utility for batch GS solving
- Solution:
  - Add `batch_solve_equilibrium(shots_list)`
  - Parallel execution (if possible)
  - Progress tracking
- Deliverables:
  - `src/pytokeq/utils/batch_solver.py`
  - Example: `examples/batch_processing.py`
  - Tests: `test_batch_solver.py`
- Output: Utility for parameter scans ✅

**#21 start: Performance profiling (1h setup)**
- Profile current v3.0 baseline
- Identify bottlenecks (Poisson, bracket, etc.)
- Document performance numbers
- (Complete next day)

---

### Day 2: Documentation & Optimization

#### Morning (小P ⚛️ + 小A 🤖 parallel)

**小P: 9:00-12:00: Issue #22 (Physics Documentation)**

**#22 Part A: Physics derivation (3-4h)**
- Morrison bracket formulation
- Elsasser variables derivation
- Toroidal coupling (why ε terms matter)
- Resistive + pressure extensions
- Why 2D+Fourier for ballooning
- Output: `docs/PHYSICS_DERIVATION.md` or Jupyter notebook
- Style: Publication supplement quality

**小A: 9:00-12:00: Issue #22 (API Documentation)**

**#22 Part B: API docs + Tutorial (3-4h)**
- Sphinx setup (autodoc from docstrings)
- API reference (all public functions)
- User tutorial:
  - Quick start (10 min example)
  - Custom ICs
  - Parameter scanning
  - RL integration
- Output: `docs/` (Sphinx) + `examples/tutorial.ipynb`

---

#### Afternoon (小P ⚛️ + 小A 🤖 協作)

**13:00-17:00: Issue #15 (GPU Acceleration) + #21 complete**

**#15: GPU acceleration (4-6h, 协作)**

**小P responsibilities (physics):**
- Identify JIT-safe functions (pure numerical kernels)
- Verify physics correctness after JIT
- Test energy conservation with GPU
- Approve GPU implementation

**小A responsibilities (optimization):**
- Add `@jax.jit` to pure functions
- Handle pytree registration (if needed)
- Benchmark CPU vs GPU
- Document speedup (target: 2-5×)

**Deliverables:**
- JIT-compiled solver (optional, behind flag)
- Performance comparison
- GPU usage guide
- Tests: same physics, faster execution

**Decision point:** If GPU speedup <2×, mark as "future work"

**#21 complete: Performance profiling (小A, 2h)**
- Complete bottleneck analysis
- Document current performance (Phase 3: 50 Hz)
- Recommendations for v3.1 (100+ Hz path)
- Output: `docs/v3.0/PERFORMANCE_REPORT.md`

---

### Day 3: 3D Extension & Final Polish

#### Morning + Afternoon (小P ⚛️)

**9:00-17:00: Issue #18 (Tearing Mode 3D)**

**#18: Add tearing mode support (3D full MHD) (6-8h)**

**Problem:** 当前只支持ballooning (pressure-driven)  
**Solution:** Add tearing IC (current-driven, resistive)

**Phase 1: 3D Harris sheet IC (3h)**
- Extend 2D Harris sheet to 3D (add toroidal coupling)
- Current sheet: J_z(r, θ, φ)
- Perturbation: m=1,2 modes
- Validate vs Issue #29 (2D tearing)

**Phase 2: 3D resistive dynamics (3h)**
- Add η (resistivity) to solver (if not already)
- Verify tearing growth rate (FKR 1963)
- Compare 2D vs 3D growth

**Phase 3: Validation (2h)**
- Growth rate validation
- Magnetic island formation
- Energy conservation check

**Deliverables:**
- `src/pim_rl/physics/v2/tearing_3d_ic.py`
- `tests/test_tearing_3d.py`
- Validation report
- Compare: 2D (Issue #29) vs 3D (Issue #18)

**Outcome:** v3.0支持两种主要instabilities (ballooning + tearing) ✅

---

### Day 3 Evening: Final Review & Release Prep

**17:00-19:00: Integration & Documentation (All)**

**小P:**
- Review all physics changes (#12, #14, #18, #20)
- Run full test suite
- Update CHANGELOG
- Write Phase 4 completion report

**小A:**
- Review all system changes (#15, #16, #21, #22)
- Build documentation site (Sphinx)
- Test tutorials
- Update README

**∞:**
- Verify all 8 issues closed
- Check milestone completion (100%)
- Prepare v3.0 release checklist
- Coordinate publication materials

---

## Prioritization Logic

### P1-High (Must-have for v3.0)

**#12, #14:** PyTokEq accuracy  
- Reason: Publication requires <5% error  
- Impact: Scientific credibility

**#22:** Documentation  
- Reason: Reproducibility for paper  
- Impact: Peer review + future collaboration

### P2-Medium (Should-have, defer if needed)

**#15:** GPU acceleration  
- Reason: Performance nice-to-have, not critical  
- Decision: If GPU speedup <2×, document and defer

**#18:** 3D tearing  
- Reason: Demonstrates generality  
- Fallback: 2D tearing (Issue #29) already sufficient

**#20:** 2D limits  
- Reason: Important for honest science  
- Impact: Clarifies scope, prevents misuse

**#16, #21:** Utilities  
- Reason: Useful but not blocking publication  
- Defer: Can be v3.1 if time constrained

---

## Risk Management

### High-Risk Issues

**#18 (3D tearing):** 6-8h estimate, 可能更长  
**Mitigation:** Start early (Day 3), 如果超时defer to v3.1

**#15 (GPU):** JAX JIT可能遇到技术障碍  
**Mitigation:** 小A负责,小P review only; if stuck, document limitation

### Contingency Plan

**If Day 3 runs late:**
- Defer #18 (3D tearing) to v3.1 → Use 2D from Issue #29
- v3.0仍然valid (has tearing mode, just 2D)

**If GPU (#15) fails:**
- Document attempt
- Note current 50 Hz performance
- Recommend v3.1 optimization path

---

## Success Criteria (Phase 4 Complete)

### Technical

- ✅ PyTokEq q-profile <5% error (#12, #14)
- ✅ 2D Morrison bracket limits documented (#20)
- ✅ Tearing mode working (2D minimum, 3D preferred) (#18)
- ✅ Performance characterized (#21)
- ⚠️ GPU acceleration (optional, if speedup >2×) (#15)
- ✅ Batch processing utility (#16)

### Documentation

- ✅ Physics derivation publication-ready (#22)
- ✅ API documentation online (Sphinx) (#22)
- ✅ User tutorials (3+ examples) (#22)
- ✅ Performance report (#21)

### Process

- ✅ All 8 issues closed (or documented deferred)
- ✅ Full test suite passing
- ✅ CHANGELOG updated
- ✅ GitHub milestone 100%

---

## Collaboration Protocol

### 小P ⚛️ Responsibilities

**Physics ownership:**
- #12, #14, #18, #20 (full)
- #15 (physics validation only)
- #22 (physics docs only)

**Deliverables:**
- Accurate physics (q-profile, tearing, limits)
- Physics documentation (derivation, validation)
- Tests passing (physics correctness)

**No involvement:**
- System utilities (#16)
- Performance engineering (#21, except validation)
- API documentation generation (#22 Sphinx部分)

---

### 小A 🤖 Responsibilities

**System ownership:**
- #16, #21 (full)
- #15 (JAX optimization only)
- #22 (API docs + tutorials)

**Deliverables:**
- Batch processing utility
- Performance profiling
- API documentation site
- User tutorials

**No involvement:**
- Physics derivation (#22 physics部分)
- Physics accuracy (#12, #14)
- 3D MHD extension (#18)

---

### Collaboration Points

**Issue #15 (GPU):**
- 小A implements JIT
- 小P validates physics unchanged
- Joint: Benchmark and document

**Issue #22 (Documentation):**
- 小P writes physics derivation
- 小A generates API docs + tutorials
- Joint: Review and integrate

**Daily sync:**
- Morning: 确认当天任务
- Evening: 进度汇报,调整计划

---

## Timeline Summary

| Day | 小P ⚛️ | 小A 🤖 | Deliverables |
|-----|--------|--------|--------------|
| **Day 1 AM** | #12 + #14 (accuracy) | - | q-profile <5% error ✅ |
| **Day 1 PM** | #20 (limits) | #16 + #21 start | 2D limits doc + batch util ✅ |
| **Day 2 AM** | #22 physics docs | #22 API docs | Physics derivation + API site ✅ |
| **Day 2 PM** | #15 physics review | #15 JAX + #21 complete | GPU (optional) + perf report ✅ |
| **Day 3 Full** | #18 (3D tearing) | Tutorial + review | Tearing mode 3D ✅ |
| **Day 3 Eve** | Integration + report | Doc site + review | Phase 4 complete 🎉 |

**Target:** 2-3 days parallel work → v3.0 Phase 4 complete ✅

---

## Decision Points for YZ

### Go/No-Go Decisions

**Decision 1: GPU acceleration scope (#15)**
- If JAX JIT speedup <2×: Document limitation, defer optimization
- If speedup >2×: Include in v3.0
- **YZ approval needed:** Defer threshold? (小P suggests >2× minimum)

**Decision 2: 3D tearing priority (#18)**
- If Day 3 runs late: Defer 3D, keep 2D from Issue #29
- **YZ approval needed:** Is 2D tearing sufficient for v3.0?

**Decision 3: Documentation depth (#22)**
- Publication supplement (high detail) vs user guide (practical)
- **YZ preference:** Both? Or prioritize one?

### Resource Allocation

**小P time commitment:**
- Day 1: 8h (#12, #14, #20)
- Day 2: 8h (#22, #15 review)
- Day 3: 8h (#18)
- **Total: 24h over 3 days**

**小A time commitment:**
- Day 1: 4h (#16, #21 start)
- Day 2: 8h (#22, #15, #21 complete)
- Day 3: 4h (tutorials, review)
- **Total: 16h over 3 days**

**YZ approval needed:** This timeline acceptable?

---

## Output Artifacts (Phase 4)

### Code

- `src/pytokeq/` - q-profile fixes (#12, #14)
- `src/pytokeq/utils/batch_solver.py` (#16)
- `src/pim_rl/physics/v2/tearing_3d_ic.py` (#18)
- `src/` - JIT optimizations (#15, optional)

### Tests

- `tests/test_q_axis_accuracy.py` (#12)
- `tests/test_q_profile_accuracy.py` (#14)
- `tests/test_batch_solver.py` (#16)
- `tests/test_tearing_3d.py` (#18)

### Documentation

- `docs/PHYSICS_DERIVATION.md` (#22, 小P)
- `docs/v3.0/2D_MORRISON_LIMITS.md` (#20)
- `docs/v3.0/PERFORMANCE_REPORT.md` (#21)
- `docs/` (Sphinx site) (#22, 小A)
- `examples/tutorial.ipynb` (#22, 小A)

### Reports

- Issue completion reports (8个)
- Phase 4 completion report
- CHANGELOG (v3.0 final)

---

## Post-Phase 4: v3.0 Release

**After all 8 issues closed:**

1. **Integration testing** (小P + 小A)
   - Full test suite
   - Multi-mode validation
   - Performance regression check

2. **Release preparation** (∞ + 小P + 小A)
   - Tag v3.0
   - Release notes
   - Publication materials
   - GitHub release

3. **Publication** (All)
   - Submit paper (structure-preserving RL for MHD)
   - Supplementary materials (docs from #22)
   - Code release (v3.0 on GitHub)

---

**Plan created by:** 小P ⚛️  
**Date:** 2026-03-24 20:54  
**Status:** Awaiting YZ approval to proceed

**YZ决策:**
1. ✅ Approve plan as-is?
2. ⚠️ Adjust priorities (#15, #18 defer threshold)?
3. 📝 Documentation depth preference (#22)?

**Ready to execute on YZ's command** 🚀⚛️
