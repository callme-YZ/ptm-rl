# PTM-RL v2.0 Roadmap

**Goal:** Structure-preserving physics + publication-ready quality

**Timeline:** 7-9 months  
**Status:** Planning (2026-03-17)

---

## Vision

Transform PTM-RL from proof-of-concept (v1.0) to **scientific tool** (v2.0):

1. **Publication-ready:** Methods publishable in JCP/CPC or PoP/NF
2. **Open-source quality:** Complete docs, benchmarks, tutorials
3. **Structure-preserving:** Long-term stability via geometric integrators

---

## Three-Phase Development Plan

### v1.1: Toroidal Geometry + Symplectic Time Integration

**Duration:** 1-2 months

**Physics Upgrades:**
- Coordinate transformation: Cylindrical (r,z) → Toroidal (R,φ,Z)
- Realistic tokamak geometry
- Reduced MHD equations in toroidal coordinates
- Metric tensor, Christoffel symbols handling

**Structure-Preserving Algorithms:**
- Replace RK4 → **Symplectic integrator**
- Preserve phase-space volume (symplectic structure)
- Energy drift: O(dt²) → O(dt⁴) or better
- Examples: Störmer-Verlet, implicit midpoint, Gauss-Legendre

**Why This First:**
- Geometry upgrade is physics foundation ✅
- Symplectic integration relatively easy to implement ✅
- Two components independent → can develop in parallel ✅
- Prepares for more complex physics (v1.2) ✅

**Parallel Work:**
- **小P:** Toroidal MHD solver core
- **小A:** RL obs/action adaptation + Offline RL research

**Deliverables:**
- Toroidal MHD solver with symplectic time integration
- RL environment adapted for toroidal geometry
- Validation: energy conservation in toroidal equilibrium
- Documentation: coordinate transformation, symplectic methods

**External Materials Needed:**
- 📄 Hairer et al. "Geometric Numerical Integration" (classic reference)
- 📄 Latest papers on "symplectic reduced MHD" (literature search)
- 📄 Standard toroidal coordinate treatment (metric tensor, Jacobian)

**Milestone:** Toroidal solver runs + RL trains successfully ✅

---

### v1.2: Energy/Helicity Conserving Spatial Discretization

**Duration:** 2-3 months (most complex)

**Physics Upgrades:**
- Time-only preserving (v1.1) → **Time + Space** preserving
- Discrete MHD satisfies: ∂E/∂t = 0, ∂K/∂t = 0 (discrete conservation laws)
- Mimetic or finite element with structure-preserving basis

**Structure-Preserving Algorithms:**
- Discrete operators satisfy continuous properties
- Examples:
  - Mimetic discretization (Palha et al.)
  - Energy-conserving schemes (Chacón et al.)
  - Finite element with compatible spaces
- **Key:** ∇·B = 0 preserved at discrete level

**Why v1.2:**
- Symplectic time (v1.1) + conservative space (v1.2) = **complete structure-preservation** ✅
- **Core scientific contribution** for publication ✅
- Guarantee of long-term stability ✅

**Critical Challenge:**
- Most complex component ⚠️
- May require external expert consultation ⚠️
- Risk of exceeding 3-month estimate ⚠️

**Parallel Work:**
- **小P:** Structure-preserving discretization (primary)
- **小A:** Sample-efficient RL methods (Offline RL, Model-based RL)
  - Concern: More complex physics → slower simulation → need better RL

**Deliverables:**
- Energy/helicity-conserving MHD solver
- Long-time stability demonstration (>10⁴ Alfvén times)
- Energy drift < 1e-6 over extended runs
- Comparison: standard vs structure-preserving discretization

**External Materials Needed:**
- 📄 Chacón "Energy- and helicity-preserving schemes for MHD" (LANL work, 2000s-2010s)
- 📄 Palha et al. "Mimetic spectral element method"
- 📄 **Latest 2020-2025 literature:** structure-preserving MHD + ML (novelty check) ⚠️
- 🤝 **Possible external expert consultation** for discretization design

**Risk Mitigation:**
- Go/No-Go decision point after v1.1
- Consider external collaboration if complexity too high
- Budget buffer time (may extend to 4 months)

**Milestone:** Energy drift < 1e-6 over 10⁴ timesteps ✅

---

### v1.3: Validation, Benchmarking, and Publication

**Duration:** 1-2 months

**Scientific Validation:**
- **Benchmark vs analytical solutions:**
  - Cylindrical tearing mode (Furth-Killeen-Rosenbluth theory)
  - Comparison with v1.0 results
  
- **Benchmark vs published results:**
  - Aydemir (1992) tearing mode growth rates
  - M3D-C1 verification cases
  - NIMROD test cases (if accessible)

- **Long-time stability tests:**
  - Run > 10⁴ Alfvén times
  - Verify energy/helicity conservation
  - Compare: structure-preserving vs standard methods

**RL Re-training and Analysis:**
- Train RL agents on v2.0 physics
- Compare with v1.0 results:
  - Sample efficiency (steps to convergence)
  - Final performance (island suppression quality)
  - Training stability
  
- **Scientific question:** Does structure-preserving physics help RL?
  - Hypothesis: More stable physics → easier learning
  - Validation: learning curves, variance analysis

**RL Benchmarking:**
- **小A responsible for:**
  - RL-specific benchmarks (sample efficiency, generalization)
  - Comparison: PPO vs Offline RL (IQL/CQL)
  - Ablation studies: reward components, hyperparameters

**Publication Preparation:**
- **Target journals:**
  - *Top-tier methods:* Journal of Computational Physics, Computer Physics Communications
  - *Physics applications:* Physics of Plasmas, Nuclear Fusion
  
- **Selling points:**
  1. Structure-preserving MHD (novel numerical method)
  2. RL on structure-preserving physics (interdisciplinary)
  3. Long-time stability demonstration
  4. Comprehensive benchmark validation

- **Paper structure:**
  - Introduction: tearing mode control + structure-preserving methods
  - Methods: v1.1-v1.2 algorithms
  - Validation: benchmarks (v1.3)
  - RL results: training, comparison
  - Discussion: physics stability → RL performance

**Open-Source Preparation:**
- **Code quality:**
  - Complete test coverage (maintain >90%)
  - CI/CD for v2.0 branches
  - Documentation updates
  
- **User-facing materials:**
  - Tutorials (beginner → advanced)
  - Reproducible examples
  - API documentation
  - Installation guide
  
- **Community readiness:**
  - Issue templates
  - Contributing guidelines
  - Example gallery

**External Materials Needed:**
- 📄 Benchmark test cases from M3D-C1, NIMROD (verification manuals)
- 📄 Aydemir (1992) and other classical tearing mode papers
- 📄 Latest RL + physics-informed methods (ML4Science literature)
- 📄 Hamiltonian Neural Networks (structure-preserving NN, if relevant)

**Milestone:** Benchmark passed + Paper draft complete ✅

---

## Overall Timeline

```
v1.0 (Done) → v1.1 (2 mo) → v1.2 (3 mo) → v1.3 (2 mo) = v2.0
   2026-03      2026-05      2026-08      2026-10
              ↓             ↓             ↓
           Toroidal +   Energy-       Validation +
           Symplectic   Conserving    Publication
```

**Total:** 7-9 months (conservative estimate)

**Buffer:** Add 1-2 months for v1.2 complexity and external review

---

## Key Risks and Mitigation

### Risk 1: v1.2 Complexity Underestimated ⚠️

**Concern:** Structure-preserving discretization is hard, 3 months may not be enough

**Mitigation:**
- Go/No-Go decision after v1.1 completion
- External expert consultation (identify candidate now)
- Possibility of simplified approach if full conservation too complex
- Budget 4 months instead of 3 if needed

---

### Risk 2: Scientific Novelty Insufficient ⚠️

**Concern:** Chacón/Palha methods are 2000s-2010s, may not be "new enough"

**Mitigation (小E recommendation):**
- **Immediate literature search:** 2020-2025 papers on structure-preserving MHD + ML
- Verify no one has done "structure-preserving MHD + RL control"
- If overlap exists → strengthen differentiation:
  - Harder benchmark problems
  - Tokamak-specific application (RL for tearing mode)
  - Comparison with production codes (M3D-C1, NIMROD)

---

### Risk 3: Physics Complexity Slows RL Training ⚠️

**Concern (小A):** More complex physics → slower simulation → RL needs more samples

**Mitigation:**
- **v1.1-v1.2 parallel work:** 小A researches sample-efficient RL
  - Offline RL (IQL, CQL) from v1.0 data
  - Model-based RL (learn physics model, plan in latent space)
  - Curriculum learning (start simple, increase complexity)
  
- If v2.0 physics too slow → explore:
  - GPU acceleration (JAX/PyTorch migration)
  - Reduced-fidelity model for exploration, high-fidelity for validation

---

### Risk 4: External Materials Inaccessible ⚠️

**Concern:** Some benchmarks (M3D-C1, NIMROD) may require collaborations

**Mitigation:**
- Prioritize publicly available cases (Aydemir, FKR theory)
- Contact刘健 professor for potential collaborations
- Worst case: use v1.0 cylindrical benchmarks + demonstrate method on toroidal

---

## Resource Requirements

### Personnel

**小P (Physics Lead):**
- Primary: v1.1-v1.2 physics implementation
- Secondary: v1.3 benchmark validation
- Estimated: 80% time over 7 months

**小A (RL Lead):**
- Primary: RL adaptation, sample-efficient methods
- Secondary: v1.3 RL benchmarking
- Estimated: 60% time over 7 months

**∞ (PM):**
- Project coordination, milestone tracking
- Documentation organization
- External material acquisition
- Estimated: 20% time over 7 months

**小E (Documentation/Research):**
- Literature search and organization
- Tutorial/documentation writing
- Novelty verification
- Estimated: 30% time over 7 months

### External Consultation (Optional)

**Target:** Structure-preserving discretization expert
- Timing: After v1.1, before v1.2 implementation
- Format: 1-2 hour consultation + email follow-up
- Candidates: TBD (literature search will identify)

### Computational Resources

**v1.1-v1.2 Development:**
- Local workstation sufficient (debugging, small tests)

**v1.3 Long-time Runs:**
- May need GPU cluster for 10⁴+ timestep runs
- Estimate: 100-500 GPU-hours total

---

## Success Metrics

### Technical Metrics

**v1.1:**
- ✅ Toroidal solver completes 1000 timesteps without crash
- ✅ Energy drift < 1e-4 (symplectic integration)
- ✅ RL trains to convergence on toroidal geometry

**v1.2:**
- ✅ Energy drift < 1e-6 over 10⁴ timesteps
- ✅ Helicity conservation < 1e-6
- ✅ ∇·B error < machine precision (mimetic/FEM)

**v1.3:**
- ✅ Growth rate matches Aydemir within 5%
- ✅ Island width evolution matches published results
- ✅ RL achieves ≥80% island suppression on v2.0 physics

### Publication Metrics

- ✅ Paper submitted to JCP/CPC or PoP/NF
- ✅ Positive reviews (address all reviewer concerns)
- ✅ Accepted for publication

### Open-Source Metrics

- ✅ Complete documentation (>20 pages)
- ✅ 3+ tutorials (beginner, intermediate, advanced)
- ✅ Test coverage >90%
- ✅ CI passing on main/develop branches

---

## Dependencies on External Work

### Critical Path Items

1. **Symplectic integrator literature** (v1.1 start)
   - Owner: 小P
   - Deadline: Before v1.1 implementation

2. **Structure-preserving discretization papers** (before v1.2)
   - Owner: 小E + 小P
   - Deadline: v1.1 completion

3. **Novelty check (2020-2025 literature)** (before v1.2)
   - Owner: 小E
   - Deadline: v1.1 mid-point

4. **Benchmark cases** (before v1.3)
   - Owner: 小P
   - Deadline: v1.2 completion

### Optional Items

- External expert consultation (v1.2)
- M3D-C1/NIMROD collaboration (v1.3)
- GPU cluster access (v1.3)

---

## Communication Plan

### Internal Milestones

**End of v1.1:**
- Demo: Toroidal solver + RL training
- Decision: Proceed to v1.2 or adjust scope?

**Mid-v1.2:**
- Review: Discretization design with external expert (if needed)
- Decision: Full conservation or simplified approach?

**End of v1.2:**
- Demo: Long-time stability (10⁴ timesteps)
- Decision: Ready for v1.3 validation?

**End of v1.3:**
- Review: Complete paper draft
- Decision: Submit or iterate?

### External Communication

**Conferences (if applicable):**
- APS-DPP 2026 (Nov): Poster/talk on v1.1-v1.2 progress
- SciPy 2027 (Jul): Talk on RL + structure-preserving physics (if v2.0 done)

**Preprints:**
- arXiv submission after paper draft complete (protect priority)

---

## Comparison: v1.0 vs v2.0

| Feature | v1.0 (Current) | v2.0 (Target) |
|---------|----------------|---------------|
| **Geometry** | Cylindrical (r,z) | Toroidal (R,φ,Z) ✅ |
| **Time Integration** | RK4 (standard) | Symplectic ✅ |
| **Spatial Discretization** | FDM (standard) | Energy-conserving ✅ |
| **Energy Drift** | O(dt²), ~0.1% | O(dt⁴), <1e-6 ✅ |
| **Long-time Stability** | 200 steps | >10⁴ steps ✅ |
| **RL Training** | PPO only | PPO + Offline RL ✅ |
| **Benchmarks** | Internal only | Published cases ✅ |
| **Publication Quality** | Proof-of-concept | JCP/PoP level ✅ |
| **Open-Source Ready** | Basic | Complete ✅ |

---

## License Plan (for v2.0 release)

**Recommendation:** MIT or Apache 2.0

**Rationale:**
- Academic-friendly
- Encourages adoption
- Compatible with most fusion/ML codebases

**Decision Point:** Before v1.3 open-source prep

---

## Notes and Decisions Log

### 2026-03-17: Initial Planning

**Participants:** YZ, 小P, 小A, 小E, ∞

**Key Decisions:**
1. ✅ Adopt 3-phase plan (v1.1 → v1.2 → v1.3)
2. ✅ Target 7-9 months timeline
3. ✅ v1.2 is critical path, may need external help
4. ✅ 小A to research sample-efficient RL in parallel
5. ✅ 小E to verify novelty via literature search

**Action Items:**
- [ ] 小P: Symplectic integrator literature search (by v1.1 start)
- [ ] 小E: 2020-2025 structure-preserving MHD+ML papers (2 weeks)
- [ ] 小A: Offline RL survey (during v1.1)
- [ ] ∞: Identify external expert candidates (by v1.1 end)

---

**Document Status:** Living document, updated at each milestone  
**Last Updated:** 2026-03-17  
**Next Review:** v1.1 completion (estimated 2026-05)
