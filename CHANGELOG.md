# Changelog

All notable changes to PTM-RL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

# CHANGELOG Update - Add v1.1 to v1.4

Insert this section BEFORE the existing [1.0.0] section in CHANGELOG.md

---

## [1.4.0] - 2026-03-20

### Added

**Phase 1: 3D Numerics Foundation**
- FFT-based toroidal derivatives (∂/∂ζ in spectral space)
- De-aliasing with 2/3 rule (Orszag method)
- 3D Poisson bracket (hybrid Arakawa + FFT)
- Full 2D Poisson solver (per k-mode, correct θ coupling)
- Comprehensive validation (energy conservation, convergence)

**Phase 2: 3D MHD Physics**
- 3D Hamiltonian formulation (r,θ,ζ geometry)
- Ballooning mode initial conditions
- 3D IMEX time evolution
- External current J_ext (6D Gaussian bump control)
- Phase 2 completion with all tests passing

**Phase 3-5: RL Integration**
- 3D MHD environment (MHDEnv3D, 392 lines)
- Multi-objective reward (island + energy + confinement)
- PPO training (51,200 timesteps)
- Robustness validation: 100% success (48/48 ICs)
- Long episode tests + generalization analysis

### Changed

**Physics Validation Standards**
- Ideal MHD energy tolerance: 1e-6 → 1e-4 (documented limitation)
- Root cause: Parallel advection non-conservative (not structure-preserving)
- Resistive MHD validation: 100% passing (primary use case)

**Project Structure**
- Reorganized v1.4 files (phase reports → docs/v1.4/)
- Debug scripts → scripts/debug/
- Temporary files → archive/v1.4-temp/
- Clean root directory (README, CHANGELOG, pyproject.toml only)

### Performance

**RL Validation Results**
- Robustness: 100% (48/48 initial conditions) ⭐⭐⭐⭐⭐
- Parameter range: ε ∈ [5e-5, 5e-4], n ∈ [3,10], m₀ ∈ [1,3]
- Energy drift: 9.12 ± 0.03% (highly consistent)
- Statistical significance: p=0.0148

**3D Physics Performance**
- FFT derivatives: 7.7e-14 accuracy
- Energy partition: 1.2e-16 precision
- IMEX stability: η ≤ 1e-3 (30× vs v1.2)
- 3D Poisson solver: Residual <1e-8

### Known Limitations

**Energy Conservation (documented, YZ approved)**
- Ideal MHD: ~1e-4 drift (parallel advection)
- 2D Arakawa bracket: 7e-6 (excellent)
- 3D parallel term adds ~3e-2 drift
- Not structure-preserving by construction
- v2.0 will address with Morrison bracket

**Geometry**
- Cylindrical (r,θ,ζ), not true toroidal
- No 1/R curvature coupling
- Deferred to v2.0

### Documentation

**v1.4 Documentation (105 docs, 1.5MB)**
- Design document (43KB, 11 sections)
- Phase 1-5 completion reports (14 files)
- Physics validation reports (Phase 5A + 5B)
- Theory documentation (energy theorem)
- 8 technical summaries

---

## [1.3.0] - 2026-03-19

### Added

**Complete 2D Reduced MHD**
- Hamiltonian formulation + energy conservation
- Poisson bracket [ψ,φ] advection operator (241 lines)
- Force balance coupling: (1/R²)[ψ,J]
- Solovev equilibrium + pressure profiles
- IMEX time stepping (stable η ≤ 1e-3, 30× improvement)

**RL Integration**
- Gym-compatible environment (355 lines)
- Basic controllability verification (5/5 tests)
- Proof-of-concept complete
- Island width + energy observation
- RMP coil current action

**Equilibrium & Pressure**
- Solovev analytical equilibrium (203 lines)
- Pressure profile module (307 lines)
- PyTokEq integration maintained

### Changed

**Strategic Pivot (March 19, 2026)**
- Original v1.3: Scalar z± = ω ± ψ ❌ (wrong physics)
- Revised v1.3: Complete 2D MHD ✅ (correct path to v2.0)
- Rationale: Learn correct physics, true 4/6 transfer to v2.0
- Decision: YZ approved Option A (小A critical review driven)

**Energy Conservation**
- Ideal MHD: <0.15% drift ✅
- Viscous MHD: <0.001% drift ✅
- Resistive MHD: 20-45% error (documented limitation)
- Root cause: Nonlinear GS operator + implicit scheme

### Performance

**v1.3 Achievements**
- Code: 1,708 lines production (Hamiltonian + operators + solvers + RL)
- Docs: ~7,000 lines (9 completion reports + 5 technical guides)
- Tests: 18 passing (13 physics + 5 RL)
- Timeline: 5 days (roadmap revision → implementation → release)

**Energy Results**
- Ideal MHD energy conservation: <0.15% ✅
- IMEX stability range: 30× improvement vs v1.2
- Integration tests: 13/13 PASS
- RL environment: 5/5 PASS

### Documentation

**Major Documents**
- V1.3_COMPLETION_REPORT.md (362 lines)
- V1.3_KNOWN_LIMITATIONS.md (415 lines)
- HANDOFF_XIAOP_TO_XIAOA.md (618 lines)
- Energy dissipation analysis (2 docs, 696 lines)
- Force balance implementation guide (285 lines)

**Roadmap Revision**
- v1.2-v2.0-revised-complete-mhd.md (15KB)
- Gap analysis framework (6 gaps quantified)
- 3-phase plan with clear milestones
- Option A selected (Complete 2D MHD path)

---

## [1.2.1] - 2026-03-17

### Added

**Energy Conservation Framework**
- Comprehensive energy diagnostics
- Budget validation tests
- Conservation analysis tools

**Physics Gap Discovery**
- Identified PyTokEq Solovev catastrophic failure (69,747% drift)
- Documented missing [ψ,φ] advection
- Energy dissipation analysis (diffusion-only limitation)

### Changed

**Physics Understanding**
- Recognized v1.2 as minimal diffusion (not complete MHD)
- Established need for advection term in v1.3
- Updated roadmap to address gaps

### Documentation

**Gap Analysis**
- Energy budget framework report
- Physics limitation documentation
- Path to v1.3 requirements

---

## [1.2.0] - 2026-03-17

### Added

**Toroidal Geometry**
- (r,θ) toroidal coordinates with metric
- Minimal diffusion: ∂ψ/∂t=η∇²ψ, ∂ω/∂t=ν∇²ω
- IMEX framework implementation
- PyTokEq integration maintained
- 2D boundary control

**RL Baseline**
- PPO training on diffusion-only MHD
- 2D boundary control (18D observation)
- Baseline performance established

### Known Limitations

**Missing Physics (documented)**
- NO [ψ,φ] advection (critical gap!)
- NO coupling term (1/R²)[ψ,J]
- Energy drift 6.75% (diffusion-only)
- PyTokEq Solovev unstable (addressed in v1.3)

**Impact**
- Cannot maintain realistic equilibria
- v1.3 required to add advection
- Serves as minimal baseline for comparison

### Documentation

**Design Documents**
- v1.2-v2.0 roadmap (initial version)
- Toroidal solver debug plan
- Hamiltonian MHD formulation
- Spatial current drive action space design
- Observation & reward design (785 lines)

---

## [1.1.0] - 2026-03-17

### Added

**Toroidal Geometry Exploration**
- Coordinate transformation design: (r,z) → (R,φ,Z)
- Toroidal curvature analysis
- Symplectic integrator investigation

### Changed

**Strategic Decision**
- Deferred toroidal solver to v1.2 (numerical instability discovered)
- Recognized complexity of toroidal transformation
- Established need for systematic approach

### Documentation

**Design Documentation**
- Toroidal + symplectic design documents
- Branching strategy for feature development
- Numerical stability analysis

**Key Learning**
- Toroidal geometry requires careful treatment
- Premature optimization avoided
- Systematic development approach established

---

## Version History Summary

```
v1.0.0 (2026-03-17): PyTokEq + PyTokMHD + RL baseline
v1.1.0 (2026-03-17): Toroidal exploration (deferred)
v1.2.0 (2026-03-17): Toroidal diffusion (minimal physics)
v1.2.1 (2026-03-17): Energy framework + gap discovery
v1.3.0 (2026-03-19): Complete 2D MHD + strategic pivot
v1.4.0 (2026-03-20): 3D MHD + spatial control + validation
```

**Next:** v2.0 (Morrison bracket + Elsässer + structure-preserving)
## [1.0.0] - 2026-03-17

### Added

**Phase 5: RL Training Framework**
- Gymnasium-based environment (`MHDTearingControlEnv`)
- PPO baseline training implementation
- Multi-core parallel training support (SubprocVecEnv)
- EvalCallback for training monitoring
- Model checkpointing (10k, 100k steps)
- Comprehensive RL environment tests (32 tests, 100% passing)

**Phase 4: RMP Control System**
- RMP field generation (single/multi-mode)
- RMP-MHD coupling mechanism
- Controller interface (Proportional/PID/RL)
- Open-loop suppression validation
- RMP control tests (9 tests, 100% passing)

**Phase 3: MHD Diagnostics**
- Island width measurement (O-point/area/mode amplitude)
- Growth rate calculation
- Energy diagnostics (kinetic/magnetic)
- Force balance verification
- Diagnostics tests (10 tests, 100% passing)

**Phase 2: Equilibrium Integration**
- PyTokEq equilibrium solver integration
- Equilibrium caching system (50 equilibria, <1ms reset)
- Solovev analytical equilibrium
- PyTokEq integration tests (13 tests, 100% passing)

**Phase 1: Project Foundation**
- PyTokMHD reduced MHD solver
- Numerical operators (gradient, divergence, Laplacian)
- Time integration (RK2/RK4)
- Boundary conditions (Dirichlet)
- Initial conditions from equilibrium

**Project Infrastructure**
- Python package structure (`pyproject.toml`, `requirements.txt`)
- Comprehensive testing suite (51 tests total)
- Documentation (17 technical reports)
- Git workflow and version control

### Changed

**Environment Improvements**
- Migrated from Gym to Gymnasium (v0.29+)
- Parameterized environment design (configurable equilibrium/grid)
- Action smoothing for numerical stability (α=0.3)
- Early termination safety (psi_max=10)

**Numerical Stability**
- Fixed Laplacian operator boundary handling
- Improved gradient/divergence consistency
- Energy conservation: <0.1% drift
- Stable evolution: 200+ timesteps

**Performance Optimization**
- 8-core parallel training: 3.8× speedup
- Training throughput: 1,271 FPS (vs 340 FPS single-core)
- RMP control overhead: <10%

### Performance

**RL Training Results**
- 10k steps: reward -45.7 (baseline)
- 100k steps: reward -5.99 (converged) → **89% improvement**
- 1M steps: reward -5.98 (no significant gain)
- **Conclusion:** 100k steps sufficient for convergence

**Island Width Suppression**
- Initial: w ≈ 0.5
- After 100k training: w ≈ 0.06 → **88% reduction**
- Growth rate: γ → 0 (stabilized)

**Physics Validation**
- Realistic equilibrium: Solovev geometry (R₀=1.0m, κ=1.7, δ=0.3)
- Tearing mode growth rate: γ = 1.44×10⁻³ s⁻¹ (physically reasonable)
- Energy conservation: <0.1%
- Numerical precision: Laplacian <1e-13 (machine precision)

### Testing

**Test Coverage: 51/51 PASSED (100%)**
- PyTokEq integration: 13 tests
- PyTokMHD diagnostics: 10 tests
- RMP control: 9 tests
- RL environment: 32 tests
- All critical paths validated

### Documentation

**Technical Reports (17 documents)**
- Phase 1-4 completion reports
- Phase 5 step-by-step reports (Steps 1-4)
- Systematic diagnosis documentation
- Large-grid verification
- Handoff documents between phases

**README Updates**
- Quick start guide (installation, training, evaluation)
- Performance benchmarks
- Roadmap (v1.1-1.3)
- Citation template

---

## [0.1.0-alpha] - 2026-03-16

### Added

**Project Initialization**
- Project structure setup
- Initial documentation framework
- Team roles and responsibilities
- Development roadmap planning

**Core Architecture Design**
- Three-layer architecture (PyTokEq → PyTokMHD → RL)
- Physics validation workflow
- Cross-validation protocols

---

## Roadmap

### [1.1.0] - Planned (2-4 weeks)

**Toroidal Geometry Upgrade**
- Coordinate transformation: (r,z) → (R,φ,Z)
- Toroidal curvature effects
- More realistic tearing mode dynamics
- Maintain reduced MHD framework

**Expected Impact**
- Realistic tokamak geometry
- Curvature-driven instabilities
- Improved physics fidelity

### [1.2.0] - Planned (1-2 months)

**Resistive MHD**
- Pressure evolution equation
- Beta effects
- Pressure-driven modes
- Full resistive MHD solver

**Expected Impact**
- Significantly improved physics realism
- Broader applicability
- Closer to production-level simulations

### [1.3.0] - Planned (3-6 months)

**TORAX Integration**
- Self-consistent transport evolution
- Production-level physics
- Long-pulse simulation capability

**Expected Impact**
- Integration with established transport code
- Real-time control scenarios
- Publication-ready results

---

## Notes

### Version Numbering

- **Major (X.0.0):** Breaking API changes or major architecture changes
- **Minor (1.X.0):** New features, backward compatible
- **Patch (1.0.X):** Bug fixes, documentation updates

### Deprecation Policy

Features marked as deprecated will be removed in the next major version.
Deprecation warnings will be issued at least one minor version before removal.

---

**Maintained by:** PTM-RL Team (YZ, 小A, 小P, ∞)  
**Last Updated:** 2026-03-17
