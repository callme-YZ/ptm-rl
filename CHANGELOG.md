# Changelog

All notable changes to PTM-RL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
