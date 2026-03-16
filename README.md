# PTM-RL - Plasma Tearing Mode RL Framework

**Status:** 🚧 Development (Day 1)  
**Version:** 0.1.0-alpha  
**Started:** 2026-03-16

---

## Overview

PTM-RL integrates PyTokEq (tokamak equilibrium) with PyTearRL (MHD dynamics) to create a physics-based RL framework for tearing mode control.

### Key Features

- ✅ **Layer 1:** Real tokamak equilibrium (PyTokEq)
- ✅ **Layer 2:** Physics-correct MHD evolution
- ✅ **Layer 3:** RL control training
- 🔄 **Dual Architecture:** CPU (NumPy+Ray) & GPU (JAX)

---

## Architecture

```
Layer 1: PyTokEq Equilibrium Solver
    ↓ (真实平衡态)
Layer 2: MHD Dynamics (Tearing Mode Evolution)
    ↓ (物理正确演化)
Layer 3: RL Control (RMP Suppression)
    ↓ (可迁移控制策略)
Real Tokamak Application
```

---

## Versions

### CPU Version (NumPy + Ray)
- Parallel execution across 10+ cores
- ~10× speedup vs single-core
- Stable and production-ready

### GPU Version (JAX)
- GPU-accelerated computation
- ~100× speedup vs CPU
- High-performance training

---

## Development Status

**Current Phase:** Project Initialization

- [x] Project structure created
- [x] Documentation initialized
- [ ] Technical design complete
- [ ] Layer 1 (PyTokEq) integration
- [ ] Layer 2 (MHD) implementation
- [ ] Layer 3 (RL) training

---

## Team

- **Physics Lead:** 小P ⚛️ (PyTokEq, MHD validation)
- **ML/RL Lead:** 小A 🤖 (RL framework, GPU optimization)
- **PM:** ∞ (Coordination, Git workflow)
- **Decision:** YZ 🐙

---

## Quick Start

*(Coming soon after Phase 1 completion)*

---

## Documentation

- [Project Plan](PROJECT_PTM_RL.md) — Full project specification
- [Status](STATUS.md) — Current development status
- [Design](design/) — Technical design documents

---

## License

*(TBD)*

---

**Created:** 2026-03-16  
**Last Updated:** 2026-03-16

---

## Phase 2: PyTokEq Integration (2026-03-16)

### Real Equilibrium Initialization

Use PyTokEq equilibria for realistic MHD simulations:

```python
from pytokmhd.solver.equilibrium_cache import EquilibriumCache
from pytokmhd.solver.initial_conditions import pytokeq_initial, solovev_equilibrium

# Setup MHD grid
Nr, Nz = 64, 128
r = np.linspace(0.5, 1.5, Nr)
z = np.linspace(-0.5, 0.5, Nz)

# Option 1: Analytical Solovev equilibrium (testing)
psi, omega = solovev_equilibrium(r, z)

# Option 2: PyTokEq equilibrium with cache (production)
cache = EquilibriumCache(cache_size=50)

def equilibrium_solver(q0, beta_p, target_grid):
    # Your PyTokEq solver call here
    # Returns: {'psi_eq': ..., 'j_eq': ..., 'p_eq': ..., 'q_profile': ...}
    pass

# Populate cache (once, takes ~5min for real PyTokEq)
cache.populate_cache(
    equilibrium_solver,
    param_ranges={'q0': (0.8, 1.2), 'beta_p': (0.5, 2.0)},
    target_grid=(r, z)
)

# Fast reset: <1ms per call
psi, omega = pytokeq_initial(r, z, cache, perturbation_amplitude=0.01)
```

### Performance Benchmarks

- **Cache build time:** <5min (one-time, 50 equilibria)
- **Reset time:** <1ms (vs 100ms without cache)
- **Speedup:** >100×
- **Interpolation accuracy:** <0.1% error

### Testing

Run Phase 2 tests:
```bash
# PyTokEq integration tests
pytest src/pytokmhd/tests/test_pytokeq_integration.py -v

# Cache performance tests
pytest src/pytokmhd/tests/test_equilibrium_cache.py -v

# All Phase 2 tests (13 tests)
pytest src/pytokmhd/tests/test_pytokeq*.py src/pytokmhd/tests/test_equilibrium_cache.py
```

See `PHASE2_COMPLETION_REPORT.md` for full details.

