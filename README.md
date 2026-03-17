# PTM-RL - Plasma Tearing Mode RL Framework

**Status:** ✅ v1.0 Complete  
**Version:** 1.0.0  
**Started:** 2026-03-16  
**v1.0 Released:** 2026-03-17

---

## Overview

PTM-RL is a physics-based reinforcement learning framework for tokamak tearing mode control, integrating realistic equilibrium (PyTokEq) with reduced MHD dynamics (PyTokMHD) and RL training.

### Core Achievement (v1.0)

**✅ End-to-end pipeline validated:**
- PyTokEq → PyTokMHD → RL chain complete
- Realistic tokamak equilibrium (Solovev geometry)
- Tearing mode suppression via RL control
- **89% reward improvement** (100k training steps)

---

## Architecture

```
Layer 1: PyTokEq Equilibrium Solver
    ↓ (Realistic tokamak geometry: R₀=1.0m, κ=1.7, δ=0.3)
Layer 2: PyTokMHD Dynamics (Reduced MHD)
    ↓ (Tearing mode evolution: γ = 1.44×10⁻³ s⁻¹)
Layer 3: RL Control Framework (Gymnasium)
    ↓ (PPO baseline: island width w~0.5 → w~0.06)
Validated Tearing Mode Suppression ✅
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/callme-YZ/ptm-rl.git
cd ptm-rl

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

### Run Training

```bash
# PPO training (100k steps, 8-core parallel)
python scripts/train_ppo_baseline.py --total-timesteps 100000 --n-envs 8

# Training time: ~78s (8-core)
# Expected reward: -5.99 (island width w~0.06)
```

### Evaluation

```python
from pytokmhd.rl import MHDTearingControlEnv
from stable_baselines3 import PPO

# Load trained model
model = PPO.load('models/ppo_baseline_100k.zip')

# Create environment
env = MHDTearingControlEnv(equilibrium_type='solovev', grid_size=64)

# Evaluate
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

print(f"Island width: {info['island_width']:.4f}")
```

---

## Phase 5: RL Training Framework ✅

### Achievements

**Core Functionality:**
- ✅ Gymnasium environment (32/32 tests PASSED)
- ✅ PPO baseline training (100k steps converged)
- ✅ **89% reward improvement** (physics validated)
- ✅ Island width suppression: w~0.5 → w~0.06
- ✅ Multi-core parallel training (8-core, 1,271 FPS)

**Physics Quality:**
- Realistic equilibrium (Solovev tokamak geometry)
- Tearing mode growth rate: γ = 1.44×10⁻³ s⁻¹ ✅
- Energy conservation: <0.1% ✅
- Numerical stability: 200+ steps ✅

**Training Performance:**
- 10k steps: reward -45.7 (baseline)
- 100k steps: reward -5.99 (converged) → **89% improvement**
- 1M steps: reward -5.98 (no significant gain)
- **Conclusion:** 100k steps sufficient for convergence ✅

### Environment Features

**Parameterized Design:**
- Configurable equilibrium types (`simple` or `solovev`)
- Adjustable grid size (32×32 to 128×128)
- Action smoothing (α=0.3) for numerical stability
- Early termination (psi_max=10) for safety

**Observation Space (25D):**
- Magnetic flux: psi statistics (6D)
- Vorticity: omega statistics (6D)
- Diagnostics: island width, growth rate, energy (7D)
- Grid info: r/z extent (4D)
- Equilibrium type (2D)

**Action Space (1D):**
- RMP amplitude: [-1, 1] (scaled to [0, 1] internally)

**Reward Function:**
```python
reward = -island_width - 0.1 * growth_rate - 0.01 * action²
```

---

## Phase 1-4: Foundation ✅

### Phase 1: PyTokEq Integration
- Real tokamak equilibrium solver
- Solovev analytical solution
- q-profile validation

### Phase 2: Equilibrium Cache
- Fast equilibrium loading (<1ms)
- Interpolation accuracy: <0.1%
- Cache build time: <5min (one-time)

### Phase 3: MHD Diagnostics (1,566 lines)
- Island width measurement
- Growth rate calculation
- Energy diagnostics
- Testing: ✅ 10/10 PASSED

### Phase 4: RMP Control System (2,271 lines)
- RMP field generation (m=2, n=1 mode)
- RMP-MHD coupling
- Controller interface (P/PID/RL)
- Testing: ✅ 9/9 PASSED

**Physics Validation:**
- Laplacian precision: <1e-13 ✅
- RMP overhead: <10% ✅
- Tearing mode growth: γ = 1.44×10⁻³ s⁻¹ ✅

---

## Performance Benchmarks

### Training Speed
- Single-core: ~340 FPS
- 8-core parallel: ~1,271 FPS → **3.8× speedup**
- 100k steps: 78 seconds (8-core)

### MHD Evolution
- Time step: dt = 1e-4
- Numerical stability: 200+ steps
- Energy conservation: <0.1%

### Control Performance
- Island width reduction: **88%** (w~0.5 → w~0.06)
- Growth rate stabilization: γ → 0
- Action efficiency: minimal actuation (<0.01)

---

## Testing

### Run All Tests
```bash
# RL environment tests (32 tests)
pytest src/pytokmhd/tests/test_rl_env.py -v

# MHD diagnostics tests (10 tests)
pytest src/pytokmhd/tests/test_diagnostics.py -v

# RMP control tests (9 tests)
pytest src/pytokmhd/tests/test_rmp_control.py -v

# All tests
pytest src/pytokmhd/tests/ -v
```

**Test Coverage:** 51/51 tests PASSED ✅

---

## Documentation

### Project Reports
- [Phase 1-4 Completion Reports](PHASE1_COMPLETION_REPORT.md)
- [Phase 5 Step 1-4 Reports](PHASE5_STEP1_FINAL_REPORT.md)
- [Gymnasium Migration](PHASE5_STEP2.5_GYMNASIUM_MIGRATION.md)
- [PyTokEq Integration](PHASE5_STEP3_PYTOKEQ_INTEGRATION.md)

### Technical Docs
- [Project Plan](PROJECT_PTM_RL.md)
- [Status Tracking](STATUS.md)
- [Systematic Diagnosis](SYSTEMATIC_DIAGNOSIS.md)

---

## Roadmap (v1.1+)

### v1.1: Toroidal Geometry Upgrade (Priority)
**Goal:** Upgrade from cylindrical (r,z) to toroidal (R,φ,Z) coordinates

**Physics Benefits:**
- Realistic toroidal curvature effects
- Curvature-driven instabilities
- More accurate tearing mode dynamics

**Timeline:** 2-4 weeks

### v1.2: Resistive MHD (Medium Term)
- Add pressure evolution
- Beta effects
- More realistic plasma dynamics

**Timeline:** 1-2 months

### v1.3: TORAX Integration (Long Term)
- Self-consistent transport evolution
- Production-level physics
- Long-pulse simulation

**Timeline:** 3-6 months

---

## Team

- **Physics Lead:** 小P ⚛️ (MHD validation, diagnostics)
- **ML/RL Lead:** 小A 🤖 (RL framework, training)
- **PM:** ∞ (Coordination, documentation)
- **Decision:** YZ 🐙 (Strategy, direction)

---

## License

MIT License (see LICENSE file)

---

## Citation

If you use PTM-RL in your research, please cite:

```bibtex
@software{ptm_rl_2026,
  title = {PTM-RL: Physics-Based Reinforcement Learning for Tokamak Tearing Mode Control},
  author = {YZ Team},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/callme-YZ/ptm-rl}
}
```

---

## Acknowledgments

- PyTokEq equilibrium solver
- Stable-Baselines3 RL library
- Gymnasium environment standard

---

**Created:** 2026-03-16  
**v1.0 Released:** 2026-03-17  
**Last Updated:** 2026-03-17

---

## Contact

For questions or collaboration:
- GitHub Issues: https://github.com/callme-YZ/ptm-rl/issues
- Project Lead: YZ
