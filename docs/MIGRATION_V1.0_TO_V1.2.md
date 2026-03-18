# Migration Guide: v1.0 → v1.2

**Target Users:** Developers upgrading from PTM-RL v1.0 to v1.2  
**Upgrade Type:** Major (breaking changes in geometry and integration)  
**Date:** 2026-03-18  
**Author:** 小P ⚛️

---

## Overview

**v1.2 Major Changes:**
1. **Geometry:** Cylindrical → Toroidal
2. **Integrator:** RK4 → Symplectic (Störmer-Verlet)
3. **Observation:** 9D → 19D (phase-resolved Fourier)
4. **Environment:** Updated API (action-aware solver)

**Migration Effort:** ~2-4 hours (depending on custom code)

**Compatibility:** **Breaking** — v1.0 code will not run without changes

---

## Quick Start

### Minimal Migration (5 minutes)

**If you only use the environment:**

```python
# v1.0
from pytokmhd.rl.mhd_env import TearingMHDEnv

env = TearingMHDEnv(
    nr=32, 
    nz=64,  # ← Cylindrical
    dt=1e-4
)

# v1.2
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv

env = ToroidalMHDEnv(
    R0=1.0,      # ← Major radius (new)
    a=0.3,       # ← Minor radius (new)
    nr=32,
    ntheta=64,   # ← Was "nz"
    dt=1e-4
)
```

**That's it!** If you only called `env.reset()` and `env.step(action)`, this is all you need.

---

## Breaking Changes

### 1. Geometry: Cylindrical → Toroidal

#### Grid Constructor

**v1.0:**
```python
from pytokmhd.geometry import Grid

grid = Grid(
    nr=32,
    nz=64,      # Axial direction
    r_min=0.0,
    r_max=1.0,
    z_min=-1.0,
    z_max=1.0
)
```

**v1.2:**
```python
from pytokmhd.geometry import ToroidalGrid

grid = ToroidalGrid(
    R0=1.0,     # Major radius (tokamak center)
    a=0.3,      # Minor radius (plasma size)
    nr=32,      # Radial points (same)
    ntheta=64   # Poloidal points (was "nz")
)
```

**Migration Notes:**
- `nz` → `ntheta` (conceptual: axial → poloidal)
- `r_min, r_max, z_min, z_max` → computed from `R0, a`
- Grid is now **periodic in θ** (was periodic in z)

#### Grid Attributes

**v1.0:**
```python
grid.r    # (nr,) 1D array
grid.z    # (nz,) 1D array
grid.R    # (nr, nz) meshgrid R = r (cylindrical)
grid.Z    # (nr, nz) meshgrid Z = z
```

**v1.2:**
```python
grid.r_grid      # (nr, ntheta) 2D array
grid.theta_grid  # (nr, ntheta) 2D array
grid.R_grid      # (nr, ntheta) R = R0 + r*cos(θ)
grid.Z_grid      # (nr, ntheta) Z = r*sin(θ)
```

**Migration:**
```python
# v1.0
r = grid.r
z = grid.z

# v1.2
r = grid.r_grid[:, 0]  # Radial coordinate (constant in θ)
theta = grid.theta_grid[0, :]  # Poloidal angle (constant in r)
```

---

### 2. Operators: Metric Corrections

#### Import Changes

**v1.0:**
```python
from pytokmhd.operators import (
    gradient, 
    laplacian, 
    poisson_bracket
)
```

**v1.2:**
```python
from pytokmhd.operators import (
    gradient_toroidal,
    laplacian_toroidal,
    poisson_bracket_toroidal
)
```

**All operators now require `grid` argument.**

#### Gradient

**v1.0:**
```python
df_dr, df_dz = gradient(f, grid.dr, grid.dz)
```

**v1.2:**
```python
df_dr, df_dtheta = gradient_toroidal(f, grid)
# df_dtheta is physical component: (1/r) ∂f/∂θ
```

**Migration:**
```python
# v1.0 code
grad_r, grad_z = gradient(psi, dr, dz)

# v1.2 equivalent
grad_r, grad_theta = gradient_toroidal(psi, grid)
# Note: grad_theta already includes 1/r factor
```

#### Laplacian

**v1.0:**
```python
lap_f = laplacian(f, grid.dr, grid.dz)
# ∇²f = ∂²f/∂r² + (1/r)∂f/∂r + ∂²f/∂z²
```

**v1.2:**
```python
lap_f = laplacian_toroidal(f, grid)
# ∇²f = (1/r)∂/∂r(r ∂f/∂r) + (1/r²)∂²f/∂θ²
```

**Key Difference:** Toroidal operator uses (r, θ) metric, not (r, z).

#### Poisson Bracket

**v1.0:**
```python
pb = poisson_bracket(f, g, grid.dr, grid.dz)
# [f,g] = ∂f/∂r ∂g/∂z - ∂f/∂z ∂g/∂r
```

**v1.2:**
```python
pb = poisson_bracket_toroidal(f, g, grid)
# [f,g] = (1/R²)(∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)
```

**Key Difference:** 1/R² factor from toroidal metric.

---

### 3. Integrator: RK4 → Symplectic

#### Constructor

**v1.0:**
```python
from pytokmhd.integrators import RK4Integrator

solver = RK4Integrator(
    grid=grid,
    dt=1e-4,
    eta=1e-6,
    nu=1e-6
)
```

**v1.2:**
```python
from pytokmhd.integrators import SymplecticIntegrator

solver = SymplecticIntegrator(
    grid=grid,  # ToroidalGrid
    dt=1e-4,
    eta=1e-6,
    nu=1e-6,
    operator_splitting=True  # New: use operator splitting
)
```

#### Initialization

**v1.0:**
```python
solver.initialize(psi0, omega0)
```

**v1.2:**
```python
solver.initialize(psi0, omega0)
# Same API! ✅
```

**Important:** `omega0` should equal `∇²psi0` for consistency.

```python
# Recommended
from pytokmhd.operators import laplacian_toroidal

psi0 = ...  # Initial flux
omega0 = laplacian_toroidal(psi0, grid)  # Consistent IC

solver.initialize(psi0, omega0)
```

#### Time Stepping

**v1.0:**
```python
solver.step()
# Advances one RK4 step
```

**v1.2:**
```python
solver.step()
# Advances one symplectic step

# NEW: Action support
action = np.array([1.2, 0.8])  # [eta_mult, nu_mult]
solver.step(action=action)
```

**Migration:**
```python
# v1.0 code
for _ in range(100):
    solver.step()

# v1.2 equivalent
for _ in range(100):
    solver.step()  # Same! Backward compatible ✅

# v1.2 with action
for _ in range(100):
    action = policy(obs)  # RL policy
    solver.step(action=action)
```

#### Energy Computation

**v1.0:**
```python
E = solver.compute_energy()
# Includes kinetic + magnetic
```

**v1.2:**
```python
E = solver.compute_energy()
# Same formula, toroidal volume element
# E = ∫[1/2 ω² + 1/2 |∇ψ|²] dV
# dV = J dr dθ dφ (J = r*R)
```

**Migration:** No code changes needed ✅

---

### 4. Observation Space: 9D → 19D

#### v1.0 Observation

**v1.0:**
```python
obs_dict = get_observation(psi, omega, grid)
# Returns dict with ~9D observation
```

**v1.2:**
```python
from pytokmhd.rl.observations import MHDObservation

obs_handler = MHDObservation(psi_eq, E_eq, grid)
obs_dict = obs_handler.get_observation(psi, omega)
```

#### Observation Vector Changes

**v1.0 (9D):**
```python
[
    psi_mode_0,    # Amplitude only
    psi_mode_1,
    ...,
    psi_mode_7,
    energy         # Total energy
]
```

**v1.2 (19D):**
```python
[
    psi_mode_0_re, psi_mode_0_im,  # Phase-resolved
    psi_mode_1_re, psi_mode_1_im,
    ...,
    psi_mode_7_re, psi_mode_7_im,  # 16D
    energy,                         # Relative (E-E_eq)/E_eq
    energy_drift,                   # Absolute |E-E_eq|/E_eq
    div_B_max                       # Constraint proxy
]
```

**Migration:**

If your RL policy expects 9D input, you need to:

**Option A: Update network input size**
```python
# v1.0
policy = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]})

# v1.2 — no change needed!
# Stable-Baselines3 auto-detects observation space ✅
policy = PPO("MlpPolicy", env)
```

**Option B: Extract v1.0-compatible observation**
```python
# If you need backward compatibility
obs_19d = obs_dict['vector']  # Full v1.2 observation

# Extract amplitude-only (like v1.0)
modes = obs_dict['psi_modes']  # (16,) complex modes
amplitudes = np.abs(modes[::2] + 1j*modes[1::2])  # (8,) amplitudes
energy = obs_19d[16]

obs_9d = np.concatenate([amplitudes, [energy]])  # v1.0-like
```

---

### 5. Environment API

#### Class Name

**v1.0:**
```python
from pytokmhd.rl.mhd_env import TearingMHDEnv

env = TearingMHDEnv(...)
```

**v1.2:**
```python
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv

env = ToroidalMHDEnv(...)
```

#### Constructor Parameters

**v1.0:**
```python
env = TearingMHDEnv(
    nr=32,
    nz=64,        # ← Cylindrical
    dt=1e-4,
    max_steps=1000,
    eta=1e-6,
    nu=1e-6
)
```

**v1.2:**
```python
env = ToroidalMHDEnv(
    R0=1.0,       # ← New: major radius
    a=0.3,        # ← New: minor radius
    nr=32,
    ntheta=64,    # ← Was "nz"
    dt=1e-4,
    max_steps=1000,
    eta=1e-6,
    nu=1e-6,
    w_energy=1.0,      # ← New: reward weight
    w_action=0.01,     # ← New: action penalty
    w_constraint=0.0   # ← New: constraint (disabled in v1.2)
)
```

**Mapping:**
```python
# v1.0 → v1.2
nr → nr (same)
nz → ntheta (conceptual change)
R0 = 1.0 (default, was implicit)
a = 0.3 (default, was r_max)
```

#### Reset Method

**v1.0:**
```python
obs = env.reset()
```

**v1.2:**
```python
obs, info = env.reset()
# Returns (observation, info_dict)
```

**Migration:**
```python
# v1.0 code
obs = env.reset()

# v1.2 (Gymnasium standard)
obs, info = env.reset()

# Or ignore info
obs, _ = env.reset()
```

#### Step Method

**v1.0:**
```python
obs, reward, done, info = env.step(action)
```

**v1.2:**
```python
obs, reward, terminated, truncated, info = env.step(action)
# Gymnasium API: "done" split into "terminated" and "truncated"
```

**Migration:**
```python
# v1.0 code
obs, reward, done, info = env.step(action)
if done:
    env.reset()

# v1.2
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    env.reset()

# Or combine
done = terminated or truncated
```

#### Action Space

**v1.0:** Not clearly defined (implementation-dependent)

**v1.2:**
```python
env.action_space  # Box([0.5, 0.5], [2.0, 2.0])
# [eta_multiplier, nu_multiplier]
```

**Migration:**

If v1.0 action was different, you need to adapt:

```python
# v1.2 action format
action = np.array([eta_mult, nu_mult])
# eta_mult, nu_mult ∈ [0.5, 2.0]
# Effective: eta = eta_base * eta_mult
```

---

## Compatibility Matrix

| Feature | v1.0 | v1.2 | Compatible? |
|---------|------|------|-------------|
| Grid coordinates | (r, z) | (R, φ, Z) | ❌ Breaking |
| Grid attributes | `.r`, `.z` | `.r_grid`, `.theta_grid` | ❌ Breaking |
| Operators | `gradient(...)` | `gradient_toroidal(..., grid)` | ❌ Breaking |
| Integrator class | `RK4Integrator` | `SymplecticIntegrator` | ❌ Breaking |
| Integrator API | `.step()` | `.step(action=None)` | ✅ Backward compatible |
| Observation dim | 9D | 19D | ❌ Breaking |
| Environment class | `TearingMHDEnv` | `ToroidalMHDEnv` | ❌ Breaking |
| `reset()` return | `obs` | `(obs, info)` | ⚠️ Partial (Gym standard) |
| `step()` return | 4 values | 5 values | ⚠️ Partial (Gym standard) |
| Action format | Implicit | `[eta_mult, nu_mult]` | ⚠️ Depends on v1.0 |

**Legend:**
- ✅ Fully compatible
- ⚠️ Partial (standard upgrade path)
- ❌ Breaking (code changes required)

---

## Step-by-Step Migration

### Step 1: Update Imports

```python
# v1.0
from pytokmhd.geometry import Grid
from pytokmhd.operators import gradient, laplacian, poisson_bracket
from pytokmhd.integrators import RK4Integrator
from pytokmhd.rl.mhd_env import TearingMHDEnv

# v1.2
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import (
    gradient_toroidal,
    laplacian_toroidal,
    poisson_bracket_toroidal
)
from pytokmhd.integrators import SymplecticIntegrator
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
```

### Step 2: Update Grid Creation

```python
# v1.0
grid = Grid(nr=32, nz=64, r_min=0.0, r_max=1.0, z_min=-1.0, z_max=1.0)

# v1.2
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
```

**Mapping r_max to a:**
```python
# If v1.0 used r_max = 0.3
a = r_max  # Minor radius
R0 = 1.0   # Choose aspect ratio R0/a (e.g., 3.3)
```

### Step 3: Update Operator Calls

```python
# v1.0
lap_psi = laplacian(psi, grid.dr, grid.dz)
grad_r, grad_z = gradient(omega, grid.dr, grid.dz)
pb = poisson_bracket(psi, phi, grid.dr, grid.dz)

# v1.2
lap_psi = laplacian_toroidal(psi, grid)
grad_r, grad_theta = gradient_toroidal(omega, grid)
pb = poisson_bracket_toroidal(psi, phi, grid)
```

### Step 4: Update Integrator

```python
# v1.0
solver = RK4Integrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)

# v1.2
solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)

# Initialize (same)
solver.initialize(psi0, omega0)

# Step (same, backward compatible)
solver.step()
```

### Step 5: Update Environment Usage

```python
# v1.0
env = TearingMHDEnv(nr=32, nz=64, dt=1e-4)
obs = env.reset()

for step in range(100):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# v1.2
env = ToroidalMHDEnv(
    R0=1.0, a=0.3, nr=32, ntheta=64, dt=1e-4,
    w_constraint=0.0  # Disable div_B proxy
)
obs, _ = env.reset()

for step in range(100):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Step 6: Update RL Training

```python
# v1.0 (Stable-Baselines3)
from stable_baselines3 import PPO

env = TearingMHDEnv(...)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)

# v1.2 — SAME! ✅
env = ToroidalMHDEnv(...)
model = PPO("MlpPolicy", env)  # Auto-detects 19D observation
model.learn(total_timesteps=100000)
```

**No RL code changes needed** if using SB3! 🎉

---

## Common Migration Issues

### Issue 1: Grid Attribute Error

**Error:**
```python
AttributeError: 'ToroidalGrid' object has no attribute 'r'
```

**Cause:** v1.0 `grid.r` was 1D, v1.2 uses `grid.r_grid` (2D)

**Fix:**
```python
# v1.0
r = grid.r

# v1.2
r = grid.r_grid[:, 0]  # Extract radial coordinate
```

---

### Issue 2: Operator Signature Mismatch

**Error:**
```python
TypeError: gradient() missing 1 required positional argument: 'grid'
```

**Cause:** v1.2 operators require `grid` parameter

**Fix:**
```python
# v1.0
grad_r, grad_z = gradient(f, dr, dz)

# v1.2
grad_r, grad_theta = gradient_toroidal(f, grid)
```

---

### Issue 3: Observation Shape Mismatch

**Error:**
```python
ValueError: Input size mismatch (expected 9, got 19)
```

**Cause:** Policy trained on v1.0 (9D) cannot use v1.2 (19D)

**Fix:**

**Option A: Retrain policy**
```python
# Train new model on v1.2
model = PPO("MlpPolicy", env_v1_2)
model.learn(...)
```

**Option B: Extract 9D subset (not recommended)**
```python
obs_19d = env.reset()[0]
obs_9d = extract_v1_0_compatible(obs_19d)  # Custom function
```

---

### Issue 4: Reset/Step Return Values

**Error:**
```python
ValueError: too many values to unpack (expected 4, got 5)
```

**Cause:** v1.2 uses Gymnasium API (5 return values)

**Fix:**
```python
# v1.0
obs, reward, done, info = env.step(action)

# v1.2
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated  # Combine if needed
```

---

## Testing Your Migration

### Minimal Test

```python
"""
Test basic v1.2 environment.
"""

from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
import numpy as np

# Create environment
env = ToroidalMHDEnv(
    R0=1.0, a=0.3, nr=32, ntheta=64, 
    dt=1e-4, max_steps=100, w_constraint=0.0
)

# Reset
obs, info = env.reset(seed=42)
assert obs.shape == (19,), f"Expected 19D, got {obs.shape}"
print(f"✅ Reset: obs shape {obs.shape}")

# Step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
assert obs.shape == (19,)
print(f"✅ Step: obs shape {obs.shape}, reward {reward:.4f}")

# Episode
obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"✅ Episode complete: {info.get('current_step')} steps")
```

**Expected output:**
```
✅ Reset: obs shape (19,)
✅ Step: obs shape (19,), reward -0.1234
✅ Episode complete: 10 steps
```

---

## Performance Comparison

### Energy Conservation

**Test:** 100 steps, small perturbation

| Metric | v1.0 (RK4) | v1.2 (Symplectic) | Improvement |
|--------|------------|-------------------|-------------|
| Energy drift | 57% | 5.7% | 10× better ✅ |
| Step time | ~5 ms | ~10 ms | 2× slower |
| **Quality/Cost** | — | — | **5× better** |

**Conclusion:** v1.2 is slower but far more accurate.

---

### Computational Cost

**Setup:** 32×64 grid, 100-step episode

| Component | v1.0 | v1.2 | Ratio |
|-----------|------|------|-------|
| Grid creation | <1 ms | ~1 ms | ~1× |
| Initialization | ~5 ms | ~10 ms | 2× |
| Time step | ~5 ms | ~10 ms | 2× |
| Energy | ~0.5 ms | ~1 ms | 2× |
| Observation | ~1 ms | ~3 ms | 3× (Fourier) |
| **Episode total** | ~0.8 s | ~1.3 s | **1.6×** |

**Recommendation:** Accept 1.6× slowdown for 10× better physics.

---

## Rollback Plan

If you need to revert to v1.0:

### Option 1: Git Branch

```bash
# Create migration branch
git checkout -b migrate-v1.2
# ... make changes ...

# If issues, rollback
git checkout main  # v1.0 branch
```

### Option 2: Virtual Environment

```bash
# Keep v1.0 environment
conda create -n ptm-rl-v1.0 python=3.9
conda activate ptm-rl-v1.0
pip install ptm-rl==1.0

# Create v1.2 environment
conda create -n ptm-rl-v1.2 python=3.9
conda activate ptm-rl-v1.2
pip install ptm-rl==1.2
```

### Option 3: Docker

```dockerfile
# Dockerfile.v1.0
FROM python:3.9
RUN pip install ptm-rl==1.0

# Dockerfile.v1.2
FROM python:3.9
RUN pip install ptm-rl==1.2
```

---

## FAQ

### Q1: Do I need to retrain my RL model?

**A:** Yes, because:
- Observation space changed (9D → 19D)
- Toroidal geometry affects dynamics
- Better to start fresh on v1.2

### Q2: Can I use v1.0 checkpoints?

**A:** No, incompatible:
- Different observation dimensions
- Different physics (cylindrical vs toroidal)

### Q3: Is there a compatibility layer?

**A:** No. v1.2 is a major upgrade, clean break preferred.

### Q4: How long does migration take?

**A:**
- Environment-only users: 5-10 minutes
- Custom solver code: 1-2 hours
- Full codebase: 2-4 hours

### Q5: What if I only want symplectic, not toroidal?

**A:** Not supported. Symplectic integrator requires toroidal grid in v1.2.

---

## Checklist

**Before Migration:**
- [ ] Backup v1.0 code (git tag/branch)
- [ ] Review breaking changes
- [ ] Plan testing strategy
- [ ] Allocate time (2-4h)

**During Migration:**
- [ ] Update imports
- [ ] Update grid creation
- [ ] Update operator calls
- [ ] Update integrator
- [ ] Update environment usage
- [ ] Fix reset/step return values

**After Migration:**
- [ ] Run minimal test
- [ ] Verify energy conservation
- [ ] Check observation shape
- [ ] Test episode rollout
- [ ] Retrain RL model
- [ ] Compare performance

**Validation:**
- [ ] All tests pass
- [ ] Energy drift < 10%
- [ ] No runtime errors
- [ ] Observation 19D
- [ ] Action 2D

---

## Support

**Issues?**
- GitHub: https://github.com/callme-YZ/ptm-rl/issues
- Docs: `docs/API_V1.2.md`
- Benchmark: `docs/V1.2_BENCHMARK_REPORT.md`

**Questions:**
- Review test files: `tests/test_step_*.py`
- Check examples: Complete workflow in `docs/API_V1.2.md`

---

**小P签字:** ⚛️  
**Date:** 2026-03-18  
**Version:** 1.2  
**Status:** Migration Guide Complete
