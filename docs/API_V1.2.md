# PTM-RL API Reference (v1.2)

**PyTokMHD + RL**: Toroidal Symplectic MHD with RL Control

**Version:** 1.2  
**Date:** 2026-03-18  
**Authors:** 小P ⚛️, 小A 🤖

---

## What's New in v1.2

**Major Upgrades:**
1. **Toroidal Geometry** — Cylindrical → Toroidal (R,φ,Z)
2. **Symplectic Integration** — RK4 → Störmer-Verlet (10× energy conservation)
3. **RL Environment** — 19D observation, 2D action, Gym-compatible

**See:** [Migration Guide](MIGRATION_V1.0_TO_V1.2.md) for upgrade path

---

## Table of Contents

1. [Geometry](#geometry)
2. [Operators](#operators)
3. [Integrators](#integrators)
4. [Diagnostics](#diagnostics)
5. [RL Components](#rl-components)
6. [Environment](#environment)

---

## Geometry

### ToroidalGrid

**Location:** `src/pytokmhd/geometry/toroidal_grid.py`

**Description:** Toroidal coordinate system with metric corrections.

#### Constructor

```python
from pytokmhd.geometry import ToroidalGrid

grid = ToroidalGrid(
    R0: float = 1.0,      # Major radius
    a: float = 0.3,       # Minor radius  
    nr: int = 32,         # Radial points
    ntheta: int = 64      # Poloidal points
)
```

#### Attributes

```python
grid.R0        # Major radius
grid.a         # Minor radius
grid.nr        # Radial grid points
grid.ntheta    # Poloidal grid points
grid.r_grid    # r coordinates (nr, ntheta)
grid.theta_grid # θ coordinates (nr, ntheta)
grid.R_grid    # R = R0 + r*cos(θ) (nr, ntheta)
grid.Z_grid    # Z = r*sin(θ) (nr, ntheta)
grid.dr        # Radial spacing
grid.dtheta    # Poloidal spacing
```

#### Methods

```python
# Metric tensor
grid.jacobian() -> np.ndarray
    """Jacobian J = r*R (nr, ntheta)"""

grid.metric_rr() -> np.ndarray
    """g_rr = 1"""

grid.metric_theta_theta() -> np.ndarray  
    """g_θθ = r²"""

grid.metric_phi_phi() -> np.ndarray
    """g_φφ = R²"""

# Volume element
grid.volume_element() -> np.ndarray
    """dV = J * dr * dθ * dφ"""
```

#### Example

```python
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
print(f"Aspect ratio: {grid.R0/grid.a:.1f}")
print(f"Grid shape: {grid.r_grid.shape}")
print(f"Jacobian range: [{grid.jacobian().min():.3f}, {grid.jacobian().max():.3f}]")
```

---

## Operators

### Toroidal Operators

**Location:** `src/pytokmhd/operators/toroidal_operators.py`

**Description:** Differential operators with toroidal metric corrections.

#### gradient_toroidal

```python
from pytokmhd.operators import gradient_toroidal

df_dr, df_dtheta = gradient_toroidal(
    f: np.ndarray,        # Field (nr, ntheta)
    grid: ToroidalGrid    # Geometry
) -> Tuple[np.ndarray, np.ndarray]
```

**Returns:**
- `df_dr`: Radial component ∂f/∂r
- `df_dtheta`: Poloidal component (1/r) ∂f/∂θ (physical component)

**Notes:**
- Second-order centered differences
- Periodic in θ
- Dirichlet BC at r boundaries (if applicable)

#### laplacian_toroidal

```python
from pytokmhd.operators import laplacian_toroidal

lap_f = laplacian_toroidal(
    f: np.ndarray,
    grid: ToroidalGrid
) -> np.ndarray
```

**Formula:**
```
∇²f = (1/r) ∂/∂r(r ∂f/∂r) + (1/r²) ∂²f/∂θ²
```

**Returns:** Laplacian field (nr, ntheta)

#### poisson_bracket_toroidal

```python
from pytokmhd.operators import poisson_bracket_toroidal

pb = poisson_bracket_toroidal(
    f: np.ndarray,
    g: np.ndarray,
    grid: ToroidalGrid
) -> np.ndarray
```

**Formula:**
```
[f,g] = (1/R²) (∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r)
```

**Used in:** Advection terms in MHD equations

---

## Integrators

### SymplecticIntegrator

**Location:** `src/pytokmhd/integrators/symplectic.py`

**Description:** Structure-preserving time integrator for Hamiltonian MHD.

#### Constructor

```python
from pytokmhd.integrators import SymplecticIntegrator

solver = SymplecticIntegrator(
    grid: ToroidalGrid,
    dt: float = 1e-4,              # Time step
    eta: float = 1e-6,             # Resistivity
    nu: float = 1e-6,              # Viscosity
    operator_splitting: bool = True # Use splitting
)
```

#### Methods

##### initialize

```python
solver.initialize(
    psi0: np.ndarray,    # Initial poloidal flux (nr, ntheta)
    omega0: np.ndarray   # Initial vorticity (nr, ntheta)
) -> None
```

**Constraint:** `omega0` should equal `∇²psi0` for consistency.

##### step

```python
solver.step(
    action: Optional[np.ndarray] = None  # RL action [eta_mult, nu_mult]
) -> None
```

**Action format:**
- `None`: Use base eta, nu (default)
- `[eta_multiplier, nu_multiplier]`: Modulate dissipation

**Updates:** `solver.psi`, `solver.omega`, `solver.t`

##### compute_energy

```python
E = solver.compute_energy() -> float
```

**Formula:**
```
E = ∫ [1/2 ω² + 1/2 |∇ψ|²] dV
```

where `dV = J dr dθ dφ`

#### Attributes

```python
solver.psi     # Current poloidal flux (nr, ntheta)
solver.omega   # Current vorticity (nr, ntheta)
solver.t       # Current time
solver.dt      # Time step
solver.eta     # Resistivity
solver.nu      # Viscosity
```

#### Example

```python
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.integrators import SymplecticIntegrator
from pytokmhd.operators import laplacian_toroidal

grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)

# Initialize with equilibrium
r_grid = grid.r_grid
psi0 = r_grid**2 * (1 - r_grid / grid.a)
omega0 = laplacian_toroidal(psi0, grid)

solver.initialize(psi0, omega0)
E0 = solver.compute_energy()

# Time evolution
for _ in range(1000):
    solver.step()

E1 = solver.compute_energy()
print(f"Energy drift: {abs(E1-E0)/E0:.2%}")
```

---

## Diagnostics

### Fourier Analysis

**Location:** `src/pytokmhd/diagnostics/fourier_analysis.py`

#### fourier_decompose

```python
from pytokmhd.diagnostics.fourier_analysis import fourier_decompose

modes = fourier_decompose(
    field: np.ndarray,     # (nr, ntheta)
    grid: ToroidalGrid,
    n_modes: int = 8       # Number of modes to extract
) -> np.ndarray           # Complex modes (n_modes,)
```

**Returns:** Radially-averaged complex Fourier modes [m=0, m=1, ..., m=n_modes-1]

**Formula:**
```
mode[m] = (1/nr) Σ_r FFT_θ(field[r, :])
```

#### reconstruct_from_modes

```python
reconstructed = reconstruct_from_modes(
    modes: np.ndarray,     # Complex modes (n_modes,)
    grid: ToroidalGrid
) -> np.ndarray           # (nr, ntheta)
```

**Use:** Verify mode extraction accuracy.

#### get_dominant_mode

```python
m_dominant, amplitude = get_dominant_mode(
    modes: np.ndarray
) -> Tuple[int, float]
```

**Returns:** Index and amplitude of largest mode.

---

## RL Components

### MHDObservation

**Location:** `src/pytokmhd/rl/observations.py`

**Description:** Extract 19D observation vector from MHD state.

#### Constructor

```python
from pytokmhd.rl.observations import MHDObservation

obs_handler = MHDObservation(
    psi_eq: np.ndarray,    # Equilibrium flux (nr, ntheta)
    E_eq: float,           # Equilibrium energy
    grid: ToroidalGrid
)
```

#### Methods

##### get_observation

```python
obs_dict = obs_handler.get_observation(
    psi: np.ndarray,       # Current flux
    omega: np.ndarray      # Current vorticity
) -> Dict[str, Any]
```

**Returns dict with:**
```python
{
    'psi_modes': np.ndarray,      # (16,) Fourier modes (8 × Re/Im)
    'energy': float,              # Relative energy (E-E_eq)/E_eq
    'energy_drift': float,        # |E-E_eq|/E_eq
    'div_B_max': float,           # max|∇²ψ| (proxy)
    'vector': np.ndarray          # (19,) flattened observation
}
```

**Observation vector (19D):**
- `[0:16]`: psi_modes (phase-resolved)
- `[16]`: energy (relative)
- `[17]`: energy_drift (absolute)
- `[18]`: div_B_max (constraint proxy)

##### get_observation_space

```python
space = obs_handler.get_observation_space() -> gym.spaces.Box
```

**Returns:** Gymnasium Box space for 19D observation.

#### Example

```python
obs_handler = MHDObservation(psi_eq, E_eq, grid)
obs_dict = obs_handler.get_observation(solver.psi, solver.omega)

print(f"Observation shape: {obs_dict['vector'].shape}")
print(f"Energy drift: {obs_dict['energy_drift']:.4f}")
print(f"Dominant mode: m={np.argmax(np.abs(obs_dict['psi_modes'][:8]))}")
```

---

### MHDAction

**Location:** `src/pytokmhd/rl/actions.py`

**Description:** Parameter modulation action handler (v1.2 limitation).

#### Constructor

```python
from pytokmhd.rl.actions import MHDAction

action_handler = MHDAction(
    eta_base: float = 1e-6,
    nu_base: float = 1e-6,
    action_bounds: Tuple[float, float] = (0.5, 2.0)
)
```

#### Methods

##### apply

```python
eta_eff, nu_eff = action_handler.apply(
    action: np.ndarray  # [eta_mult, nu_mult] ∈ [0.5, 2.0]²
) -> Tuple[float, float]
```

**Returns:**
```python
eta_eff = eta_base * action[0]
nu_eff = nu_base * action[1]
```

**Clips action to bounds automatically.**

##### get_action_space

```python
space = action_handler.get_action_space() -> gym.spaces.Box
```

**Returns:** Box([0.5, 0.5], [2.0, 2.0])

#### v1.2 Limitation

**Parameter modulation is NOT a physical actuator:**
- Cannot transfer learned policy to real tokamak
- Purpose: Framework validation only
- v2.0 will add spatial current drive `J_ext(r,θ)`

See: `docs/v1.1/V1.3_IMPROVEMENTS.md`

---

## Environment

### ToroidalMHDEnv

**Location:** `src/pytokmhd/rl/mhd_env_v1_2.py`

**Description:** Gymnasium-compatible RL environment for MHD control.

#### Constructor

```python
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv

env = ToroidalMHDEnv(
    R0: float = 1.0,
    a: float = 0.3,
    nr: int = 32,
    ntheta: int = 64,
    dt: float = 1e-4,
    max_steps: int = 1000,
    eta: float = 1e-6,
    nu: float = 1e-6,
    w_energy: float = 1.0,         # Energy weight
    w_action: float = 0.01,        # Action penalty
    w_constraint: float = 0.0      # Constraint weight (v1.2: disabled)
)
```

#### Spaces

```python
env.observation_space  # Box(-inf, inf, (19,), float32)
env.action_space       # Box([0.5, 0.5], [2.0, 2.0], float32)
```

#### Methods

##### reset

```python
obs, info = env.reset(
    seed: Optional[int] = None,
    options: Optional[dict] = None
) -> Tuple[np.ndarray, dict]
```

**Options:**
```python
{
    'perturbation_amplitude': float,  # Default: 0.01
    'perturbation_mode': int          # Default: 2
}
```

**Returns:**
- `obs`: 19D observation vector
- `info`: {'E_eq', 'perturbation_amplitude', 'perturbation_mode'}

##### step

```python
obs, reward, terminated, truncated, info = env.step(
    action: np.ndarray  # (2,) [eta_mult, nu_mult]
) -> Tuple[np.ndarray, float, bool, bool, dict]
```

**Returns:**
- `obs`: New observation
- `reward`: Scalar reward
- `terminated`: Episode ended (energy explosion)
- `truncated`: Max steps reached
- `info`: Detailed metrics

#### Reward Function

**v1.2 (w_constraint=0):**
```python
reward = -w_energy * energy_drift - w_action * ||action - 1||²
```

**Components:**
- Energy: Minimize drift from equilibrium
- Action: Penalize large deviations from [1, 1]
- Constraint: Disabled (div_B proxy issue)

#### Termination

**Terminated (failure):**
- `energy_drift > 1.0` (100% deviation)

**Truncated:**
- `current_step >= max_steps`

**Success bonus (+10):**
- If truncated AND `energy_drift < 0.01`

#### Example

```python
import gymnasium as gym
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv

env = ToroidalMHDEnv(
    nr=32, ntheta=64, dt=1e-4, max_steps=100,
    w_energy=1.0, w_action=0.01, w_constraint=0.0
)

obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")
print(f"Equilibrium energy: {info['E_eq']:.6e}")

for step in range(100):
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended at step {step+1}")
        print(f"Final energy drift: {info['energy_drift']:.4f}")
        break
```

---

## Complete Example Workflow

```python
"""
Complete v1.2 example: MHD simulation with RL control.
"""

import numpy as np
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.integrators import SymplecticIntegrator
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
from stable_baselines3 import PPO

# 1. Manual simulation (no RL)
print("=== Manual Simulation ===")
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)

# Equilibrium + perturbation
r_grid = grid.r_grid
theta_grid = grid.theta_grid
psi_eq = r_grid**2 * (1 - r_grid / grid.a)
psi_pert = 0.01 * r_grid * np.sin(2 * theta_grid)
psi0 = psi_eq + psi_pert
omega0 = laplacian_toroidal(psi0, grid)

solver.initialize(psi0, omega0)
E0 = solver.compute_energy()

# Time evolution
for _ in range(100):
    solver.step()

E1 = solver.compute_energy()
print(f"Energy drift (no control): {abs(E1-E0)/E0:.2%}")

# 2. RL Environment
print("\n=== RL Environment ===")
env = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=100, w_constraint=0.0)

# Random baseline
obs, _ = env.reset(seed=42)
episode_return = 0
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:
        break

print(f"Random policy return: {episode_return:.4f}")
print(f"Final energy drift: {info['energy_drift']:.4f}")

# 3. RL Training
print("\n=== RL Training ===")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50000)

# Evaluate
obs, _ = env.reset(seed=42)
episode_return = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:
        break

print(f"PPO policy return: {episode_return:.4f}")
print(f"Final energy drift: {info['energy_drift']:.4f}")
```

---

## Performance Characteristics

### Computational Cost

**Grid: 32×64 (2048 points)**

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| step() | ~10 | Single timestep |
| compute_energy() | ~1 | Energy integral |
| fourier_decompose() | ~2 | 8-mode FFT |
| get_observation() | ~3 | Full 19D obs |

**100-step episode:** ~1.3 seconds

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Grid (32×64) | ~1 MB | r, θ, R, Z grids |
| State (psi, omega) | ~64 KB | 2 × 2048 × float64 |
| Solver | ~2 MB | Incl. Poisson matrix |
| Environment | ~3 MB | Total |

### Scaling

**Time per step ∝ nr × ntheta × log(nr×ntheta)**
- 32×64: ~10 ms
- 64×128: ~40 ms (4× points, 4× time)
- 128×256: ~160 ms

---

## Known Limitations (v1.2)

### 1. div_B Proxy

**Issue:** `div_B_max` uses Laplacian proxy (∇²ψ), not true ∇·B

**Impact:**
- Values ~10⁷ (vs true ∇·B ~ 0)
- Cannot use in reward (dominates)
- Set `w_constraint=0.0`

**Fix:** v1.3 will implement true ∇·B computation

### 2. Equilibrium

**Issue:** `r²(1-r/a)` is cylindrical, not toroidal equilibrium

**Impact:**
- ~15% energy transient
- Force balance approximate

**Fix:** v1.3 Solovev equilibrium

### 3. Action Space

**Issue:** Parameter modulation (η, ν), not physical actuator

**Impact:**
- Cannot transfer to real tokamak
- Framework validation only

**Fix:** v2.0 spatial current drive J_ext(r,θ)

See: `docs/v1.1/V1.3_IMPROVEMENTS.md` for roadmap

---

## References

### Documentation

- [Migration Guide](MIGRATION_V1.0_TO_V1.2.md)
- [v1.3 Improvements](v1.1/V1.3_IMPROVEMENTS.md)
- [Design Document](v1.1/design/v1.1-toroidal-symplectic-design-v2.1.md)

### Code

- GitHub: https://github.com/callme-YZ/ptm-rl
- Branch: `feature/v1.2-hamiltonian`
- Tests: `tests/test_step_*.py`

### Physics

- Toroidal geometry: Grad-Shafranov equation
- Symplectic integration: Hamiltonian mechanics
- Reduced MHD: Hazeltine & Meiss (2003)

---

**小P签字:** ⚛️  
**Date:** 2026-03-18  
**Version:** 1.2
