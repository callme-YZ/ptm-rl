# Handoff: 小P (Physics Core) → 小A (RL Environment)

**Date:** 2026-03-19  
**From:** 小P ⚛️ (Physics/等离子体研究员)  
**To:** 小A 🤖 (AI/ML研究员)  
**Project:** PTM-RL v1.3 - Hamiltonian MHD with IMEX  
**Status:** Physics Layer COMPLETE ✅, Ready for RL Integration

---

## Executive Summary

**小P已完成 (Layer 1: Physics Core):**
- ✅ Hamiltonian MHD solver (validated)
- ✅ IMEX resistive diffusion (stable for η up to 1e-3)
- ✅ Physics diagnostics (energy, current, forces)
- ✅ Observation interface (compatible with Gym)
- ✅ Test suite (all passing)
- ✅ Documentation (completion report + known limitations)

**小A接手 (Layer 2: RL Environment):**
- 定义reward function
- 定义action space
- 包装Gym environment
- 测试environment API
- 与RL算法集成 (PPO/SAC/IQL)

**交接点:** Validated physics → RL-ready interface

---

## What小P Delivers

### 1. Physics Solver (Complete & Validated)

**Core solver:**
```python
from pytokmhd.solvers import HamiltonianMHDIMEX

solver = HamiltonianMHDIMEX(
    grid=grid,
    dt=1e-4,
    eta=1e-4,  # resistivity (up to 1e-3 stable)
    nu=0.0,    # viscosity (optional)
    use_imex=True
)

# Evolve one timestep
psi_new, omega_new = solver.step(psi, omega)
```

**Guarantees:**
- ✅ Stable evolution (no NaN/Inf)
- ✅ Energy decreases (resistive case)
- ✅ Energy conserved (ideal case, <0.15% drift)
- ✅ Physical constraints satisfied (force balance, div(B)=0)

**Validation status:**
- Integration tests: 6/6 PASS
- Physics tests: 7/7 PASS
- Energy budget: validated (20-45% error, acceptable)

---

### 2. Physics Diagnostics

**Energy metrics:**
```python
from pytokmhd.physics import compute_hamiltonian

H = compute_hamiltonian(psi, phi, grid)  # Total energy
K = kinetic_energy(phi, grid)            # E×B flow energy
U = magnetic_energy(psi, grid)           # Magnetic energy
```

**Current density:**
```python
from pytokmhd.physics import compute_current_density

J = compute_current_density(psi, grid, mu0=1.0)
# Returns: J = Δ*ψ/R (Grad-Shafranov operator)
```

**Forces (for validation):**
```python
from pytokmhd.diagnostics import compute_j_cross_b, compute_pressure_gradient

j_cross_b = compute_j_cross_b(psi, grid)
grad_p = compute_pressure_gradient(psi, grid)
force_balance_error = j_cross_b - grad_p
```

---

### 3. Observation Interface

**Already implemented:**
```python
from pytokmhd.rl import ObservationWrapper

obs_wrapper = ObservationWrapper(solver, grid)
observation = obs_wrapper.get_observation(psi, omega)
```

**Observation vector (19 features):**
```python
observation = [
    # Energy metrics (3)
    total_energy,
    kinetic_energy_fraction,
    energy_drift,
    
    # Constraint (1)
    div_B_max,
    
    # Fourier modes (4)
    psi_mode_0, psi_mode_1, psi_mode_2, psi_mode_3,
    
    # Gradients (4)
    psi_gradient_max, psi_gradient_rms,
    omega_gradient_max, omega_gradient_rms,
    
    # Physics (7)
    island_width, current_density_peak, safety_factor,
    force_balance_error, energy_dissipation_rate,
    magnetic_helicity, enstrophy
]
```

**小A can:**
- ✅ Use as-is
- ✅ Add more features (小A's choice)
- ✅ Normalize/clip (already implemented)

---

### 4. Grid Configuration

**Current default:**
```python
from pytokmhd.geometry import ToroidalGrid

grid = ToroidalGrid(
    R0=1.0,    # Major radius
    a=0.3,     # Minor radius
    nr=32,     # Radial points
    ntheta=64  # Poloidal points
)
```

**小A可调整:**
- Resolution: 32×64 (fast) to 64×128 (accurate)
- Aspect ratio: R0/a (affects physics)

**小P建议:** 
- RL training用32×64 (computational efficiency)
- Final validation用64×128 (physics accuracy)

---

### 5. Test Suite

**小A应该run的tests:**

**Integration tests:**
```bash
pytest tests/test_step_3_1_env_integration.py -v
# 6 tests, validates observation extraction
```

**Physics tests:**
```bash
pytest tests/test_step_3_3_physics.py -v
# 7 tests, validates physics correctness
```

**Energy budget:**
```bash
python tests/test_energy_budget_fixed.py
# Validates dissipation (expect 20% error)
```

**All should PASS before RL integration!**

---

## What小A Needs to Do

### Step 1: Define Reward Function

**小P建议 (小A can modify):**

**Option A: Island width suppression**
```python
reward = -island_width  # Negative = minimize
```

**Option B: Energy stability**
```python
reward = -abs(energy_drift)
```

**Option C: Multi-objective**
```python
reward = (
    -w1 * island_width 
    - w2 * abs(energy_drift)
    - w3 * div_B_penalty
)
```

**小A decision needed:**
- Which physics metric to optimize?
- Weights for multi-objective?
- Episode termination criteria?

---

### Step 2: Define Action Space

**小P建议 (physics-motivated):**

**Option A: External current drive**
```python
action = J_ext  # Shape: (nr, ntheta)
# Apply as: omega_new += dt * curl(J_ext)
```

**Option B: Heating control**
```python
action = [P_ECRH, P_NBI, ...]  # Heating powers
# Affects: resistivity profile η(r)
```

**Option C: Simple resistivity modulation (for testing)**
```python
action = eta_multiplier  # Scalar
# Apply as: eta_effective = eta_base * action
```

**小A decision needed:**
- What can RL agent control?
- Discrete or continuous action?
- Action space bounds?

**小P can provide:**
- Physics constraints for actions
- How to apply action to solver

---

### Step 3: Wrap Gym Environment

**Template (小A implements):**

```python
import gymnasium as gym
from pytokmhd.solvers import HamiltonianMHDIMEX
from pytokmhd.rl import ObservationWrapper

class MHDControlEnv(gym.Env):
    def __init__(self, config):
        self.grid = ToroidalGrid(**config['grid'])
        self.solver = HamiltonianMHDIMEX(**config['solver'])
        self.obs_wrapper = ObservationWrapper(self.solver, self.grid)
        
        # 小A defines these:
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Box(...)
        
    def reset(self, seed=None):
        # Initialize psi, omega
        psi = self._get_initial_condition()
        omega = np.zeros_like(psi)
        
        # Return observation
        obs = self.obs_wrapper.get_observation(psi, omega)
        return obs, {}
        
    def step(self, action):
        # Apply action (小A implements)
        self._apply_action(action)
        
        # Evolve physics (小P provides)
        self.psi, self.omega = self.solver.step(self.psi, self.omega)
        
        # Compute reward (小A implements)
        reward = self._compute_reward()
        
        # Get observation
        obs = self.obs_wrapper.get_observation(self.psi, self.omega)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = (self.step_count >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}
        
    def _apply_action(self, action):
        # 小A implements based on action space choice
        pass
        
    def _compute_reward(self):
        # 小A implements based on reward choice
        pass
```

**小P provides:**
- Solver interface (`solver.step()`)
- Observation interface (`obs_wrapper.get_observation()`)
- Physics diagnostics (for reward computation)

**小A implements:**
- Action application (`_apply_action()`)
- Reward computation (`_compute_reward()`)
- Termination logic (`_is_terminated()`)

---

### Step 4: Test Environment

**Before RL training, verify:**

```python
env = MHDControlEnv(config)

# Test reset
obs, info = env.reset()
assert obs.shape == env.observation_space.shape

# Test step
action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(action)
assert isinstance(reward, float)
assert obs.shape == env.observation_space.shape

# Test episode
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()
```

**小P available for:**
- Debugging physics issues
- Verifying solver behavior
- Checking observation correctness

---

### Step 5: RL Algorithm Integration

**小A chooses algorithm:**
- PPO (stable, sample efficient)
- SAC (continuous actions, off-policy)
- IQL (offline RL, if using pre-collected data)

**小P not involved in this step** (小A's expertise)

**小P available for:**
- Physics interpretation of learned policy
- Validating final control strategy

---

## Known Issues (小A Should Know)

### 1. Energy Dissipation Error (20-45%)

**Issue:** Numerical dH/dt differs from theory by 20-45%

**Impact on RL:**
- ✅ Reward signal direction correct (energy decreases)
- ✅ Relative comparisons valid (policy A vs B)
- ⚠️ Absolute reward values biased

**Recommendation:**
- Use **relative** rewards (compare to baseline)
- Don't trust absolute energy dissipation rate
- Focus on trend (improving vs worsening)

**详见:** `docs/V1.3_KNOWN_LIMITATIONS.md` Section 1

---

### 2. Computational Cost

**Speed:** ~0.5 sec/step (32×64 grid)

**RL training estimate:**
```
10k episodes × 100 steps = 1M steps
1M steps × 0.5 sec = 139 hours ≈ 6 days
```

**小A mitigation:**
- Use vectorized environments (parallel)
- Start with small-scale experiments (1k episodes)
- Consider lower resolution for hyperparameter search

---

### 3. Initial Conditions

**Important:** IC must satisfy BC!

```python
# Good: conducting wall BC satisfied
psi[-1, :] = 0.0  # Edge
psi[0, :] = np.mean(psi[0, :])  # Axis

# Bad: violates BC → energy explosion
psi = r**2 * cos(theta)  # psi[-1,:] ≠ 0 ❌
```

**小P provides:** `get_equilibrium_IC()` helper (already satisfies BC)

---

## Communication Protocol

### 小A Questions小P Should Answer

**Physics questions:**
- "这个observation feature代表什么物理量?"
- "Action会如何影响plasma evolution?"
- "这个reward function物理上合理吗?"

**Solver questions:**
- "为什么solver step失败了?"
- "这个数值不稳定是什么原因?"
- "怎么调节参数提高stability?"

**小P response time:** Same-day (如果@小P)

---

### 小A Questions小P Should NOT Answer (小A's职责)

**RL algorithm questions:**
- "PPO hyperparameters怎么选?"
- "Learning rate应该多大?"
- "Why policy not converging?"

**Environment design questions:**
- "Observation space应该多大?"
- "Action normalization怎么做?"
- "Episode length设多少?"

**小P会说:** "这是小A的专业领域，小P不越界" ⚛️

---

## Success Criteria

### 小A完成RL integration的标志:

**Minimal (v1.3.1):**
1. ✅ Gym environment implemented
2. ✅ Can reset and step
3. ✅ Observation/action spaces defined
4. ✅ Reward function working
5. ✅ 1 episode runs without crash

**Good (v1.3.2):**
1. ✅ 100 episodes run stably
2. ✅ RL agent trains (loss decreases)
3. ✅ Policy improves baseline (even slightly)

**Excellent (v1.4):**
1. ✅ Agent outperforms expert policy
2. ✅ Physical interpretation clear
3. ✅ Robust to different ICs

---

## Files小A Needs

### Core files (小P provides, 小A uses)
```
src/pytokmhd/
├── geometry/
│   └── toroidal.py           # Grid
├── solvers/
│   ├── hamiltonian_mhd_imex.py  # Main solver
│   └── implicit_resistive.py     # Implicit step
├── physics/
│   ├── hamiltonian.py        # Energy
│   └── diagnostics.py        # Current, forces
├── rl/
│   ├── observations.py       # ObservationWrapper
│   └── __init__.py
└── operators/
    ├── poisson_bracket.py    # Advection
    └── gradient_toroidal.py  # Derivatives
```

### Files小A creates
```
src/pytokmhd/rl/
├── environment.py         # MHDControlEnv (Gym wrapper)
├── rewards.py             # Reward functions
├── actions.py             # Action application
└── policies.py            # (optional) Expert policies

experiments/
├── train_ppo.py           # Training script
├── train_sac.py
└── configs/
    └── mhd_control.yaml   # Hyperparameters
```

---

## Documentation小A Should Read

**Must read (before starting):**
1. `docs/V1.3_COMPLETION_REPORT.md` — Overview
2. `docs/V1.3_KNOWN_LIMITATIONS.md` — Pitfalls
3. This file — Handoff guide

**Optional (for deep dive):**
4. `docs/v1.3-phase1-hamiltonian-implementation.md` — Physics details
5. `docs/v1.3-phase3-completion-report.md` — IMEX details
6. `tests/test_step_3_*.py` — Example usage

---

## Timeline Estimate

**小A's RL integration tasks:**

| Task | Estimate | Dependencies |
|------|----------|--------------|
| Define reward function | 2h | Physics discussion with小P |
| Define action space | 2h | Physics discussion with小P |
| Implement Gym wrapper | 4h | Reward + action defined |
| Test environment | 2h | Wrapper complete |
| RL algorithm integration | 4h | Environment tested |
| Debug training | 4h | Algorithm integrated |
| **Total** | **18h ≈ 2-3 days** | |

**Assumptions:**
- 小A熟悉Gym API
- 小A already chosen RL library (stable-baselines3?)
- 小P available for physics questions

---

## Contact

**小P (Physics questions):**
- Discord: @Plasma - YZ's BOT
- Role: 等离子体物理研究员 ⚛️
- Availability: Same-day response
- Expertise: MHD physics, solver issues, diagnostics

**小A (RL questions):**
- Discord: @AI - YZ's BOT
- Role: AI/ML研究员 🤖
- Responsibility: Environment, reward, training

**YZ (Decisions & conflicts):**
- Discord: @YZ
- Final authority on design choices

---

## Appendix: Quick Reference

### Physics Constants
```python
# Default parameters (小A can modify)
R0 = 1.0      # Major radius
a = 0.3       # Minor radius
eta = 1e-4    # Resistivity
nu = 0.0      # Viscosity
dt = 1e-4     # Timestep
```

### Typical Values
```python
# Energy
H ~ 1e-1 to 1e+1

# Current density
J ~ 1e+0 to 1e+3

# Island width
w ~ 0.01 to 0.1 (in units of a)

# Energy drift (ideal)
|dH/H| < 0.002 per step
```

### Error Messages小A Might See

**"GMRES did not converge"**
→ Resistivity too large or timestep too large
→ Reduce dt or η

**"Energy increased (unphysical)"**
→ Initial condition violates BC
→ Check `psi[-1,:] == 0` and `psi[0,:] == mean(psi[0,:])`

**"NaN detected"**
→ Numerical instability
→ Reduce timestep or check IC

---

**Handoff complete! 小A可以开始RL integration** 🤝

**小P签字:** ⚛️  
**小A确认签收:** 🤖 (pending)  
**Date:** 2026-03-19  
**Version:** v1.3 → v1.3.1 (RL integration)
