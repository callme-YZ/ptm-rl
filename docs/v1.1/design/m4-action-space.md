# M4 Action Space Design

**Project:** PTM-RL v1.1 - M4 RL Integration  
**Author:** 小P ⚛️  
**Date:** 2026-03-17

## Executive Summary

Action space for toroidal MHD control.

**v1.1 Critical Limitation:** Current solver has NO control inputs!

**Solution:**
- v1.1: Minimal (parameter modulation: η, ν)
- v1.2: Ideal (spatial current drive, heating)



---

## ⚠️ CRITICAL LIMITATION: v1.1 Action is NOT Realistic

**v1.1 uses parameter modulation (η, ν multipliers):**
- ❌ NOT physical actuators (no RMP, ECCD, NBI)
- ❌ NOT transferable to real tokamak control
- ❌ Learned policies WILL NOT work in v1.2

**Purpose:** Framework validation ONLY

**v1.2 will use realistic spatial current drive** ✅

---


## 1. v1.1 Solver Limitations

### 1.1 Current Implementation

**Equations:**
```python
∂ψ/∂t = -η*J    # J = -∇²ψ
∂ω/∂t = -ν*∇²ω
```

**Problem:** No control terms!

**Solver API:**
```python
class ToroidalMHDSolver:
    def __init__(self, grid, dt, eta, nu):
        self.eta = eta  # Fixed resistivity
        self.nu = nu    # Fixed viscosity
    
    def step(self):
        # No action input!
        dpsi_dt = -self.eta * self.compute_J()
        domega_dt = -self.nu * laplacian(omega)
```

### 1.2 What's Missing

**Control inputs needed:**
```python
∂ψ/∂t = [ψ,φ] - η*J + J_ext(r,θ,t)  # External current
∂ω/∂t = [ω,φ] - ν*∇²ω + S(r,θ,t)    # Heating source
```

**Solver extension required:**
```python
def step(self, action):
    J_ext = action['current_drive']
    dpsi_dt = ... + J_ext
```

## 2. v1.1 Minimal Action Space

### 2.1 Parameter Modulation (Feasible)

**Idea:** Modulate η, ν in time

```python
action = {
    'eta_multiplier': float,  # Range: [0.5, 2.0]
    'nu_multiplier': float,   # Range: [0.5, 2.0]
}
```

**Implementation:**
```python
# In solver.step()
eta_effective = self.eta * action['eta_multiplier']
nu_effective = self.nu * action['nu_multiplier']

dpsi_dt = -eta_effective * J
domega_dt = -nu_effective * laplacian(omega)
```

**Physics:**
- Increase η → faster resistive diffusion
- Decrease η → slower diffusion
- Similar for ν (viscosity)

**Control effect:**
- ✅ Can modulate diffusion rates
- ⚠️ Indirect control (not physical actuators)
- ⚠️ Limited control authority

### 2.2 Gym Action Space

```python
from gym.spaces import Box

action_space = Box(
    low=np.array([0.5, 0.5]),
    high=np.array([2.0, 2.0]),
    shape=(2,),
    dtype=np.float32
)
```

**Normalization (for RL):**
```python
# Map [-1,1] → [0.5, 2.0]
action_normalized = (action + 1) / 2 * 1.5 + 0.5
```

### 2.3 Solver Extension (Minimal)

**Modify ToroidalMHDSolver:**

```python
def compute_rhs(self, psi, omega, action):
    # Apply action modulation
    eta_eff = self.eta * action[0]
    nu_eff = self.nu * action[1]
    
    # Compute RHS with effective parameters
    J = -laplacian_toroidal(psi, self.grid)
    dpsi_dt = -eta_eff * J
    
    lap_omega = laplacian_toroidal(omega, self.grid)
    domega_dt = -nu_eff * lap_omega
    
    return dpsi_dt, domega_dt

def step(self, action=None):
    if action is None:
        action = [1.0, 1.0]  # Default: no modulation
    
    self.psi, self.omega = self.integrator.step(
        self.psi, self.omega, 
        lambda p, o: self.compute_rhs(p, o, action)
    )
```

**Implementation effort:** ~50 lines

## 3. v1.2 Ideal Action Space

### 3.1 Spatial Current Drive

**Physics:**
```python
∂ψ/∂t = [ψ,φ] - η*J + J_ext(r,θ)
```

**Action:**
```python
action = {
    'J_ext': np.ndarray (nr, ntheta),  # Spatial profile
}
```

**Control:**
- ✅ Direct current injection
- ✅ Spatially localized
- ✅ Realistic actuator (ECCD, NBI)

**Challenges:**
- High-dimensional action space (64×128 = 8192D)
- Need dimensionality reduction (e.g., radial basis functions)

### 3.2 Localized Heating

**Physics:**
```python
∂ω/∂t = [ω,φ] - ν*∇²ω + S(r,θ)
```

**Action:**
```python
action = {
    'heating': np.ndarray (nr, ntheta),
}
```

**Control:**
- ✅ Local heating modulation
- ✅ Affects pressure (→ current)

### 3.3 Reduced-Dimension Action (Practical)

**Radial basis functions:**

```python
# Action: amplitudes of basis functions
action = np.array([a_1, a_2, ..., a_K])  # K basis functions

# Reconstruct spatial profile
J_ext(r,θ) = Σ a_k * φ_k(r,θ)
```

**Example basis:**
```python
φ_k(r,θ) = exp(-((r-r_k)/σ)²) * cos(m_k * θ)
# K = 8 basis functions → 8D action
```

**Dimension:** 8D (tractable)

## 4. Action Space Comparison

| Aspect | v1.1 Minimal | v1.2 Ideal |
|--------|-------------|-----------|
| Type | Parameter modulation | Spatial current |
| Dimension | 2D | 8D (reduced) |
| Physics | Indirect | Direct |
| Implementation | Easy (~50 lines) | Hard (~500 lines) |
| Control authority | Limited | Strong |
| Realism | Low | High |

**Recommendation:** v1.1 minimal for framework, v1.2 for realistic control

## 5. Physics Validation

### 5.1 v1.1 Minimal (Parameter Modulation)

**✅ Can it control?**
- Yes, indirectly (modulate diffusion rates)

**⚠️ Is it realistic?**
- No (real tokamaks don't modulate η, ν)

**✅ Is it useful?**
- Yes, for framework validation
- RL can learn to use available control

**Physics interpretation:**
- Increasing η ~ Enhanced resistivity (e.g., turbulence)
- Decreasing η ~ Improved confinement
- Not physical actuators, but captures trade-off

### 5.2 v1.2 Ideal (Current Drive)

**✅ Realistic actuator:**
- ECCD: electron cyclotron current drive
- NBI: neutral beam injection

**✅ Direct control:**
- J_ext directly modifies ∂ψ/∂t

**✅ Physics-informed:**
- Spatial localization matters
- Mode coupling via Poisson bracket

## 6. Implementation Requirements

### 6.1 v1.1: Modify ToroidalMHDSolver

**Changes needed:**

1. **compute_rhs signature:**
   ```python
   def compute_rhs(self, psi, omega, action):  # Add action parameter
   ```

2. **Apply modulation:**
   ```python
   eta_eff = self.eta * action[0]
   nu_eff = self.nu * action[1]
   ```

3. **Integrator call:**
   ```python
   self.integrator.step(..., lambda p,o: self.compute_rhs(p,o,action))
   ```

**Estimated effort:** 1-2 hours

### 6.2 v1.2: Add Control Terms

**Changes needed:**

1. **Implement J_ext reconstruction:**
   ```python
   def reconstruct_J_ext(action, basis_functions, grid):
       J_ext = np.zeros_like(grid.r_grid)
       for k, a_k in enumerate(action):
           J_ext += a_k * basis_functions[k]
       return J_ext
   ```

2. **Add to RHS:**
   ```python
   dpsi_dt = poisson_bracket(psi, phi) - eta*J + J_ext
   ```

3. **Implement Poisson bracket:**
   ```python
   def poisson_bracket(f, g, grid):
       df_dr, df_dtheta = gradient_toroidal(f, grid)
       dg_dr, dg_dtheta = gradient_toroidal(g, grid)
       return df_dr*dg_dtheta - df_dtheta*dg_dr
   ```

**Estimated effort:** 1-2 days

## 7. Recommendations for 小A

### 7.1 v1.1 Implementation (Phase 3)

**Step 1:** Modify ToroidalMHDSolver
- Add action parameter to compute_rhs
- Implement parameter modulation
- Test: action=[1.0,1.0] should match no-action

**Step 2:** Environment Integration
```python
class ToroidalMHDEnv:
    def step(self, action):
        # Normalize action
        action_scaled = self.scale_action(action)
        
        # Solver step with action
        self.solver.step(action_scaled)
        
        # Get observation & reward
        obs = self.get_observation()
        reward = self.compute_reward(obs, action)
        
        return obs, reward, done, info
```

**Step 3:** Sanity Check
- Zero action: system should evolve naturally
- Extreme action: should see parameter effect

### 7.2 Physics Review Points

**小P will review:**
1. ✅ Action scaling correct? ([0.5, 2.0] range)
2. ✅ Solver modification preserves physics?
3. ✅ Energy evolution makes sense under control?

**Review meeting:** After 小A completes Step 2

## 8. Known Limitations (v1.1)

### 8.1 Indirect Control

**Not realistic:** Real tokamaks use:
- RF heating (ECCD, ICRH)
- Neutral beam injection (NBI)
- Pellet injection

**v1.1 uses:** Parameter modulation (abstract)

**Impact:** 
- ✅ RL learns control framework
- ❌ Learned policy NOT transferable to real tokamak

### 8.2 Limited Authority

**Parameter range:** [0.5, 2.0] (factor of 4)

**Effect:** Modest control (not strong suppression)

**Why limit:**
- Physics validity (extreme η, ν unrealistic)
- Numerical stability

## 9. v1.2 Action Design Preview

### 9.1 Basis Function Selection

**Candidate bases:**
1. **Gaussian radial:**
   ```python
   φ(r) = exp(-((r-r_k)/σ)²)
   ```

2. **Polynomial:**
   ```python
   φ(r) = r^k * (1-r)^m
   ```

3. **Fourier poloidal:**
   ```python
   φ(θ) = cos(m*θ), sin(m*θ)
   ```

**Recommendation:** Gaussian × Fourier (8D total)

### 9.2 Action Space (v1.2)

```python
action_space = Box(
    low=-1.0, high=1.0,
    shape=(8,),  # 4 radial × 2 poloidal
    dtype=np.float32
)
```

---

**Status:** Complete ✅  
**v1.1 Action:** Parameter modulation (2D)  
**v1.2 Action:** Spatial current drive (8D reduced)  
**Implementation:** 小A Phase 3
