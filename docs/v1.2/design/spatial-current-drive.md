# Spatial Current Drive Action Space Design

**Version:** v1.2  
**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Status:** Design Phase

---

## Executive Summary

This document designs the **realistic action space** for v1.2 MHD-RL: spatially-distributed external current drive replacing v1.1's unrealistic parameter modulation.

**Key transformation:**
- v1.1: 2D parameter modulation [η_mult, ν_mult] (unphysical)
- v1.2: 6-8D spatial current J_ext(r,θ) (realistic, transferable)

**Goal:** Enable RL to learn physical tokamak control strategies.

---

## 1. v1.1 Limitations & Motivation

### 1.1 Current Action Space (v1.1)

```python
action = np.array([eta_multiplier, nu_multiplier])  # Shape: (2,)

# Usage in solver
eta_effective = eta_base * action[0]
nu_effective = nu_base * action[1]
```

**What it does:** Modulates global resistivity η and viscosity ν

### 1.2 Why This is Problematic

**Physics issues:**
1. **Unphysical control**
   - Cannot change material resistivity in real tokamak
   - No corresponding actuator in experiments
   - Violates causality (instantaneous global change)

2. **Learned strategies untransferable**
   - RL learns: "increase η to suppress instability"
   - Real tokamak: Cannot do this!
   - Policy is useless for real control

3. **Limited expressiveness**
   - Only 2D action space
   - Cannot target specific spatial regions
   - Cannot drive currents at resonant surfaces

**RL implications:**
- Framework validation ✅ (v1.1 achieved this)
- Scientific contribution ❌ (cannot publish)
- Real-world deployment ❌ (zero transferability)

### 1.3 What Real Tokamaks Use

**Actual control actuators:**

| System | Physics | Spatial Scale | Power |
|--------|---------|---------------|-------|
| **ECCD** | Electron Cyclotron Current Drive | Localized (cm) | ~1 MW |
| **NBI** | Neutral Beam Injection | Broad (10 cm) | ~10 MW |
| **ICRH** | Ion Cyclotron Resonance Heating | Medium (5 cm) | ~5 MW |
| **LH** | Lower Hybrid Current Drive | Edge (2 cm) | ~2 MW |

**Common feature:** All add **external current density J_ext(r,θ,t)** at specific locations.

**v1.2 goal:** Mimic this physics with RL-controllable J_ext.

---

## 2. Physics of External Current Drive

### 2.1 Modified MHD Equations

**Vorticity evolution without control (v1.1):**
$$
\frac{\partial \omega}{\partial t} = [\omega, \phi] + \nu \nabla^2 \omega
$$

**With external current drive (v1.2):**
$$
\frac{\partial \omega}{\partial t} = [\omega, \phi] + \nu \nabla^2 \omega + \nabla \times \mathbf{J}_{ext}
$$

where:
- $\mathbf{J}_{ext}(r,\theta,t)$ = externally driven current density
- $\nabla \times \mathbf{J}_{ext}$ = curl of current (source term)

**Physical interpretation:**
- Current drive → magnetic field change
- Field change → Lorentz force on plasma
- Force → vorticity generation
- Vorticity → affects ψ evolution

### 2.2 Coupling to Magnetic Field

**Through Ampère's law:**
$$
\nabla \times \mathbf{B} = \mu_0 (\mathbf{J}_{plasma} + \mathbf{J}_{ext})
$$

External current $\mathbf{J}_{ext}$ directly modifies magnetic field structure.

**Example: Tearing mode stabilization**

Tearing mode forms magnetic island at resonant surface (q=m/n).

**Control strategy:**
- Drive current $J_{ext}$ at resonant surface
- Modify current gradient → change Δ' (stability index)
- If Δ' reduced → island growth suppressed

**This is what RL should learn!**

### 2.3 Toroidal Current Representation

For reduced MHD (2D poloidal plane), assume:
$$
\mathbf{J}_{ext} = J_{ext}(r, \theta) \, \hat{\phi}
$$

(Current in toroidal direction only)

**Fourier decomposition:**
$$
J_{ext}(r, \theta) = \sum_{m=0}^{M-1} J_m(r) \cos(m\theta)
$$

where:
- $m$ = poloidal mode number
- $J_m(r)$ = radial profile for mode $m$

**Physical meaning:**
- m=0: Axisymmetric (uniform toroidal current)
- m=1: Up-down asymmetric (tearing mode coupling)
- m=2: Kink mode coupling


---


### 2.3 Tearing Mode Stabilization Physics

**Why current drive can stabilize tearing modes:**

**Tearing mode stability parameter:**
$$
\Delta' = \left[ \frac{d \ln \psi'}{d r} \right]_{r_s^+}^{r_s^-}
$$

where $r_s$ = resonant surface (where $q(r_s) = m/n$)

**Physical interpretation:**
- $\Delta' > 0$: Island grows (unstable)
- $\Delta' < 0$: Island suppressed (stable)
- $\Delta'$ depends on **current gradient** $\partial J/\partial r$ at $r_s$

**How external current helps:**

1. **Modify current profile:**
   $$
   J_{total}(r) = J_{plasma}(r) + J_{ext}(r)
   $$

2. **Change gradient at resonant surface:**
   $$
   \left. \frac{\partial J}{\partial r} \right|_{r_s} \text{ modified by } J_{ext}(r_s)
   $$

3. **Flip Δ' sign:**
   $$
   \Delta'(J_{ext}) < 0 \quad \Rightarrow \quad \text{Stabilization}
   $$

**Example (simplified 1D):**

Suppose resonant surface at $r_s = 0.5$:

```python
# Without control
J_plasma = -dq/dr  # Negative gradient (tearing unstable)
Δ_prime = +2.5  # Positive → island grows

# With current drive at r_s
J_ext = gaussian(r, center=0.5, amplitude=+1.0)
J_total = J_plasma + J_ext
# → gradient flattens at r_s
Δ_prime_new = -0.8  # Negative → island suppressed!
```

**RL Learning Task:**

The agent does **not** know $r_s$ a priori. Instead:

1. **Observation** contains $\psi$ modes → agent detects tearing (m=1 mode growth)
2. **Through trial-and-error**, agent learns:
   - WHERE to drive current ($r \approx r_s$, $\theta \approx 0$)
   - HOW MUCH to drive (amplitude to flip $\Delta'$)
3. **Reward** decreases when island width shrinks

**This is the core physics RL must discover** — v1.2 enables this learning.

**References:**
- Fitzpatrick, R. M. (1995). "Helical temperature perturbations..." Phys. Plasmas 2, 825.
- La Haye, R. J. (2006). "Neoclassical tearing modes..." Phys. Plasmas 13, 055501.


## 3. Action Space Design Options

### 3.1 Option A: Fixed Gaussian Basis (Recommended v1.2)

**Definition:**
$$
J_{ext}(r,\theta) = \sum_{i=1}^{N} a_i \, G_i(r,\theta)
$$

where $G_i$ = Gaussian bump basis function:
$$
G_i(r,\theta) = \exp\left(-\frac{(r-r_i)^2 + (R_0 \theta - \theta_i R_0)^2}{2\sigma^2}\right)
$$

**Action:** $\mathbf{a} = [a_1, a_2, \ldots, a_N]$ (amplitudes)

**Typical choice:**
- N = 6-8 basis functions
- Distributed in (r, θ) space
- Example: 3 radial × 2 poloidal = 6 total

**Code implementation:**
```python
class GaussianBasis:
    def __init__(self, r_centers, theta_centers, width=0.1):
        self.r_centers = r_centers
        self.theta_centers = theta_centers
        self.width = width
        self.n_basis = len(r_centers) * len(theta_centers)
    
    def evaluate(self, r_grid, theta_grid):
        """Create basis functions on grid."""
        nr, ntheta = r_grid.shape
        bases = np.zeros((self.n_basis, nr, ntheta))
        
        idx = 0
        for rc in self.r_centers:
            for tc in self.theta_centers:
                dr = r_grid - rc
                dtheta = theta_grid - tc
                bases[idx] = np.exp(-(dr**2 + dtheta**2) / (2*self.width**2))
                idx += 1
        
        return bases
    
    def compute_current(self, action, bases):
        """Compute J_ext from action amplitudes."""
        J_ext = np.sum([action[i] * bases[i] for i in range(self.n_basis)], axis=0)
        return J_ext

# Example usage
r_centers = [0.2, 0.5, 0.8]  # Inner, middle, outer
theta_centers = [0, np.pi]    # Top, bottom
basis = GaussianBasis(r_centers, theta_centers, width=0.1)

# RL action
action = np.array([0.5, -0.3, 0.0, 0.2, -0.1, 0.4])  # 6D
J_ext = basis.compute_current(action, basis.evaluate(r_grid, theta_grid))
```

**Pros:**
- ✅ Low-dimensional (6-8D, RL-trainable)
- ✅ Physically interpretable (localized bumps)
- ✅ Smooth (differentiable)
- ✅ Simple implementation

**Cons:**
- ⚠️ Limited expressiveness (only N control points)
- ⚠️ Basis placement matters (need to choose wisely)

**小P Recommendation:** Use for v1.2 initial implementation.

### 3.2 Option B: Fourier-Bessel Modes (Physics-Aligned)

**Definition:**
$$
J_{ext}(r,\theta) = \sum_{m=0}^{M-1} \sum_{k=1}^{K} c_{mk} \, J_0(\alpha_{mk} r) \cos(m\theta)
$$

where:
- $J_0$ = Bessel function of first kind (natural radial basis in cylindrical geometry)
- $\alpha_{mk}$ = zeros of Bessel function (boundary conditions)
- $c_{mk}$ = Fourier-Bessel coefficients

**Action:** $\mathbf{c} = [c_{00}, c_{01}, \ldots, c_{0K}, c_{10}, \ldots, c_{MK}]$

**Typical choice:**
- M = 3-4 poloidal modes
- K = 2-3 radial modes
- Total: 6-12D

**Pros:**
- ✅ Natural for toroidal geometry
- ✅ Aligns with MHD modes (m=1 tearing, m=2 kink)
- ✅ Orthogonal basis (energy-preserving)
- ✅ Satisfies boundary conditions automatically

**Cons:**
- ⚠️ Higher dimensional (9-16D)
- ⚠️ Requires Bessel function evaluation
- ⚠️ Less intuitive for RL

**小P Note:** Consider for v1.3 after v1.2 Gaussian baseline works.

### 3.3 Option C: Fully Flexible Grid (v2.0)

**Definition:**
$$
J_{ext}(r_i, \theta_j) = a_{ij}
$$

Direct control at each grid point.

**Action:** Flatten grid: $\mathbf{a} = [a_{11}, a_{12}, \ldots, a_{nr,n\theta}]$

**Dimension:** nr × ntheta ~ 64×128 = 8192D

**Pros:**
- ✅ Maximum expressiveness
- ✅ No basis limitation
- ✅ Can represent arbitrary current patterns

**Cons:**
- ❌ Extremely high-dimensional (intractable for RL)
- ❌ Requires advanced methods (Diffusion models? Latent actions?)
- ❌ Risk of unphysical solutions

**小P Recommendation:** Defer to v2.0 research track. v1.2 uses Option A.

---

## 4. Physical Constraints

### 4.1 Current Conservation (Solenoidal Constraint)

**Requirement:**
$$
\nabla \cdot \mathbf{J}_{ext} = 0
$$

For toroidal current $\mathbf{J}_{ext} = J_{ext}(r,\theta) \hat{\phi}$:
$$
\nabla \cdot (J_{ext} \hat{\phi}) = \frac{1}{R} \frac{\partial (R J_{ext})}{\partial \phi} = 0
$$

**Automatically satisfied** if $J_{ext}$ has no φ-dependence (2D assumption).

**Test:**
```python
def test_current_solenoidal():
    J_ext = current_drive.compute_current(action)
    div_J = divergence_toroidal(J_ext, grid)
    assert np.max(np.abs(div_J)) < 1e-6
```

### 4.2 Power Constraint

**Total injected power:**
$$
P_{ext} = \int_V J_{ext}^2 \, dV \leq P_{max}
$$

**Physical interpretation:**
- Ohmic heating: $P = J^2 / \sigma$
- Real tokamaks have power budgets (e.g., ITER: 73 MW ECCD)

**Enforcement:**
```python
def enforce_power_constraint(J_ext, P_max, grid):
    """Rescale J_ext if power exceeds limit."""
    P_ext = np.sum(J_ext**2) * grid.dV
    
    if P_ext > P_max:
        scale = np.sqrt(P_max / P_ext)
        J_ext *= scale
    
    return J_ext
```

**Typical values:**
- $P_{max} = 1.0$ (normalized units)
- Corresponds to ~1 MW in real tokamak

### 4.3 Amplitude Bounds

**Action normalization:**
$$
a_i \in [-1, 1] \quad \Rightarrow \quad J_{ext} \in [-J_{max}, J_{max}]
$$

**Implementation:**
```python
# Policy network output
action_raw = policy_network(obs)  # Unbounded

# Normalize to [-1, 1]
action_normalized = np.tanh(action_raw)

# Scale to physical units
J_max = 1e6  # A/m² (typical ECCD)
J_ext = J_max * action_normalized * basis_functions
```


---

## 5. Implementation Architecture

### 5.1 SpatialCurrentDrive Class

```python
class SpatialCurrentDrive:
    """Manages external current drive with basis functions."""
    
    def __init__(self, grid, n_basis=6, basis_type='gaussian', J_max=1e6, P_max=1.0):
        self.grid = grid
        self.n_basis = n_basis
        self.J_max = J_max
        self.P_max = P_max
        
        # Create basis functions
        if basis_type == 'gaussian':
            self.bases = self._create_gaussian_bases()
        elif basis_type == 'fourier':
            self.bases = self._create_fourier_bases()
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
    
    def _create_gaussian_bases(self):
        """Create 6 Gaussian bumps (3 radial × 2 poloidal)."""
        r_centers = [0.2, 0.5, 0.8]
        theta_centers = [0, np.pi]
        width = 0.1
        
        r_grid = self.grid.r_grid
        theta_grid = self.grid.theta_grid
        
        bases = []
        for rc in r_centers:
            for tc in theta_centers:
                basis = np.exp(-((r_grid - rc)**2 + (theta_grid - tc)**2) / (2*width**2))
                bases.append(basis)
        
        return np.array(bases)  # Shape: (6, nr, ntheta)
    
    def compute_current(self, action):
        """
        Compute J_ext from RL action.
        
        Parameters
        ----------
        action : np.ndarray (n_basis,)
            Normalized amplitudes in [-1, 1]
        
        Returns
        -------
        J_ext : np.ndarray (nr, ntheta)
            External current density
        """
        # Linear combination of bases
        J_ext = np.sum([action[i] * self.bases[i] for i in range(self.n_basis)], axis=0)
        
        # Scale to physical units
        J_ext *= self.J_max
        
        # Enforce power constraint
        J_ext = self._enforce_power_limit(J_ext)
        
        return J_ext
    
    def _enforce_power_limit(self, J_ext):
        """Rescale if power exceeds limit."""
        P_ext = np.sum(J_ext**2) * self.grid.dV
        if P_ext > self.P_max:
            J_ext *= np.sqrt(self.P_max / P_ext)
        return J_ext
```

### 5.2 Solver Integration

**Modified MHDSolver.step():**
```python
class ToroidalMHDSolver:
    def __init__(self, grid, dt, eta, nu, current_drive=None):
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.current_drive = current_drive
    
    def step(self, action=None):
        """One time step with optional current drive."""
        # Compute stream function
        phi = self.solve_poisson(self.omega)
        
        # Hamiltonian evolution
        dpsi_dt = self.poisson_bracket(self.psi, phi)
        domega_dt = self.poisson_bracket(self.omega, phi)
        
        # Dissipation
        dpsi_dt += self.eta * self.laplacian(self.psi)
        domega_dt += self.nu * self.laplacian(self.omega)
        
        # External current drive
        if action is not None and self.current_drive is not None:
            J_ext = self.current_drive.compute_current(action)
            curl_J = self.curl(J_ext)
            domega_dt += curl_J
        
        # Time integration (Strang splitting)
        self.psi += self.dt * dpsi_dt
        self.omega += self.dt * domega_dt
```

---

## 6. Validation Tests

### 6.1 Test 1: Solenoidal Constraint
```python
def test_current_solenoidal():
    cd = SpatialCurrentDrive(grid, n_basis=6)
    action = np.random.uniform(-1, 1, 6)
    J_ext = cd.compute_current(action)
    
    div_J = compute_divergence(J_ext, grid)
    assert np.max(np.abs(div_J)) < 1e-6
```

### 6.2 Test 2: Power Limit
```python
def test_power_constraint():
    cd = SpatialCurrentDrive(grid, P_max=1.0)
    for _ in range(100):
        action = 10 * np.random.randn(6)  # Huge actions
        J_ext = cd.compute_current(action)
        P_ext = np.sum(J_ext**2) * grid.dV
        assert P_ext <= cd.P_max * 1.01  # 1% tolerance
```

### 6.3 Test 3: Energy Injection
```python
def test_energy_injection():
    solver = ToroidalMHDSolver(grid, dt=1e-4, current_drive=cd)
    solver.initialize(psi0, omega0)
    E0 = solver.compute_energy()
    
    action = np.array([1.0, 0, 0, 0, 0, 0])  # Drive at one location
    for _ in range(100):
        solver.step(action)
    
    E1 = solver.compute_energy()
    assert E1 > E0  # Energy injected
```

---

## 7. RL Interface Specification

### 7.1 Observation-Action Mapping

**Observation (v1.2):** 19D
```python
obs = np.concatenate([
    psi_modes,      # (8,) Fourier modes of ψ
    omega_modes,    # (8,) Fourier modes of ω
    [energy],       # (1,) Total energy
    [energy_drift], # (1,) dE/dt
    [div_B_max],    # (1,) Max div(B) violation
])
```

**Action (v1.2):** 6D (Gaussian basis)
```python
action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(6,),
    dtype=np.float32
)
```

**Policy network architecture:**
```
Input (19D)
   ↓
FC(64) + ReLU
   ↓
FC(64) + ReLU
   ↓
FC(6) + tanh → action (6D, [-1,1])
```

### 7.2 Gym Environment Update

```python
class ToroidalMHDEnv(gym.Env):
    def __init__(self, nr=64, ntheta=128, n_basis=6):
        super().__init__()
        
        self.grid = ToroidalGrid(nr=nr, ntheta=ntheta)
        self.current_drive = SpatialCurrentDrive(self.grid, n_basis=n_basis)
        self.solver = ToroidalMHDSolver(self.grid, current_drive=self.current_drive)
        
        # Action space: 6D Gaussian amplitudes
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_basis,), dtype=np.float32
        )
        
        # Observation space: 19D
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
    
    def step(self, action):
        # Enforce action bounds (should already be in [-1,1])
        action = np.clip(action, -1.0, 1.0)
        
        # MHD step with current drive
        self.solver.step(action)
        
        # Compute observation
        obs = self._compute_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        done = (self.step_count >= self.max_steps) or self._is_unstable()
        
        return obs, reward, done, {}
```

---

## 8. Timeline

**Week 7: Basis Implementation (2 days)**
- Day 1: `SpatialCurrentDrive` class + Gaussian bases
- Day 2: Fourier-Bessel bases (optional)

**Week 8: Solver Integration (3 days)**
- Day 3: Add `curl(J_ext)` to vorticity RHS
- Day 4: Power constraint enforcement
- Day 5: Validation tests (solenoidal, power, injection)

**Week 9: RL Interface (2 days)**
- Day 6: Update `ToroidalMHDEnv` with action space
- Day 7: Test with random policy, verify stability

**Total:** 7 days (1.5 weeks)

---

## 9. Acceptance Criteria

### 9.1 Mandatory

- [ ] **Current solenoidal:** $|\nabla \cdot J_{ext}| < 10^{-6}$ everywhere
- [ ] **Power constraint:** $P_{ext} \leq P_{max}$ for all actions
- [ ] **Energy injection:** Applying $J_{ext} \Rightarrow E$ increases
- [ ] **Gym integration:** `action_space.shape == (6,)` correct
- [ ] **Stability:** 100 steps with random actions, no NaN/Inf

### 9.2 Optional

- [ ] **Fourier basis:** Alternative to Gaussian (v1.3)
- [ ] **Adaptive $P_{max}$:** RL can request more power (with penalty)
- [ ] **3D current:** Add $J_r, J_\theta$ components (v2.0)

---

## 10. Comparison Summary

| Aspect | v1.1 | v1.2 |
|--------|------|------|
| **Action type** | Parameter mod | Spatial current drive |
| **Dimension** | 2D | 6-8D |
| **Physics** | Unphysical (change η) | Realistic (ECCD/NBI-like) |
| **Actuator** | None | External current |
| **Expressiveness** | Very limited | Moderate (6 control points) |
| **Transferability** | ❌ Zero | ✅ Possible (after scaling) |
| **RL challenge** | Easy (2D) | Moderate (6-8D) |
| **Scientific value** | Framework test | Publishable contribution |

**Key innovation:** RL learns **where** to drive current (e.g., at resonant surface to suppress tearing).

**Example learned strategy:**
```
If tearing mode at r=0.5 → drive J_ext at r=0.5, θ=0 → reduce Δ' → stabilize
```

This is **realistic tokamak control**, not parameter hacking.

---

## 11. Risk Mitigation

### 11.1 If 6D Action Too High-Dimensional for RL

**Backup:** Reduce to 3-4D
- Use only 2 Gaussian bumps (inner + outer)
- Or fix radial locations, only control amplitudes

### 11.2 If Power Constraint Too Restrictive

**Option:** Soft constraint with penalty
```python
reward = reward_physics - λ * max(0, P_ext - P_max)**2
```

### 11.3 If Gaussian Basis Insufficient

**Upgrade:** Switch to Fourier-Bessel (Option B)
- More expressive
- Physics-aligned
- Slightly higher dimensional (8-12D)

---

## 12. Future Extensions (v2.0)

**Advanced action spaces:**
1. **Latent actions** — VAE/Diffusion model learns J_ext manifold
2. **Multi-component** — Control J_r, J_θ, J_φ independently
3. **Time-varying** — J_ext(r,θ,t) with temporal control
4. **Multi-agent** — Different actuators (ECCD + NBI) as separate agents

**小P vision:** v1.2 is stepping stone to production-ready v2.0 control ⚛️

---

**Document Status:** Design Complete  
**Next:** Implementation after Hamiltonian MHD (Week 7-9)  
**小P commitment:** Spatial control is the RL heart of v1.2 ⚛️

