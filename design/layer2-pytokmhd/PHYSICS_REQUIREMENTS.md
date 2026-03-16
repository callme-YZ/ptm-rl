# PyTokMHD Physics Requirements

**Author:** 小P ⚛️  
**Date:** 2026-03-16  
**Purpose:** 定义PyTokMHD必须满足的物理准确性标准

---

## 1. Physics Model要求

### 1.1 MHD方程完整性

**必须实现:**

**Reduced MHD系统:**
```
∂ψ/∂t = -[φ, ψ] + η∇²ψ                    (Induction)
∂ω/∂t = -[φ, ω] + ν∇²ω + (J×B)_pol       (Vorticity)
∇²φ = -ω                                   (Poisson)
```

**Where:**
- ψ: Poloidal flux function
- ω: Vorticity (∇²φ)
- φ: Stream function (velocity potential)
- [f,g] = ∂_r f ∂_z g - ∂_z f ∂_r g (Poisson bracket)
- J = ∇×B: Current density
- η: Resistivity (normalized)
- ν: Viscosity (normalized)

**禁止简化:**
- ❌ 不能忽略 J×B force
- ❌ 不能忽略 nonlinear terms [φ,ψ], [φ,ω]
- ❌ 不能使用线性化方程

**理由:** 撕裂模是非线性现象,线性化无法捕捉岛宽饱和。

---

### 1.2 几何要求

**Cylindrical geometry (r, θ, z):**

**坐标系:**
```
r: Radial coordinate [0, r_max]
θ: Poloidal angle [0, 2π]
z: Axial (toroidal) coordinate [0, L_z]
```

**轴对称假设:**
- ∂/∂θ = 0 (axisymmetric)
- 简化3D → 2D (r-z plane)

**Toroidal effects (可选,Phase 2):**
- Toroidal curvature → drift terms
- 暂时忽略 (cylindrical approximation)

**边界条件:**
```
r = 0:     正则性条件 (ψ, ω finite)
r = r_max: 固定边界 (conducting wall)
z = 0, L_z: 周期性 (toroidal symmetry)
```

---

### 1.3 物理参数范围

**Normalized units:**

| Parameter | Symbol | Typical Range | Physical Meaning |
|-----------|--------|---------------|------------------|
| 磁雷诺数 | S = τ_R/τ_A | 10⁴ - 10⁸ | Resistive vs Alfvén time |
| Lundquist数 | S_L = v_A L / η | 10⁶ - 10⁹ | Same as S |
| Reynolds数 | Re = v L / ν | 10⁴ - 10⁶ | Inertial vs viscous |
| 归一化电阻 | η | 10⁻⁵ - 10⁻³ | Magnetic diffusivity |
| 归一化粘滞 | ν | 10⁻⁶ - 10⁻⁴ | Kinematic viscosity |

**PyTokMHD参数选择 (典型):**
```python
eta = 1e-5      # Moderate resistivity
nu = 1e-6       # Low viscosity (Re ~ 10⁵)
dt = 1e-4       # CFL-safe time step
```

**物理意义:**
- Small η → 撕裂模增长慢 (realistic)
- Small ν → 高雷诺数湍流 (tokamak-like)

---

## 2. 守恒律要求

### 2.1 Energy Conservation

**总能量演化:**
```
E_total = E_magnetic + E_kinetic

E_magnetic = (1/2) ∫ B² dV
E_kinetic = (1/2) ∫ ρ v² dV

dE/dt = -∫(η J² + ν |∇v|²) dV  (dissipation only)
```

**验收标准:**
```python
# Measured energy change
dE_measured = E(t+dt) - E(t)

# Expected dissipation
dissipation = -∫(eta * J**2 + nu * grad_v**2) dV * dt

# Relative error
error = |dE_measured - dissipation| / |dissipation|

# Pass if:
assert error < 0.01  # 1% tolerance
```

**Why important:**
- 能量不守恒 → 数值instability
- 1%精度 → 足够验证physics正确

---

### 2.2 Magnetic Flux Conservation

**Total flux through any closed surface:**
```
Φ = ∫ B·dS = const (if no resistivity on boundary)
```

**In flux coordinates:**
```
ψ(r, z, t) → ψ_axis(t), ψ_edge(t)

Δψ = ψ_edge - ψ_axis (poloidal flux)
```

**验收标准:**
```python
# Flux change (interior, away from resistive layer)
Dpsi_bulk = psi_bulk(t+dt) - psi_bulk(t)

# Should be small (diffusion only)
assert |Dpsi_bulk| / |psi_total| < 1e-4
```

---

### 2.3 ∇·B = 0 (Solenoidal Constraint)

**Mathematical:**
```
B = ∇ψ × ẑ + B_z ẑ  (in cylindrical coords)

→ ∇·B = 0 automatically (if ψ smooth)
```

**Numerical verification:**
```python
div_B = (1/r) * d(r*Br)/dr + dBz/dz

# Finite difference error
assert np.abs(div_B).max() < 1e-6
```

**Why ∇·B matters:**
- ∇·B ≠ 0 → 非物理 magnetic monopoles
- 数值误差来源: grid discretization, interpolation

---

## 3. 撕裂模物理要求

### 3.1 Linear Growth Phase

**FKR Theory (Furth-Killeen-Rosenbluth 1963):**

**Growth rate:**
```
γ = c * η^(3/5) * Δ'^(4/5)

Where:
  Δ' = [d(ln ψ')/dr]_{r_s^+}^{r_s^-}  (tearing stability index)
  c ≈ 0.55 (numerical constant)
```

**Island width evolution:**
```
w(t) = w0 * exp(γ * t)  (linear phase, w << r_s)
```

**验收标准:**
```python
# Measure γ from simulation
log_w = np.log(w_history)
gamma_measured, _ = np.polyfit(t_history, log_w, deg=1)

# FKR prediction
gamma_FKR = 0.55 * eta**(3/5) * Delta_prime**(4/5)

# Relative error
error = |gamma_measured - gamma_FKR| / gamma_FKR

# Pass if:
assert error < 0.20  # 20% tolerance (theory vs simulation)
```

**Why 20% tolerance?**
- FKR是slab geometry近似
- Cylindrical geometry有 ~ O(r_s/R) corrections
- 数值误差 ~ 5-10%

---

### 3.2 Nonlinear Saturation

**Rutherford Regime (w > w_critical):**

**Saturated island width:**
```
dw/dt = c_R * Δ' * r_s - c_NL * w³/r_s²

→ w_sat ~ (Δ' * r_s²)^(1/2)  (when dw/dt = 0)
```

**Saturation timescale:**
```
τ_NL ~ τ_R = r_s² / η  (Resistive timescale)
```

**验收标准:**
```python
# Evolve until saturation
w_sat = w_history[-1]  # Final width

# Rutherford prediction
w_theory = np.sqrt(Delta_prime * r_s**2)

# Factor-of-2 tolerance
assert 0.5 < w_sat / w_theory < 2.0
```

**Why factor-of-2?**
- Nonlinear physics less precise
- Depends on pressure, toroidal effects (忽略的项)
- Order-of-magnitude agreement sufficient

---

### 3.3 Rational Surface Requirements

**Safety factor profile:**
```
q(r) = (r * B_z) / (R_0 * B_θ)

Rational surfaces: q(r_s) = m/n (integer m, n)
```

**Tearing modes occur at rational surfaces where:**
```
Δ' > 0  (unstable)
```

**PyTokMHD必须:**
1. ✅ 从PyTokEq读取 q(r)
2. ✅ 找到 rational surface r_s where q(r_s) = m/n
3. ✅ 在 r_s附近施加扰动
4. ✅ 验证岛宽在 r_s处最大

**验收标准:**
```python
# Measure island location
r_island = find_island_center(psi)

# Compare with rational surface
r_s = find_rational_surface(q_profile, m=2, n=1)

# Tolerance: within 1 grid spacing
assert |r_island - r_s| < 2 * dr
```

---

## 4. 外部控制要求

### 4.1 RMP (Resonant Magnetic Perturbation)

**External field from coils:**
```
B_ext = B_RMP(I_coil, r, z, θ)

Total field:
B_total = B_plasma + B_ext
```

**RMP physics:**
- Resonant when RMP mode matches tearing mode (m, n)
- Phase-locking: RMP can suppress or amplify island
- Control power: P ~ I_coil²

**验收标准:**
```python
# Test 1: RMP suppression (correct phase)
w_no_RMP = measure_island(run_simulation(I_RMP=0))
w_with_RMP = measure_island(run_simulation(I_RMP=0.1, phase=0))

assert w_with_RMP < w_no_RMP  # Suppression

# Test 2: Phase-dependence
w_wrong_phase = measure_island(run_simulation(I_RMP=0.1, phase=pi))

assert w_wrong_phase > w_no_RMP  # Amplification (wrong phase)
```

---

### 4.2 External Field Requirements

**Implementation:**
```python
class RMPCoil:
    def compute_field(self,
                     I_coil: float,
                     r: np.ndarray,
                     z: np.ndarray,
                     mode: Tuple[int, int]) -> Tuple[np.ndarray, ...]:
        """
        Compute RMP field from external coils
        
        Args:
            I_coil: Coil current [A]
            r, z: Grid coordinates
            mode: (m, n) helical mode
            
        Returns:
            Br_ext, Bz_ext: External field components
            
        Physics model:
            Simple helical coil approximation
            (more realistic: Biot-Savart from coil geometry)
        """
        pass
```

**验收标准:**
- RMP field满足 ∇·B = 0
- Mode structure正确 (m, n harmonics)
- Amplitude scaling: B_RMP ∝ I_coil

---

## 5. 初始条件要求

### 5.1 PyTokEq平衡态

**Input from Layer 1:**
```python
equilibrium = {
    'psi': (Nr, Nz),       # Poloidal flux
    'j_tor': (Nr, Nz),     # Toroidal current
    'pressure': (Nr, Nz),  # Plasma pressure
    'q_profile': (Nr,)     # Safety factor
}
```

**Physics consistency checks:**
```python
# 1. Force balance: J×B = ∇p
J_cross_B = compute_lorentz_force(equilibrium)
grad_p = compute_pressure_gradient(equilibrium)

assert np.allclose(J_cross_B, grad_p, rtol=0.01)

# 2. q-profile reasonable
q_axis = q_profile[0]
q_edge = q_profile[-1]

assert 0.8 < q_axis < 1.2  # Typically near 1
assert 2.0 < q_edge < 5.0  # Edge safety factor
```

---

### 5.2 Perturbation要求

**Helical perturbation at rational surface:**
```python
# Mode numbers
m, n = 2, 1  # (poloidal, toroidal)

# Amplitude
amp = 1e-5  # Small (linear regime initially)

# Spatial profile
r_s = find_rational_surface(q_profile, q_target=m/n)
perturbation = amp * exp(-((r - r_s)/σ)²) * sin(m*θ - n*φ)
```

**Requirements:**
1. ✅ ∇·B = 0 (solenoidal)
2. ✅ Localized at r_s (width σ ~ 0.1 * r_s)
3. ✅ Random phase (seed-controlled)
4. ✅ Amplitude << equilibrium (linear start)

**验收标准:**
```python
# Initial perturbation small
psi_pert = psi_init - psi_eq
assert np.abs(psi_pert).max() / np.abs(psi_eq).max() < 0.01  # <1%

# Solenoidal
B_pert = compute_B_from_psi(psi_pert)
div_B_pert = compute_divergence(B_pert)
assert np.abs(div_B_pert).max() < 1e-6
```

---

## 6. 数值精度要求

### 6.1 空间分辨率

**Grid requirements:**

**最小分辨率:**
```
Nr ≥ 32   (径向)
Nz ≥ 64   (轴向)
```

**推荐分辨率:**
```
Nr = 64
Nz = 128
```

**Convergence test:**
```python
# Run at multiple resolutions
results = {
    32:  run_simulation(Nr=32, Nz=64),
    64:  run_simulation(Nr=64, Nz=128),
    128: run_simulation(Nr=128, Nz=256)
}

# Measure convergence
gamma_32 = results[32]['growth_rate']
gamma_64 = results[64]['growth_rate']
gamma_128 = results[128]['growth_rate']

# Richardson extrapolation
error_64 = |gamma_64 - gamma_128| / gamma_128

# Pass if converged
assert error_64 < 0.05  # 5% convergence
```

---

### 6.2 时间步长

**CFL condition:**
```
dt < C_CFL * min(
    dr² / (4*eta),      # Diffusion limit
    dr² / (4*nu),       # Viscous limit
    dr / v_max          # Advection limit
)

C_CFL = 0.5  (safety factor)
```

**Adaptive time stepping (可选):**
```python
def adaptive_dt(psi, omega, dr, dz):
    """Compute CFL-limited time step"""
    
    # Velocity estimate
    v = compute_velocity(omega)
    v_max = np.abs(v).max()
    
    # CFL conditions
    dt_diff = 0.25 * dr**2 / eta
    dt_visc = 0.25 * dr**2 / nu
    dt_adv = 0.5 * dr / (v_max + 1e-10)
    
    return min(dt_diff, dt_visc, dt_adv)
```

---

### 6.3 Integration Accuracy

**RK4 truncation error:**
```
Local error ~ O(dt⁵)
Global error ~ O(dt⁴)
```

**Verification test:**
```python
# Compare dt vs dt/2
result_dt = run_simulation(dt=1e-4, steps=1000)
result_dt2 = run_simulation(dt=5e-5, steps=2000)

# Richardson estimate of error
error = |result_dt['w_final'] - result_dt2['w_final']| / result_dt2['w_final']

# 4th-order convergence
assert error < 16 * (dt/2)**4 / dt**4  # ≈ 1/16
```

---

## 7. Diagnostics要求

### 7.1 Island Width Measurement

**Algorithm:**

**Poincaré section method:**
```python
def measure_island_width(psi, psi_eq, r_s):
    """
    Measure island width on Poincaré section
    
    Steps:
        1. Compute perturbed flux: δψ = ψ - ψ_eq
        2. Find O-point (local max of δψ at r ≈ r_s)
        3. Find X-point (local min of δψ)
        4. Island width: w = 2 * |r_O - r_X|
    """
    delta_psi = psi - psi_eq
    
    # Restrict to region near r_s
    mask = (r > r_s - 0.2) & (r < r_s + 0.2)
    
    # Find extrema
    i_O = np.argmax(delta_psi[mask])
    i_X = np.argmin(delta_psi[mask])
    
    r_O = r[mask][i_O]
    r_X = r[mask][i_X]
    
    w = 2 * abs(r_O - r_X)
    
    return w
```

**验收标准:**
```python
# Grid convergence of island width
w_32 = measure_island_width(..., Nr=32)
w_64 = measure_island_width(..., Nr=64)

# Should converge
assert |w_64 - w_32| / w_64 < 0.1  # 10% change
```

---

### 7.2 Growth Rate Calculation

**Method: Exponential fit in linear phase**

```python
def compute_growth_rate(w_history, t_history):
    """
    Fit w(t) = w0 * exp(γ*t)
    
    Linear fit: log(w) = log(w0) + γ*t
    """
    # Use only linear phase (w < 0.1 * r_s)
    linear_phase = np.array(w_history) < 0.1 * r_s
    
    t = np.array(t_history)[linear_phase]
    log_w = np.log(np.array(w_history)[linear_phase])
    
    # Linear regression
    gamma, log_w0 = np.polyfit(t, log_w, deg=1)
    
    return gamma
```

**验收标准:**
- 至少10个数据点 in linear phase
- R² > 0.95 (good exponential fit)

---

## 8. Performance要求

### 8.1 Computational Efficiency

**Acceptable performance (NumPy baseline):**
```
Grid: 64×128
Time steps: 10,000 (t ~ 1.0 τ_R)
CPU time: < 10s per simulation (single core)
```

**If slower:**
- Profile code (cProfile)
- Optimize hotspots (Poisson solver, RHS evaluation)
- Consider JAX/GPU (Phase 2)

---

### 8.2 Memory Requirements

**Typical memory footprint:**
```
State arrays: 
  psi: Nr×Nz×8 bytes = 64×128×8 = 64 KB
  omega: 64 KB
  phi: 64 KB
  
Total per state: ~200 KB

History storage (optional):
  1000 snapshots × 200 KB = 200 MB
```

**Acceptable:**
- < 1 GB RAM for single simulation
- 可扩展 to batch simulations (10-100 parallel)

---

## 9. Summary of Requirements

### Critical (Must-have):
- [x] Reduced MHD方程完整实现
- [x] ∇·B < 1e-6
- [x] 能量守恒 < 1%
- [x] FKR growth rate < 20% error
- [x] PyTokEq平衡态集成
- [x] Island width可测量

### Important (Should-have):
- [ ] Rutherford saturation (factor-of-2)
- [ ] RMP响应正确
- [ ] Grid convergence验证
- [ ] CFL-safe time stepping

### Nice-to-have:
- [ ] Adaptive time stepping
- [ ] Toroidal effects
- [ ] Advanced diagnostics (current profile, q evolution)

---

**这些要求是PyTokMHD的physics质量标准。**

**小P签字: 2026-03-16 ⚛️**
