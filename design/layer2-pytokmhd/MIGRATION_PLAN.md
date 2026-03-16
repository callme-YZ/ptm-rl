# PyTearRL → PyTokMHD Migration Plan

**Author:** 小P ⚛️ + 小A 🤖  
**Date:** 2026-03-16  
**Purpose:** 从PyTearRL简化版本迁移到PyTokMHD真实物理版本的详细计划

---

## 1. 迁移目标

### 1.1 为什么迁移?

**PyTearRL现状 (Simplified):**
- ✅ 可运行的MHD solver
- ✅ RL environment框架
- ✅ Baseline训练完成
- ❌ Harris sheet平衡态 (非真实)
- ❌ 笛卡尔几何 (非tokamak)
- ❌ 简化物理 (缺少关键项)

**PyTokMHD目标 (Realistic):**
- ✅ PyTokEq真实平衡态
- ✅ 柱坐标几何 (tokamak-like)
- ✅ 完整reduced MHD
- ✅ 可验证的物理 (FKR benchmark)

**Why not start from scratch?**
- PyTearRL有大量可复用代码 (MHD solver核心 ~ 90%)
- 已有RL环境接口设计
- 数值方法已验证 (RK4, Poisson solver)

**Strategy: 改造,不重写** ✅

---

### 1.2 关键差异分析

| Component | PyTearRL | PyTokMHD | Migration Strategy |
|-----------|----------|----------|-------------------|
| **平衡态** | Harris sheet | PyTokEq | 替换初始化函数 |
| **几何** | Cartesian (x,y) | Cylindrical (r,z) | 修改operators |
| **边界** | Periodic | r固定 + z周期 | 修改BC处理 |
| **MHD方程** | Simplified | Full reduced MHD | 补充missing terms |
| **Diagnostics** | Island width (简化) | Poincaré + FKR | 重写diagnostics |
| **RL interface** | Gym API | Same ✅ | 保持不变 |

**复用率估算: ~85%代码可保留**

---

## 2. 代码审计

### 2.1 PyTearRL现有实现分析

**文件结构:**
```
/workspace-xiaoa/
├── dynamic_tearing_env_final.py       ← RL environment (复用90%)
├── tearing_mode_current_sheet.py     ← MHD solver (改造70%)
├── train_tearing_rl.py                ← Training script (复用100%)
└── [其他测试文件]
```

**检查 tearing_mode_current_sheet.py:**

**可复用部分 ✅:**
1. `laplacian()` — ∇²算子 (需改造为cylindrical)
2. `step()` — RK4时间积分 (保持不变)
3. `compute_current()` — J=∇×B (需改造)
4. Poisson solver框架 (保持)

**需要替换 ❌:**
1. `initialize_current_sheet()` → `initialize_from_pytokeq()`
2. Harris sheet平衡态 → PyTokEq输入
3. 笛卡尔gradients → cylindrical derivatives

**需要补充 ➕:**
1. PyTokEq集成接口
2. Helical perturbation生成
3. Improved diagnostics (Poincaré map)
4. FKR benchmark tests

---

### 2.2 关键函数映射

#### 2.2.1 初始化函数

**PyTearRL (current):**
```python
def initialize_current_sheet(self, B0=1.0, r0=0.5, a=0.1, epsilon=0.05):
    """
    Harris sheet equilibrium:
    Bz = B0
    Bθ = B0 * tanh((r-r0)/a)
    """
    # Hardcoded analytical profile
    Btheta = B0 * np.tanh((r - r0) / a)
    ...
```

**PyTokMHD (target):**
```python
def initialize_from_equilibrium(self,
                               psi_eq: np.ndarray,
                               j_tor: np.ndarray,
                               q_profile: np.ndarray,
                               perturbation: Dict):
    """
    Initialize from PyTokEq solution:
    1. Load equilibrium fields
    2. Add helical perturbation at q = m/n surface
    3. Compute consistent vorticity
    """
    # Read PyTokEq output
    # Find rational surface
    # Add perturbation
    # Return psi_init, omega_init
```

**Migration:** 完全替换,无法复用

---

#### 2.2.2 Operators (最关键)

**Cartesian → Cylindrical transformation:**

| Operator | Cartesian | Cylindrical |
|----------|-----------|-------------|
| **Gradient** | (∂/∂x, ∂/∂y) | (∂/∂r, (1/r)∂/∂θ, ∂/∂z) |
| **Laplacian** | ∂²/∂x² + ∂²/∂y² | (1/r)∂/∂r(r∂/∂r) + ∂²/∂z² |
| **Divergence** | ∂f_x/∂x + ∂f_y/∂y | (1/r)∂(rf_r)/∂r + ∂f_z/∂z |
| **Poisson bracket** | [f,g]_xy | [f,g]_rz |

**现有代码 (PyTearRL):**
```python
def laplacian(self, f):
    """∇²f Cartesian"""
    lap = np.zeros_like(f)
    for i in range(1, self.Nr-1):
        for j in range(self.Nz):
            radial = (f[i+1,j] - 2*f[i,j] + f[i-1,j]) / self.dr**2
            axial = (f[i,jp] - 2*f[i,j] + f[i,jm]) / self.dz**2
            lap[i,j] = radial + axial
    return lap
```

**修改为 (PyTokMHD):**
```python
def laplacian_cylindrical(self, f):
    """∇²f Cylindrical"""
    lap = np.zeros_like(f)
    for i in range(1, self.Nr-1):
        r = self.r[i]
        for j in range(self.Nz):
            # Radial: (1/r) d/dr(r df/dr)
            df_dr_plus = (f[i+1,j] - f[i,j]) / self.dr
            df_dr_minus = (f[i,j] - f[i-1,j]) / self.dr
            r_plus = (self.r[i] + self.r[i+1]) / 2
            r_minus = (self.r[i-1] + self.r[i]) / 2
            
            radial = (r_plus*df_dr_plus - r_minus*df_dr_minus) / (r*self.dr)
            
            # Axial: same as before
            axial = (f[i,jp] - 2*f[i,j] + f[i,jm]) / self.dz**2
            
            lap[i,j] = radial + axial
    
    # Special treatment at r=0 (axis)
    lap[0,:] = 4*(f[1,:] - f[0,:]) / self.dr**2
    
    return lap
```

**Migration strategy:**
- 保留函数结构
- 修改radial部分 (1/r因子)
- 添加r=0边界处理

---

#### 2.2.3 Induction Equation

**Current (PyTearRL):**
```python
def induction(self, Br, Btheta, Bz, vr, vtheta, vz):
    dBr = self.eta * self.laplacian(Br) + advection_terms
    dBtheta = self.eta * self.laplacian(Btheta) + advection_terms
    dBz = self.eta * self.laplacian(Bz) + advection_terms
    return dBr, dBtheta, dBz
```

**Needed (PyTokMHD):**
```python
def induction_cylindrical(self, psi, phi):
    """
    ∂ψ/∂t = -[φ, ψ] + η∇²ψ
    
    Where:
        [φ,ψ] = (∂φ/∂r)(∂ψ/∂z) - (∂φ/∂z)(∂ψ/∂r)  (Poisson bracket)
    """
    # Resistive term
    dpsi_dt = self.eta * self.laplacian_cylindrical(psi)
    
    # Advection term
    poisson_bracket = self.compute_poisson_bracket(phi, psi)
    dpsi_dt -= poisson_bracket
    
    return dpsi_dt
```

**Migration:**
- Simplify from (Br, Bθ, Bz) → single flux function ψ
- 改用Poisson bracket formulation
- **Conceptual change, but simpler!**

---

### 2.3 Complexity Assessment

**Function-level complexity:**

| Function | Lines (PyTearRL) | Lines (PyTokMHD) | Change | Effort |
|----------|-----------------|-----------------|--------|--------|
| `__init__` | 20 | 25 | +Grid params | Low |
| `initialize_*` | 80 | 120 | Complete rewrite | High |
| `laplacian` | 30 | 40 | Cylindrical | Medium |
| `induction` | 100 | 60 | Simplify to ψ | Medium |
| `momentum` | 80 | 60 | Simplify to ω | Medium |
| `step` (RK4) | 40 | 40 | No change ✅ | None |
| `compute_current` | 40 | 30 | Cylindrical | Low |
| Diagnostics | 50 | 150 | Add Poincaré | High |

**Total估算:**
- 保留: ~200 lines (step, basic structure)
- 修改: ~300 lines (operators, equations)
- 新增: ~200 lines (PyTokEq interface, diagnostics)

**Total: ~700 lines (vs 500 in PyTearRL)**

**Time估算: 2-3周**

---

## 3. 迁移步骤

### Step 1: 创建PyTokMHD骨架 (Day 1-2)

**Tasks:**
1. 创建新目录 `src/pytokmhd/solver/`
2. 复制 `tearing_mode_current_sheet.py` → `mhd_solver.py`
3. 重命名class: `CurrentSheetTearing` → `PyTokMHDSolver`
4. 保留:
   - `__init__`
   - `step` (RK4框架)
   - 基础grid定义

**Deliverable:**
```python
# src/pytokmhd/solver/mhd_solver.py
class PyTokMHDSolver:
    def __init__(self, grid, eta, nu, dt):
        # Copied from PyTearRL
        pass
    
    def step(self, psi, omega, dt):
        # RK4 framework (unchanged)
        pass
```

**Validation:** 代码可import,无syntax errors

---

### Step 2: 改造Operators为Cylindrical (Day 3-5)

**Tasks:**
1. 修改 `laplacian()` 添加 (1/r) 因子
2. 实现 `compute_poisson_bracket(f, g)`
3. 修改 `compute_divergence()` 为cylindrical
4. 添加 r=0 边界特殊处理

**Test:**
```python
def test_laplacian_cylindrical():
    """Test against known analytical solution"""
    # ∇²(r²) = 4 in cylindrical
    r = np.linspace(0.1, 1.0, 32)
    z = np.linspace(0, 2.0, 64)
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    f = R**2
    lap_f = solver.laplacian_cylindrical(f)
    
    # Should be ~4 everywhere
    assert np.allclose(lap_f[1:-1, 1:-1], 4.0, rtol=0.05)
```

**Validation:**
- Laplacian test通过
- ∇·B analytical test通过

---

### Step 3: 实现PyTokEq集成 (Day 6-8)

**Tasks:**
1. 创建 `equilibrium_interface.py`
2. 实现 `load_pytokeq_solution()`
3. 实现 `find_rational_surface(q_profile, m, n)`
4. 实现 `add_helical_perturbation()`
5. Grid interpolation (如需要)

**Code:**
```python
# src/pytokmhd/external_field/equilibrium_interface.py

def load_pytokeq_solution(filepath: str) -> Dict:
    """Load PyTokEq equilibrium from file"""
    import pickle
    with open(filepath, 'rb') as f:
        eq = pickle.load(f)
    return eq

def find_rational_surface(r: np.ndarray,
                         q_profile: np.ndarray,
                         m: int, n: int) -> float:
    """Find radius where q(r) = m/n"""
    q_target = m / n
    idx = np.argmin(np.abs(q_profile - q_target))
    return r[idx]

def add_helical_perturbation(psi_eq, grid, r_s, mode, amp, seed):
    """Add (m,n) helical perturbation at r_s"""
    m, n = mode
    Nr, Nz = psi_eq.shape
    
    # Radial profile (Gaussian at r_s)
    r = grid.r
    r_profile = np.exp(-((r - r_s)/0.1)**2)
    
    # Helical pattern
    z = grid.z
    theta = 2*np.pi*z / grid.Lz
    helical = np.sin(m * theta)
    
    # Perturbation
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2*np.pi)
    
    psi_pert = r_profile[:, None] * np.sin(m*theta[None,:] + phase)
    psi_init = psi_eq + amp * psi_pert
    
    return psi_init
```

**Test:**
```python
def test_pytokeq_integration():
    # Load Solovev test case
    eq = load_pytokeq_solution('test_data/solovev_eq.pkl')
    
    # Find rational surface
    r_s = find_rational_surface(eq['r'], eq['q_profile'], m=2, n=1)
    
    # Should be near expected location
    assert 0.4 < r_s < 0.6  # Typical for q=2 surface
    
    # Add perturbation
    psi_init = add_helical_perturbation(eq['psi'], grid, r_s, (2,1), 1e-5, 42)
    
    # Check perturbation small
    delta_psi = psi_init - eq['psi']
    assert np.abs(delta_psi).max() / np.abs(eq['psi']).max() < 0.01
```

**Validation:**
- PyTokEq数据可读取
- Rational surface定位准确
- Perturbation满足物理要求

---

### Step 4: 改造MHD Equations (Day 9-12)

**Tasks:**
1. Simplify from (Br, Bθ, Bz) → (ψ, ω) formulation
2. 实现 `induction_equation(psi, phi)`
3. 实现 `vorticity_equation(omega, psi, phi)`
4. 实现 `poisson_equation(omega)` → solve for φ
5. 组装完整RHS

**Code structure:**
```python
def compute_rhs(self, psi, omega):
    """Compute dψ/dt and dω/dt"""
    
    # 1. Solve Poisson equation for φ
    phi = self.solve_poisson(omega)
    
    # 2. Induction equation
    dpsi_dt = self.induction_equation(psi, phi)
    
    # 3. Vorticity equation
    domega_dt = self.vorticity_equation(omega, psi, phi)
    
    return dpsi_dt, domega_dt

def induction_equation(self, psi, phi):
    """∂ψ/∂t = -[φ,ψ] + η∇²ψ"""
    dpsi_dt = -self.poisson_bracket(phi, psi)
    dpsi_dt += self.eta * self.laplacian_cylindrical(psi)
    return dpsi_dt

def vorticity_equation(self, omega, psi, phi):
    """∂ω/∂t = -[φ,ω] + ν∇²ω + (J×B)_pol"""
    domega_dt = -self.poisson_bracket(phi, omega)
    domega_dt += self.nu * self.laplacian_cylindrical(omega)
    
    # Lorentz force term
    J = self.compute_current(psi)
    B = self.compute_B_field(psi)
    JxB = self.compute_lorentz_force(J, B)
    
    domega_dt += JxB
    return domega_dt
```

**Test:**
```python
def test_mhd_equations():
    """Test against equilibrium (should not evolve)"""
    # Start from PyTokEq equilibrium (no perturbation)
    psi_eq = load_equilibrium()
    omega_eq = np.zeros_like(psi_eq)  # Equilibrium: ω=0
    
    # Evolve 1 step
    dpsi_dt, domega_dt = solver.compute_rhs(psi_eq, omega_eq)
    
    # Should remain in equilibrium (rhs ≈ 0)
    assert np.abs(dpsi_dt).max() < 1e-6
    assert np.abs(domega_dt).max() < 1e-6
```

**Validation:**
- Equilibrium不演化 (Force balance test)
- 能量守恒 < 1%
- ∇·B < 1e-6

---

### Step 5: 实现Diagnostics (Day 13-15)

**Tasks:**
1. 实现 `measure_island_width()` — Poincaré方法
2. 实现 `compute_growth_rate()` — 指数拟合
3. 实现 `check_energy_conservation()`
4. 添加FKR benchmark测试

**Code:**
```python
# src/pytokmhd/diagnostics/tearing_mode.py

class TearingModeDiagnostics:
    def measure_island_width(self, psi, psi_eq, r_s):
        """Poincaré section method"""
        delta_psi = psi - psi_eq
        
        # Find O-point and X-point
        # ... (detailed in PYTOKMHD_DESIGN.md)
        
        return w
    
    def compute_growth_rate(self, w_history, t_history):
        """Fit w ~ exp(γt)"""
        log_w = np.log(w_history)
        gamma, _ = np.polyfit(t_history, log_w, deg=1)
        return gamma
```

**Test:**
```python
def test_island_measurement():
    """Test against known island configuration"""
    # Create artificial island
    psi_test = create_test_island(w_known=0.1)
    
    # Measure
    w_measured = diagnostics.measure_island_width(psi_test, psi_eq, r_s)
    
    # Should match
    assert abs(w_measured - w_known) / w_known < 0.05
```

**Validation:**
- Island width measurement稳定
- Growth rate拟合 R² > 0.95

---

### Step 6: RL Environment集成 (Day 16-18)

**Tasks:**
1. 修改 `dynamic_tearing_env_final.py` 使用PyTokMHD solver
2. 更新 `reset()` — 调用PyTokEq
3. 更新 `step()` — 使用新solver
4. 保持observation/action/reward API不变

**Code changes:**
```python
# OLD (PyTearRL):
from tearing_mode_current_sheet import CurrentSheetTearing
solver = CurrentSheetTearing(Nr=64, Nz=64)
Br, Bt, Bz, vr, vt, vz = solver.initialize_current_sheet(epsilon=0.05)

# NEW (PyTokMHD):
from pytokmhd.solver import PyTokMHDSolver
from pytokmhd.equilibrium import load_pytokeq_equilibrium

solver = PyTokMHDSolver(grid, eta=1e-5, nu=1e-6)
eq = load_pytokeq_equilibrium(cache_key=(beta_p, I_p))
psi, omega = solver.initialize_from_equilibrium(eq, perturbation_config)
```

**Validation:**
- Environment可运行
- Observation shape正确
- Random action不crash

---

### Step 7: 验证和Benchmark (Day 19-21)

**Tasks:**
1. FKR benchmark测试
2. Conservation laws测试
3. 对比PyTearRL vs PyTokMHD结果
4. 生成validation report

**Tests:**
```python
def test_fkr_benchmark():
    """Validate against FKR theory"""
    solver = PyTokMHDSolver(...)
    
    # Run simulation
    w_history, t_history = run_tearing_simulation(solver)
    
    # Measure growth rate
    gamma_measured = compute_growth_rate(w_history, t_history)
    
    # FKR prediction
    gamma_FKR = 0.55 * eta**(3/5) * Delta_prime**(4/5)
    
    # Validate
    error = abs(gamma_measured - gamma_FKR) / gamma_FKR
    assert error < 0.20, f"FKR error: {error:.1%}"
    
    print(f"✅ FKR benchmark passed: γ={gamma_measured:.3f}, theory={gamma_FKR:.3f}")
```

**Deliverable:**
- `validation_report.md` — 所有测试结果
- Plots: w(t), γ vs theory, energy conservation

---

## 4. 回归测试

### 4.1 与PyTearRL对比

**Sanity checks:**
```python
def test_simplified_case_matches_pytearrl():
    """
    When PyTokMHD uses Harris sheet equilibrium,
    should reproduce PyTearRL results
    """
    # PyTearRL result (baseline)
    env_old = DynamicTearingEnv(...)  # PyTearRL
    obs_old, _ = env_old.reset(seed=42)
    
    # PyTokMHD with Harris sheet (compatibility mode)
    env_new = PyTokMHDEnv(equilibrium_type='harris_sheet', ...)
    obs_new, _ = env_new.reset(seed=42)
    
    # Should match initial conditions
    assert np.allclose(obs_old, obs_new, rtol=0.05)
```

**Why important:**
- 确认迁移没有破坏working code
- Debugging reference

---

### 4.2 Physics Consistency

**Cross-validation:**
1. ✅ 能量守恒 (new vs old)
2. ✅ ∇·B (new < old,因为cylindrical更准确)
3. ✅ Island growth qualitative behavior

**Acceptance:**
- PyTokMHD物理更准确 (expected)
- 但基本behavior一致 (岛增长、RMP抑制)

---

## 5. Risk Mitigation

### Risk 1: Cylindrical operators bug

**Symptom:** ∇·B很大,能量不守恒

**Mitigation:**
- 每个operator独立测试 (analytical solutions)
- 与FEniCS/Firedrake cylindrical reference对比

**Fallback:**
- 如果cylindrical太难debug,暂时用Cartesian + warning

---

### Risk 2: PyTokEq集成复杂

**Symptom:** Grid不匹配,数据格式问题

**Mitigation:**
- Week 3专门处理集成
- 先用Solovev analytical solution测试
- Interpolation用成熟库 (scipy)

**Fallback:**
- 先手动生成compatible grid
- Phase 2再做通用interpolation

---

### Risk 3: Performance下降

**Symptom:** PyTokMHD比PyTearRL慢10×

**Mitigation:**
- Profile找hotspots
- Cylindrical operators优化 (vectorization)

**Fallback:**
- 接受2-3×慢 (physics换性能)
- Phase 2: JAX加速

---

## 6. Rollout Plan

### 6.1 Alpha版本 (Week 1-2)

**Goal:** 核心solver可运行

**Features:**
- [x] Cylindrical operators
- [x] Basic MHD equations
- [x] RK4 integrator
- [x] Conservation tests

**Not included:**
- PyTokEq integration (hardcoded equilibrium)
- Advanced diagnostics

---

### 6.2 Beta版本 (Week 3-4)

**Goal:** PyTokEq集成 + diagnostics

**Features:**
- [x] PyTokEq equilibrium loading
- [x] Perturbation generation
- [x] Island width measurement
- [x] FKR benchmark

**Not included:**
- RL environment (manual testing only)

---

### 6.3 Release版本 (Week 5-6)

**Goal:** 完整功能 + RL集成

**Features:**
- [x] RL environment updated
- [x] External control (RMP)
- [x] Documentation
- [x] Validation report

**Ready for:**
- 小A接手Layer 3 RL训练
- Production experiments

---

## 7. Success Metrics

### Code metrics:
- [ ] 85%+ code reuse from PyTearRL
- [ ] <1000 total lines (manageable)
- [ ] 100% test coverage (critical functions)

### Physics metrics:
- [ ] ∇·B < 1e-6 ✅
- [ ] Energy conservation < 1% ✅
- [ ] FKR γ error < 20% ✅

### Integration metrics:
- [ ] PyTokEq → PyTokMHD pipeline works
- [ ] RL environment API unchanged
- [ ] 小A验收通过 (API可用性)

---

## 8. Timeline Summary

```
Week 1: Skeleton + Operators
Week 2: MHD equations + basic tests
Week 3: PyTokEq integration
Week 4: Diagnostics + FKR benchmark
Week 5: RL environment + RMP control
Week 6: Validation + documentation

Total: 6 weeks to PyTokMHD v1.0
```

**Buffer: +2 weeks for unexpected issues**

---

## 9. Handoff to 小A

### 接口保证:

**小A不需要关心:**
- PyTokMHD内部实现
- Cylindrical vs Cartesian
- Numerical methods

**小A只需要:**
```python
# Same API as PyTearRL
from pytokmhd.environment import PyTokMHDEnv

env = PyTokMHDEnv(config)
obs, info = env.reset(seed=42)
obs, reward, done, trunc, info = env.step(action)
```

**Physics validation由小P负责** ✅

---

## 10. Conclusion

**Migration is feasible:**
- 85% code reuse
- 6 weeks total
- Low risk (staged rollout)

**Physics upgrade significant:**
- Real equilibrium (PyTokEq)
- Validated physics (FKR)
- Production-ready

**Ready to start Week 1 implementation.**

**小P + 小A签字: 2026-03-16 ⚛️🤖**
