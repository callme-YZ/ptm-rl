# PyTokMHD Layer 2 Design Document

**Author:** 小P ⚛️ (Physics Lead)  
**Date:** 2026-03-16  
**Status:** Design Phase  
**Purpose:** Layer 2 (MHD Evolution) 架构设计和技术选择

---

## 1. 目标和范围

### 1.1 Layer 2职责

**PyTokMHD = Physics-realistic MHD evolution layer**

**核心功能:**
1. **Time evolution:** 从PyTokEq平衡态演化MHD系统
2. **Tearing mode simulation:** 撕裂模的非线性演化
3. **External control:** RMP coils等外部控制场
4. **Physics diagnostics:** Island width, growth rate, energy conservation

**边界定义:**
- **Input (from Layer 1):** PyTokEq平衡态 (ψ_eq, j_tor, p, q)
- **Output (to Layer 3):** MHD状态 (ψ, ω, B) + diagnostics (w, γ, ...)
- **不包括:** RL算法、policy、training (Layer 3职责)

---

### 1.2 设计原则

**Physics correctness优先:**
1. **真实MHD物理** — 不简化方程
2. **守恒律验证** — 能量、磁通量、拓扑
3. **可验证benchmark** — 与理论/实验对照
4. **清晰physics API** — 易于理解和维护

**与PyTearRL的关键差异:**

| 特性 | PyTearRL (Simplified) | PyTokMHD (Realistic) |
|------|----------------------|---------------------|
| **平衡态** | Harris sheet | PyTokEq真实平衡态 |
| **几何** | 笛卡尔2D | 柱坐标 + toroidal effects |
| **MHD方程** | 简化resistive MHD | 完整resistive MHD |
| **边界条件** | 周期性 | 真实tokamak边界 |
| **验证** | Analytical | FKR theory + benchmarks |

---

## 2. 架构设计

### 2.1 模块结构

```
src/pytokmhd/
├── __init__.py
├── solver/                      ← MHD演化求解器
│   ├── __init__.py
│   ├── mhd_equations.py         ← Reduced MHD equations
│   ├── time_integrator.py       ← RK4/Implicit integrators
│   ├── boundary.py              ← 边界条件处理
│   └── poisson_solver.py        ← ∇²ψ = -ω solver
├── diagnostics/                 ← 撕裂模诊断
│   ├── __init__.py
│   ├── tearing_mode.py          ← Island width, growth rate
│   ├── island_width.py          ← Poincaré map + separatrix
│   └── energy_conservation.py   ← 守恒律检查
├── validation/                  ← 物理验证
│   ├── __init__.py
│   ├── benchmark.py             ← FKR benchmark
│   └── conservation_tests.py    ← 能量/磁通守恒测试
└── external_field/              ← 外部控制场
    ├── __init__.py
    ├── rmp_coils.py             ← RMP coil fields
    └── equilibrium_field.py     ← 背景平衡场管理
```

---

### 2.2 核心API设计

#### 2.2.1 MHDSolver接口

```python
class PyTokMHDSolver:
    """
    Reduced MHD solver for tokamak tearing modes
    
    Physics model: Reduced resistive MHD in (r, θ, z) cylindrical coords
    
    Variables:
        ψ(r,z,t): Poloidal flux (r-z plane, axisymmetric)
        ω(r,z,t): Vorticity (stream function for poloidal flow)
        
    Equations:
        ∂ψ/∂t = η∇²ψ - [ψ, φ]                (Induction)
        ∂ω/∂t = ν∇²ω - [ω, φ] + J×B          (Vorticity)
        ∇²φ = -ω                              (Stream function)
        
    Where [f,g] = (∂f/∂r)(∂g/∂z) - (∂f/∂z)(∂g/∂r) (Poisson bracket)
    """
    
    def __init__(self,
                 grid: Grid,
                 eta: float = 1e-5,      # Resistivity
                 nu: float = 1e-6,       # Viscosity
                 dt: float = 1e-4):      # Time step
        """
        Initialize MHD solver
        
        Args:
            grid: Computational grid (r, z)
            eta: Magnetic diffusivity (normalized)
            nu: Kinematic viscosity (normalized)
            dt: Time step size (CFL-limited)
        """
        pass
    
    def initialize_from_equilibrium(self,
                                   psi_eq: np.ndarray,
                                   j_tor: np.ndarray,
                                   perturbation: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize MHD state from PyTokEq equilibrium
        
        Args:
            psi_eq: Equilibrium poloidal flux from PyTokEq
            j_tor: Toroidal current density
            perturbation: {
                'mode': (m, n),           # Helical mode numbers
                'amplitude': float,       # Perturbation amplitude
                'seed': int               # Random seed for phase
            }
            
        Returns:
            psi_init: Initial poloidal flux (equilibrium + perturbation)
            omega_init: Initial vorticity
            
        Physics:
            1. Add helical perturbation at rational surface q = m/n
            2. Ensure ∇·B = 0 (solenoidal constraint)
            3. Compute consistent vorticity field
        """
        pass
    
    def step(self,
            psi: np.ndarray,
            omega: np.ndarray,
            external_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve MHD state by one time step
        
        Args:
            psi: Current poloidal flux
            omega: Current vorticity
            external_field: External B field (e.g., RMP coils)
            
        Returns:
            psi_new: Updated flux
            omega_new: Updated vorticity
            
        Method: RK4 time integration
        """
        pass
    
    def compute_diagnostics(self,
                           psi: np.ndarray,
                           omega: np.ndarray) -> Dict[str, float]:
        """
        Compute physics diagnostics
        
        Returns:
            {
                'island_width': float,      # Magnetic island width
                'growth_rate': float,       # Instantaneous γ
                'magnetic_energy': float,   # ∫B²dV
                'kinetic_energy': float,    # ∫v²dV
                'div_B_max': float          # ∇·B conservation
            }
        """
        pass
```

---

#### 2.2.2 Diagnostics API

```python
class TearingModeDiagnostics:
    """撕裂模物理诊断工具"""
    
    def measure_island_width(self,
                            psi: np.ndarray,
                            psi_eq: np.ndarray,
                            q_profile: np.ndarray,
                            mode: Tuple[int, int]) -> float:
        """
        Measure magnetic island width
        
        Method: Poincaré section + separatrix detection
        
        Algorithm:
            1. Find rational surface r_s where q(r_s) = m/n
            2. Compute perturbed flux ψ_pert = ψ - ψ_eq
            3. Find O-point (max ψ_pert) and X-point (min ψ_pert)
            4. Island width w = 2 * |r_O - r_X|
            
        Args:
            psi: Current poloidal flux
            psi_eq: Equilibrium flux (from PyTokEq)
            q_profile: Safety factor q(r)
            mode: (m, n) mode numbers
            
        Returns:
            w: Island width [normalized units]
        """
        pass
    
    def compute_growth_rate(self,
                           w_history: List[float],
                           t_history: List[float]) -> float:
        """
        Compute tearing mode growth rate γ
        
        Method: Fit w(t) ~ w0 * exp(γt) in linear phase
        
        Args:
            w_history: Island width time series
            t_history: Corresponding time points
            
        Returns:
            γ: Growth rate (1/time)
        """
        pass
```

---

### 2.3 Physics Model选择

#### 2.3.1 Reduced MHD方程

**为什么选Reduced MHD?**

**优点 ✅:**
1. **Physics-complete** for tearing modes
2. **Computationally tractable** (2D instead of 3D full MHD)
3. **Well-validated** in tokamak literature
4. **Captures key physics:**
   - Magnetic reconnection
   - Island formation
   - Resistive effects
   - External control response

**方程组:**

```
Induction equation:
∂ψ/∂t = -[φ, ψ] + η∇²ψ

Vorticity equation:
∂ω/∂t = -[φ, ω] + ν∇²ω + J×B poloidal projection

Poisson equation:
∇²φ = -ω
```

**Physical meaning:**
- ψ: Poloidal flux → describes magnetic field lines
- ω: Vorticity → describes plasma flow
- φ: Stream function → velocity potential
- [f,g]: Poisson bracket → convective nonlinearity

---

#### 2.3.2 边界条件

**Realistic tokamak boundaries:**

```python
# 径向边界 (r方向)
r = r_min:  ψ = 0,  ∂ψ/∂r = 0    # 磁轴 (regularity)
r = r_max:  ψ = ψ_edge (fixed)     # 边界通量 (conducting wall)

# 轴向边界 (z方向)  
z = 0, L_z:  周期性边界条件         # Toroidal periodicity
```

**与PyTearRL对比:**
- PyTearRL: 全周期性 → 非物理
- PyTokMHD: 径向固定 + 轴向周期 → 真实tokamak

---

#### 2.3.3 数值方法

**空间离散: Finite Difference**

**选择理由:**
- ✅ PyTearRL已实现并验证
- ✅ 简单、可复用90%代码
- ✅ 足够精确 (64×128 grid)

**时间积分: RK4**

```python
def rk4_step(psi, omega, dt):
    """4th-order Runge-Kutta"""
    k1_psi, k1_omega = rhs(psi, omega)
    k2_psi, k2_omega = rhs(psi + 0.5*dt*k1_psi, omega + 0.5*dt*k1_omega)
    k3_psi, k3_omega = rhs(psi + 0.5*dt*k2_psi, omega + 0.5*dt*k2_omega)
    k4_psi, k4_omega = rhs(psi + dt*k3_psi, omega + dt*k3_omega)
    
    psi_new = psi + dt/6 * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)
    omega_new = omega + dt/6 * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    
    return psi_new, omega_new
```

**CFL条件:**
```
dt < min(dr²/η, dr²/ν, dr/v_max)
```

---

## 3. PyTokEq集成方案

### 3.1 数据流

```
Layer 1 (PyTokEq)
    ↓ 
    ψ_eq(r,z), j_tor(r,z), p(r,z), q(r)
    ↓
Layer 2 (PyTokMHD)
    ↓
    Initialize: ψ_init = ψ_eq + perturbation
    ↓
    Evolve: ψ(t), ω(t) using MHD equations
    ↓
    Diagnostics: w(t), γ, energies
    ↓
Layer 3 (RL Environment)
```

---

### 3.2 集成接口实现

```python
def integrate_pytokeq_equilibrium(
    eq_solution: Dict,          # PyTokEq output
    perturbation_config: Dict   # Perturbation parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PyTokEq equilibrium to PyTokMHD initial condition
    
    Args:
        eq_solution: {
            'psi': (Nr, Nz),     # Poloidal flux
            'j_tor': (Nr, Nz),   # Toroidal current
            'pressure': (Nr, Nz),
            'q_profile': (Nr,)   # Safety factor
        }
        perturbation_config: {
            'mode': (2, 1),
            'amplitude': 1e-5,
            'seed': 42
        }
        
    Returns:
        psi_init: Initial flux with perturbation
        omega_init: Consistent vorticity field
        
    Steps:
        1. Extract ψ_eq from PyTokEq
        2. Find rational surface q = m/n
        3. Add helical perturbation
        4. Compute vorticity ω = -∇²φ from force balance
    """
    
    psi_eq = eq_solution['psi']
    q_profile = eq_solution['q_profile']
    
    # Find rational surface
    m, n = perturbation_config['mode']
    r_s = find_rational_surface(q_profile, q_target=m/n)
    
    # Add perturbation
    amp = perturbation_config['amplitude']
    seed = perturbation_config['seed']
    
    psi_pert = create_helical_perturbation(
        grid=grid,
        r_s=r_s,
        mode=(m, n),
        amplitude=amp,
        seed=seed
    )
    
    psi_init = psi_eq + psi_pert
    
    # Compute consistent vorticity
    # From equilibrium: J×B = ∇p → ω_eq ≈ 0 (force balance)
    # Add perturbation vorticity from perturbed current
    omega_init = compute_initial_vorticity(psi_init, psi_eq)
    
    return psi_init, omega_init
```

---

### 3.3 Grid对齐

**问题:** PyTokEq和PyTokMHD可能使用不同grid

**解决方案: 插值**

```python
def interpolate_to_mhd_grid(
    psi_eq_pytokeq: np.ndarray,
    grid_pytokeq: Grid,
    grid_mhd: Grid
) -> np.ndarray:
    """
    Interpolate PyTokEq solution to PyTokMHD grid
    
    Method: 2D cubic spline interpolation
    """
    from scipy.interpolate import RectBivariateSpline
    
    spline = RectBivariateSpline(
        grid_pytokeq.r,
        grid_pytokeq.z,
        psi_eq_pytokeq
    )
    
    psi_eq_mhd = spline(grid_mhd.r, grid_mhd.z, grid=True)
    
    return psi_eq_mhd
```

---

## 4. 验证策略

### 4.1 Physics Benchmarks

**Benchmark 1: FKR Theory**

**Test case:** Linear tearing mode in slab geometry

**Expected:**
```
Growth rate: γ ≈ 0.55 * η^(3/5) * Δ'^(4/5)
Island width: w ~ w0 * exp(γt) in linear phase
```

**Validation:**
- Measure γ from w(t) time series
- Compare with FKR prediction
- **Acceptance:** |γ_measured - γ_FKR| / γ_FKR < 20%

---

**Benchmark 2: Rutherford Regime**

**Test case:** Nonlinear island saturation

**Expected:**
```
Saturated width: w_sat ~ (Δ' * r_s)^(1/2)
Timescale: τ_NL ~ τ_R = r_s² / η
```

**Validation:**
- Evolve until w(t) saturates
- Compare w_sat with Rutherford theory
- **Acceptance:** 0.5 < w_sat / w_theory < 2.0

---

### 4.2 Conservation Laws

**Test 1: Energy Conservation (with resistivity)**

```python
def test_energy_conservation():
    """
    Total energy: E = E_magnetic + E_kinetic
    
    Energy evolution:
    dE/dt = -∫(η J² + ν |∇v|²) dV  (dissipation)
    
    Test: dE/dt should match resistive/viscous dissipation
    """
    E_mag = 0.5 * ∫ B² dV
    E_kin = 0.5 * ∫ ρ v² dV
    E_total = E_mag + E_kin
    
    dissipation = ∫ (η * J² + ν * |∇v|²) dV
    
    dE_dt_measured = (E_total[t+1] - E_total[t]) / dt
    
    assert |dE_dt_measured + dissipation| < 1e-3 * |dE_dt_measured|
```

---

**Test 2: ∇·B = 0**

```python
def test_divergence_free():
    """
    Magnetic field must be solenoidal
    
    B = ∇ψ × ẑ + B_z ẑ
    ∇·B = 0 (automatically for flux function)
    
    Test: Numerical divergence < 1e-6
    """
    div_B = compute_divergence(B_r, B_z)
    
    assert np.abs(div_B).max() < 1e-6
```

---

### 4.3 RMP响应测试

**Test case:** Island suppression by external RMP

**Setup:**
```python
# 1. Grow tearing mode to w0
# 2. Apply RMP at t = t_control
# 3. Measure Δw = w_final - w0
```

**Expected physics:**
- RMP phase-locked to island → suppression
- Wrong phase → amplification
- **Acceptance:** Sign of Δw consistent with RMP phase

---

## 5. 性能考虑

### 5.1 计算复杂度

**Per time step:**
```
Poisson solve: O(N² log N)  (FFT method)
RHS evaluation: O(N²)        (finite difference)
RK4 overhead: 4× RHS calls

Total: O(N² log N) per step
```

**Typical parameters:**
```
Grid: 64×128 → N² ≈ 8K
Time steps: 10K (for t ~ 1.0 τ_R)

Total ops: ~100M per simulation
Time: ~1s on CPU (NumPy)
```

---

### 5.2 优化策略

**Phase 1: NumPy实现**
- Simple, maintainable
- ~1s per simulation acceptable for development
- **Sufficient for validation and small-scale RL**

**Phase 2 (可选): JAX加速**
- JIT compilation
- GPU parallelization
- **10-100× speedup → large-scale training**

---

## 6. 实现里程碑

### Phase 1: 核心Solver (Week 1-2)

**Deliverables:**
- [ ] `mhd_equations.py` — RHS函数实现
- [ ] `time_integrator.py` — RK4 integrator
- [ ] `poisson_solver.py` — ∇²ψ solver
- [ ] `boundary.py` — 边界条件
- [ ] Basic test: Harris sheet evolution

**Validation:**
- ∇·B < 1e-6
- Energy conservation within 1%
- Stable for 1000 time steps

---

### Phase 2: PyTokEq集成 (Week 3)

**Deliverables:**
- [ ] `equilibrium_interface.py` — PyTokEq → PyTokMHD
- [ ] Grid interpolation
- [ ] Perturbation generation
- [ ] Test: Solovev equilibrium evolution

**Validation:**
- Initial condition physics-consistent
- Equilibrium preserved (no spurious growth)

---

### Phase 3: Diagnostics (Week 4)

**Deliverables:**
- [ ] `tearing_mode.py` — Island width measurement
- [ ] `energy_conservation.py` — Conservation tests
- [ ] Growth rate calculation
- [ ] Benchmark against FKR theory

**Validation:**
- FKR benchmark: γ within 20%
- Island width measurement converged

---

### Phase 4: External Control (Week 5)

**Deliverables:**
- [ ] `rmp_coils.py` — External field calculation
- [ ] RMP response tests
- [ ] Control effectiveness metrics

**Validation:**
- RMP suppresses island (correct phase)
- Control power scaling correct

---

### Phase 5: RL Interface (Week 6)

**Deliverables:**
- [ ] Clean API for Layer 3 (RL environment)
- [ ] Documentation
- [ ] Example usage
- [ ] Performance profiling

**Validation:**
- API易用性 (小A review)
- Physics correctness (小P验证)

---

## 7. Risk Mitigation

### Risk 1: 数值不稳定

**Mitigation:**
- Start with coarse grid (32×64)
- Use conservative CFL condition (dt = 0.5 * dt_max)
- Implement adaptive time stepping (如果需要)

---

### Risk 2: PyTokEq集成复杂

**Mitigation:**
- Phase 2前先完成standalone solver
- 使用简单Solovev平衡态测试集成
- Grid插值验证独立进行

---

### Risk 3: 验证失败

**Mitigation:**
- 预留2周buffer for debugging
- 如FKR benchmark不通过,降级到slab geometry
- 与小A协调:即使PyTokMHD延迟,PyTearRL可继续

---

## 8. Success Criteria

**最低标准 (MVP):**
- [ ] MHD solver运行稳定 (1000+ steps)
- [ ] ∇·B < 1e-5
- [ ] PyTokEq集成成功 (Solovev case)
- [ ] Island width可测量

**理想标准:**
- [ ] FKR benchmark通过 (γ < 20% error)
- [ ] 能量守恒 < 1%
- [ ] RMP控制有效
- [ ] 小A验收通过 (API可用性)

---

## 9. Timeline

```
Week 1-2: Core solver + basic tests
Week 3:   PyTokEq integration
Week 4:   Diagnostics + FKR benchmark
Week 5:   External control (RMP)
Week 6:   RL interface + documentation
```

**Total: 6 weeks to production-ready PyTokMHD**

---

## 10. Summary

**PyTokMHD = Physics-realistic Layer 2**

**Key design decisions:**
1. ✅ Reduced MHD方程 (physics-complete, tractable)
2. ✅ Finite difference + RK4 (复用PyTearRL经验)
3. ✅ Realistic boundaries (与PyTearRL差异化)
4. ✅ 严格验证 (FKR + conservation laws)
5. ✅ Clean API for Layer 3集成

**Ready for implementation.**

**小P签字: 2026-03-16 ⚛️**
