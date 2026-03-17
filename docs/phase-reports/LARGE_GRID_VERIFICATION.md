# Phase 4 Validation Tests - Large Grid Verification Report

**日期:** 2026-03-16  
**任务:** 验证"Validation tests失败是因为小网格分辨率不足"的假设  
**执行者:** 小P ⚛️ (Subagent)

---

## 执行摘要

**假设状态:** ❌ **被推翻**

**结论:** 
- 大网格和小网格都失败
- 失败原因**不是**网格分辨率不足
- 问题出在**物理初始条件**或**数值稳定性**

---

## 测试配置

### Test 1: Small Grid (Baseline)
```python
Nr = 32
Nz = 64
n_steps = 50
```

### Test 2: Large Grid (Verification)
```python
Nr = 64
Nz = 128
n_steps = 100
```

**物理参数（相同）:**
- Mode: m=2, n=1 (tearing mode)
- q-profile: q(r) = 1.5 + 1.5(r/Lr)²
- RMP amplitude: 0.05

---

## 测试结果

### Small Grid (32×64, 50 steps)

**Result:** FAIL ❌

**Diagnostics:**
```python
{
    'gamma_free': 0.0,
    'gamma_rmp': 0.0,
    'reduction': 0.0,
    'w_free_final': 0.0,
    'w_rmp_final': 0.0,
    'w_free_history': [0., 0., 0., 0., 0.],
    'w_rmp_history': [0., 0., 0., 0., 0.],
    'success': False
}
```

**关键问题:**
- 磁岛宽度全程为0
- 没有观察到增长 (γ = 0)

---

### Large Grid (64×128, 100 steps)

**Result:** FAIL ❌

**Diagnostics:**
```python
{
    'gamma_free': 0.0,
    'gamma_rmp': 0.0,
    'reduction': 0.0,
    'w_free_final': 0.0,
    'w_rmp_final': 0.0,
    'w_free_history': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'w_rmp_history': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'success': False
}
```

**关键问题:**
- 与小网格完全相同的失败模式
- 磁岛宽度全程为0
- 没有观察到增长 (γ = 0)

---

## 对比分析

| 配置 | 网格 | γ_free | γ_rmp | Reduction | w_final | Status |
|------|------|--------|-------|-----------|---------|--------|
| Small | 32×64 | 0.0 | 0.0 | 0.0% | 0.0 | FAIL ❌ |
| Large | 64×128 | 0.0 | 0.0 | 0.0% | 0.0 | FAIL ❌ |

**关键观察:**
- 完全相同的失败模式
- 网格分辨率提升**4倍**（Nr×2, Nz×2）没有任何改善
- 演化时间加倍（50→100 steps）也没有改善

---

## 根本原因分析

### 1. 数值不稳定性 ⚠️

**警告信息:**
```
RuntimeWarning: overflow encountered in multiply
RuntimeWarning: invalid value encountered in subtract/add
```

**来源:**
- `mhd_equations.py:133` - Poisson bracket计算
- `rmp_coupling.py:154` - 时间演化
- 各种导数计算

**说明:** 数值已经出现 NaN/Inf，导致后续计算全部失败

---

### 2. 初始条件问题 🔍

**修复的Bug（执行过程中发现）:**

#### Bug 1: `find_rational_surface` 返回值解包错误
```python
# 错误代码
r_s, _ = find_rational_surface(q_profile, r_values, m/n)

# 修复后
r_s = find_rational_surface(r_values, q_profile, m/n)
```

**原因:** `initial_conditions.py` 中的 `find_rational_surface` 只返回单个float，不是tuple

---

#### Bug 2: Solovev equilibrium 维度不匹配
```python
# 错误代码
psi_eq = solovev_equilibrium(r_grid, z_grid)  # r_grid, z_grid 是 2D meshgrid

# 修复后
if r_grid.ndim == 2:
    r_1d = r_grid[:, 0]
    z_1d = z_grid[0, :]
else:
    r_1d = r_grid
    z_1d = z_grid
psi_eq, omega_eq = solovev_equilibrium(r_1d, z_1d)
```

**原因:** `solovev_equilibrium` 期望1D数组，但传入了2D meshgrid

---

#### Bug 3: Monitor 传递2D网格而非1D数组
```python
# 错误代码
monitor_free.update(psi_free, omega_free, t, R, Z, q_profile)

# 修复后
monitor_free.update(psi_free, omega_free, t, r, z, q_profile)
```

**原因:** `compute_island_width` → `find_rational_surface` 需要1D数组，但传入了2D meshgrid

---

### 3. 物理初始条件可能不正确 🚨

**问题:**
- 即使修复了上述bugs，磁岛宽度仍为0
- 说明 `setup_tearing_mode` 生成的初始扰动可能太小或不正确

**当前初始条件:**
```python
# Tearing mode perturbation
theta = np.arctan2(Z, R - 1.0)
delta_psi = w_0 * np.exp(-((R - 1.0 - r_s)**2) / (0.1**2)) * np.cos(m * theta)
```

**可能的问题:**
- `w_0 = 0.01` 可能太小
- 扰动形式可能不对（应该是 `cos(m*θ - n*φ)`）
- Rational surface位置 `r_s` 可能不对

---

## 结论和建议

### 结论 ✅

**假设验证结果:**
- ❌ 假设"网格分辨率不足"被**推翻**
- ✅ 确认问题**不是网格分辨率**
- ⚠️ 真正原因是**初始条件**或**数值稳定性**

---

### 建议 🎯

#### 立即行动（优先级P0）

1. **修复初始条件生成:**
   - 重新设计 `setup_tearing_mode`
   - 增大初始扰动幅度 `w_0`
   - 验证 rational surface 位置
   - 确保扰动形式正确（m-n mode coupling）

2. **数值稳定性诊断:**
   - 添加中间诊断：每一步检查 psi, omega 是否有 NaN/Inf
   - 降低时间步长 `dt`
   - 检查边界条件是否正确

3. **分步验证:**
   - **Step 1:** 先验证 Solovev equilibrium 是否正确（能量守恒、div(B)=0）
   - **Step 2:** 验证 tearing mode 扰动是否正确生成
   - **Step 3:** 验证无RMP时磁岛能否增长
   - **Step 4:** 再测试RMP控制效果

#### 后续工作（优先级P1）

4. **单元测试:**
   - `test_solovev_equilibrium()` - 验证平衡态
   - `test_tearing_mode_perturbation()` - 验证扰动生成
   - `test_island_width_calculation()` - 验证岛宽计算

5. **参考文献验证:**
   - 查阅经典tearing mode论文（Furth-Killeen-Rosenbluth 1963）
   - 对照文献中的初始条件设置
   - 验证我们的实现是否符合理论

---

## 不建议 ❌

**不要:**
- ❌ 继续增大网格（已验证无效）
- ❌ 增加演化时间（问题在初始条件，不在演化）
- ❌ 尝试其他validation tests（它们会有同样的问题）
- ❌ 带着这些bugs进入M2里程碑

**必须:**
- ✅ 先修复初始条件
- ✅ 建立可靠的物理验证基线
- ✅ 确保无RMP情况下磁岛能正常增长

---

## 交付物

**已完成:**
- ✅ `verify_large_grid.py` - 验证脚本
- ✅ `verify_large_grid.log` - 执行日志
- ✅ `LARGE_GRID_VERIFICATION.md` - 本报告

**Bug修复（已提交）:**
- ✅ `initial_conditions.py` - 修复 `setup_tearing_mode` 维度问题
- ✅ `validation.py` - 修复 monitor 调用参数

---

## M2提交建议

**当前状态:** 🚨 **不适合提交M2**

**原因:**
- Phase 4 validation tests 全部失败
- 失败原因是基础物理问题，不是网格问题
- 需要回到 Phase 1-2 重新验证基础物理

**建议里程碑:**
- **M1 (重做):** 修复初始条件，验证 Solovev equilibrium + tearing mode
- **M2 (当前):** Phase 4 validation tests 通过（至少1个test）
- **M3 (未来):** RL环境集成和基础训练

**时间线预估:**
- M1重做：需要深入调试和理论验证
- 不应设定具体时间（按SOUL.md原则，不预估时间）
- 当前状态：阻塞于初始条件问题，需要理论推导验证

---

**报告完成时间:** 2026-03-16 16:45 GMT+8  
**执行者:** 小P ⚛️ (Subagent)  
**审核:** 等待主agent和YZ确认
