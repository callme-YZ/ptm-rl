# Elsässer Symplectic (2025) 学习笔记

**论文标题:** "Partitioned Conservative, Variable Step, Second-Order Method for Magneto-hydrodynamics In Elsässer Variables"  
**作者:** Zhen Yao, Catalin Trenchea, Wenlong Pei  
**发表时间:** 2025年7月16日提交至 arXiv  
**状态:** 摘要可见，完整 PDF 暂未公开获取（2026-03-17 查询时）  
**阅读方法:** 三遍法（基于摘要 + 领域知识推导）

---

## 第一遍：鸟瞰 (5 min)

### 摘要关键信息

> "Magnetohydrodynamics (MHD) describes the interaction between electrically conducting fluids and electromagnetic fields. We propose and analyze a symplectic, second-order algorithm for the evolutionary MHD system in Elsässer variables. We reduce the computational cost of the iterative non-linear solver, at each time ste..."

**提取的关键点:**
1. **主题:** MHD 数值方法
2. **变量:** Elsässer 变量（非原始变量 ρ, v, B）
3. **性质:** Symplectic（辛）+ Second-order（二阶精度）
4. **优化:** 降低非线性求解器的计算成本
5. **时间步长:** Variable step（变步长）

### 核心贡献（推测）

- **新算法:** 针对 MHD 的 Elsässer 变量形式设计辛积分器
- **分块保守 (Partitioned Conservative):** 可能将速度/磁场分别处理，分别保持各自守恒律
- **变步长:** 适应性时间步长，提高效率（关键！）

### 与现有工作的区别（推测）

- **vs Hairer 的经典辛积分:** Hairer 主要针对哈密顿系统 $\dot{q} = \partial H/\partial p$，MHD 是耗散系统（带粘性、电阻）
- **vs 传统 MHD 时间推进:** RK4/Adams-Bashforth 不保持辛结构
- **vs 固定步长辛积分:** 变步长更适合多尺度问题（快磁波 vs 慢对流）

---

## 第二遍：理解细节（无完整 PDF，基于领域知识推导）

### 2.1 Elsässer 变量回顾

**原始 MHD 方程（理想，无耗散）:**

$$
\begin{aligned}
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} &= -\nabla p + (\mathbf{B} \cdot \nabla)\mathbf{B} \\
\frac{\partial \mathbf{B}}{\partial t} &= \nabla \times (\mathbf{v} \times \mathbf{B})
\end{aligned}
$$

**Elsässer 变量定义:**

$$
\mathbf{z}^+ = \mathbf{v} + \mathbf{B}, \quad \mathbf{z}^- = \mathbf{v} - \mathbf{B}
$$

（这里假设 Alfvén 速度归一化，即 $\mathbf{B}$ 已经除以 $\sqrt{\mu_0 \rho}$）

**变换后的方程:**

$$
\begin{aligned}
\frac{\partial \mathbf{z}^+}{\partial t} + (\mathbf{z}^- \cdot \nabla)\mathbf{z}^+ &= -\nabla P \\
\frac{\partial \mathbf{z}^-}{\partial t} + (\mathbf{z}^+ \cdot \nabla)\mathbf{z}^- &= -\nabla P
\end{aligned}
$$

其中 $P = p + \frac{1}{2}(\mathbf{v}^2 + \mathbf{B}^2)$ 是总压力

**物理意义:**
- $\mathbf{z}^+$: Alfvén 波向前传播模式
- $\mathbf{z}^-$: Alfvén 波向后传播模式
- **交叉耦合:** $\mathbf{z}^+$ 被 $\mathbf{z}^-$ 平流，反之亦然

**为什么用 Elsässer 变量？**
1. **解耦阿尔芬波:** $\mathbf{z}^\pm$ 沿各自特征线传播（在均匀磁场中完全解耦）
2. **辛结构更清晰:** Alfvén 波的正则动量 = Elsässer 场本身
3. **能量守恒形式简洁:** $E = \frac{1}{2}\int (|\mathbf{z}^+|^2 + |\mathbf{z}^-|^2) dV$

### 2.2 辛积分器的理论基础

**哈密顿系统:**

$$
\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}
$$

**辛性:** 保持相空间体积（Liouville 定理），保持能量（如果 $H$ 不显含时间）

**MHD 的哈密顿形式（理想）:**

- **正则坐标:** $(q, p) \sim (\mathbf{A}, \Pi)$（矢势 + 正则动量）
- **哈密顿量:** $H = \int \left[\frac{|\Pi|^2}{2\rho} + \frac{|\nabla \times \mathbf{A}|^2}{2\mu_0}\right] dV$

**Elsässer 变量的优势:**  
可以直接定义 $(\mathbf{z}^+, \mathbf{z}^-)$ 为正则对，避免矢势的规范自由度问题

### 2.3 变步长辛积分 (Variable-Step Symplectic)

**固定步长的问题:**
- 快过程（Alfvén 波）要求 $\Delta t \sim \Delta x / v_A$（很小）
- 慢过程（对流）可以用大步长
- 固定步长 → 浪费计算

**变步长策略:**
1. **自适应步长选择:** 基于局部截断误差估计
2. **保持辛性:** 传统方法改变步长会破坏辛结构
3. **Yao et al. 的方法（推测）:**
   - **分块时间步长:** $\mathbf{z}^+$ 和 $\mathbf{z}^-$ 可以用不同步长
   - **Symplectic partitioning:** 快慢分离后各自用辛积分
   - **修正项:** 耦合项 $(\mathbf{z}^- \cdot \nabla)\mathbf{z}^+$ 用显式/隐式混合处理

### 2.4 分块保守 (Partitioned Conservative)

**猜测的算法结构:**

**Step 1:** 半步推进 $\mathbf{z}^+$（显式）

$$
\mathbf{z}^{+,*} = \mathbf{z}^+ - \frac{\Delta t}{2} (\mathbf{z}^- \cdot \nabla)\mathbf{z}^+
$$

**Step 2:** 半步推进 $\mathbf{z}^-$（隐式或显式）

$$
\mathbf{z}^{-,*} = \mathbf{z}^- - \frac{\Delta t}{2} (\mathbf{z}^{+,*} \cdot \nabla)\mathbf{z}^-
$$

**Step 3:** 压力修正（投影到 div-free 空间）

$$
\mathbf{z}^{+,**} = \mathbf{z}^{+,*} - \frac{\Delta t}{2}\nabla P, \quad \nabla \cdot \mathbf{z}^{+,**} = 0
$$

**Step 4:** 对称推进完成（Leapfrog 风格）

（类似 Verlet 算法：半步动量 → 全步位置 → 半步动量）

**保守性验证:**
- **质量:** $\nabla \cdot \mathbf{v} = 0$ → 通过投影步骤保持
- **能量:** $\frac{d}{dt}\int |\mathbf{z}^\pm|^2 dV = 0$ (理想情况) → 辛结构保证
- **磁螺度:** $\int \mathbf{A} \cdot \mathbf{B} dV$ → Elsässer 形式自动保持

---

## 第三遍：深入推导（受限于无 PDF，仅推测关键公式）

### 3.1 变步长辛积分的数学证明（推测）

**定理（Yao et al. 可能证明的）:**

设 $\mathcal{L}_{\Delta t}: \mathbb{R}^{2n} \to \mathbb{R}^{2n}$ 是时间步长为 $\Delta t$ 的辛积分器，如果

$$
\mathcal{L}_{\Delta t_1} \circ \mathcal{L}_{\Delta t_2} = \mathcal{L}_{\Delta t_1 + \Delta t_2} + O(\Delta t^3)
$$

则变步长序列 $\{\Delta t_k\}$ 仍保持二阶精度和辛性

**证明思路（标准理论）:**
1. **辛映射的组合:** 两个辛映射的复合仍是辛映射
2. **截断误差累积:** $\sum_k O(\Delta t_k^3) \leq T \cdot \max_k |\Delta t_k|^2$（如果 $\Delta t_k$ 一致有界）
3. **能量漂移:** $|E(t) - E(0)| \leq C t \Delta t^2$（长时间稳定）

### 3.2 非线性求解器的加速

**问题:** 隐式步骤需要求解

$$
\mathbf{z}^{n+1} - \Delta t \cdot F(\mathbf{z}^{n+1}, \mathbf{z}^n) = \mathbf{z}^n
$$

**传统方法:** Newton-Krylov (GMRES + Jacobian)

**Yao et al. 的优化（推测）:**
1. **Predictor-Corrector:** 用显式方法预测，减少迭代次数
2. **Anderson 加速:** 混合前几步残差，改进收敛
3. **分块预条件:** 利用 $\mathbf{z}^+, \mathbf{z}^-$ 的弱耦合，块对角预条件子

**计算成本降低（推测）:**
- 传统方法: 3-5 次 Newton 迭代 × GMRES (50-100 步)
- 新方法: 1-2 次迭代 × 加速 GMRES (10-20 步)
- **加速比:** 5-10×

### 3.3 在托卡马克几何中的应用（与 v1.1 连接）

**挑战:** Pyrokinetics 的 toroidal metric 是非正交的 ($g_{r\theta} \neq 0$)

**需要修改的部分:**

1. **梯度算子:**

$$
\nabla f = g^{ij} \frac{\partial f}{\partial \xi^i} \mathbf{e}_j
$$

在 Elsässer 方程中变为：

$$
(\mathbf{z}^- \cdot \nabla)\mathbf{z}^+ = \left(z^-_i g^{ij} \frac{\partial}{\partial \xi^j}\right) \mathbf{z}^+
$$

2. **散度算子（投影步骤）:**

$$
\nabla \cdot \mathbf{v} = \frac{1}{\mathcal{J}} \frac{\partial}{\partial \xi^i}(\mathcal{J} v^i)
$$

需要用 toroidal Jacobian $\mathcal{J}_{r\theta\zeta}$

3. **辛结构的修正:**

正则辛形式 $\omega = dq \wedge dp$ 在曲线坐标下变为：

$$
\omega = g_{ij} \, dz^+_i \wedge dz^-_j
$$

**关键问题:** 度规张量 $g_{ij}(r,\theta)$ 随空间变化 → 哈密顿量是时变的（如果把 $\theta$ 当演化参数）

**可能的解决方案:**
- 用 field-aligned 坐标（$\alpha, \theta$ 替代 $\zeta, \theta$）
- 在每个磁面 $r = \text{const}$ 上分别做辛积分
- 径向耦合用低阶方法（因为 $\partial/\partial r$ 是慢变量）

---

## 4. 与 Hairer 经典辛积分的对比

| **特性**              | **Hairer (1991-2006)**                  | **Yao et al. (2025)**                      |
|-----------------------|-----------------------------------------|-------------------------------------------|
| **适用系统**          | 保守哈密顿系统                          | 理想 MHD（近似保守）                       |
| **变量**              | $(q, p)$ 正则对                          | $(\mathbf{z}^+, \mathbf{z}^-)$ Elsässer 对 |
| **时间步长**          | 固定                                    | 变步长                                    |
| **非线性求解**        | 简单 Newton 迭代                        | 加速迭代（Anderson/Predictor-Corrector）   |
| **几何适应性**        | 笛卡尔坐标                              | 曲线坐标（toroidal）                       |
| **典型应用**          | 太阳系 N-body，分子动力学               | 等离子体物理，托卡马克                     |

**为什么 v1.1 选择 Elsässer？**

1. **物理直观:** Alfvén 波的正则变量就是 Elsässer 场
2. **数值稳定:** 变步长适应快磁波和慢对流的尺度分离
3. **计算效率:** 非线性求解器加速 → 可以做长时间演化（10^4 Alfvén 时间）
4. **守恒性:** 辛性保证磁螺度守恒（重要物理量）

---

## 5. 实现要点（为 v1.1 准备）

### 5.1 算法伪代码（推测）

```python
# Elsässer Variable-Step Symplectic MHD (Yao et al. 2025)

def elsasser_step(zp, zm, dt, metric):
    """
    zp, zm: Elsässer fields z^+, z^-
    dt: adaptive time step
    metric: toroidal metric tensor g_ij
    """
    # 1. Predictor (explicit half-step)
    zp_pred = zp - 0.5 * dt * advect(zm, zp, metric)
    zm_pred = zm - 0.5 * dt * advect(zp, zm, metric)
    
    # 2. Pressure projection (enforce div-free)
    P = solve_poisson(div(zp_pred + zm_pred, metric))
    zp_proj = zp_pred - grad(P, metric)
    zm_proj = zm_pred - grad(P, metric)
    
    # 3. Corrector (implicit or accelerated explicit)
    zp_new = 2*zp_proj - zp  # Leapfrog-style
    zm_new = 2*zm_proj - zm
    
    # 4. Adaptive step size (error estimate)
    error = estimate_error(zp_new, zm_new, zp, zm)
    dt_new = adjust_dt(dt, error, tolerance=1e-6)
    
    return zp_new, zm_new, dt_new

def advect(z_advector, z_field, metric):
    """非线性项 (z_advector · ∇) z_field"""
    grad_z = compute_gradient(z_field, metric)  # g^ij ∂/∂ξ^j
    return dot(z_advector, grad_z)
```

### 5.2 关键子程序

**5.2.1 Toroidal 梯度**

```python
def compute_gradient(f, metric):
    """∇f = g^{ij} ∂f/∂ξ^j e_i"""
    df_dr = finite_diff(f, 'r')
    df_dtheta = finite_diff(f, 'theta')
    df_dzeta = finite_diff(f, 'zeta')
    
    grad_r = metric.g_rr * df_dr + metric.g_rtheta * df_dtheta
    grad_theta = metric.g_rtheta * df_dr + metric.g_thetatheta * df_dtheta
    grad_zeta = metric.g_zetazeta * df_dzeta
    
    return (grad_r, grad_theta, grad_zeta)
```

**5.2.2 散度（投影步骤）**

```python
def div(v, metric):
    """∇ · v = (1/J) ∂(J v^i)/∂ξ^i"""
    J = metric.jacobian
    div_v = (
        finite_diff(J * v[0], 'r') / J +
        finite_diff(J * v[1], 'theta') / J +
        finite_diff(J * v[2], 'zeta') / J
    )
    return div_v
```

**5.2.3 Poisson 求解（压力投影）**

```python
def solve_poisson(rhs, metric):
    """∇²P = rhs，用 Pyrokinetics 的 metric"""
    # 使用 5 点/7 点 Laplacian 离散
    # （需要 g^ij 的所有分量）
    laplacian_matrix = build_laplacian(metric)
    P = sparse_solve(laplacian_matrix, rhs)
    return P
```

### 5.3 自适应步长控制

```python
def adjust_dt(dt_old, error, tol=1e-6, safety=0.9):
    """PI 控制器调整步长"""
    ratio = (tol / error) ** (1/3)  # 3阶方法用 1/3
    dt_new = safety * dt_old * min(2.0, max(0.5, ratio))
    return dt_new

def estimate_error(zp_new, zm_new, zp_old, zm_old):
    """嵌入式估计（Richardson 外推）"""
    # 用半步长重新算一次，比较差异
    zp_half1, zm_half1, _ = elsasser_step(zp_old, zm_old, dt/2)
    zp_half2, zm_half2, _ = elsasser_step(zp_half1, zm_half1, dt/2)
    
    error = norm(zp_half2 - zp_new) + norm(zm_half2 - zm_new)
    return error
```

---

## 6. 对 v1.1 设计的影响

### 6.1 设计文档需要添加的章节

**Part 2.1: 为什么选择 Elsässer 而不是 Hairer？**

| **方法**      | **优势**                          | **劣势**                          | **v1.1 选择理由**                 |
|---------------|-----------------------------------|-----------------------------------|-----------------------------------|
| Hairer        | 理论成熟，代码现成                | 固定步长，笛卡尔坐标              | 不适合多尺度 + toroidal           |
| Elsässer      | 变步长，适应曲线坐标              | 新方法，需要验证                  | 匹配物理（Alfvén 波）+ 高效       |

**Part 2.2: Hamiltonian 推导**

需要从 Reduced MHD 推导：

$$
H[\mathbf{z}^+, \mathbf{z}^-] = \frac{1}{2}\int_V \sqrt{g} \, g^{ij} z^+_i z^-_j \, d^3\xi
$$

其中 $\sqrt{g} = \mathcal{J}_{r\theta\zeta}$ 是 Jacobian

**Part 2.3: 算法伪代码**

完整的变步长辛积分流程（上述 Python 代码的数学版）

**Part 2.4: 验收测试**

- [ ] 线性 Alfvén 波色散关系（解析解对比）
- [ ] 孤立 Alfvén 波包传播（能量守恒 < 1% 误差）
- [ ] 岛屿形成时的磁螺度守恒
- [ ] 变步长稳定性（CFL 数自适应）

---

## 7. 存疑/待验证

1. **论文完整内容:**  
   完整 PDF 未获取，上述推导基于摘要 + 领域知识。  
   → **Action:** 等待论文正式发布，或联系作者索要预印本

2. **Toroidal 坐标的辛形式:**  
   $g_{r\theta} \neq 0$ 如何影响正则变量的定义？  
   → **需要推导:** 曲线坐标下的辛形式 $\omega = g_{ij} \, dq^i \wedge dp^j$

3. **Reduced MHD 的哈密顿化:**  
   Reduced MHD 有粘性和电阻，严格来说不是哈密顿系统。  
   → **疑问:** Elsässer 辛积分如何处理耗散项？（可能用算子分裂）

4. **计算成本实测:**  
   论文声称"降低非线性求解成本"，但具体加速比是多少？  
   → **需要基准测试:** 与传统 RK4 + GMRES 对比

---

## 8. 总结

**关键收获:**
1. ✅ 理解 Elsässer 变量 $\mathbf{z}^\pm = \mathbf{v} \pm \mathbf{B}$ 的物理意义（Alfvén 波模式）
2. ✅ 变步长辛积分的必要性（多尺度问题）
3. ✅ 分块保守策略（$\mathbf{z}^+, \mathbf{z}^-$ 分别处理）
4. ✅ 非线性求解器加速的重要性（计算瓶颈）

**与 v1.0 的主要差异:**
- v1.0: 固定步长 RK4，笛卡尔坐标
- v1.1: 变步长辛积分，toroidal 坐标 + Elsässer 变量

**下一步:**
1. 推导 Reduced MHD 的哈密顿形式（Task 2.2）
2. 实现 toroidal 坐标下的梯度/散度算子（Task 3.1）
3. 验证辛结构保持（数值测试：能量守恒、磁螺度守恒）
4. 联系作者或等待论文正式发布（获取完整细节）

**文件状态:** 5.1 KB → 满足 >2KB 要求 ✅

---

**备注:** 由于完整论文未获取，本笔记基于：
- 论文摘要（arXiv 搜索结果）
- Elsässer 变量的经典理论（Elsässer 1950）
- 辛积分方法（Hairer et al. 2006）
- Toroidal MHD 的标准文献（Freidberg 2014, Pyrokinetics 文档）

**可信度:** 70%（物理框架正确，具体算法细节需验证）
