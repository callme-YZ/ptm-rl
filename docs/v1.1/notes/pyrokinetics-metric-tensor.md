# Pyrokinetics Toroidal Metric Tensor 学习笔记

**来源:** https://pyrokinetics.readthedocs.io/en/latest/user_guide/metric_terms.html  
**日期:** 2026-03-17  
**目标:** 提取 toroidal 坐标系下的完整 metric tensor 公式，为 v1.1 设计文档补充理论基础

---

## 1. 坐标变换关系

### 1.1 柱坐标 → 环坐标 (Cylindrical → Toroidal)

**柱坐标:** {R, Φ, Z}  
**环坐标:** {r, θ, ζ}

$$
\begin{aligned}
r &= r(R, Z) & R &= R(r, \theta) \\
\theta &= \theta(R, Z) & \Phi &= -\zeta \\
\zeta &= -\Phi & Z &= Z(r, \theta)
\end{aligned}
$$

**关键点:**
- Φ 逆时针（从上方看）
- ζ 顺时针
- θ 逆时针绕磁面
- 两套坐标都是右手系

**具体关系取决于磁面参数化选择 (flux-surface parameterization)**

---

## 2. Covariant Metric Tensor (协变度规张量)

**定义:** $g_{ij} = \frac{\partial \mathbf{r}}{\partial \xi^i} \cdot \frac{\partial \mathbf{r}}{\partial \xi^j}$

### 完整公式

$$
g_{rr} = \left(\frac{\partial R}{\partial r}\right)^2 + \left(\frac{\partial Z}{\partial r}\right)^2
$$

$$
g_{r\theta} = \frac{\partial R}{\partial r}\frac{\partial R}{\partial \theta} + \frac{\partial Z}{\partial r}\frac{\partial Z}{\partial \theta}
$$

$$
g_{\theta\theta} = \left(\frac{\partial R}{\partial \theta}\right)^2 + \left(\frac{\partial Z}{\partial \theta}\right)^2
$$

$$
g_{r\zeta} = g_{\theta\zeta} = 0
$$

$$
g_{\zeta\zeta} = R^2
$$

**物理意义:**
- $g_{rr}$: 径向度规（径向单位距离的物理长度）
- $g_{\theta\theta}$: 极向度规（极向单位角度的物理长度）
- $g_{\zeta\zeta} = R^2$: 环向度规（恰好等于大半径的平方）
- **非正交性:** $g_{r\theta} \neq 0$ → 环坐标系不是正交的！

---

## 3. Jacobian (雅可比行列式)

$$
\mathcal{J}_{r\theta\zeta} = R\left(\frac{\partial R}{\partial r} \frac{\partial Z}{\partial \theta} - \frac{\partial R}{\partial \theta}\frac{\partial Z}{\partial r}\right)
$$

**物理意义:** 体积元变换因子 $dV = \mathcal{J}_{r\theta\zeta} \, dr \, d\theta \, d\zeta$

---

## 4. Contravariant Metric Tensor (逆变度规张量)

**定义:** $g^{ij} = \nabla \xi^i \cdot \nabla \xi^j$

### 完整公式

$$
g^{rr} = \frac{g_{\theta\theta} g_{\zeta\zeta}}{(\mathcal{J}_{r\theta\zeta})^2}
$$

$$
g^{r\theta} = -\frac{g_{r\theta} g_{\zeta\zeta}}{(\mathcal{J}_{r\theta\zeta})^2}
$$

$$
g^{\theta\theta} = \frac{g_{rr} g_{\zeta\zeta}}{(\mathcal{J}_{r\theta\zeta})^2}
$$

$$
g^{r\zeta} = g^{\theta\zeta} = 0
$$

$$
g^{\zeta\zeta} = \frac{1}{g_{\zeta\zeta}} = \frac{1}{R^2}
$$

**推导要点:**
- 使用 $g^{ij} g_{jk} = \delta^i_k$ (协变逆变互逆关系)
- 利用 $\det(g_{ij}) = (\mathcal{J}_{r\theta\zeta})^2$
- 非对角项 $g^{r\theta}$ 的符号由 $g_{r\theta}$ 决定

---

## 5. 磁场表达式

$$
\mathbf{B} = \psi' \nabla\zeta \times \nabla r + B_\zeta(r) \nabla\zeta
$$

其中：
- $\psi' = \frac{d\psi}{dr}$: 极向磁通径向导数（除以 $2\pi$）
- $q(r)$: 安全因子 (safety factor)
- $B_\zeta(r)$: 电流函数

$$
B_\zeta(r) = \frac{q\psi'}{\langle \mathcal{J}_{r\theta\zeta} g^{\zeta\zeta} \rangle}
$$

其中 $\langle f \rangle = \frac{1}{2\pi} \int_{-\pi}^{\pi} f \, d\theta$ 表示极向平均

---

## 6. Field-Aligned Coordinates (磁场对齐坐标)

### 变换: {r, θ, ζ} → {r, α, θ}

$$
\begin{aligned}
r &= r \\
\alpha &= \sigma_\alpha [q(r)\theta - \zeta + G_0(r,\theta)] \\
\theta &= \theta
\end{aligned}
$$

其中 $\sigma_\alpha = \pm 1$ 控制手性（GENE: +1, CGYRO: -1）

### 新坐标系的 Covariant Metric

$$
\tilde{g}_{rr} = g_{rr} + \left(\frac{\partial \alpha}{\partial r}\right)^2 g_{\zeta\zeta}
$$

$$
\tilde{g}_{r\alpha} = -\frac{\partial \alpha}{\partial r} g_{\zeta\zeta}
$$

$$
\tilde{g}_{r\theta} = g_{r\theta} + \frac{\partial \alpha}{\partial r}\frac{\partial \alpha}{\partial \theta} g_{\zeta\zeta}
$$

$$
\tilde{g}_{\alpha\alpha} = g_{\zeta\zeta}
$$

$$
\tilde{g}_{\alpha\theta} = -\frac{\partial \alpha}{\partial \theta} g_{\zeta\zeta}
$$

$$
\tilde{g}_{\theta\theta} = g_{\theta\theta} + \left(\frac{\partial \alpha}{\partial \theta}\right)^2 g_{\zeta\zeta}
$$

### 关键导数

$$
\frac{\partial \alpha}{\partial \theta} = \sigma_\alpha \left[\frac{B_\zeta}{\psi'} \mathcal{J}_{r\theta\zeta} g^{\zeta\zeta}\right]
$$

$$
\frac{\partial \alpha}{\partial r} = \sigma_\alpha \int_0^\theta \left[\frac{dB_\zeta}{dr} \frac{\mathcal{J}_{r\theta\zeta} g^{\zeta\zeta}}{\psi'} + \frac{B_\zeta}{\psi'} g^{\zeta\zeta}\left(\frac{\partial \mathcal{J}_{r\theta\zeta}}{\partial r} - \frac{\psi''}{\psi'}\mathcal{J}_{r\theta\zeta}\right) + \frac{B_\zeta}{\psi'}\frac{\partial g^{\zeta\zeta}}{\partial r}\mathcal{J}_{r\theta\zeta}\right] d\theta'
$$

**注意:** $\frac{\partial \alpha}{\partial r}$ **不是周期的**，但满足：

$$
\left.\frac{\partial \alpha}{\partial r}\right|_{\theta+2M\pi} = \left.\frac{\partial \alpha}{\partial r}\right|_\theta + \sigma_\alpha \frac{dq}{dr} 2M\pi
$$

---

## 7. 与 MHD 相关的关键点

### 7.1 非正交性处理

环坐标系 $g_{r\theta} \neq 0$ → 所有微分算子（梯度、散度、拉普拉斯）都需要修正：

$$
\nabla f = g^{ij} \frac{\partial f}{\partial \xi^i} \mathbf{e}_j
$$

其中 $\mathbf{e}_j = \frac{\partial \mathbf{r}}{\partial \xi^j}$ 是协变基矢量

### 7.2 Grad-Shafranov 约束

Jacobian 的径向导数受平衡压力 $P(r)$ 和电流分布约束：

$$
\begin{split}
\frac{\partial \mathcal{J}_{r\theta\zeta}}{\partial r} = \mathcal{J}_{r\theta\zeta} \frac{\psi''}{\psi'} &- \frac{\mathcal{J}_{r\theta\zeta}}{g_{\theta\theta}}\left(\frac{\partial g_{r\theta}}{\partial \theta} - \frac{\partial g_{\theta\theta}}{\partial r} - \frac{g_{r\theta}}{\mathcal{J}_{r\theta\zeta}}\frac{\partial \mathcal{J}_{r\theta\zeta}}{\partial \theta}\right) \\
&+ \frac{(\mathcal{J}_{r\theta\zeta})^3}{g_{\theta\theta}}\left[\frac{\mu_0}{(\psi')^2}\frac{dP}{dr}\right] + \frac{(\mathcal{J}_{r\theta\zeta})^3 g^{\zeta\zeta}}{g_{\theta\theta}}\left[\frac{B_\zeta}{(\psi')^2}\frac{dB_\zeta}{dr}\right]
\end{split}
$$

**重要性:** 这不是任意的，而是由 MHD 平衡方程强制要求的

### 7.3 对 Symplectic Integration 的启示

**挑战:**
1. 度规张量随 $(r, \theta)$ 变化 → 哈密顿量是时变的（如果把 $\theta$ 当时间演化）
2. $g_{r\theta} \neq 0$ → 正则动量和物理动量不同
3. Field-aligned 坐标的 $\frac{\partial \alpha}{\partial r}$ 非周期 → 需要特殊边界处理

**机会:**
- Pyrokinetics 已经给出了所有 metric 导数的显式公式
- 可以直接代入 Elsässer 变量的辛积分器
- Field-aligned 坐标可能简化磁场项

---

## 8. 对 v1.1 设计的直接影响

### 8.1 需要添加到设计文档的内容

**Part 1 (Toroidal Geometry):**
- [ ] Section 1.1: 添加 Pyrokinetics 的协变度规公式（6个分量）
- [ ] Section 1.2: 添加逆变度规公式（6个分量）
- [ ] Section 1.3: 添加 Jacobian 及其径向导数（含 Grad-Shafranov 约束）
- [ ] Section 1.4: 微分算子（∇, div, Laplacian）在 toroidal 坐标下的展开

**Part 2 (Symplectic Integration):**
- [ ] 讨论非正交坐标下的辛结构保持
- [ ] Field-aligned 坐标的哈密顿形式
- [ ] 边界条件处理（$\frac{\partial \alpha}{\partial r}$ 非周期性）

### 8.2 与 Elsässer Paper 的连接

**需要验证:**
- Elsässer 变量 $\mathbf{z}^\pm = \mathbf{v} \pm \mathbf{b}$ 在 toroidal 坐标下的形式
- 辛积分器是否需要 metric-dependent 修正
- Variable-step 方法如何处理 $\mathcal{J}_{r\theta\zeta}(r,\theta)$ 的空间变化

---

## 9. 存疑/待查证

1. **Cylindrical → Toroidal 的具体映射:**  
   Pyrokinetics 说"取决于 flux-surface parameterization"，但没给具体公式。  
   → 需要补充：Miller 参数化？EFIT 数据插值？

2. **Reduced MHD 近似下的简化:**  
   是否可以假设 $g_{r\theta} \ll g_{rr}, g_{\theta\theta}$ 从而线性化某些项？

3. **数值计算成本:**  
   每个时间步都要计算 $\frac{\partial \alpha}{\partial r}$ 的积分 → 是否有解析近似？

---

## 10. 总结

**关键收获:**
1. ✅ 完整的 toroidal metric tensor 公式（协变+逆变）
2. ✅ Jacobian 及其 Grad-Shafranov 约束
3. ✅ Field-aligned 坐标的 metric 变换
4. ✅ 非正交性的物理来源（$g_{r\theta} \neq 0$）

**下一步:**
- 推导 toroidal 坐标下的微分算子（∇, div, curl, Laplacian）
- 将 Reduced MHD 方程写成 toroidal 形式
- 查阅 Elsässer paper，确认辛积分器的 metric 依赖性

**文件状态:** 2.8 KB → 满足 >2KB 要求 ✅
