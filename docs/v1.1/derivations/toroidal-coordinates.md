# Toroidal Coordinate Transformation for MHD

**目标:** 推导 toroidal 坐标系下的 metric tensor、Christoffel symbols 和微分算子，为 symplectic MHD 提供几何基础。

---

## 1. Coordinate Definition

### 1.1 Cylindrical vs Toroidal Coordinates

**Cylindrical coordinates (r, φ, z):**
- 适合无限长圆柱等离子体
- 不适合 tokamak（有环向封闭性）

**Toroidal coordinates (R, φ, Z):**
- R: major radius (from tokamak center to flux surface)
- φ: toroidal angle (围绕主轴旋转)
- Z: vertical position

**坐标变换（toroidal → Cartesian）:**

$$
\begin{aligned}
x &= R \cos\varphi \\
y &= R \sin\varphi \\
z &= Z
\end{aligned}
$$

### 1.2 Jacobian

Jacobian 行列式衡量体积元素的变换：

$$
J = \frac{\partial(x,y,z)}{\partial(R,\varphi,Z)} = R
$$

**推导:**

$$
\begin{vmatrix}
\partial x/\partial R & \partial x/\partial\varphi & \partial x/\partial Z \\
\partial y/\partial R & \partial y/\partial\varphi & \partial y/\partial Z \\
\partial z/\partial R & \partial z/\partial\varphi & \partial z/\partial Z
\end{vmatrix}
=
\begin{vmatrix}
\cos\varphi & -R\sin\varphi & 0 \\
\sin\varphi &  R\cos\varphi & 0 \\
0          &  0            & 1
\end{vmatrix}
= R
$$

**体积元素:**

$$
dV = J \, dR \, d\varphi \, dZ = R \, dR \, d\varphi \, dZ
$$

### 1.3 Basis Vectors

**Covariant basis vectors:**

$$
\mathbf{e}_R = \frac{\partial \mathbf{r}}{\partial R} = \cos\varphi \, \hat{x} + \sin\varphi \, \hat{y}
$$

$$
\mathbf{e}_\varphi = \frac{\partial \mathbf{r}}{\partial \varphi} = -R\sin\varphi \, \hat{x} + R\cos\varphi \, \hat{y}
$$

$$
\mathbf{e}_Z = \frac{\partial \mathbf{r}}{\partial Z} = \hat{z}
$$

**长度:**

$$
|\mathbf{e}_R| = 1, \quad |\mathbf{e}_\varphi| = R, \quad |\mathbf{e}_Z| = 1
$$

**正交性验证:**

$$
\mathbf{e}_R \cdot \mathbf{e}_\varphi = 0, \quad \mathbf{e}_R \cdot \mathbf{e}_Z = 0, \quad \mathbf{e}_\varphi \cdot \mathbf{e}_Z = 0
$$

---

## 2. Metric Tensor

### 2.1 Definition

Metric tensor $g_{ij}$ 定义为：

$$
g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j
$$

### 2.2 Calculation

$$
g_{RR} = \mathbf{e}_R \cdot \mathbf{e}_R = 1
$$

$$
g_{\varphi\varphi} = \mathbf{e}_\varphi \cdot \mathbf{e}_\varphi = R^2
$$

$$
g_{ZZ} = \mathbf{e}_Z \cdot \mathbf{e}_Z = 1
$$

$$
g_{R\varphi} = g_{RZ} = g_{\varphi Z} = 0 \quad (\text{正交性})
$$

**Metric tensor matrix:**

$$
g_{ij} = 
\begin{pmatrix}
1 & 0 & 0 \\
0 & R^2 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

**Inverse metric tensor (contravariant):**

$$
g^{ij} = 
\begin{pmatrix}
1 & 0 & 0 \\
0 & 1/R^2 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

### 2.3 Line Element

$$
ds^2 = g_{ij} \, dx^i \, dx^j = dR^2 + R^2 \, d\varphi^2 + dZ^2
$$

**物理意义:** 在 toroidal 坐标系中，toroidal 方向的长度被 R 拉伸。

---

## 3. Christoffel Symbols

### 3.1 Definition

Christoffel symbols $\Gamma^\lambda_{\mu\nu}$ 描述坐标系弯曲，定义为：

$$
\Gamma^\lambda_{\mu\nu} = \frac{1}{2} g^{\lambda\sigma} \left( \frac{\partial g_{\sigma\mu}}{\partial x^\nu} + \frac{\partial g_{\sigma\nu}}{\partial x^\mu} - \frac{\partial g_{\mu\nu}}{\partial x^\sigma} \right)
$$

### 3.2 Non-zero Components

**计算关键偏导数:**

$$
\frac{\partial g_{\varphi\varphi}}{\partial R} = 2R, \quad \text{其他偏导为 0}
$$

**$\Gamma^\varphi_{R\varphi}$ (和 $\Gamma^\varphi_{\varphi R}$，对称):**

$$
\Gamma^\varphi_{R\varphi} = \frac{1}{2} g^{\varphi\varphi} \left( \frac{\partial g_{\varphi R}}{\partial \varphi} + \frac{\partial g_{\varphi\varphi}}{\partial R} - \frac{\partial g_{R\varphi}}{\partial \varphi} \right)
$$

$$
= \frac{1}{2} \cdot \frac{1}{R^2} \cdot 2R = \frac{1}{R}
$$

**$\Gamma^R_{\varphi\varphi}$:**

$$
\Gamma^R_{\varphi\varphi} = \frac{1}{2} g^{RR} \left( \frac{\partial g_{R\varphi}}{\partial \varphi} + \frac{\partial g_{R\varphi}}{\partial \varphi} - \frac{\partial g_{\varphi\varphi}}{\partial R} \right)
$$

$$
= \frac{1}{2} \cdot 1 \cdot (-2R) = -R
$$

**Summary:**

$$
\Gamma^\varphi_{R\varphi} = \Gamma^\varphi_{\varphi R} = \frac{1}{R}
$$

$$
\Gamma^R_{\varphi\varphi} = -R
$$

$$
\text{All other } \Gamma^\lambda_{\mu\nu} = 0
$$

### 3.3 Physical Interpretation

- **$\Gamma^\varphi_{R\varphi} = 1/R$**: 当沿 R 方向移动时，toroidal 角速度需要修正（因为 R 变化）
- **$\Gamma^R_{\varphi\varphi} = -R$**: 纯 toroidal 运动产生向心加速度（圆周运动）

---

## 4. Differential Operators

### 4.1 Gradient

**Contravariant form:**

$$
\nabla f = g^{ij} \frac{\partial f}{\partial x^j} \mathbf{e}_i
$$

**Components:**

$$
\nabla f = \frac{\partial f}{\partial R} \mathbf{e}_R + \frac{1}{R^2} \frac{\partial f}{\partial \varphi} \mathbf{e}_\varphi + \frac{\partial f}{\partial Z} \mathbf{e}_Z
$$

**Physical form (使用 normalized basis $\hat{\mathbf{e}}_\varphi = \mathbf{e}_\varphi/R$):**

$$
\nabla f = \frac{\partial f}{\partial R} \hat{\mathbf{R}} + \frac{1}{R} \frac{\partial f}{\partial \varphi} \hat{\boldsymbol{\varphi}} + \frac{\partial f}{\partial Z} \hat{\mathbf{Z}}
$$

### 4.2 Divergence

**General formula:**

$$
\nabla \cdot \mathbf{A} = \frac{1}{\sqrt{g}} \frac{\partial}{\partial x^i} \left( \sqrt{g} \, A^i \right)
$$

其中 $\sqrt{g} = \sqrt{\det g_{ij}} = R$。

**For contravariant components $(A^R, A^\varphi, A^Z)$:**

$$
\nabla \cdot \mathbf{A} = \frac{1}{R} \frac{\partial}{\partial R}(R A^R) + \frac{1}{R} \frac{\partial A^\varphi}{\partial \varphi} + \frac{\partial A^Z}{\partial Z}
$$

**Expanded:**

$$
\nabla \cdot \mathbf{A} = \frac{\partial A^R}{\partial R} + \frac{A^R}{R} + \frac{1}{R} \frac{\partial A^\varphi}{\partial \varphi} + \frac{\partial A^Z}{\partial Z}
$$

**Physical interpretation:** 第二项 $A^R/R$ 是 toroidal 几何的修正项（径向流的发散需考虑 R 的变化）。

### 4.3 Curl

**General formula (contravariant components):**

$$
(\nabla \times \mathbf{A})^i = \epsilon^{ijk} \frac{1}{\sqrt{g}} \frac{\partial A_k}{\partial x^j}
$$

其中 $\epsilon^{ijk}$ 是 Levi-Civita symbol，$A_k = g_{kl} A^l$ (covariant components)。

**Components:**

$$
(\nabla \times \mathbf{A})^R = \frac{1}{R} \frac{\partial A_Z}{\partial \varphi} - \frac{\partial A_\varphi}{\partial Z}
$$

$$
(\nabla \times \mathbf{A})^\varphi = \frac{\partial A_R}{\partial Z} - \frac{\partial A_Z}{\partial R}
$$

$$
(\nabla \times \mathbf{A})^Z = \frac{1}{R} \left( \frac{\partial (R A_\varphi)}{\partial R} - \frac{\partial A_R}{\partial \varphi} \right)
$$

**Note:** $A_\varphi = R^2 A^\varphi$ (因为 $g_{\varphi\varphi} = R^2$)。

### 4.4 Laplacian

**Scalar Laplacian:**

$$
\nabla^2 f = \nabla \cdot (\nabla f)
$$

**Calculation:**

$$
\nabla f = \left( \frac{\partial f}{\partial R}, \frac{1}{R^2} \frac{\partial f}{\partial \varphi}, \frac{\partial f}{\partial Z} \right)
$$

$$
\nabla^2 f = \frac{1}{R} \frac{\partial}{\partial R} \left( R \frac{\partial f}{\partial R} \right) + \frac{1}{R} \frac{\partial}{\partial \varphi} \left( \frac{1}{R^2} \frac{\partial f}{\partial \varphi} \right) + \frac{\partial^2 f}{\partial Z^2}
$$

**Expanded:**

$$
\nabla^2 f = \frac{\partial^2 f}{\partial R^2} + \frac{1}{R} \frac{\partial f}{\partial R} + \frac{1}{R^2} \frac{\partial^2 f}{\partial \varphi^2} + \frac{\partial^2 f}{\partial Z^2}
$$

**物理意义:** $1/R$ 修正项来自 toroidal 几何的曲率。

---

## 5. MHD Equations in Toroidal Coordinates

### 5.1 Continuity Equation

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
$$

$$
\frac{\partial \rho}{\partial t} + \frac{1}{R} \frac{\partial}{\partial R}(R \rho v^R) + \frac{1}{R} \frac{\partial(\rho v^\varphi)}{\partial \varphi} + \frac{\partial(\rho v^Z)}{\partial Z} = 0
$$

### 5.2 Momentum Equation (Simplified)

Navier-Stokes in toroidal coordinates includes Christoffel symbol corrections:

$$
\frac{D v^R}{Dt} = f^R + v^\varphi v^\varphi \Gamma^R_{\varphi\varphi} = f^R - \frac{(v^\varphi)^2}{R}
$$

$$
\frac{D v^\varphi}{Dt} = f^\varphi + 2 v^R v^\varphi \Gamma^\varphi_{R\varphi} = f^\varphi + \frac{2 v^R v^\varphi}{R}
$$

其中 $f^i$ 是 Lorentz force + pressure gradient。

**物理意义:**
- $-v^\varphi v^\varphi / R$: 向心加速度（toroidal 旋转）
- $2 v^R v^\varphi / R$: Coriolis-like 效应（径向运动影响 toroidal 速度）

### 5.3 Induction Equation

$$
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
$$

在 toroidal 坐标下，curl 和 Laplacian 需使用上述公式，特别是 $1/R$ 修正项。

---

## 6. Key Properties for Symplectic Integration

### 6.1 Metric Tensor Determinant

$$
\det(g_{ij}) = R^2
$$

$$
\sqrt{g} = R
$$

**重要性:** Volume-preserving (辛形式保持) 需要 $\sqrt{g}$ 的正确处理。

### 6.2 Christoffel Symbols and Symplectic Form

辛结构保持要求 Poisson bracket 在坐标变换下不变：

$$
\{f, g\} = \frac{\partial f}{\partial q^i} \frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i} \frac{\partial g}{\partial q^i}
$$

在 toroidal 坐标下，需验证：

$$
d\omega = d(p_i \, dq^i) = 0
$$

**关键:** Christoffel symbols 不改变辛形式（因为它们来自坐标变换，而辛形式是几何不变量）。

### 6.3 Hamiltonian Form

MHD Hamiltonian 在 toroidal 坐标下：

$$
H = \int \left( \frac{1}{2\rho} |\mathbf{p}|^2 + \frac{1}{2\mu_0} |\mathbf{B}|^2 \right) \sqrt{g} \, dR \, d\varphi \, dZ
$$

其中 $\sqrt{g} = R$ 确保体积积分正确。

---

## 7. Summary

**已推导:**

1. ✅ Metric tensor: $g_{ij} = \text{diag}(1, R^2, 1)$
2. ✅ Christoffel symbols: $\Gamma^\varphi_{R\varphi} = 1/R$, $\Gamma^R_{\varphi\varphi} = -R$
3. ✅ Differential operators: $\nabla$, $\nabla \cdot$, $\nabla \times$, $\nabla^2$
4. ✅ MHD equations in toroidal form (continuity, momentum, induction)

**关键修正项:**

- **Divergence:** $A^R/R$ (径向流修正)
- **Laplacian:** $\partial f/\partial R / R$ (曲率修正)
- **Momentum:** $-v^\varphi v^\varphi / R$ (向心力), $2v^R v^\varphi / R$ (Coriolis)

**下一步:** 结合 Hamiltonian formulation (derivations/symplectic-mhd-hamiltonian.md)，构造完整 symplectic integrator。

---

**文档大小:** 约 7.2 KB  
**LaTeX 公式:** ✅ 完整且可渲染
