# Inverse of Differential Operators - 学习笔记

**创建时间:** 2026-03-18  
**目标:** 理解如何从forward differential operator构建inverse solver (Poisson求解器)

---

## Part 1: 理论基础

### 1.1 Differential Operators as Linear Maps

**核心概念:**

- **Forward operator**: L: f → Lf
  - 例如: L = ∇² (Laplacian)
  - 作用: 给定函数 f，计算其微分结果 Lf
  - 性质: 线性算子 L(af + bg) = aL(f) + bL(g)

- **Inverse operator**: L⁻¹: g → f
  - 定义: L⁻¹g = f such that Lf = g
  - 物理意义: 已知源 g (如电荷密度)，求场 f (如电势)
  - 例子: ∇²φ = ρ → φ = ∇⁻²ρ

**Inverse存在性条件:**

1. **算子可逆性 (Operator Invertibility)**
   - L必须是满射和单射
   - 核空间 Ker(L) = {f: Lf = 0} 必须是 {0}
   - 对于Laplacian: 需要排除常数解 (∇²c = 0)

2. **边界条件的关键角色**
   - Dirichlet (φ=0 on boundary): 唯一确定解
   - Neumann (∂φ/∂n=0): 解差一个常数 (需要额外约束)
   - Periodic: 类似Neumann，需要约束平均值
   - **边界条件 = 选择唯一解的机制**

3. **Well-posedness**
   - Existence: 对所有 g ∈ range(L)，存在解 f
   - Uniqueness: 解唯一
   - Stability: 小的 δg 导致小的 δf (连续依赖性)

---

### 1.2 Matrix Representation (Discretization)

**从连续算子到离散矩阵:**

**步骤1: 网格化 (Gridding)**
- 连续域 Ω → 离散网格点 {x₁, x₂, ..., xₙ}
- 函数 f(x) → 向量 **f** = [f₁, f₂, ..., fₙ]ᵀ

**步骤2: Stencil 推导**
- 用Taylor展开近似导数
- 例如2阶中心差分:
  ```
  ∂²f/∂x² ≈ (f_{i+1} - 2f_i + f_{i-1}) / Δx²
  ```
- 1D Laplacian stencil: [-1, 2, -1] / Δx²

**步骤3: Sparse Matrix 构建**
- 每个网格点 i → 矩阵第 i 行
- Stencil coefficients → 矩阵元素
- 例如1D Laplacian (5个点，Dirichlet BC):
  ```
  L = (1/Δx²) * [ 2  -1   0   0   0]
                 [-1   2  -1   0   0]
                 [ 0  -1   2  -1   0]
                 [ 0   0  -1   2  -1]
                 [ 0   0   0  -1   2]
  ```

**步骤4: 应用边界条件**
- **Dirichlet (φ=0)**: 
  - 边界点不出现在矩阵中 (已知值)
  - 或者第一行/最后一行直接 = [1, 0, ..., 0]
- **Periodic**:
  - 第一行最后一列 = -1 (wrap around)
  - 最后一行第一列 = -1
- **Neumann (∂φ/∂n=0)**:
  - 用单边差分或ghost point
  - 导致matrix singular (需要约束)

**为什么边界条件改变matrix structure?**
- BC决定了stencil在边界的形式
- Dirichlet: 减少未知数 (矩阵更小)
- Periodic: 增加非对角元素 (wrap-around)
- Neumann: 改变边界行的系数 (ghost point消去)

---

### 1.3 Nested Differential Operators

**问题: ∇·(A∇f) where A = A(x)**

**挑战:**
- 内层: ∇f (gradient)
- 外层: ∇·(A·) (divergence with coefficient)
- A可能是tensor (各向异性)

**Discretization策略:**

**方法1: Product Rule展开**
```
∇·(A∇f) = A∇²f + ∇A·∇f
```
- 第一项: A在网格点上的值 × Laplacian stencil
- 第二项: ∇A 用差分 × ∇f 用差分
- 问题: 可能不保守 (不守恒)

**方法2: Finite Volume (Conservative)**
```
∫_V ∇·(A∇f) dV = ∮_S A∇f·n dS
```
- 在cell faces计算flux: A_{i+1/2} (f_{i+1}-f_i)/Δx
- 保证局部守恒
- A_{i+1/2} 用harmonic mean: 2/(1/A_i + 1/A_{i+1})

**Stencil for ∇·(A∇f) in 1D:**
```
[∇·(A∇f)]_i ≈ (1/Δx²) [A_{i+1/2}(f_{i+1}-f_i) - A_{i-1/2}(f_i-f_{i-1})]
            = (1/Δx²) [A_{i+1/2}f_{i+1} - (A_{i+1/2}+A_{i-1/2})f_i + A_{i-1/2}f_{i-1}]
```
- 系数position-dependent!
- Matrix每一行不同

**2D/3D扩展:**
- 每个方向分别处理
- 例如cylindrical: ∇·(A∇f) = (1/r)∂/∂r(rA∂f/∂r) + ...
- 需要metric factors (Jacobian)

---

## Part 2: Poisson Equation Specific

### 2.1 Standard Poisson: ∇²φ = ρ

**Cartesian Coordinates (简单)**

**1D:**
```
∂²φ/∂x² = ρ
Stencil: (φ_{i+1} - 2φ_i + φ_{i-1})/Δx² = ρ_i
```

**2D (5-point stencil):**
```
∂²φ/∂x² + ∂²φ/∂y² = ρ

     φ_{i,j+1}
        |
φ_{i-1,j} - φ_{i,j} - φ_{i+1,j}
        |
     φ_{i,j-1}

Coefficients: [0, 1, 0]
              [1,-4, 1] / Δx²  (assuming Δx=Δy)
              [0, 1, 0]
```

**3D (7-point stencil):**
- 中心: -6
- 6个邻居: +1 each

---

**Cylindrical Coordinates (r, θ, z)**

**Laplacian修正:**
```
∇²φ = (1/r)∂/∂r(r∂φ/∂r) + (1/r²)∂²φ/∂θ² + ∂²φ/∂z²
    = ∂²φ/∂r² + (1/r)∂φ/∂r + (1/r²)∂²φ/∂θ² + ∂²φ/∂z²
```

**关键: metric tensor corrections**
- 第一项: ∂²φ/∂r² (标准差分)
- 第二项: (1/r)∂φ/∂r (额外的1/r因子)
- 第三项: (1/r²)∂²φ/∂θ² (1/r²缩放)

**Stencil in r-direction:**
```
∂²φ/∂r² + (1/r)∂φ/∂r ≈ (φ_{i+1} - 2φ_i + φ_{i-1})/Δr² + (1/r_i)(φ_{i+1} - φ_{i-1})/(2Δr)
```
- 组合成非对称stencil: [(1-Δr/2r_i), -2, (1+Δr/2r_i)] / Δr²

**r=0特殊处理:**
- 用L'Hospital或regularity条件
- 通常: ∂φ/∂r|_{r=0} = 0

---

**Toroidal Coordinates (R, Z, φ)**

**Laplacian (假设轴对称 ∂/∂φ=0):**
```
∇²ψ = ∂²ψ/∂R² + ∂²ψ/∂Z² + (1/R)∂ψ/∂R
```
- 类似cylindrical
- 额外的(1/R)项 = metric correction

**Product rule处理:**
```
(1/R)∂ψ/∂R 在discretization时:
方法1: 直接差分 → (1/R_i)(ψ_{i+1,j} - ψ_{i-1,j})/(2ΔR)
方法2: Finite volume → 保证守恒
```

**Metric factors:**
- Jacobian J = R
- Volume element: dV = R dR dZ dφ
- 影响matrix构建

---

### 2.2 From laplacian_toroidal to Matrix

**Forward Operator (已有):**
```python
def laplacian_toroidal(f, dR, dZ, R):
    """
    Given f, compute ∇²f in toroidal geometry
    """
    # R-direction: second derivative + (1/R) first derivative
    d2f_dR2 = (f[i+1,j] - 2*f[i,j] + f[i-1,j]) / dR**2
    df_dR = (f[i+1,j] - f[i-1,j]) / (2*dR)
    
    # Z-direction: standard second derivative
    d2f_dZ2 = (f[i,j+1] - 2*f[i,j] + f[i,j-1]) / dZ**2
    
    # Combine
    laplacian = d2f_dR2 + (1/R[i]) * df_dR + d2f_dZ2
```

**Inverse Operator (构建Matrix L):**

**目标:** L·φ = ω (给定ω求φ)

**Step 1: Extract Stencil Coefficients**

对于点(i,j)，识别contributions:
```
Coefficient of φ[i-1,j]: c_L = 1/dR² - 1/(2*R[i]*dR)
Coefficient of φ[i,j]:   c_C = -2/dR² - 2/dZ²
Coefficient of φ[i+1,j]: c_R = 1/dR² + 1/(2*R[i]*dR)
Coefficient of φ[i,j-1]: c_D = 1/dZ²
Coefficient of φ[i,j+1]: c_U = 1/dZ²
```

**Step 2: Flatten to 1D indexing**
- 2D grid (NR, NZ) → 1D vector (NR*NZ,)
- Index mapping: k = i*NZ + j (row-major)
- 或: k = i + j*NR (column-major)

**Step 3: Fill Sparse Matrix**
```python
from scipy.sparse import lil_matrix

N = NR * NZ
L = lil_matrix((N, N))

for i in range(NR):
    for j in range(NZ):
        k = i*NZ + j  # current point
        
        # Center
        L[k, k] = -2/dR**2 - 2/dZ**2
        
        # R-neighbors (if not boundary)
        if i > 0:
            k_left = (i-1)*NZ + j
            L[k, k_left] = 1/dR**2 - 1/(2*R[i]*dR)
        if i < NR-1:
            k_right = (i+1)*NZ + j
            L[k, k_right] = 1/dR**2 + 1/(2*R[i]*dR)
        
        # Z-neighbors (if not boundary)
        if j > 0:
            k_down = i*NZ + (j-1)
            L[k, k_down] = 1/dZ**2
        if j < NZ-1:
            k_up = i*NZ + (j+1)
            L[k, k_up] = 1/dZ**2

L = L.tocsr()  # convert to efficient format
```

**Step 4: Solve Linear System**
```python
from scipy.sparse.linalg import spsolve

phi = spsolve(L, omega.flatten())
phi = phi.reshape((NR, NZ))
```

**关键点:**
- Stencil coefficients directly from forward code
- Position-dependent coefficients (R[i])
- Sparse matrix storage (CSR/CSC)
- Boundary conditions handled separately

---

### 2.3 Boundary Condition Handling

**Dirichlet (φ=0 at boundary):**

**方法1: 消去边界点**
- 只对内部点建立方程
- 边界点值代入右端项
- Matrix size: (N_interior, N_interior)

**方法2: 显式约束**
- 边界点也在matrix中
- 边界行: L[k_boundary, k_boundary] = 1, 其他=0
- 右端: b[k_boundary] = 0
- Matrix size: (N_total, N_total)

**例子 (1D, 5 points, BC: φ[0]=φ[4]=0):**
```
方法1 (只内部点):
L = [ 2 -1  0]     φ = [φ[1]]     b = [ρ[1]]
    [-1  2 -1]         [φ[2]]         [ρ[2]]
    [ 0 -1  2]         [φ[3]]         [ρ[3]]

方法2 (包含边界):
L = [ 1  0  0  0  0]     φ = [φ[0]]     b = [  0  ]
    [-1  2 -1  0  0]         [φ[1]]         [ρ[1]]
    [ 0 -1  2 -1  0]         [φ[2]]         [ρ[2]]
    [ 0  0 -1  2 -1]         [φ[3]]         [ρ[3]]
    [ 0  0  0  0  1]         [φ[4]]         [  0  ]
```

---

**Periodic BC (φ[0] = φ[N-1]):**

**方法: Wrap-around connections**
```
L = [ 2 -1  0  0 -1]     (第一行: 左邻居是最后一个点)
    [-1  2 -1  0  0]
    [ 0 -1  2 -1  0]
    [ 0  0 -1  2 -1]
    [-1  0  0 -1  2]     (最后一行: 右邻居是第一个点)
```

**问题: Matrix singular!**
- Ker(L) = {常数}
- 需要额外约束: ∑φ_i = 0 或 φ[0] = 0

**解决:**
- Add constraint: 最后一行改为 [1, 1, 1, ..., 1], b[-1] = 0
- 或用pseudo-inverse/least-squares

---

**Neumann BC (∂φ/∂n = 0):**

**方法1: Ghost point**
```
∂φ/∂x|_{x=0} = 0 → φ[-1] = φ[1]
代入stencil: (φ[1] - 2φ[0] + φ[-1])/Δx² = (2φ[1] - 2φ[0])/Δx²
```
- 边界行stencil: [2, -2, 0, ...]

**方法2: One-sided difference**
```
∂φ/∂x|_{x=0} ≈ (-3φ[0] + 4φ[1] - φ[2])/(2Δx) = 0
→ -3φ[0] + 4φ[1] - φ[2] = 0
```
- 边界行: [-3, 4, -1, 0, ...]
- But: 这是约束条件，不是Poisson方程

**通常做法:**
- 内部点: 标准Laplacian
- 边界点: Neumann条件替换
- Matrix singular → 需要额外约束 (fix one point or add ∑φ=0)

---

## Part 3: 实践方法

### 3.1 Recipe for Extracting Stencil

**步骤1: 读Forward Operator代码**
- 找到每个导数的差分公式
- 例如: `d2f_dR2 = (f[i+1,j] - 2*f[i,j] + f[i-1,j]) / dR**2`

**步骤2: 识别每个Neighbor的Contribution**
- 列出所有涉及的grid points: (i-1,j), (i,j), (i+1,j), (i,j-1), (i,j+1)
- 记录每个点前的系数

**步骤3: 记录Coefficients (可能position-dependent)**
- 例如: `coef[i+1,j] = 1/dR² + 1/(2*R[i]*dR)`
- 注意: R[i]随i变化 → 每行不同

**步骤4: 构建Sparse Matrix**
- 用`scipy.sparse.lil_matrix`逐行填充
- 转换为CSR格式提高求解效率
- 应用边界条件

**伪代码:**
```python
# Step 1-3: Extract stencil
def get_stencil_coefficients(i, j, R, dR, dZ):
    c_center = -2/dR**2 - 2/dZ**2
    c_left = 1/dR**2 - 1/(2*R[i]*dR)
    c_right = 1/dR**2 + 1/(2*R[i]*dR)
    c_down = 1/dZ**2
    c_up = 1/dZ**2
    return c_center, c_left, c_right, c_down, c_up

# Step 4: Build matrix
L = lil_matrix((N, N))
for i, j in interior_points:
    k = index_2d_to_1d(i, j)
    c_center, c_left, c_right, c_down, c_up = get_stencil_coefficients(i, j, R, dR, dZ)
    
    L[k, k] = c_center
    L[k, index_2d_to_1d(i-1, j)] = c_left
    L[k, index_2d_to_1d(i+1, j)] = c_right
    L[k, index_2d_to_1d(i, j-1)] = c_down
    L[k, index_2d_to_1d(i, j+1)] = c_up

# Apply BC
apply_boundary_conditions(L, boundary_type)

# Solve
phi = spsolve(L, rhs)
```

---

### 3.2 Common Pitfalls

**1. Forgetting Jacobian/Metric Factors**
- ❌ 用Cartesian stencil在cylindrical/toroidal
- ✅ 包含(1/r)或(1/R)项
- **检查:** 手动验证一个简单解 (如φ=r²) 是否满足∇²φ的预期值

**2. Mixing Up Signs (Forward vs Inverse)**
- ❌ Forward: ∇²φ → Inverse也用∇²
- ✅ Inverse: 构建matrix使得 L·φ = ∇²φ
- **检查:** Forward和Inverse组合应该return identity: L⁻¹(L(φ)) = φ

**3. Boundary Conditions Inconsistent**
- ❌ Forward用periodic，Inverse用Dirichlet
- ✅ 两者必须完全一致
- **检查:** 边界的stencil修改在forward和inverse中对应

**4. Index Mapping Errors (2D→1D)**
- ❌ 混淆row-major和column-major
- ✅ 统一约定: k = i*NZ + j 或 k = i + j*NR
- **检查:** 打印小矩阵的neighbor connections，手动验证

**5. Sparse Matrix Format Confusion**
- ❌ 用dense matrix (内存爆炸)
- ❌ 用lil_matrix求解 (效率低)
- ✅ 构建用lil，求解前转CSR: `L.tocsr()`

**6. Singular Matrix (Periodic/Neumann BC)**
- ❌ 直接调用`spsolve` → LinAlgError
- ✅ 添加约束: fix one point or ∑φ=0
- **检查:** 用`np.linalg.matrix_rank(L.toarray())`验证full rank (对小矩阵)

**7. Grid Spacing Inconsistency**
- ❌ Forward用dR=0.1，Inverse用dR=0.2
- ✅ 完全相同的grid定义
- **检查:** 打印grid arrays，确认一致

**8. Not Testing with Known Solutions**
- ❌ 直接用于复杂问题
- ✅ 先测试manufactured solution: φ_exact → ρ = ∇²φ_exact → φ_numerical → compare
- **检查:** L2 norm of error < tolerance

---

## 总结 & Checklist

### 关键概念回顾

1. **线性算子 → 矩阵**: Differential operator discretization via stencils
2. **边界条件 → Matrix structure**: BC决定唯一性和matrix形式
3. **Nested operators**: Product rule或finite volume处理
4. **Metric corrections**: Cylindrical/toroidal需要Jacobian因子
5. **Forward → Inverse**: Extract stencil → Build matrix → Solve linear system

### Before Implementing Checklist

- [ ] Forward operator已验证 (与理论解对比)
- [ ] Stencil coefficients手动推导一遍
- [ ] 边界条件明确定义 (Dirichlet/Neumann/Periodic)
- [ ] Index mapping统一约定
- [ ] Metric factors (1/r, 1/R)正确包含
- [ ] Sparse matrix格式选择 (lil → CSR)
- [ ] Manufactured solution测试准备好
- [ ] Boundary处理与forward一致

### Debugging Checklist

- [ ] Print小矩阵 (如5×5) 检查structure
- [ ] 验证对称性 (如果理论上对称)
- [ ] Check matrix rank (是否singular)
- [ ] 用简单解测试 (如φ=x², φ=sin(πx))
- [ ] 对比forward(inverse(ρ)) 与 ρ
- [ ] 检查守恒性 (如果用finite volume)

---

**学习完成时间:** 2026-03-18  
**下一步:** 应用到toroidal Poisson solver实现  
**参考代码:** `laplacian_toroidal` → `poisson_solve_toroidal`
