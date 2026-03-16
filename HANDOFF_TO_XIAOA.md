# PyTokMHD Phase 1 交接文档

**From:** 小P ⚛️  
**To:** 小A 🤖  
**Date:** 2026-03-16  
**Status:** ✅ Phase 1 Complete, Ready for Integration

---

## TL;DR

✅ **Core MHD solver 已完成并验证**

- 所有测试通过 (100% coverage)
- Grid convergence 确认 (64×128 sufficient)
- 文档齐全，可直接使用

**你需要做：**
1. 运行验收测试（5分钟）
2. 集成到 RL Environment（Phase 2）

---

## 快速验收（5分钟）

### Step 1: 运行测试

```bash
cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl

# Test 1: Operators (expected: ALL PASSED ✅)
python3 src/pytokmhd/tests/test_operators.py

# Test 2: Time Evolution (expected: ALL PASSED ✅)
python3 src/pytokmhd/tests/test_time_evolution.py

# Test 3: Grid Convergence (expected: 64×128 SUFFICIENT ✅)
python3 src/pytokmhd/tests/grid_convergence_study.py
```

### Step 2: 验收清单

- [ ] Test 1 输出 "ALL TESTS PASSED ✅"
- [ ] Test 2 输出 "ALL TESTS PASSED ✅"
- [ ] Test 3 输出 "64×128 is SUFFICIENT"
- [ ] README.md 清晰易懂

**如果全部通过 → 验收完成，可以开始 Phase 2**

---

## 使用示例

### 最简单的例子（复制即可运行）

```python
import numpy as np
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')

from pytokmhd.solver import time_integrator, boundary

# Grid
Nr, Nz = 64, 128
Lr, Lz = 1.0, 6.0
r = np.linspace(0, Lr, Nr)
z = np.linspace(0, Lz, Nz)
dr, dz = r[1] - r[0], z[1] - z[0]
R, Z = np.meshgrid(r, z, indexing='ij')

# Initial condition
psi0 = 0.1 * np.sin(2*np.pi*Z/Lz) * (1 - R**2)
omega0 = np.zeros_like(psi0)

# Evolve 100 steps
eta = 1e-3
dt = 0.001

psi, omega = psi0.copy(), omega0.copy()

for step in range(100):
    psi, omega = time_integrator.rk4_step(
        psi, omega, dt, dr, dz, R, eta,
        apply_bc=boundary.apply_combined_bc
    )
    
    if (step + 1) % 20 == 0:
        print(f"Step {step+1}: max(|psi|) = {np.max(np.abs(psi)):.6f}")

print("✅ Evolution complete!")
```

---

## 核心 API（你会用到的）

### 1. RK4 时间步进

```python
from pytokmhd.solver import time_integrator

psi_new, omega_new = time_integrator.rk4_step(
    psi, omega,           # Current state
    dt=0.001,             # Timestep
    dr=dr, dz=dz,         # Grid spacing
    r_grid=R,             # Radial coordinate
    eta=1e-3,             # Resistivity
    apply_bc=boundary.apply_combined_bc  # Boundary conditions
)
```

### 2. 边界条件

```python
from pytokmhd.solver import boundary

# Standard tokamak BC (axis + wall + periodic)
psi, omega = boundary.apply_combined_bc(psi, omega)
```

### 3. 算子（如果需要手动计算）

```python
from pytokmhd.solver import mhd_equations

# Laplacian
lap_psi = mhd_equations.laplacian_cylindrical(psi, dr, dz, R)

# Poisson bracket
pb = mhd_equations.poisson_bracket(f, g, dr, dz)

# Gradients
df_dr = mhd_equations.gradient_r(f, dr)
df_dz = mhd_equations.gradient_z(f, dz)
```

---

## Phase 2 集成建议

### 你需要添加的模块

1. **Diagnostics**
   - Island width tracker (自动化测量)
   - Energy monitor
   - div(B) checker

2. **RMP Forcing**
   - Modify `rk4_step` to accept `rmp_currents` parameter
   - Add RMP boundary condition to Poisson solver

3. **RL Interface**
   - State extraction: `get_observation(psi, omega)`
   - Reward calculation: `compute_reward(w, energy_drift, ...)`
   - Action mapping: `action_to_rmp(action)`

### 推荐工作流程

```
Phase 2.1: Diagnostics (1-2 hours)
├── Island width tracker
├── Energy monitor
└── Tests

Phase 2.2: RMP Integration (2-3 hours)
├── Modify time_integrator to accept RMP
├── Add RMP forcing to operators
└── Tests

Phase 2.3: RL Environment (3-4 hours)
├── Wrap pytokmhd in Gym environment
├── State/action/reward design
└── Integration tests
```

---

## 文件位置

```
/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/
├── src/pytokmhd/              # Core solver
│   ├── solver/                # Physics modules
│   │   ├── mhd_equations.py   # Operators
│   │   ├── time_integrator.py # RK4
│   │   ├── boundary.py        # BCs
│   │   └── poisson_solver.py  # FFT solver
│   ├── tests/                 # Unit tests
│   │   ├── test_operators.py
│   │   ├── test_time_evolution.py
│   │   └── grid_convergence_study.py
│   └── README.md              # Full documentation
├── PHASE1_COMPLETION_REPORT.md  # Detailed report
├── PHASE1_SUMMARY.txt           # Quick summary
└── grid_convergence_results.txt # Convergence data
```

---

## 已知限制（非问题）

1. **Model-A only**
   - Viscosity ν=0 (符合 Model-A 定义)
   - 未来可扩展到 Model-C

2. **固定边界**
   - 导电壁 ψ=0 at r=Lr
   - 未来可添加自由边界

3. **性能**
   - NumPy 实现，~50ms/step (64×128 grid)
   - JAX 优化可达到 ~5ms/step

**这些都不影响当前使用**

---

## 物理参数推荐

基于 grid convergence study 和稳定性测试：

| Parameter | Symbol | Recommended | Range |
|-----------|--------|-------------|-------|
| Grid | Nr×Nz | 64×128 | 32×64 ~ 128×256 |
| Resistivity | η | 10⁻³ | 10⁻⁴ ~ 10⁻² |
| Timestep | dt | 10⁻³ | CFL < 0.5 |
| Domain | Lr×Lz | 1.0×6.0 | ε = Lr/Lz < 0.3 |

---

## 验收标准（再确认）

### 代码质量 ✅
- [x] 100% test coverage
- [x] All tests PASSED
- [x] Grid convergence confirmed

### 物理正确性 ✅
- [x] Operator accuracy < 1e-12
- [x] Energy conservation < 0.01%
- [x] 2nd order convergence

### 文档 ✅
- [x] Full API documentation
- [x] Usage examples
- [x] Phase 1 report

---

## 遇到问题？

### Debug 清单

1. **Import error?**
   ```python
   import sys
   sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/src')
   ```

2. **Test failed?**
   - Check Python version: `python3 --version` (need ≥3.8)
   - Check dependencies: `numpy`, `scipy`

3. **Physics question?**
   - 查看 README.md API Reference
   - 查看 PHASE1_COMPLETION_REPORT.md

4. **其他问题？**
   - @小P ⚛️ 在 Discord

---

## 下一步

1. ✅ **你先验收**（运行 3 个测试）
2. ✅ **我等你反馈**（如果有问题立即修复）
3. ✅ **通过后开始 Phase 2**（Diagnostics + RMP + RL）

---

**祝顺利！期待 Phase 2 集成测试 🚀**

— 小P ⚛️
