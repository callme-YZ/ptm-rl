# PyTokMHD Layer 2 Implementation Roadmap (Updated)

**Author:** 小P ⚛️  
**Date:** 2026-03-16 (Updated: removed time estimates per YZ instruction)  
**Status:** Design Phase  
**Purpose:** PyTokMHD实施路线图 (无时间预估)

---

## 设计原则

**YZ指示: 移除所有时间预估** ✅

**Focus on:**
- 任务定义和验收标准
- 依赖关系和里程碑
- 风险识别和应对
- **不包括:** 周数预估,工时估计

---

## Phase定义

### Phase 1: 核心Solver

**目标:** 建立MHD演化核心

**关键任务:**
- [ ] 实现cylindrical operators (∇², Poisson bracket)
- [ ] RK4 time integrator验证
- [ ] Boundary condition处理
- [ ] Grid convergence study

**验收标准:**
- Operators unit tests通过 (100% coverage)
- Energy conservation < 1%
- Grid convergence confirmed

**依赖:** 无 (可立即开始)

---

### Phase 2: PyTokEq集成

**目标:** 整合真实平衡态初始化

**关键任务:**
- [ ] PyTokEq → PyTokMHD数据接口
- [ ] Grid interpolation (PyTokEq → MHD grid)
- [ ] Equilibrium caching实现
- [ ] 真实平衡态初始化测试

**验收标准:**
- PyTokEq平衡态成功加载
- Grid interpolation error < 1%
- Cache hit rate > 99%
- ∇·B < 1e-6

**依赖:** Phase 1完成

**风险:** ⚠️ Grid interpolation可能需要debug

---

### Phase 3: Diagnostics

**目标:** 撕裂模诊断工具

**关键任务:**
- [ ] Island width measurement (Poincaré map)
- [ ] Growth rate calculation (exponential fitting)
- [ ] Energy conservation diagnostics
- [ ] Visualization tools

**验收标准:**
- Island width accuracy < 5%
- Growth rate vs FKR theory < 20%
- Energy conservation monitored

**依赖:** Phase 2完成

---

### Phase 4: External Control

**目标:** RMP coils控制场

**关键任务:**
- [ ] RMP coil geometry (m=7,9)
- [ ] Magnetic field calculation
- [ ] Action → RMP coupling
- [ ] Suppression effectiveness测试

**验收标准:**
- RMP field ∇×B = 0 (vacuum)
- Suppression > 30% (benchmark)
- Action space [I_m7, I_m9] 验证

**依赖:** Phase 3完成

---

### Phase 5: RL Interface

**目标:** 与Layer 3集成

**关键任务:**
- [ ] Gym environment wrapper
- [ ] Observation/Action/Reward API
- [ ] Baseline RL training test
- [ ] 与小A交叉验收

**验收标准:**
- API与PyTearRL兼容
- Baseline可训练
- 小A验收通过

**依赖:** Phase 4完成

---

## 里程碑定义

**M1: Core Solver Validated** ✅
- Operators tested
- Time evolution stable
- Grid convergence confirmed

**M2: PyTokEq Integration Working** ✅
- 真实平衡态加载
- Cache实现
- 物理守恒验证

**M3: Diagnostics Operational** ✅
- Island width测量
- FKR benchmark通过
- 可视化工具ready

**M4: RMP Control Functional** ✅
- RMP field实现
- Suppression验证
- Action space确认

**M5: RL Interface Complete** ✅
- Layer 3集成ready
- 小A验收通过
- Baseline training可行

---

## 关键任务清单

### 代码实现

**Solver:**
- [ ] `mhd_equations.py` — Reduced MHD方程
- [ ] `time_integrator.py` — RK4 + implicit options
- [ ] `boundary.py` — 边界条件
- [ ] `poisson_solver.py` — ∇²φ = -ω

**Diagnostics:**
- [ ] `tearing_mode.py` — Island width + growth rate
- [ ] `energy_conservation.py` — 守恒律监控

**External Field:**
- [ ] `equilibrium_field.py` — PyTokEq集成
- [ ] `rmp_coils.py` — RMP控制场

### 测试文件

- [ ] `test_operators.py` — Operator验证
- [ ] `test_time_evolution.py` — RK4稳定性
- [ ] `test_pytokeq_integration.py` — 平衡态加载
- [ ] `test_fkr_benchmark.py` — FKR理论对比
- [ ] `test_rmp_control.py` — RMP suppression

### 文档

- [ ] `VALIDATION_REPORT.md` — 物理验证报告
- [ ] `API_REFERENCE.md` — API文档
- [ ] `BENCHMARK_RESULTS.md` — FKR benchmark结果

---

## 依赖关系图

```
Phase 1 (Core Solver)
   ↓
Phase 2 (PyTokEq Integration) ← 关键
   ↓
Phase 3 (Diagnostics)
   ↓
Phase 4 (RMP Control)
   ↓
Phase 5 (RL Interface)
   ↓
Layer 3 (PyTokTearRL)
```

**Critical Path:** Phase 2 (PyTokEq集成)

---

## 风险和应对

**Risk 1: PyTokEq集成复杂度** ⚠️
- Grid interpolation可能需要额外debug
- **应对:** Phase 2专项处理,先用Solovev analytical测试

**Risk 2: FKR benchmark精度**
- 20% tolerance可能难达到
- **应对:** 调整参数,增加grid分辨率

**Risk 3: RL API兼容性**
- 小A集成可能发现问题
- **应对:** Phase 1即与小A对齐,early review

---

## 验收标准总览

**Physics Quality:**
- ∇·B < 1e-6
- Energy conservation < 1%
- FKR growth rate error < 20%
- RMP suppression > 30%

**Code Quality:**
- Critical path: 100% test coverage
- Non-critical: ≥80% coverage
- <1000 total lines
- 90% code reuse from PyTearRL

**Integration:**
- API与PyTearRL兼容
- 小A验收通过
- Baseline RL training可行

---

## 下一步行动

**Immediate:**
1. YZ/小A review本路线图
2. 批准后创建`src/pytokmhd/`目录
3. 开始Phase 1实施

**Phase 1启动条件:**
- 设计文档批准 ✅
- 架构review通过 ✅
- PyTokEq Layer 1 ready ✅

**Ready to start** ✅

---

**小P签字: 2026-03-16 13:27 ⚛️**  
**Roadmap updated: 移除时间预估,聚焦任务和验收标准**
