# v2.0 Release收尾清单

**Date:** 2026-03-21 20:52  
**Goal:** 准备v2.0.0 GitHub Release  
**Coordinator:** YZ + ∞

---

## 当前状态

**Branch:** `feature/v2.0-elsasser`  
**Tag:** `v2.0.0-phase1` (已存在)  
**Working tree:** Clean ✅

**已完成:**
- Physics layer (PyTokEq + Morrison bracket) ✅
- Baseline RL training (+32.1%) ✅
- Physics validation (C1-C3) ✅
- Phase 1 tag pushed to GitHub ✅

---

## YZ需要决策 (4项Critical)

### **决策1: README策略** ⭐⭐⭐⭐⭐

**问题:** Root README.md还是v1.0内容

**Option A: v2.0主README** (∞推荐)
- Root README改为v2.0介绍
- v1.x移到docs/legacy/
- v2.0是production → 应该是主README

**Option B: Overview**
- Root README介绍整个项目
- v2.0作为"最新版本"
- 链接到experiments/v2.0/

**Option C: 保持v1.0**
- Root README不动
- v2.0只在experiments/v2.0/

**YZ选择:** [ ]

---

### **决策2: CHANGELOG** ⭐⭐⭐⭐⭐

**需要添加v2.0.0条目**

**草稿 (YZ批准/修改):**

```markdown
## [2.0.0] - 2026-03-21

### Physics Layer (Phase 1)

**Added:**
- Elsässer MHD formulation (z± = u ± B)
- Morrison bracket structure-preserving numerics
- PyTokEq Solovev equilibrium integration
- Physics validation suite (C1-C3)

**Physics Validation:**
- Growth rate γ=1.29 (GTC-consistent)
- Energy conservation 0.38% drift (92% better than v1.4)
- Stable 100-step episodes (+30% vs v1.4)

**RL Results (Baseline PPO):**
- +32.1% improvement (uncontrolled vs RL)
- Multi-objective training (island width + energy)
- 40 FPS throughput (8-core)

**Breaking Changes:**
- New environment: `MHDElsasserEnv`
- 113D observation space (vs 18D in v1.4)
- Requires PyTokEq

**Known Limitations:**
- 2D reduced MHD
- Single Solovev equilibrium
- Ablation study (Standard FD vs Morrison) in progress
```

**YZ批准:** [ ] YES  [ ] Modify: _________

---

### **决策3: Code位置** ⭐⭐⭐

**问题:** v2.0在experiments/是否OK?

**Option A: 保持experiments/v2.0/** (∞推荐)
- 原因: 与v1.x并行开发
- README说明清楚即可

**Option B: 移到src/pytokmhd/v2/**
- 更"正式"
- 但可能confusing

**YZ选择:** [ ]

---

### **决策4: Merge顺序** ⭐⭐⭐⭐

**Option A: develop → main**
```bash
git checkout develop
git merge feature/v2.0-elsasser --no-ff
git checkout main
git merge develop --no-ff
git tag v2.0.0
```

**Option B: 直接 feature → main**
```bash
git checkout main
git merge feature/v2.0-elsasser --no-ff
git tag v2.0.0
```

**YZ选择:** [ ]

---

## 可选决策 (不阻塞release)

### **决策5: CI/CD** ⭐⭐

**需要GitHub Actions吗?**
- [ ] YES → 小A设置pytest + validation
- [ ] NO → 手动验证已足够

### **决策6: Roadmap** ⭐

**需要写未来计划吗?**
- [ ] YES → 写v2.x roadmap (3D, EFIT, etc.)
- [ ] NO → 跳过

---

## 执行责任分工

### **YZ:**
- ✅ 决策上述4项Critical
- ✅ Review CHANGELOG草稿
- ✅ 批准release

### **∞:**
- ⏳ 根据YZ决策更新README/CHANGELOG
- ⏳ 准备Release Notes
- ⏳ 协调小A检查

### **小A:**
- ⏳ Review experiments/v2.0/README.md
- ⏳ 确认Quick Start可运行
- ⏳ 检查requirements

### **小P:**
- ⏸️ (Optional) Review physics描述

---

## Release Notes草稿 (GitHub)

**Title:** v2.0.0 - Elsässer MHD + Structure-Preserving RL

```markdown
## v2.0.0: Physics-Faithful MHD-RL Framework

**Major Release:** Complete rewrite with structure-preserving numerics and realistic equilibrium.

### Highlights

🔬 **Physics Validated:**
- Morrison bracket MHD (0.38% energy drift, 92% better than v1.4)
- PyTokEq Solovev equilibrium (β=0.17, realistic)
- Growth rate γ=1.29 (GTC-consistent)

🤖 **RL Baseline:**
- +32.1% island width suppression
- Multi-objective control
- 40 FPS training

📦 **Production-Ready:**
- Stable 100-step episodes
- Gymnasium-compatible
- Comprehensive validation

### Breaking Changes

- New `MHDElsasserEnv` API
- Requires PyTokEq
- 113D observation space

### Known Limitations

- 2D reduced MHD (3D planned)
- Single equilibrium (EFIT planned)
- Ablation study ongoing (paper)

### Installation

See `experiments/v2.0/README.md`

### Citation

```
YZ et al. (2024). Structure-Preserving Reinforcement Learning 
for Tokamak Tearing Mode Control. 
arXiv:XXXX.XXXXX (submitted to PPCF)
```

**Full changelog:** CHANGELOG.md
```

---

## 执行步骤 (YZ决策后)

**Step 1: 文档更新** (∞负责)
- [ ] 更新README (按YZ决策)
- [ ] 更新CHANGELOG
- [ ] Commit & push

**Step 2: 验证** (小A负责)
- [ ] Review experiments/v2.0/README
- [ ] 测试Quick Start可运行
- [ ] 检查requirements

**Step 3: Merge** (YZ/∞执行)
- [ ] 按YZ决策merge (develop or main)
- [ ] Tag v2.0.0
- [ ] Push to GitHub

**Step 4: GitHub Release** (∞协助)
- [ ] Draft release
- [ ] 粘贴Release Notes
- [ ] Publish

**Step 5: 验收** (YZ)
- [ ] 检查GitHub Release页面
- [ ] 检查README显示正确
- [ ] 批准完成

---

## 与论文解耦

**v2.0 Release:** 代码部分 → 可立即进行  
**论文工作:** Experiment 1 + 论文调整 → 独立进行

**论文可以cite:** v2.0.0 GitHub release

---

## 当前阻塞

**等待:** YZ决策4项Critical (README/CHANGELOG/Code位置/Merge顺序)

**准备好后:** 1-2步即可release ✅

---

**Last Updated:** 2026-03-21 20:52  
**Status:** Awaiting YZ decisions
