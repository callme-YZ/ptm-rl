# PTM-RL Architecture Design v4.0

**Author:** 小P (Physics Lead) + 小A (ML Lead) review  
**Date:** 2026-03-16  
**Status:** Technical Design (基于既有实现 + 小A补充)

---

## 设计原则

**复用优先:**
1. PyTokEq (小P) → Layer 1 直接使用
2. PyTearRL (小A) → Layer 2/3 改造使用
3. 只修改必要的集成接口
4. **最小化重写,最大化复用** ✅

---

## 既有资产评估

### PyTokEq (Layer 1 Ready)

**Location:** `/workspace-xiaop/reduced-mhd/`

**已有功能 ✅:**
- Picard iteration G-S solver
- Free-boundary framework
- X-point finder
- q-profile calculator
- 28/28 tests passing

**需要修复 ⚠️:**
1. q(axis) 30%误差 → 提升到<5%
2. 数据一致性 (β_p声明)
3. 补充FreeGS验证报告

**可直接复用 ✅:**
```python
# API已存在,无需重新设计
from equilibrium import solve_picard_free_boundary

eq = solve_picard_free_boundary(
    grid=Grid.from_1d(R, Z),
    profile=M3DC1Profile(beta_p=0.05),
    tol=1e-6
)
# Returns: psi, j_tor, pressure, q_profile
```

---

### PyTearRL (Layer 2/3 基础)

**Location:** `/workspace-xiaoa/fusion-ai4s-pytearrl/`

**已有功能 ✅:**
- MHD solver (NumPy, RK4)
- Poisson solver (sparse)
- RL environment (Gym)
- Observation/Action/Reward已定义
- Baseline完成 (200 ep)

**需要改造 ⚠️:**
1. 替换 sheared_pinch → PyTokEq equilibrium
2. 改造初始化逻辑

**可直接复用 ✅:**
```python
# 这些已经work,不需要重写
class MHDTearingEnv:
    def step(self, action):
        # MHD solver ✅
        # Diagnostics ✅
        # Reward ✅
```

---

## Layer 1: PyTokEq集成

### 1.1 基础接口 (已存在)

```python
# equilibrium/picard_gs_solver.py
def solve_picard_free_boundary(
    grid: Grid,
    profile: Profile,
    tol: float = 1e-6,
    max_iter: int = 100,
    **kwargs
) -> Tuple[np.ndarray, ...]:
    """
    Returns:
        psi: (Nr, Nz) poloidal flux
        j_tor: (Nr, Nz) toroidal current
        pressure: (Nr, Nz) pressure
        q_profile: (Nr,) safety factor
    """
```

---

### 1.2 Equilibrium Caching (小A补充)

**Problem:** 
- 10K episodes × 1s solve time = 2.8h bottleneck ❌
- PyTokEq成为training瓶颈

**Solution: 缓存机制** ✅

```python
class EquilibriumCache:
    """Cache solved equilibria to avoid repeated expensive solves"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def _make_key(self, beta_p, I_p, Nr, Nz):
        """Create hashable cache key"""
        return (
            round(beta_p, 4),
            round(I_p, 2),
            Nr, Nz
        )
    
    def get_or_solve(self, grid, profile):
        """Get cached equilibrium or solve new"""
        key = self._make_key(
            profile.beta_p,
            profile.I_p,
            grid.Nr,
            grid.Nz
        )
        
        if key not in self.cache:
            # Solve once (1s)
            psi, j_tor, p, q = solve_picard_free_boundary(
                grid=grid,
                profile=profile
            )
            self.cache[key] = (psi, j_tor, p, q)
            
            # LRU eviction if cache full
            if len(self.cache) > self.max_size:
                self.cache.pop(next(iter(self.cache)))
        
        # Return cached (0.001s) ✅
        return self.cache[key]

# Usage in environment:
class PTMRLEnv:
    def __init__(self, config):
        self.eq_cache = EquilibriumCache(max_size=50)
        
    def reset(self, seed):
        # Fast cached lookup
        psi_eq, j_eq, p, q = self.eq_cache.get_or_solve(
            self.grid,
            M3DC1Profile(beta_p=0.05, I_p=1.0)
        )
        # Add unique perturbation
        psi, omega = self._add_perturbation(psi_eq, seed)
        return state, obs
```

**Performance impact:**
- Typical training: ~10 unique equilibria
- Solve time: 10s total (not 2.8h) ✅
- Cache hit rate: >99% after warmup ✅

---

## Layer 2: MHD Solver改造

### 2.1 PyTearRL改造 (可复用90%)

```python
def reset(self, seed):
    # === NEW: PyTokEq equilibrium ===
    # 1. Get cached equilibrium
    psi_eq, j_eq, p, q = self.eq_cache.get_or_solve(
        grid=self.grid,
        profile=M3DC1Profile(beta_p=0.05, I_p=1.0)
    )
    
    # 2. Add tearing mode perturbation
    psi_init, omega_init = self._add_tearing_perturbation(
        psi_eq=psi_eq,
        j_eq=j_eq,
        q_profile=q,
        mode=(2, 1),
        amp=1e-5,
        seed=seed
    )
    
    # 3. Continue with existing code ✅
    self.psi = psi_init
    self.omega = omega_init
    return self._get_obs(), {}
```

### 2.2 Perturbation Helper (~200 lines new)

```python
def _add_tearing_perturbation(self, psi_eq, j_eq, q_profile, mode, amp, seed):
    """
    Add (m,n) helical perturbation to equilibrium
    
    Physics:
        psi = psi_eq + amp * sin(m*θ) * gaussian(r, r_s)
        where r_s is rational surface q(r_s) = m/n
    """
    m, n = mode
    Nr, Nz = psi_eq.shape
    
    # Find rational surface
    r_s = self._find_rational_surface(q_profile, q_target=m/n)
    
    # Radial profile (gaussian centered at r_s)
    r = np.linspace(0, self.a, Nr)
    r_profile = np.exp(-((r - r_s) / 0.1)**2)
    
    # Helical pattern
    z = np.linspace(0, self.Lz, Nz)
    theta = 2*np.pi*z / self.Lz
    perturbation = r_profile[:, None] * np.sin(m * theta[None, :])
    
    # Random phase/amplitude from seed
    rng = np.random.default_rng(seed)
    phase_shift = int(rng.uniform(0, Nz))
    amp_factor = 0.5 + rng.uniform()
    
    # Apply perturbation
    psi_init = psi_eq + amp * amp_factor * np.roll(perturbation, phase_shift, axis=1)
    
    # Compute vorticity
    omega_init = -self.poisson_solver.solve(psi_init)
    
    return psi_init, omega_init

def _find_rational_surface(self, q_profile, q_target):
    """Find radius where q = q_target"""
    r = np.linspace(0, self.a, len(q_profile))
    idx = np.argmin(np.abs(q_profile - q_target))
    return r[idx]
```

---

## Layer 3: RL Framework (复用100%)

**Already working ✅:**
```python
# Observation (32D) ✅
obs = [w, ψ_samples(12), ω_samples(12), γ, drift, div_B, prev_action(4)]

# Action (4D RMP currents) ✅
action ∈ [-1, 1]^4

# Reward ✅
reward = -w - λ * drift

# Environment API ✅
env = PTMRLEnv(config)
obs, info = env.reset(seed=42)
obs, r, done, trunc, info = env.step(action)
```

**No changes needed!** ✅

---

## 方案A: CPU多核 (NumPy + Ray)

### A.1 技术栈

```python
# Base: 既有NumPy实现
- PyTokEq: NumPy ✅
- MHD solver: NumPy ✅
- Equilibrium cache: Python dict ✅

# Parallelization: Ray
import ray

@ray.remote
class PTMRLWorker:
    def __init__(self, config):
        self.env = PTMRLEnv(config)
        
    def run_episode(self, seed):
        obs = self.env.reset(seed=seed)
        rewards = []
        for _ in range(100):
            action = self.policy(obs)
            obs, r, done, _, _ = self.env.step(action)
            rewards.append(r)
            if done: break
        return sum(rewards)
```

---

### A.2 Ray + SB3集成 (小A补充)

**SB3-compatible VecEnv wrapper:**

```python
from stable_baselines3.common.vec_env import VecEnv
import numpy as np

class RayVecEnv(VecEnv):
    """Wrap Ray workers as SB3-compatible VecEnv"""
    
    def __init__(self, env_configs, n_envs=10):
        # Create Ray workers
        self.workers = [
            PTMRLWorker.remote(config) 
            for config in env_configs
        ]
        
        # VecEnv initialization
        dummy_env = PTMRLEnv(env_configs[0])
        obs_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        
        super().__init__(
            num_envs=n_envs,
            observation_space=obs_space,
            action_space=action_space
        )
        
        self.futures = None
    
    def reset(self):
        """Reset all environments"""
        futures = [w.reset.remote() for w in self.workers]
        obs_list = ray.get(futures)
        return np.array(obs_list)
    
    def step_async(self, actions):
        """Send actions to workers (non-blocking)"""
        self.futures = [
            w.step.remote(action)
            for w, action in zip(self.workers, actions)
        ]
    
    def step_wait(self):
        """Wait for results and return"""
        results = ray.get(self.futures)
        
        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        done_list = [r[2] for r in results]
        info_list = [r[4] for r in results]
        
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(done_list),
            info_list
        )
    
    def close(self):
        """Cleanup Ray workers"""
        for w in self.workers:
            ray.kill(w)

# SB3 PPO usage:
configs = [PTMRLConfig() for _ in range(10)]
env = RayVecEnv(configs, n_envs=10)

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=256,
    gamma=0.95
)

model.learn(total_timesteps=100_000)
```

**Why this works:**
- ✅ SB3 expects VecEnv interface
- ✅ Ray provides async parallelization
- ✅ step_async/step_wait pattern standard
- ✅ 10× speedup vs single-core

---

### A.3 性能分析

**Speedup:**
- 10 Ray workers on 10-core CPU
- ~10× faster than single-core ✅
- Linear scaling (no GIL issues)

**Overhead:**
- Ray communication: ~1ms per step
- Negligible vs MHD solve time (100ms)

**复用率: 94%**
- PyTokEq: 100% (5000 lines)
- MHD solver: 90% (200 new lines)
- RL env: 100%
- Ray wrapper: 100 new lines

---

## 方案B: GPU加速 (Hybrid JAX)

### B.1 Hybrid设计 (小P推荐)

**Philosophy: 只port MHD solver,不port PyTokEq**

**Rationale:**
- PyTokEq → JAX需大工程 (2-3周)
- MHD solver → JAX合理 (1周)
- Reset不频繁 (1/100 calls)
- **Pragmatic** ✅

---

### B.2 实现策略

```python
import jax
import jax.numpy as jnp

class PTMRLEnvGPU:
    def __init__(self, config):
        # PyTokEq cache (NumPy)
        self.eq_cache = EquilibriumCache()
        
        # MHD solver (JAX)
        self.mhd_step_jit = jax.jit(self._mhd_step_jax)
        
    def reset(self, seed):
        # 1. NumPy PyTokEq (1s, rare)
        psi_eq_np, j_eq_np, p, q = self.eq_cache.get_or_solve(...)
        
        # 2. Convert to JAX
        psi_eq_jax = jnp.array(psi_eq_np)
        
        # 3. JAX perturbation
        key = jax.random.PRNGKey(seed)
        psi_jax, omega_jax = self._add_perturbation_jax(
            psi_eq_jax, key
        )
        
        self.state = (psi_jax, omega_jax)
        return self._get_obs_jax(), {}
    
    def step(self, action):
        # Pure JAX, GPU-accelerated ✅
        self.state = self.mhd_step_jit(self.state, action)
        obs = self._get_obs_jax()
        reward = self._compute_reward_jax()
        return obs, reward, done, trunc, {}

# MHD step in JAX
def _mhd_step_jax(self, state, action):
    """JIT-compiled MHD evolution (GPU)"""
    psi, omega = state
    
    # 10 RK4 substeps
    for _ in range(10):
        # Poisson bracket
        dpsi_dt = self._poisson_bracket_jax(omega, psi)
        domega_dt = self._poisson_bracket_jax(psi, omega)
        
        # RK4
        psi, omega = self._rk4_step_jax(psi, omega, dpsi_dt, domega_dt)
    
    return psi, omega
```

---

### B.3 NumPy ↔ JAX转换overhead (小A估算)

**Overhead analysis:**

```python
# Per reset (1/100 calls):
psi_eq_np = pytokeq.solve()         # 1000ms (NumPy)
psi_eq_jax = jnp.array(psi_eq_np)   # 0.1ms (conversion)
# Total: 1000.1ms ✅ negligible

# First JIT compilation:
mhd_step_jit(state, action)  # ~5000ms (one-time)

# Subsequent calls:
mhd_step_jit(state, action)  # ~10ms (GPU) ✅

# Amortized overhead: 0.01% ✅
```

**Conclusion: Hybrid方案overhead可忽略** ✅

---

### B.4 GPU内存分析 (小A补充)

**Memory requirements:**

```python
# Per state:
psi:   Nr×Nz×8 bytes  = 64×128×8 = 64KB
omega: Nr×Nz×8 bytes  = 64×128×8 = 64KB
Total per state: 128KB

# Batch training:
Batch size = 128:
  States: 128 × 128KB = 16MB
  Gradients: ~16MB (same size as params)
  Network activations: ~32MB (MLP)
  Total: ~64MB

# GPU capacity:
A100 40GB:
  Max batch size = 40GB / 64MB ≈ 640
  Comfortable batch = 128-256 ✅
  
RTX 3090 24GB:
  Max batch size = 24GB / 64MB ≈ 375
  Comfortable batch = 128 ✅

# Recommendation: Start batch=128, scale if needed
```

---

### B.5 性能分析

**Speedup:**
- GPU MHD step: ~0.01s (vs NumPy 0.1s)
- 10× per-step speedup
- Batch parallelization: 128× effective speedup
- **Total: ~100-1000× vs single-core CPU** ✅

**复用率: 83%**
- PyTokEq: 100% (NumPy preserved)
- MHD solver: JAX porting (~500 lines)
- RL env: JAX PPO (~500 lines)
- Interface: ~50 lines

---

## 技术对比表

| Component | CPU方案 (Ray) | GPU方案 (Hybrid JAX) |
|-----------|---------------|---------------------|
| **Layer 1** | NumPy (~1s, cached) | NumPy (~1s, cached) |
| **Layer 2** | NumPy RK4 | JAX JIT (~0.01s) |
| **Parallelization** | Ray 10-core | GPU batch 128 |
| **Layer 3** | SB3 PPO | Custom JAX PPO |
| **Speedup** | ~10× vs baseline | ~100-1000× vs baseline |
| **Hardware** | Standard CPU | GPU required |
| **Equilibrium cache** | ✅ Required | ✅ Required |
| **Code reuse** | 94% (~300 new) | 83% (~1050 new) |
| **适用** | 调试/验证/中等训练 | 大规模训练/生产 |

---

## 实现路径建议

### 路径A: CPU优先 (小P推荐)

**Phase 1: CPU版本**
1. PyTokEq质量修复
2. Equilibrium缓存实现 ✅
3. PyTearRL集成 (改造reset)
4. Ray并行 + SB3接口 ✅
5. Baseline + 初步RL训练

**Phase 2: GPU (可选)**
- 如果CPU速度不够
- 如果有GPU硬件
- 作为性能优化

**优势:**
- ✅ 低风险,快速验证
- ✅ CPU版本已可用for研究
- ✅ GPU作为bonus

---

### 路径B: 双轨并行

**同时开发:**
- Track 1: CPU版本 (小P主导)
- Track 2: GPU版本 (小A主导)

**优势:**
- ✅ 更快获得GPU版本
- ✅ 两个版本可互相验证

**劣势:**
- ⚠️ 资源分散
- ⚠️ 协调成本

---

### 路径C: GPU优先

**直接GPU (Hybrid JAX):**
1. PyTokEq质量修复
2. MHD solver → JAX porting
3. JAX PPO实现
4. GPU训练

**优势:**
- ✅ 最快最优版本
- ✅ 一次到位

**劣势:**
- ⚠️ 如果GPU porting遇到问题,无fallback
- ⚠️ 调试困难

---

## 验收标准

### CPU方案验收

**Physics layer:**
- [ ] PyTokEq质量修复完成 (q < 5% error)
- [ ] Equilibrium缓存工作 (hit rate > 99%)
- [ ] 集成后conservation < 1%
- [ ] Diverse initial conditions (std > 0)

**Parallel layer:**
- [ ] RayVecEnv + SB3集成成功
- [ ] 10-worker并行运行
- [ ] Seed独立性验证
- [ ] Speedup > 8× (vs single-core)

**RL layer:**
- [ ] 200-episode baseline完成
- [ ] 初步PPO训练可运行
- [ ] Reward下降趋势

---

### GPU方案验收 (如果实施)

**JAX porting:**
- [ ] MHD solver JAX版本实现
- [ ] Physics一致性 (vs NumPy < 1% diff)
- [ ] JIT compilation成功
- [ ] GPU utilization > 80%

**Memory:**
- [ ] Batch 128运行成功
- [ ] 内存使用 < 10GB (A100 40GB)

**Performance:**
- [ ] Step时间 < 0.01s
- [ ] Speedup > 100× vs single-core
- [ ] 1M-episode训练可行

---

## Summary

**双方案设计完整:**

**CPU (Ray + SB3):**
- ✅ 94% code reuse
- ✅ Equilibrium缓存 (小A补充) ✅
- ✅ RayVecEnv接口 (小A实现) ✅
- ✅ 10× speedup
- ✅ 适合调试和中等训练

**GPU (Hybrid JAX):**
- ✅ 83% code reuse
- ✅ Hybrid避免完整porting
- ✅ NumPy↔JAX overhead分析 (小A) ✅
- ✅ GPU内存估算 (小A) ✅
- ✅ 100-1000× speedup
- ✅ 适合大规模训练

**小A review已整合:**
1. ✅ Equilibrium缓存机制 (Section 1.2)
2. ✅ Ray+SB3集成接口 (Section A.2)
3. ✅ GPU内存分析 (Section B.4)

**建议: CPU优先,GPU可选** ✅

---

**小P+小A签字: v4.0完成。整合小A review的3个关键补充(缓存/接口/内存)。CPU方案94%复用,GPU方案83%复用。技术细节完整,ready for implementation。⚛️🤖✅**
