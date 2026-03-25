#!/usr/bin/env python3
"""
Issue #21: Comprehensive Performance Profiling

Systematic profiling of MHD solver to identify optimization opportunities.

Phase 4 Day 1 Morning (3h)

Author: 小A 🤖
Date: 2026-03-25
Issue: #21
"""

import sys
sys.path.insert(0, 'src')

import time
import cProfile
import pstats
import io
import numpy as np
import jax
import jax.numpy as jnp
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pytokmhd.rl.classical_controllers import make_baseline_agent
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 80)
print("Issue #21: Comprehensive Performance Profiling")
print("=" * 80)
print()

# ==============================================================================
# Setup
# ==============================================================================

print("Setting up environment...")
env = make_hamiltonian_mhd_env(
    nr=32, ntheta=64, nz=8,
    dt=1e-4, max_steps=1000,
    eta=0.05, nu=1e-4,
    normalize_obs=False
)

psi, phi = create_tearing_ic(nr=32, ntheta=64)
env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env.current_step = 0
env.obs_computer.reset()
obs = env.obs_computer.compute_observation(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env._last_obs = obs
env._last_psi = jnp.array(psi, dtype=jnp.float32)
env._last_phi = jnp.array(phi, dtype=jnp.float32)
env.psi = jnp.array(psi, dtype=jnp.float32)
env.phi = jnp.array(phi, dtype=jnp.float32)

agent = make_baseline_agent('pid', env.action_space, Kp=5.0, Ki=0.5, Kd=0.01, target=0.0, dt=1e-4)

print("✅ Setup complete\n")

# Warm-up
print("Warming up (JIT compilation)...")
for _ in range(10):
    _ = env.step(agent.act(obs), compute_obs=False)
print("✅ Warm-up complete\n")

# ==============================================================================
# Profiling 1: Component Breakdown
# ==============================================================================

print("=" * 80)
print("Profiling 1: Component-level Timing Breakdown")
print("=" * 80)
print()

n_steps = 100

# Profile individual components
print("Measuring individual component times (100 steps each)...\n")

# 1. Physics step only (no observation)
print("1. Physics step (cached observation):")
times_physics = []
for _ in range(n_steps):
    start = time.perf_counter()
    obs, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=False)
    end = time.perf_counter()
    times_physics.append((end - start) * 1000)
times_physics = np.array(times_physics)
print(f"   Mean: {times_physics.mean():.2f} ms")
print(f"   P50:  {np.percentile(times_physics, 50):.2f} ms")
print(f"   P95:  {np.percentile(times_physics, 95):.2f} ms")
print()

# 2. Full observation (Poisson solve)
print("2. Full observation (with Poisson):")
times_obs = []
for _ in range(20):  # Fewer samples (slow)
    start = time.perf_counter()
    obs, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=True)
    end = time.perf_counter()
    times_obs.append((end - start) * 1000)
times_obs = np.array(times_obs)
print(f"   Mean: {times_obs.mean():.2f} ms")
print(f"   P50:  {np.percentile(times_obs, 50):.2f} ms")
print(f"   P95:  {np.percentile(times_obs, 95):.2f} ms")
print()

# 3. Policy inference
print("3. Policy inference (PID):")
times_policy = []
for _ in range(1000):
    start = time.perf_counter()
    _ = agent.act(env._last_obs)
    end = time.perf_counter()
    times_policy.append((end - start) * 1000)
times_policy = np.array(times_policy)
print(f"   Mean: {times_policy.mean():.3f} ms")
print(f"   P50:  {np.percentile(times_policy, 50):.3f} ms")
print(f"   P95:  {np.percentile(times_policy, 95):.3f} ms")
print()

# Summary
print("📊 Component Breakdown:")
print(f"   Physics step:      {times_physics.mean():.2f} ms  (bottleneck if >10 ms)")
print(f"   Full observation:  {times_obs.mean():.2f} ms  (Poisson overhead)")
print(f"   Policy inference:  {times_policy.mean():.3f} ms  (negligible)")
print(f"   Observation overhead: {times_obs.mean() - times_physics.mean():.2f} ms")
print()

# ==============================================================================
# Profiling 2: Python cProfile (CPU hotspots)
# ==============================================================================

print("=" * 80)
print("Profiling 2: Python cProfile (CPU Hotspots)")
print("=" * 80)
print()

print("Running cProfile on 50 physics steps...")

profiler = cProfile.Profile()
profiler.enable()

# Profile 50 steps
for _ in range(50):
    obs, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=False)

profiler.disable()

# Analyze results
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(30)  # Top 30 functions

profile_output = s.getvalue()
print()
print("Top 30 functions by cumulative time:")
print(profile_output)
print()

# Save detailed profile
with open('results/issue21_cprofile_detailed.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
print("✅ Detailed cProfile saved to: results/issue21_cprofile_detailed.txt")
print()

# ==============================================================================
# Profiling 3: Memory Usage
# ==============================================================================

print("=" * 80)
print("Profiling 3: Memory Usage Analysis")
print("=" * 80)
print()

import psutil
import os

process = psutil.Process(os.getpid())

# Measure memory before
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# Run 1000 steps
for _ in range(1000):
    obs, r, term, trunc, info = env.step(agent.act(env._last_obs), compute_obs=False)

# Measure memory after
mem_after = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before: {mem_before:.1f} MB")
print(f"Memory after:  {mem_after:.1f} MB")
print(f"Memory delta:  {mem_after - mem_before:.1f} MB")
print()

if abs(mem_after - mem_before) > 10:
    print("⚠️ Potential memory leak detected (>10 MB growth)")
else:
    print("✅ No significant memory leak")
print()

# ==============================================================================
# Profiling 4: JAX Compilation Overhead
# ==============================================================================

print("=" * 80)
print("Profiling 4: JAX Compilation Overhead")
print("=" * 80)
print()

print("Measuring first call (with compilation) vs subsequent calls...")

# Reset environment to trigger recompilation
env2 = make_hamiltonian_mhd_env(nr=32, ntheta=64, nz=8, dt=1e-4, max_steps=1000, eta=0.05, nu=1e-4, normalize_obs=False)
env2.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
env2.current_step = 0
env2._last_obs = obs
env2._last_psi = jnp.array(psi, dtype=jnp.float32)
env2._last_phi = jnp.array(phi, dtype=jnp.float32)

# First call (with compilation)
start = time.perf_counter()
obs2, r, term, trunc, info = env2.step(agent.act(env2._last_obs), compute_obs=False)
first_call_time = (time.perf_counter() - start) * 1000

# Subsequent calls (compiled)
subsequent_times = []
for _ in range(10):
    start = time.perf_counter()
    obs2, r, term, trunc, info = env2.step(agent.act(env2._last_obs), compute_obs=False)
    subsequent_times.append((time.perf_counter() - start) * 1000)

print(f"First call (with JIT):  {first_call_time:.2f} ms")
print(f"Subsequent (compiled):  {np.mean(subsequent_times):.2f} ms")
print(f"Compilation overhead:   {first_call_time - np.mean(subsequent_times):.2f} ms")
print()

# ==============================================================================
# Summary Report
# ==============================================================================

print("=" * 80)
print("SUMMARY: Performance Profiling Results")
print("=" * 80)
print()

print("📊 Key Findings:\n")

# 1. Bottleneck analysis
bottleneck = "Physics step" if times_physics.mean() > times_policy.mean() * 100 else "Policy"
print(f"1. Primary bottleneck: {bottleneck}")
print(f"   - Physics step: {times_physics.mean():.2f} ms")
print(f"   - Policy: {times_policy.mean():.3f} ms")
print()

# 2. Observation overhead
obs_overhead_pct = (times_obs.mean() - times_physics.mean()) / times_obs.mean() * 100
print(f"2. Observation overhead: {obs_overhead_pct:.1f}%")
print(f"   - Full step: {times_obs.mean():.2f} ms")
print(f"   - Cached step: {times_physics.mean():.2f} ms")
print()

# 3. Performance metrics
freq_cached = 1000.0 / times_physics.mean()
freq_full = 1000.0 / times_obs.mean()
print(f"3. Achievable frequencies:")
print(f"   - Cached obs: {freq_cached:.1f} Hz")
print(f"   - Full obs:   {freq_full:.1f} Hz")
print()

# 4. Optimization targets
print("4. Optimization targets (ranked by impact):")
if times_obs.mean() - times_physics.mean() > 100:
    print(f"   ⭐ #1: Poisson solver ({times_obs.mean() - times_physics.mean():.1f} ms overhead)")
if times_physics.mean() > 15:
    print(f"   ⭐ #2: Physics step ({times_physics.mean():.2f} ms, target <10 ms)")
if first_call_time > 1000:
    print(f"   • Compilation overhead ({first_call_time:.0f} ms first call)")
print()

# 5. Recommendations
print("5. Optimization recommendations:")
print()
if times_obs.mean() - times_physics.mean() > 100:
    print("   🎯 HIGH PRIORITY: Optimize Poisson solver")
    print("      - Current: {:.0f} ms".format(times_obs.mean() - times_physics.mean()))
    print("      - Options: Fast FFT, approximate solve, GPU")
    print()
if times_physics.mean() > 15:
    print("   🎯 MEDIUM PRIORITY: Optimize physics step")
    print("      - Current: {:.1f} ms".format(times_physics.mean()))
    print("      - Options: JAX JIT, GPU, reduce resolution")
    print()
if first_call_time - np.mean(subsequent_times) > 500:
    print("   💡 LOW PRIORITY: Reduce compilation overhead")
    print("      - Pre-compile critical paths")
    print("      - Cache compiled functions")
    print()

print("=" * 80)
print("Profiling Complete")
print("=" * 80)
print()
print("📁 Detailed results saved to:")
print("   - results/issue21_cprofile_detailed.txt")
print()
