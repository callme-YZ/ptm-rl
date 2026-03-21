"""Profile v2.0 environment to find bottlenecks"""

import time
from mhd_elsasser_env import MHDElsasserEnv

print("=" * 60)
print("v2.0 Environment Performance Profile")
print("=" * 60 + "\n")

# Create env
print("1. Environment creation...")
t0 = time.time()
env = MHDElsasserEnv(
    grid_shape=(16, 32, 16),
    n_coils=4,
    max_episode_steps=100
)
t_create = time.time() - t0
print(f"   Time: {t_create:.2f}s\n")

# Reset
print("2. First reset (includes BOUT++ init)...")
t0 = time.time()
obs, info = env.reset()
t_reset1 = time.time() - t0
print(f"   Time: {t_reset1:.2f}s\n")

# Second reset
print("3. Second reset (should be faster if cached)...")
t0 = time.time()
obs, info = env.reset()
t_reset2 = time.time() - t0
print(f"   Time: {t_reset2:.2f}s\n")

# Steps
print("4. Episode steps (10 steps)...")
step_times = []
for i in range(10):
    t0 = time.time()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    t_step = time.time() - t0
    step_times.append(t_step)
    
    if i == 0:
        print(f"   First step: {t_step:.3f}s (may include JIT)")

avg_step = sum(step_times[1:]) / len(step_times[1:])  # Exclude first
print(f"   Average step (2-10): {avg_step:.3f}s")
print(f"   Total 10 steps: {sum(step_times):.2f}s\n")

# Summary
print("=" * 60)
print("Performance Summary")
print("=" * 60)
print(f"Environment creation: {t_create:.2f}s")
print(f"First reset:          {t_reset1:.2f}s")
print(f"Second reset:         {t_reset2:.2f}s")
print(f"Average step time:    {avg_step:.3f}s")
print(f"Steps per second:     {1/avg_step:.1f}")
print(f"\nEstimated 100-step episode: {t_reset2 + 100*avg_step:.1f}s")
print(f"Episodes per hour:          {3600/(t_reset2 + 100*avg_step):.0f}")

# Bottleneck analysis
print("\n" + "=" * 60)
print("Bottleneck Analysis")
print("=" * 60)
if t_reset1 > 1.0:
    print(f"⚠️  MAJOR: First reset slow ({t_reset1:.1f}s)")
    print("    → BOUT++ metric/field-aligned initialization")
    print("    → FIX: Cache these objects")
if t_reset2 > 0.5:
    print(f"⚠️  Subsequent reset slow ({t_reset2:.1f}s)")
    print("    → Ballooning IC recreation")
    print("    → FIX: Cache initial state template")
if avg_step > 0.1:
    print(f"⚠️  Steps slow ({avg_step:.2f}s)")
    print("    → Physics computation heavy")
    print("    → FIX: JAX JIT compilation, smaller grid")

print("\n✅ Profile complete")
