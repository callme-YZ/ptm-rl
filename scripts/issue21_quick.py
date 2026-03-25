#!/usr/bin/env python3
import sys; sys.path.insert(0, 'src')
import time, numpy as np, jax.numpy as jnp
from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pim_rl.physics.v2.tearing_ic import create_tearing_ic

print("=" * 70)
print("Issue #21: Quick Deep Analysis")
print("=" * 70)
print()

# Resolution scaling (32, 48, 64)
print("Resolution Scaling Test (30 steps, nr>=32 constraint):\n")
print("Nr  Nθ   Nz  | Points  | Time (ms) | Freq (Hz) | Scaling")
print("-" * 70)

for nr, ntheta, nz in [(32,64,8), (48,96,12), (64,128,16)]:
    env = make_hamiltonian_mhd_env(nr=nr, ntheta=ntheta, nz=nz, dt=1e-4, max_steps=100, eta=0.05, nu=1e-4, normalize_obs=False)
    psi, phi = create_tearing_ic(nr=nr, ntheta=ntheta)
    env.mhd_solver.initialize(jnp.array(psi, dtype=jnp.float32), jnp.array(phi, dtype=jnp.float32))
    
    for _ in range(10): env.step(np.array([1.0, 1.0]), compute_obs=False)  # warm-up
    
    times = []
    for _ in range(30):
        start = time.perf_counter()
        env.step(np.array([1.0, 1.0]), compute_obs=False)
        times.append((time.perf_counter() - start) * 1000)
    
    mean = np.mean(times)
    freq = 1000/mean
    N = nr*ntheta*nz
    scale = f"{mean/16.76:.2f}×" if nr==32 else f"{mean/(32*64*8)*(nr*ntheta*nz):.2f}×"
    print(f"{nr:2d}  {ntheta:3d}  {nz:2d} | {N:7d} | {mean:9.2f} | {freq:9.1f} | {scale}")

print()

# JAX JIT check
print("JAX & JIT Status:\n")
try:
    with open('src/pim_rl/physics/v2/complete_solver_v2.py') as f:
        code = f.read()
    has_jit = '@jax.jit' in code or '@jit' in code
    print(f"@jax.jit in complete_solver_v2.py: {'Yes ✅' if has_jit else 'No ❌ (NEEDS OPTIMIZATION!)'}")
    if not has_jit:
        print("   → Expected 2-5× speedup from adding JIT")
        print("   → This is PRIMARY optimization target for Issue #15")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 70)
print("Quick Analysis Complete")
print("=" * 70)
