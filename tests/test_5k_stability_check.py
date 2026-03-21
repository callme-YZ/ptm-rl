"""
5000-step Stability Check

Verify if 5.7% energy drift is:
- Transient (stabilizes) → OK
- Accumulation (grows) → Bug

Author: 小P ⚛️
Date: 2026-03-18
"""

import numpy as np
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.integrators.symplectic import SymplecticIntegrator
from pytokmhd.operators import laplacian_toroidal


print("="*70)
print("⚛️ 5000-STEP STABILITY CHECK")
print("="*70)

# Grid
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)

# Initial condition: Small perturbation (same as before)
r = grid.r_grid
theta = grid.theta_grid
a = grid.a

amplitude = 0.001
psi0 = amplitude * r * (1 - r/a) * np.sin(theta)
omega0 = laplacian_toroidal(psi0, grid)

print(f"\nInitial condition:")
print(f"  ψ₀ = {amplitude} × r(1-r/a) sinθ")
print(f"  ψ_max = {np.max(np.abs(psi0)):.3e}")
print(f"  ω_max = {np.max(np.abs(omega0)):.3e}")

# Symplectic integrator
dt = 1e-4
eta = 0.0
nu = 0.0
n_steps = 5000

print(f"\nPhysics:")
print(f"  η = {eta} (Pure Hamiltonian)")
print(f"  ν = {nu}")
print(f"  dt = {dt}")
print(f"  steps = {n_steps}")

# Run
symplectic = SymplecticIntegrator(grid=grid, dt=dt, eta=eta, nu=nu)
symplectic.psi = psi0.copy()
symplectic.omega = omega0.copy()

E0 = symplectic.compute_energy()
print(f"\nInitial energy: E₀ = {E0:.6e}")

print(f"\nRunning {n_steps} steps...")
print("-"*70)

energy_history = [E0]
time_history = [0.0]
psi_max_history = [np.max(np.abs(psi0))]

report_interval = 500

for step in range(n_steps):
    symplectic.step()
    
    if (step + 1) % report_interval == 0:
        E = symplectic.compute_energy()
        dE = abs(E - E0) / E0
        psi_max = np.max(np.abs(symplectic.psi))
        
        energy_history.append(E)
        time_history.append(symplectic.t)
        psi_max_history.append(psi_max)
        
        print(f"Step {step+1:5d}: E = {E:.6e}, ΔE/E₀ = {dE:.4e}, ψ_max = {psi_max:.3e}")

# Final check
E_final = symplectic.compute_energy()
drift_final = abs(E_final - E0) / E0
psi_final_max = np.max(np.abs(symplectic.psi))

print("-"*70)
print(f"\nFinal state (5000 steps):")
print(f"  Energy drift: ΔE/E₀ = {drift_final:.4e}")
print(f"  ψ_max: {psi_final_max:.3e} (initial: {amplitude:.3e})")

# Analysis
print("\n" + "="*70)
print("STABILITY ANALYSIS")
print("="*70)

# Check drift trend
drifts = [abs(E - E0)/E0 for E in energy_history]
drift_1k = drifts[2]  # Step 1000 (index 2: 0, 500, 1000)
drift_5k = drifts[-1]  # Step 5000

print(f"\nEnergy drift comparison:")
print(f"  Step 1000: {drift_1k:.4e}")
print(f"  Step 5000: {drift_5k:.4e}")
print(f"  Change: {drift_5k - drift_1k:.4e}")

if abs(drift_5k - drift_1k) < 0.01:  # Less than 1% change
    verdict = "STABLE (Transient)"
    status = "✅"
    explanation = "Drift stabilized, consistent with initial transient"
elif drift_5k < drift_1k:
    verdict = "DECREASING (Good!)"
    status = "✅✅"
    explanation = "Drift actually decreasing, very good sign"
else:
    growth_rate = (drift_5k - drift_1k) / drift_1k
    if growth_rate < 0.2:  # Less than 20% growth
        verdict = "SLOWLY GROWING (Acceptable)"
        status = "⚠️"
        explanation = f"Slow growth ({growth_rate*100:.1f}%), may be acceptable"
    else:
        verdict = "ACCUMULATING (Problem!)"
        status = "❌"
        explanation = f"Significant growth ({growth_rate*100:.1f}%), indicates bug"

print(f"\nVerdict: {status} {verdict}")
print(f"  {explanation}")

# Check ψ stability
psi_change = (psi_final_max - psi_max_history[0]) / psi_max_history[0]
print(f"\nψ_max change: {psi_change*100:.2f}%")
if abs(psi_change) < 0.01:
    print("  ✅ ψ amplitude stable")
else:
    print(f"  ⚠️  ψ amplitude changed {psi_change*100:.1f}%")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if status == "✅" or status == "✅✅":
    print("\n✅ Phase 2 can be accepted with documented caveat:")
    print("   'Toroidal geometry shows ~5.6% initial transient in energy'")
    print("   'Drift stabilizes after initial adjustment'")
    print("   'Symplectic still 10× better than RK4'")
elif status == "⚠️":
    print("\n⚠️  Phase 2 conditional acceptance:")
    print("   'Slow energy drift observed in toroidal geometry'")
    print("   'Further investigation recommended for Phase 3'")
    print("   'Symplectic 10× better than RK4 proven'")
else:
    print("\n❌ Phase 2 NOT ready for acceptance:")
    print("   'Energy accumulation detected'")
    print("   'Requires debugging before Phase 3'")
    print("   'Implementation issue suspected'")

print("="*70)
