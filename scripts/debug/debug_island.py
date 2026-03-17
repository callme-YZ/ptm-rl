#!/usr/bin/env python3
"""Debug island width calculation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pytokmhd.diagnostics.magnetic_island import (
    extract_flux_surface, find_extrema, compute_island_width
)

# Create test case
def create_perturbed_solovev(Nr=64, Nz=64, delta=0.1, m=2):
    r = np.linspace(0.1, 1.0, Nr)
    z = np.linspace(-0.5, 0.5, Nz)
    
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    psi_0 = R**2 + 0.5 * Z**2
    theta = np.arctan2(Z, R)
    delta_psi = delta * R**m * np.cos(m * theta)
    
    psi = psi_0 + delta_psi
    
    q0 = 1.0
    q = q0 * (1 + r**2)
    
    return psi, r, z, q

# Test
delta = 0.1
psi, r, z, q = create_perturbed_solovev(delta=delta)

print("Grid shape:", psi.shape)
print("r range:", r.min(), "to", r.max())
print("z range:", z.min(), "to", z.max())
print("q range:", q.min(), "to", q.max())

# Find rational surface
from pytokmhd.diagnostics.rational_surface import find_rational_surface
r_s, acc = find_rational_surface(q, r, q_target=2.0)
print(f"\nRational surface: r_s = {r_s:.4f}")

# Extract flux at r_s
psi_theta, theta = extract_flux_surface(psi, r, z, r_s)
print(f"Flux values: min={psi_theta.min():.4f}, max={psi_theta.max():.4f}")
print(f"Flux variation: {psi_theta.max() - psi_theta.min():.6f}")

# Find extrema
extrema = find_extrema(psi_theta, theta)
print(f"\nO-points: {len(extrema['o_points'])}")
print(f"X-points: {len(extrema['x_points'])}")

if len(extrema['o_points']) > 0:
    print("O-point values:", [v for _, v in extrema['o_points']])
if len(extrema['x_points']) > 0:
    print("X-point values:", [v for _, v in extrema['x_points']])

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Poincare section
R, Z = np.meshgrid(r, z, indexing='ij')
ax1.contour(R, Z, psi, levels=20, colors='b')
theta_circ = np.linspace(0, 2*np.pi, 100)
ax1.plot(r_s * np.cos(theta_circ), r_s * np.sin(theta_circ), 'r--', linewidth=2)
ax1.set_xlabel('R')
ax1.set_ylabel('Z')
ax1.set_title('Poincare Section')
ax1.set_aspect('equal')

# Flux along surface
ax2.plot(theta, psi_theta, 'b-', linewidth=2)
if len(extrema['o_points']) > 0:
    o_idx = [i for i, _ in extrema['o_points']]
    o_vals = [v for _, v in extrema['o_points']]
    ax2.plot(theta[o_idx], o_vals, 'ro', markersize=10, label='O-points')
if len(extrema['x_points']) > 0:
    x_idx = [i for i, _ in extrema['x_points']]
    x_vals = [v for _, v in extrema['x_points']]
    ax2.plot(theta[x_idx], x_vals, 'gx', markersize=12, markeredgewidth=3, label='X-points')
ax2.set_xlabel('θ')
ax2.set_ylabel('ψ')
ax2.set_title('Flux along Rational Surface')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('/Users/yz/.openclaw/workspace-xiaoa/ptm-rl/debug_island.png', dpi=150)
print("\nPlot saved to debug_island.png")

# Compute width
w, r_s_out, phase = compute_island_width(psi, r, z, q, m=2, n=1)
print(f"\nIsland width: w = {w:.6f}")
print(f"Phase: {phase:.4f} rad")
