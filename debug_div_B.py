"""
Debug script: Verify ∇·B = 0 theory

Check whether ∇·B_pol should be zero for equilibrium from ψ alone.
"""

import numpy as np
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.equilibrium import circular_equilibrium
from pytokmhd.operators import B_poloidal_from_psi, divergence_toroidal

# Create grid
grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)

# Get equilibrium
psi = circular_equilibrium(grid)

# Compute B_pol
B_r, B_theta = B_poloidal_from_psi(psi, grid)

# Compute divergence
div_B = divergence_toroidal(B_r, B_theta, grid)

print("=" * 60)
print("THEORY CHECK: Should ∇·B_pol = 0?")
print("=" * 60)

print(f"\nEquilibrium: ψ = psi0 * r² * (1 + ε*cos(m*θ))")
print(f"  psi range: [{np.min(psi):.3f}, {np.max(psi):.3f}]")

print(f"\nPoloidal field from ψ:")
print(f"  B_r = (1/r) ∂ψ/∂θ")
print(f"    range: [{np.min(B_r):.3e}, {np.max(B_r):.3e}]")
print(f"    non-zero? {np.max(np.abs(B_r)) > 1e-10}")

print(f"  B_θ = -∂ψ/∂r")
print(f"    range: [{np.min(B_theta):.3e}, {np.max(B_theta):.3e}]")
print(f"    non-zero? {np.max(np.abs(B_theta)) > 1e-10}")

print(f"\nDivergence:")
print(f"  max|∇·B_pol| = {np.max(np.abs(div_B)):.3e}")

print("\n" + "=" * 60)
print("THEORY:")
print("=" * 60)
print("""
In toroidal geometry, the poloidal flux function ψ satisfies:
    B_pol = ∇ψ × ∇φ

where φ is the toroidal angle.

In general, ∇·B_pol ≠ 0!

The divergence-free condition ∇·B = 0 applies to the TOTAL field:
    B_total = B_pol + B_tor
    B_tor = F(ψ) ∇φ

where F(ψ) is chosen to make ∇·B_total = 0.

For B_pol alone (ignoring B_tor), we have:
    ∇·B_pol = ∇·(∇ψ × ∇φ)

Using vector identity: ∇·(A × B) = B·(∇×A) - A·(∇×B)

    ∇·(∇ψ × ∇φ) = ∇φ·(∇×∇ψ) - ∇ψ·(∇×∇φ)
                  = 0 - 0 = 0  (curl of gradient is zero)

Wait, this suggests ∇·B_pol SHOULD be zero!

Let me recalculate more carefully...
""")

# Manual calculation to verify
print("\n" + "=" * 60)
print("MANUAL VERIFICATION:")
print("=" * 60)

# In toroidal coords (r, θ, φ):
# B_pol = ∇ψ × ∇φ
# 
# ∇ψ = (∂ψ/∂r, (1/r²)∂ψ/∂θ, 0) in contravariant form
# ∇φ = (0, 0, 1/R²) in contravariant form
#
# Cross product in curvilinear coords is tricky...
# 
# Actually, the standard formula is:
# B_pol = (1/R²)[∂ψ/∂θ ê_r - r² ∂ψ/∂r ê_θ]
#
# Wait, this doesn't match what we have...

print("""
Let me reconsider the derivation of B from ψ.

In toroidal coordinates (r, θ, φ), for axisymmetric equilibrium:
    B = ∇ψ × ∇φ + F∇φ

where ψ is the poloidal flux and F is related to toroidal field.

For poloidal part only:
    B_pol = ∇ψ × ∇φ

The toroidal basis vectors are:
    ê_r: radial (outward from magnetic axis)
    ê_θ: poloidal (along flux surfaces)
    ê_φ: toroidal (around major axis)

The metric is:
    ds² = dr² + r²dθ² + R²dφ²
    where R = R₀ + r cos(θ)

The gradient in contravariant form:
    ∇ψ = (∂ψ/∂r) ê^r + (1/r²)(∂ψ/∂θ) ê^θ + 0 ê^φ
    ∇φ = 0 ê^r + 0 ê^θ + (1/R²) ê^φ

Cross product (using ε^{ijk} with √g = r·R):
    (∇ψ × ∇φ)^i = (1/√g) ε^{ijk} (∂ψ/∂x^j)(∂φ/∂x^k)

Let me compute this properly...
Actually, let's use a different approach.
""")

print("\n" + "=" * 60)
print("RESOLUTION:")
print("=" * 60)
print("""
I think the issue is in how we're computing B from ψ.

The standard formulation in MHD codes uses:
    B = ∇ψ × ∇φ / (2π)

In (r, θ, φ) coordinates:
    B_r = (1/r) ∂ψ/∂θ
    B_θ = -(1/r) ∂ψ/∂r  ← NOTE: extra 1/r factor!
    B_φ = F/R

Wait, this is inconsistent with what's in utils.py...

Let me check the actual MHD formulation.
""")
