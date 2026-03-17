"""
Verify the correct formula for B_poloidal from ψ.

References:
- Freidberg "Ideal MHD" (2014), Appendix B
- Goedbloed & Poedts "Principles of MHD" (2004), Ch. 4
"""

import numpy as np

print("=" * 70)
print("TOROIDAL MHD: B from ψ")
print("=" * 70)

print("""
COORDINATE SYSTEM:
------------------
Toroidal coordinates (r, θ, φ):
  r:  minor radius (from magnetic axis)
  θ:  poloidal angle
  φ:  toroidal angle

Cylindrical coordinates (R, Z, φ):
  R = R₀ + r cos(θ)
  Z = r sin(θ)

Metric tensor (orthogonal):
  ds² = dr² + r² dθ² + R² dφ²
  
Jacobian:
  √g = r·R


MAGNETIC FIELD REPRESENTATION:
------------------------------
For axisymmetric equilibrium (∂/∂φ = 0):

  B = ∇ψ × ∇φ + F(ψ) ∇φ

where:
  ψ: poloidal flux function
  F(ψ): related to toroidal field (F = R·B_φ)


POLOIDAL FIELD (ignoring toroidal part):
-----------------------------------------
  B_pol = ∇ψ × ∇φ

In toroidal coordinates:
  ∇ψ = ∂ψ/∂r ê_r + (1/r) ∂ψ/∂θ ê_θ  (covariant components)
  ∇φ = (1/R) ê_φ                      (covariant component)

Cross product (in orthogonal coords):
  ê_r × ê_θ = (1/√g) ê^φ = ê_φ/(r·R)
  ê_θ × ê_φ = (1/√g) ê^r = ê_r/(r·R)
  ê_φ × ê_r = (1/√g) ê^θ = ê_θ/(r·R)

Therefore:
  ∇ψ × ∇φ = (∂ψ/∂r ê_r + (1/r)∂ψ/∂θ ê_θ) × ((1/R) ê_φ)
          = (∂ψ/∂r)(1/R)(ê_r × ê_φ) + (1/r)(∂ψ/∂θ)(1/R)(ê_θ × ê_φ)
          = (∂ψ/∂r)(1/R)(-ê_θ/(r·R)) + (1/r)(∂ψ/∂θ)(1/R)(ê_r/(r·R))

Wait, this is getting messy. Let me use the standard result:

STANDARD FORMULA (from Freidberg):
-----------------------------------
In toroidal coords (r, θ, φ):

  B_r = (1/r) ∂ψ/∂θ
  
  B_θ = -(1/r) ∂ψ/∂r
  
  B_φ = F/R

Wait, this has a factor of 1/r in B_θ term!


Actually, let me check in CYLINDRICAL coords first, which is clearer.
""")

print("\n" + "=" * 70)
print("IN CYLINDRICAL COORDINATES (R, Z, φ):")
print("=" * 70)

print("""
For axisymmetric equilibrium in (R, Z, φ):

  B = ∇ψ × ∇φ + F∇φ

where ψ = ψ(R, Z).

Poloidal part:
  B_pol = ∇ψ × ∇φ

In cylindrical coords:
  ∇ψ = ∂ψ/∂R ê_R + ∂ψ/∂Z ê_Z
  ∇φ = (1/R) ê_φ

  ∇ψ × ∇φ = (∂ψ/∂R ê_R + ∂ψ/∂Z ê_Z) × ((1/R) ê_φ)
          = (1/R) ∂ψ/∂R (ê_R × ê_φ) + (1/R) ∂ψ/∂Z (ê_Z × ê_φ)
          = (1/R) ∂ψ/∂R (-ê_Z) + (1/R) ∂ψ/∂Z (ê_R)
          = (1/R) ∂ψ/∂Z ê_R - (1/R) ∂ψ/∂R ê_Z

Therefore in cylindrical coords:
  B_R = (1/R) ∂ψ/∂Z
  B_Z = -(1/R) ∂ψ/∂R
  B_φ = F/R


Now transform to toroidal coords:
  R = R₀ + r cos(θ)
  Z = r sin(θ)

  ∂ψ/∂R = (∂ψ/∂r)(∂r/∂R) + (∂ψ/∂θ)(∂θ/∂R)
  ∂ψ/∂Z = (∂ψ/∂r)(∂r/∂Z) + (∂ψ/∂θ)(∂θ/∂Z)

From inverse transformation:
  r = √[(R-R₀)² + Z²]
  θ = arctan(Z/(R-R₀))

  ∂r/∂R = (R-R₀)/r = cos(θ)
  ∂r/∂Z = Z/r = sin(θ)
  ∂θ/∂R = -Z/r² = -sin(θ)/r
  ∂θ/∂Z = (R-R₀)/r² = cos(θ)/r

Therefore:
  ∂ψ/∂R = ∂ψ/∂r cos(θ) - (∂ψ/∂θ)(sin(θ)/r)
  ∂ψ/∂Z = ∂ψ/∂r sin(θ) + (∂ψ/∂θ)(cos(θ)/r)

Substituting into B_R, B_Z:
  B_R = (1/R) ∂ψ/∂Z
      = (1/R)[∂ψ/∂r sin(θ) + (∂ψ/∂θ)(cos(θ)/r)]
  
  B_Z = -(1/R) ∂ψ/∂R
      = -(1/R)[∂ψ/∂r cos(θ) - (∂ψ/∂θ)(sin(θ)/r)]

Now transform B_R, B_Z to B_r, B_θ:
  B_r = B_R cos(θ) + B_Z sin(θ)
  B_θ = -B_R sin(θ) + B_Z cos(θ)

Substituting...
  B_r = (1/R)[∂ψ/∂r sin(θ) + (∂ψ/∂θ)(cos(θ)/r)] cos(θ)
      + (1/R)[-∂ψ/∂r cos(θ) + (∂ψ/∂θ)(sin(θ)/r)] sin(θ)
      = (1/R)[(∂ψ/∂θ)(cos²(θ)/r) + (∂ψ/∂θ)(sin²(θ)/r)]
      = (1/R)(∂ψ/∂θ)/r
      = (1/r)(1/R) ∂ψ/∂θ

  B_θ = -(1/R)[∂ψ/∂r sin(θ) + (∂ψ/∂θ)(cos(θ)/r)] sin(θ)
      + (1/R)[-∂ψ/∂r cos(θ) + (∂ψ/∂θ)(sin(θ)/r)] cos(θ)
      = (1/R)[-∂ψ/∂r (sin²(θ) + cos²(θ))]
      = -(1/R) ∂ψ/∂r

FINAL RESULT:
  B_r = (1/(r·R)) ∂ψ/∂θ
  B_θ = -(1/R) ∂ψ/∂r

NOT what we have in utils.py!
""")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("""
The correct formula for B_pol from ψ in toroidal coords is:

  B_r = (1/(r·R)) ∂ψ/∂θ  ← has R factor!
  
  B_θ = -(1/R) ∂ψ/∂r     ← has R factor!

where R = R₀ + r cos(θ).

Our current implementation in utils.py has:
  B_r = (1/r) ∂ψ/∂θ      ← WRONG! Missing R factor
  B_θ = -∂ψ/∂r            ← WRONG! Missing R factor

This is why ∇·B ≠ 0!
""")
