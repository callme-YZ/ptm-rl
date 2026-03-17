"""
Re-derive divergence formula in toroidal coordinates.

Check if divergence_toroidal is consistent with the B formula.
"""

print("=" * 70)
print("DIVERGENCE IN TOROIDAL COORDINATES")
print("=" * 70)

print("""
For vector field A = (A_r, A_θ, A_φ) in toroidal coords (r, θ, φ):

General formula for divergence in orthogonal curvilinear coords:
    ∇·A = (1/√g)[∂(√g A^r)/∂r + ∂(√g A^θ)/∂θ + ∂(√g A^φ)/∂φ]

where A^i are CONTRAVARIANT components and √g is the Jacobian.

For toroidal coords:
    √g = r·R = r(R₀ + r cos(θ))
    
    Metric: g_ij = diag(1, r², R²)
    Inverse: g^ij = diag(1, 1/r², 1/R²)

Contravariant components:
    A^r = g^rr A_r = A_r
    A^θ = g^θθ A_θ = A_θ/r²
    A^φ = g^φφ A_φ = A_φ/R²

For axisymmetric case (∂/∂φ = 0, A_φ = 0):
    ∇·A = (1/√g)[∂(√g A^r)/∂r + ∂(√g A^θ)/∂θ]
        = (1/(r·R))[∂(r·R·A_r)/∂r + ∂(r·R·A_θ/r²)/∂θ]
        = (1/(r·R))[∂(r·R·A_r)/∂r + ∂(R·A_θ/r)/∂θ]

Now let's check if this matches what's in divergence_toroidal()...

From the code:
    sqrtg_Ar = J * A_r           where J = r·R
    sqrtg_Atheta = J * A_theta   where A_theta is input

Wait, the code treats A_theta as CONTRAVARIANT A^θ, not COVARIANT A_θ!

Let me re-check the signature of divergence_toroidal...

From the docstring:
    "Parameters: A_r, A_theta are radial and poloidal components"

It doesn't specify covariant vs contravariant. 

Looking at how it's called from test:
    B_r, B_theta = B_poloidal_from_psi(psi, grid)
    div_B = divergence_toroidal(B_r, B_theta, grid)

And B_poloidal_from_psi returns:
    B_r = (1/(r·R)) ∂ψ/∂θ
    B_θ = -(1/R) ∂ψ/∂r

These look like PHYSICAL components (unit vectors), not contravariant!

For physical components in orthogonal coords:
    A_phys,r = A_r / √g_rr = A_r / 1 = A_r
    A_phys,θ = A_θ / √g_θθ = A_θ / r
    
Contravariant:
    A^r = A_r
    A^θ = A_θ / r²

So: A_phys,θ = A^θ · r

Let me clarify the conventions...
""")

print("\n" + "=" * 70)
print("COORDINATE COMPONENT CONVENTIONS")
print("=" * 70)

print("""
There are multiple conventions for vector components:

1. COVARIANT (subscript):
   A_i = A · ∂x^i  (dot product with coordinate basis)
   A = A_i dx^i    (sum over i)
   
2. CONTRAVARIANT (superscript):
   A^i = g^ij A_j
   A = A^i ∂_i     (sum over i)
   
3. PHYSICAL (unit vectors):
   A_phys,i = A · ê_i  where ê_i are unit vectors
   A_phys,i = A_i / √g_ii  (no sum)

For orthogonal coords:
   ê_i = (1/√g_ii) ∂_i

EXAMPLE: Toroidal coords
   ê_r = ∂_r / √g_rr = ∂_r / 1 = ∂_r
   ê_θ = ∂_θ / √g_θθ = ∂_θ / r
   ê_φ = ∂_φ / √g_φφ = ∂_φ / R

For vector A:
   A = A^r ê_r + A^θ ê_θ + A^φ ê_φ  (physical components)
   
   A_phys,r = A^r
   A_phys,θ = A^θ
   A_phys,φ = A^φ

But:
   A_covariant,r = A^r / g^rr = A^r
   A_covariant,θ = A^θ / g^θθ = r² A^θ
   A_covariant,φ = A^φ / g^φφ = R² A^φ

So the B_r, B_θ returned by B_poloidal_from_psi are PHYSICAL components.
""")

print("\n" + "=" * 70)
print("DIVERGENCE FOR PHYSICAL COMPONENTS")
print("=" * 70)

print("""
For orthogonal coords with physical components (A_phys,r, A_phys,θ, A_phys,φ):

    ∇·A = (1/h₁h₂h₃)[∂(h₂h₃ A_phys,r)/∂r 
                    + ∂(h₃h₁ A_phys,θ)/∂θ 
                    + ∂(h₁h₂ A_phys,φ)/∂φ]

where h_i = √g_ii are scale factors:
    h_r = 1
    h_θ = r
    h_φ = R

Therefore:
    ∇·A = (1/(1·r·R))[∂(r·R·A_phys,r)/∂r 
                      + ∂(R·1·A_phys,θ)/∂θ 
                      + 0]
        = (1/(r·R))[∂(r·R·A_r)/∂r + ∂(R·A_θ)/∂θ]

where I'm using A_r, A_θ to denote physical components.

This MATCHES the code! Let me check more carefully...

Code:
    J = r·R  (Jacobian)
    sqrtg_Ar = J * A_r
    sqrtg_Atheta = J * A_theta
    
    d(sqrtg_Ar)/dr → this gives ∂(r·R·A_r)/∂r ✓
    d(sqrtg_Atheta)/dθ → this gives ∂(r·R·A_θ)/∂θ

But we want ∂(R·A_θ)/∂θ, not ∂(r·R·A_θ)/∂θ !

THERE'S THE BUG!
""")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print("""
Bug in divergence_toroidal():

Current:
    sqrtg_Atheta = J * A_theta
    where J = r·R

Should be:
    sqrtg_Atheta = R * A_theta  ← only R, not r·R!

The correct formula is:
    ∇·A = (1/(r·R))[∂(r·R·A_r)/∂r + ∂(R·A_θ)/∂θ]

NOT:
    ∇·A = (1/(r·R))[∂(r·R·A_r)/∂r + ∂(r·R·A_θ)/∂θ]  ← WRONG!
""")
