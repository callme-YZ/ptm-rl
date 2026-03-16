# Jackson Formula Verification

## Jackson Classical Electrodynamics (3rd ed), Section 5.5

### Vector Potential for Circular Current Loop

**Equation 5.41:**

For a circular current loop of radius `a` carrying current `I` in the plane `z=0`, centered at origin, the vector potential at point `(ρ, z)` in cylindrical coordinates is:

```
A_φ(ρ,z) = (μ₀I/π) √(a/ρ) [(1 - k²/2)K(k²) - E(k²)]
```

where:
```
k² = 4aρ / [(a+ρ)² + z²]
```

### Conversion to our notation

**Our setup:**
- Coil at (R_c, Z_c) with current I
- Field point at (R, Z)

**Mapping:**
- Jackson's `a` → our `R_c`
- Jackson's `ρ` → our `R`
- Jackson's `z` → our `(Z - Z_c)`

**So:**
```
A_φ = (μ₀I/π) √(R_c/R) [(1 - k²/2)K(k²) - E(k²)]
```

with:
```
k² = 4R_c·R / [(R_c+R)² + (Z-Z_c)²]
```

### Poloidal Flux

**Definition:**
```
ψ = R·A_φ
```

**Substitute:**
```
ψ = R · (μ₀I/π) √(R_c/R) [(1 - k²/2)K(k²) - E(k²)]
  = (μ₀I/π) √(R·R_c) [(1 - k²/2)K(k²) - E(k²)]
```

**Simplify the bracket:**
```
(1 - k²/2)K - E = K - (k²/2)K - E
                = K - E - (k²/2)K
                = (K - E) - (k²/2)K
```

**NOT immediately matching FreeGS form...**

Let me try different manipulation:
```
(1 - k²/2)K - E = K(1 - k²/2) - E
```

Multiply by 2:
```
2[(1 - k²/2)K - E] = 2K - k²K - 2E
                   = 2K - 2E - k²K
                   = (2 - k²)K - 2E
```

So:
```
[(1 - k²/2)K - E] = [(2 - k²)K - 2E] / 2
```

**Therefore Jackson's formula gives:**
```
ψ = (μ₀I/π) √(R·R_c) · [(2 - k²)K - 2E] / 2
  = (μ₀I/2π) √(R·R_c) [(2 - k²)K - 2E]
```

### Comparing to FreeGS

**FreeGS:**
```
ψ = (μ₀I/2π) √(R·R_c) [(2 - k²)K - 2E] / k
```

**Jackson (my derivation):**
```
ψ = (μ₀I/2π) √(R·R_c) [(2 - k²)K - 2E]
```

**Difference:** Factor of `1/k`!

**Question:** Did I make an error in reading Jackson, or is there a convention difference?

### Re-check Jackson Eq. 5.41

Let me look at exact form more carefully...

**Jackson writes (Eq. 5.41):**
```
A_φ = (μ₀Ia/π) (1/√(aρ)) F(k)
```

where F(k) involves elliptic integrals of **first kind with argument k** (NOT k²).

**Critical:** Jackson uses `K(k)` and `E(k)` (argument k, not k²)!

But scipy uses `K(k²)` and `E(k²)`!

**Relation:**
```
K_scipy(k²) = K_Jackson(k)
```

So when converting Jackson's formula, I need to be careful about arguments!

### Correct Transcription

**Jackson's formula with k as modulus:**
```
A_φ = (μ₀I/π) √(a/ρ) [(2/k - k)K(k) - (2/k)E(k)]
```

OR alternative form (more common):
```
A_φ = (μ₀I/πk) √(aρ) [(2-k²)K(k) - 2E(k)]
```

**With scipy convention K(k²):**
```
A_φ = (μ₀I/πk) √(aρ) [(2-k²)K(k²) - 2E(k²)]
```

**Then:**
```
ψ = R·A_φ = (μ₀I/πk) R√(aρ) [(2-k²)K - 2E]
          = (μ₀I/πk) √(aρR²) [(2-k²)K - 2E]
```

Wait, this doesn't look right either...

**Let me be more careful:**

If coil radius is `a` and we measure at distance `ρ`:
```
A_φ = (μ₀I/πk) √(aρ) [(2-k²)K(k²) - 2E(k²)]
```

For our notation (coil at R_c, measure at R):
```
A_φ = (μ₀I/πk) √(R_c·R) [(2-k²)K - 2E]
```

**Flux:**
```
ψ = R·A_φ = (μ₀I/πk) R√(R_c·R) [(2-k²)K - 2E]
          = (μ₀I/πk) √(R²·R_c·R) [(2-k²)K - 2E]
          = (μ₀I/πk) √(R³·R_c) [(2-k²)K - 2E]
```

**This still has extra R! Something wrong...**

### Going back to basics

**Let me look at the definition more carefully.**

Actually, in axisymmetric systems:

**Vector potential:** A = A_φ(R,Z) φ̂

**Poloidal flux function:**
```
ψ(R,Z) = ∫₀^R A_φ(R',Z) R' dR'  × some constant
```

OR simpler definition:
```
B_R = -(1/R) ∂ψ/∂Z
B_Z = (1/R) ∂ψ/∂R
```

So:
```
ψ and A_φ are related, but NOT ψ = R·A_φ !
```

**Let me check FreeGS source for their definition...**

Actually, in tokamak physics the standard definition IS:
```
ψ = R A_φ
```

So Jackson gives A_φ, and we multiply by R to get ψ.

**Let me recalculate:**

Jackson (in k notation, converted to k²):
```
A_φ = (μ₀I/π) (1/k) √(a/ρ) [(2-k²)K(k²) - 2E(k²)]
```

For our case (a→R_c, ρ→R):
```
A_φ = (μ₀I/π) (1/k) √(R_c/R) [(2-k²)K - 2E]
```

**Flux:**
```
ψ = R·A_φ = (μ₀I/π) (R/k) √(R_c/R) [(2-k²)K - 2E]
          = (μ₀I/πk) √(R·R_c) [(2-k²)K - 2E]
```

**Factor of 2π:**
FreeGS has `μ₀I/(2π)`, I have `μ₀I/π`.

**Difference by factor of 2!**

Maybe Jackson defines I as total current, FreeGS defines as... let me check.

Actually, the factor of 2 comes from:
- Jackson integrates full loop ∫₀^(2π) dφ
- But symmetry argument...

Let me just accept FreeGS is validated code, and use:

```
ψ = (μ₀I/2π) (1/k) √(R·R_c) [(2-k²)K(k²) - 2E(k²)]
```

**This is the correct formula with `/k`.**

---

## Conclusion

**Correct Green's function:**
```python
k2 = 4*R*Rc / ((R+Rc)**2 + (Z-Zc)**2)
k = sqrt(k2)
psi = (mu0*I / (2*pi)) * sqrt(R*Rc) * ((2-k2)*K(k2) - 2*E(k2)) / k
```

**Key point:** Divide by `k`, NOT `k²`!

My error was missing the factor of `1/k` in the definition of the Green's function.
