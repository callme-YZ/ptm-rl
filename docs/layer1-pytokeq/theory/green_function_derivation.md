# Green's Function for Circular Coil - Complete Derivation

**Date:** 2026-03-11  
**Purpose:** Rigorous derivation of ψ and its derivatives for circular current loop

---

## Physical Setup

**Configuration:**
- Circular coil of radius `a` at height `Z_c` in (R,Z) coordinates
- Coil carries current `I` (positive = counter-clockwise viewed from above)
- Coil located at `R = R_c`, `Z = Z_c`
- Evaluate flux ψ at field point `(R, Z)`

**Cylindrical coordinates:** `(R, φ, Z)`
- `R`: distance from symmetry axis
- `φ`: toroidal angle
- `Z`: vertical position

**Axisymmetry:** All quantities independent of `φ`

---

## Starting Point: Vector Potential

**Biot-Savart law for vector potential:**

For a current element `I d𝓁` at position `r'`, the vector potential at `r` is:

```
A(r) = (μ₀I/4π) ∮ d𝓁' / |r - r'|
```

**For circular coil:**
- Coil position: `r' = (R_c cos φ', R_c sin φ', Z_c)`
- Current element: `d𝓁' = R_c dφ' φ̂'`
- Field point: `r = (R cos φ, R sin φ, Z)`

**Distance:**
```
|r - r'|² = R² + R_c² - 2RR_c cos(φ-φ') + (Z-Z_c)²
```

**By symmetry:** `A = A_φ(R,Z) φ̂` (only toroidal component)

**Poloidal flux:**
```
ψ(R,Z) = R A_φ(R,Z)
```

---

## Standard Result (Jackson Ch. 5)

After integration over φ', the result is expressed in terms of **complete elliptic integrals**:

```
ψ(R,Z) = (μ₀I/π) √(RR_c) [(2-k²)K(k²) - 2E(k²)] / k²
```

where:

```
k² = 4RR_c / [(R+R_c)² + (Z-Z_c)²]
```

**Notation:**
- `K(k²)` = Complete elliptic integral of 1st kind
- `E(k²)` = Complete elliptic integral of 2nd kind
- Both are functions of modulus parameter `k²` (NOT `k`)

**scipy.special convention:**
```python
K = scipy.special.ellipk(k²)  # Takes k² as argument
E = scipy.special.ellipe(k²)
```

---

## Alternative Form

Define:
```
G(k²) ≡ [(2-k²)K(k²) - 2E(k²)] / k²
```

Then:
```
ψ = (μ₀I/π) √(RR_c) G(k²)
```

**Note:** My original formula had extra factor! Let me verify...

**Checking dimensions:**
- `[ψ] = Wb/rad = T·m²`
- `[μ₀I] = T·m` (correct)
- `[√(RR_c)] = m`
- `[G] = dimensionless`
- Product: `T·m·m = T·m²` ✓

**Prefactor check:**

Jackson Eq. 5.41: `A_φ = (μ₀I/4π) ... `
But ψ = R A_φ, and there's integral ∫dφ' = 2π for symmetric case
So: `ψ ~ (μ₀I/4π) × 2π × ... = (μ₀I/2π) × ...`

**CORRECTION:** Standard formula is:

```
ψ = (μ₀I/2π) √(RR_c) G(k²)
```

NOT `μ₀I/π`!

---

## Corrected Formula

```
ψ(R,Z) = (μ₀I_c/2π) √(RR_c) [(2-k²)K(k²) - 2E(k²)] / k²
```

This matches my original implementation ✓

---

## Derivatives

### Strategy

Use chain rule:
```
∂ψ/∂R = ∂ψ/∂R|_k² + (∂ψ/∂k²)(∂k²/∂R)
∂ψ/∂Z = (∂ψ/∂k²)(∂k²/∂Z)
```

### Step 1: Derivatives of k²

```
k² = 4RR_c / D

where D = (R+R_c)² + (Z-Z_c)²
```

**∂k²/∂R:**
```
∂k²/∂R = 4R_c [D - 4RR_c·2(R+R_c)] / D²
       = 4R_c [D - 8R(R+R_c)R_c] / D²
       = 4R_c [(R+R_c)² + (Z-Z_c)² - 8R(R+R_c)R_c] / D²
       = 4R_c [(R+R_c)² - 8R(R+R_c)R_c + (Z-Z_c)²] / D²
```

Simplify numerator:
```
(R+R_c)² - 8R(R+R_c)R_c
= (R+R_c)[(R+R_c) - 8RR_c]
= (R+R_c)[R + R_c - 8RR_c]

Hmm, this doesn't simplify nicely...
```

**Let me recalculate carefully:**

```
∂/∂R [4RR_c / D] = 4R_c / D - 4RR_c / D² · (∂D/∂R)

∂D/∂R = 2(R+R_c)

∂k²/∂R = 4R_c/D - 4RR_c·2(R+R_c)/D²
       = 4R_c [1/D - 2R(R+R_c)/D²]
       = 4R_c [D - 2R(R+R_c)] / D²
       = 4R_c [(R+R_c)² + (Z-Z_c)² - 2R(R+R_c)] / D²
```

Simplify:
```
(R+R_c)² - 2R(R+R_c) = (R+R_c)[(R+R_c) - 2R]
                      = (R+R_c)(R_c - R)
                      = (R_c² - R²)
```

So:
```
∂k²/∂R = 4R_c [(R_c² - R²) + (Z-Z_c)²] / D²
       = 4R_c [R_c² - R² + (Z-Z_c)²] / D²
```

**∂k²/∂Z:**
```
∂D/∂Z = 2(Z - Z_c)

∂k²/∂Z = -4RR_c · 2(Z-Z_c) / D²
       = -8RR_c(Z-Z_c) / D²
```

---

### Step 2: Derivatives of G(k²)

```
G(k²) = [(2-k²)K - 2E] / k²
```

**Need:** `dG/dk²`

**Using:**
```
dK/dk² = [E/(1-k²) - K] / (2k²)  (Abramowitz & Stegun 17.3.11)
dE/dk² = (E - K) / (2k²)         (Abramowitz & Stegun 17.3.12)
```

**Differentiate G:**
```
dG/dk² = d/dk² {[(2-k²)K - 2E] / k²}

Use quotient rule:
dG/dk² = [k²·d/dk²((2-k²)K - 2E) - ((2-k²)K - 2E)·2k] / k⁴
```

**Compute numerator term 1:**
```
d/dk²[(2-k²)K - 2E] = -K + (2-k²)dK/dk² - 2dE/dk²
                    = -K + (2-k²)[E/(1-k²) - K]/(2k²) - 2(E-K)/(2k²)
                    = -K + [(2-k²)E/(1-k²) - (2-k²)K - (E-K)]/(2k²)
```

**This is getting very messy... **

Let me look for standard result in literature...

---

## Alternative: Use Existing Validated Code

**FreeGS implementation (GitHub: freegs/greenfunc.py):**

They provide:
- `psi(R, Z, R_c, Z_c, I)` - flux
- `psi_greens(R, Z, R_c, Z_c)` - Green's function
- `Br(R, Z, R_c, Z_c, I)` - radial field
- `Bz(R, Z, R_c, Z_c, I)` - vertical field

**All analytical, tested against benchmarks**

**Option:** 
1. Study their implementation
2. Understand the formulas
3. Reimplement with full understanding
4. Cross-validate

This is **learning from production code**, not copying blindly.

---

## Decision Point

**Path A: Complete derivation from scratch** (4-6 hours)
- Finish derivative algebra
- Implement all formulas
- Validate step-by-step

**Path B: Study + understand FreeGS** (2-3 hours)
- Read their implementation
- Understand each formula
- Verify against literature
- Reimplement with full documentation

**Path C: Hybrid** (3-4 hours)
- Get derivatives from FreeGS
- Verify they match theory
- Implement myself with understanding
- Test thoroughly

---

## Recommendation

**Use Path B (Study FreeGS):**

**Why:**
1. FreeGS is **production-tested** (used in real tokamak simulations)
2. Their formulas are **validated** against benchmarks
3. **Learning from experts** is scientific method
4. I will **understand fully** before implementing
5. **Faster** than deriving from scratch
6. **Same rigor** if I verify their formulas

**How:**
1. Download FreeGS source
2. Read `greenfunc.py` line-by-line
3. Match formulas to theory (Jackson/Wesson)
4. Write my own implementation
5. Test: my results == FreeGS results
6. Document all formulas with references

**This is NOT taking shortcuts:**
- I will understand every formula
- I will implement myself
- I will test thoroughly
- Just using FreeGS as **reference**, not black box

---

**YZ, approve this approach?**
