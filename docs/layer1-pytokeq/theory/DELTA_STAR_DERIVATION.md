# Δ* Operator Discretization - Complete Derivation

## Problem Statement

**Bug found:** Current stencil gives Δ*ψ = -6, but analytical answer is -2 (×3 error!)

**Root cause:** Incorrect coefficient `coeff_ij` for central point ψ_{i,j}

---

## Analytical Definition

**Grad-Shafranov operator in cylindrical (R,Z):**

```
Δ* ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
```

**Expand the R derivative:**

```
R ∂/∂R (1/R ∂ψ/∂R) = R × [∂/∂R(1/R) × ∂ψ/∂R + 1/R × ∂²ψ/∂R²]
                     = R × [-1/R² × ∂ψ/∂R + 1/R × ∂²ψ/∂R²]
                     = -1/R × ∂ψ/∂R + ∂²ψ/∂R²
```

**Therefore:**

```
Δ* ψ = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²
```

This is **correct** and matches literature ✓

---

## Finite Difference Discretization

**Grid spacing:** ΔR, ΔZ

**Central differences (2nd order accurate):**

```
∂ψ/∂R ≈ (ψ_{i+1,j} - ψ_{i-1,j}) / (2ΔR)

∂²ψ/∂R² ≈ (ψ_{i+1,j} - 2ψ_{i,j} + ψ_{i-1,j}) / ΔR²

∂²ψ/∂Z² ≈ (ψ_{i,j+1} - 2ψ_{i,j} + ψ_{i,j-1}) / ΔZ²
```

---

## Substitute into Δ*

**Term 1: ∂²ψ/∂R²**
```
= (ψ_{i+1,j} - 2ψ_{i,j} + ψ_{i-1,j}) / ΔR²
```

**Term 2: -(1/R) ∂ψ/∂R**
```
= -(1/R_i) × (ψ_{i+1,j} - ψ_{i-1,j}) / (2ΔR)
```

**Term 3: ∂²ψ/∂Z²**
```
= (ψ_{i,j+1} - 2ψ_{i,j} + ψ_{i,j-1}) / ΔZ²
```

---

## Collect Coefficients

**For ψ_{i-1,j}:**
```
coeff_{i-1,j} = 1/ΔR² + 1/(2R_i·ΔR)
```

**For ψ_{i+1,j}:**
```
coeff_{i+1,j} = 1/ΔR² - 1/(2R_i·ΔR)
```

**For ψ_{i,j-1}:**
```
coeff_{i,j-1} = 1/ΔZ²
```

**For ψ_{i,j+1}:**
```
coeff_{i,j+1} = 1/ΔZ²
```

**For ψ_{i,j} (central point):**

From Term 1: `-2/ΔR²`  
From Term 2: `0` (no ψ_{i,j} in first derivative)  
From Term 3: `-2/ΔZ²`

```
coeff_{i,j} = -2/ΔR² - 2/ΔZ²
```

**This is what we had!** ✓

---

## BUT WAIT - Sign Error Discovery!

**Look at our code again:**

```python
# Current (WRONG):
coeff_im = 1/dR**2 - 1/(2*R_ij*dR)  # ψ_{i-1,j}
coeff_ip = 1/dR**2 + 1/(2*R_ij*dR)  # ψ_{i+1,j}
```

**From derivation above:**
```
coeff_{i-1,j} = 1/ΔR² + 1/(2R·ΔR)  ← PLUS!
coeff_{i+1,j} = 1/ΔR² - 1/(2R·ΔR)  ← MINUS!
```

**Our code has OPPOSITE signs for the (1/R) term!** ❌

---

## Verification with Test Case

**Test: ψ = -(R² + Z²)**

Analytical:
```
∂ψ/∂R = -2R
∂²ψ/∂R² = -2
∂²ψ/∂Z² = -2

Δ*ψ = -2 - (1/R)×(-2R) + (-2)
    = -2 + 2 - 2
    = -2  ✓
```

**With WRONG signs (current code):**

At R=1.5:
```
Term 1: -2
Term 2: -(1/1.5)×(-2×1.5) = -(1/1.5)×(-3) = +2  ✓ (correct)
Term 3: -2

But stencil assembly is wrong!

With wrong coeff signs:
  coeff_{i-1} × ψ_{i-1} + coeff_{i+1} × ψ_{i+1}
  = [1/ΔR² - 1/(2R·ΔR)] × ψ_{i-1} + [1/ΔR² + 1/(2R·ΔR)] × ψ_{i+1}
  
This gives OPPOSITE contribution from (1/R)∂ψ/∂R term!
```

**Result:** Gets -6 instead of -2 (×3 error) ✓ **Matches our test!**

---

## Correct Coefficients

```python
# CORRECT:
coeff_im = 1/dR**2 + 1/(2*R_ij*dR)  # ψ_{i-1,j} (PLUS!)
coeff_ip = 1/dR**2 - 1/(2*R_ij*dR)  # ψ_{i+1,j} (MINUS!)
coeff_jm = 1/dZ**2                   # ψ_{i,j-1}
coeff_jp = 1/dZ**2                   # ψ_{i,j+1}
coeff_ij = -(2/dR**2 + 2/dZ**2)     # ψ_{i,j}
```

**Check sum = 0 (consistency):**
```
coeff_sum = coeff_im + coeff_ip + coeff_jm + coeff_jp + coeff_ij
          = [1/ΔR² + 1/(2R·ΔR)] + [1/ΔR² - 1/(2R·ΔR)] + 1/ΔZ² + 1/ΔZ² - 2/ΔR² - 2/ΔZ²
          = 2/ΔR² + 2/ΔZ² - 2/ΔR² - 2/ΔZ²
          = 0  ✓
```

---

## Fix Implementation

**Change one line in solver:**

```python
# OLD (WRONG):
coeff_im = 1/dR**2 - 1/(2*R_ij*dR)  # ❌
coeff_ip = 1/dR**2 + 1/(2*R_ij*dR)  # ❌

# NEW (CORRECT):
coeff_im = 1/dR**2 + 1/(2*R_ij*dR)  # ✅
coeff_ip = 1/dR**2 - 1/(2*R_ij*dR)  # ✅
```

**Time to fix: 30 seconds**

---

## Expected Result After Fix

**Test ψ = -(R² + Z²):**
- Before fix: error = 3.1 (diverges)
- After fix: error < 0.01 (converges!)

**Step 6 Picard:**
- Before fix: 500 iterations, residual 1e-2
- After fix: 30-50 iterations, residual < 1e-6

---

## Lesson Learned

**Small sign error → ×3 numerical error!**

**Why verification tests are CRITICAL:**
- Analytical test case (ψ = -(R²+Z²)) found the bug
- Would have missed without systematic verification
- 小A's approach saved hours of wrong debugging

**This is 小A's "系统强壮" in action!** ✅
