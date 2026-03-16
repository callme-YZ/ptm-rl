# Profile Second Derivatives Derivation

**Date:** 2026-03-11  
**Purpose:** Analytical second derivatives for Newton Jacobian

---

## 1. Constant Profile

**Definitions:**
```
p'(ψ) = p1  (constant)
f·f'(ψ) = f1/2  (constant)
```

**First derivatives:**
```
∂p'/∂ψ = 0
∂(f·f')/∂ψ = 0
```

**Second derivatives:**
```
p''(ψ_N) = ∂²p'/∂ψ_N² = 0
ff''(ψ_N) = ∂²(f·f')/∂ψ_N² = 0
```

**Simple!** ✓

---

## 2. Quadratic Profile

**Definitions:**
```
p'(ψ) = p0 + p1·ψ_N
f·f'(ψ) = f0 + f1·ψ_N
```

where `ψ_N = (ψ - ψ_ma)/(ψ_x - ψ_ma)`

**First derivatives w.r.t. ψ_N:**
```
∂p'/∂ψ_N = p1
∂(f·f')/∂ψ_N = f1
```

**Second derivatives:**
```
p''(ψ_N) = ∂²p'/∂ψ_N² = 0
ff''(ψ_N) = ∂²(f·f')/∂ψ_N² = 0
```

**Also zero!** ✓

---

## 3. Luxon-Brown Profile

**This is the tricky one!**

**Definitions:**
```
p'(ψ_N) = (α·β/r₀) · (1 - ψ_N^δ)^γ
f·f'(ψ_N) = α·(1-β)·μ₀r₀ · (1 - ψ_N^δ)^γ
```

**Let:**
```
F(ψ_N) = (1 - ψ_N^δ)^γ
```

**Then:**
```
p'(ψ_N) = C_p · F(ψ_N)   where C_p = α·β/r₀
f·f'(ψ_N) = C_f · F(ψ_N)  where C_f = α·(1-β)·μ₀r₀
```

### Step 1: First derivative of F

**Chain rule:**
```
dF/dψ_N = d/dψ_N [(1 - ψ_N^δ)^γ]
        = γ·(1 - ψ_N^δ)^(γ-1) · d/dψ_N(1 - ψ_N^δ)
        = γ·(1 - ψ_N^δ)^(γ-1) · (-δ·ψ_N^(δ-1))
        = -γδ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-1)
```

**Therefore:**
```
∂p'/∂ψ_N = C_p · dF/dψ_N
         = -C_p·γδ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-1)

∂(f·f')/∂ψ_N = C_f · dF/dψ_N
             = -C_f·γδ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-1)
```

### Step 2: Second derivative of F

**Apply product rule to:**
```
dF/dψ_N = -γδ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-1)
```

**Let:**
```
u = ψ_N^(δ-1)
v = (1 - ψ_N^δ)^(γ-1)
```

**Then:**
```
dF/dψ_N = -γδ·u·v
```

**Product rule:**
```
d²F/dψ_N² = -γδ·[u'·v + u·v']
```

**Compute u':**
```
u' = d/dψ_N [ψ_N^(δ-1)]
   = (δ-1)·ψ_N^(δ-2)
```

**Compute v':**
```
v' = d/dψ_N [(1 - ψ_N^δ)^(γ-1)]
   = (γ-1)·(1 - ψ_N^δ)^(γ-2)·(-δ·ψ_N^(δ-1))
   = -(γ-1)δ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-2)
```

**Substitute:**
```
d²F/dψ_N² = -γδ·[(δ-1)·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-1) 
                + ψ_N^(δ-1)·[-(γ-1)δ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-2)]]
```

**Simplify first term:**
```
T1 = -γδ·(δ-1)·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-1)
```

**Simplify second term:**
```
T2 = -γδ·ψ_N^(δ-1)·[-(γ-1)δ·ψ_N^(δ-1)·(1 - ψ_N^δ)^(γ-2)]
   = γδ·(γ-1)δ·ψ_N^(2δ-2)·(1 - ψ_N^δ)^(γ-2)
   = γδ²(γ-1)·ψ_N^(2δ-2)·(1 - ψ_N^δ)^(γ-2)
```

**Combine:**
```
d²F/dψ_N² = -γδ(δ-1)·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-1) 
          + γδ²(γ-1)·ψ_N^(2δ-2)·(1 - ψ_N^δ)^(γ-2)
```

**Factor out common terms:**
```
d²F/dψ_N² = γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2) · 
           [-(δ-1)·(1 - ψ_N^δ) + δ(γ-1)·ψ_N^δ]
```

**Simplify bracket:**
```
-(δ-1)·(1 - ψ_N^δ) + δ(γ-1)·ψ_N^δ
= -(δ-1) + (δ-1)·ψ_N^δ + δ(γ-1)·ψ_N^δ
= -(δ-1) + [(δ-1) + δ(γ-1)]·ψ_N^δ
= -(δ-1) + [δ-1 + δγ - δ]·ψ_N^δ
= -(δ-1) + [δγ - 1]·ψ_N^δ
```

**Final form:**
```
d²F/dψ_N² = γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2) · 
           [-(δ-1) + (δγ - 1)·ψ_N^δ]
```

**Therefore:**
```
p''(ψ_N) = C_p · d²F/dψ_N²
         = (α·β/r₀)·γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2)·
           [-(δ-1) + (δγ - 1)·ψ_N^δ]

ff''(ψ_N) = C_f · d²F/dψ_N²
          = α·(1-β)·μ₀r₀·γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2)·
            [-(δ-1) + (δγ - 1)·ψ_N^δ]
```

---

## Edge Cases

**For ψ_N = 0:**
- `ψ_N^(δ-2) → 0` (if δ > 2)
- `ψ_N^(δ-2) → ∞` (if δ < 2)

**Standard parameters: δ=2.0**
- `ψ_N^0 = 1`
- Well-defined! ✓

**For ψ_N = 1 (at separatrix):**
- `(1 - ψ_N^δ) = 0`
- `(1 - ψ_N^δ)^(γ-2) → 0` or `∞` depending on γ

**If γ > 2:** second derivative → 0 at edge (good)  
**If γ < 2:** second derivative → ∞ at edge (bad)

**Standard parameters: γ=1.4 < 2**
- Need to handle edge carefully!

**Solution:** Clip ψ_N to [0, 0.999] to avoid singularity

---

## Implementation Summary

**For each profile:**

1. **ConstantProfile:**
   - `p'' = 0`
   - `ff'' = 0`

2. **QuadraticProfile:**
   - `p'' = 0`
   - `ff'' = 0`

3. **LuxonBrownProfile:**
   - Clip ψ_N to [0, 0.999]
   - Compute F, F', F'' using formulas above
   - Return C·F''
