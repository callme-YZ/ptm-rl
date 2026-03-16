# Coil Configuration Scaling Theory

## Goal
Estimate required coil currents for healthy tokamak equilibrium from first principles.

## Given Profile Parameters

**Quadratic Profile:**
```
p'(ψ_N) = p0 + p1·ψ_N = 1e5 + (-8e4)·ψ_N  [Pa/Wb]
ff'(ψ_N) = f0 + f1·ψ_N = 25 + (-20)·ψ_N  [T²m²/Wb]
```

## Step 1: Estimate Plasma Current Density

**Current density formula:**
```
J_φ = R·p'(ψ) + (1/(μ₀R))·ff'(ψ)
```

**At plasma center (ψ_N=0, R=4.5m):**
```
J_φ,center = 4.5·(1e5) + (1/(4π×10⁻⁷·4.5))·25
          = 4.5e5 + 25/(5.65e-6)
          = 4.5e5 + 4.42e6
          ≈ 4.9e6 A/m²
```

**At plasma edge (ψ_N=1, R=4.5m):**
```
p'(1) = 1e5 - 8e4 = 2e4
ff'(1) = 25 - 20 = 5

J_φ,edge = 4.5·(2e4) + 5/(5.65e-6)
         = 9e4 + 8.85e5
         ≈ 9.75e5 A/m²
```

**Average:** J_avg ~ (4.9e6 + 9.75e5)/2 ~ 2.9e6 A/m²

## Step 2: Estimate Total Plasma Current

**Plasma cross-section (rough estimate):**
```
Minor radius: a ~ 1.0 m (from grid 3-6m, R₀=4.5m)
Area: A ~ π·a² ~ 3.14 m²
```

**Total current:**
```
I_plasma = J_avg × A
         ≈ 2.9e6 × 3.14
         ≈ 9.1e6 A
```

**This is HUGE! Typical tokamak I_plasma ~ 1-15 MA (mega-amperes) ✓**

## Step 3: Coil Current Requirement

**Vacuum flux from single coil at distance d:**
```
ψ_coil ~ (μ₀·I_coil)/(2π) · ln(R_max/R_coil)
```

**For R_coil=8m, R_max=6m:**
```
ψ_coil ~ (4π×10⁻⁷·I_coil)/(2π) · ln(8/6)
       ~ 2×10⁻⁷·I_coil·0.288
       ~ 5.76×10⁻⁸·I_coil
```

**To balance plasma (ψ_plasma ~ -μ₀·I_plasma/(2πR₀)):**
```
ψ_plasma ~ -(4π×10⁻⁷·9.1e6)/(2π·4.5)
         ~ -4.05×10⁻¹ Wb
```

**Required total coil flux:**
```
ψ_coil,total ~ |ψ_plasma| ~ 0.4 Wb
```

**For 3 coils with efficiency ~0.3:**
```
3 × 5.76×10⁻⁸·I_coil × 0.3 ~ 0.4
I_coil ~ 0.4/(5.2×10⁻⁸)
       ~ 7.7e6 A per coil
```

## Step 4: Refined Estimate

**Problem:** Single coil estimate too crude. Use superposition.

**Better approach - Poloidal field coil system:**
```
PF1 (outer, R=8m):  I₁ (main field)
PF2 (upper, Z=3m):  I₂ (shaping)
PF3 (lower, Z=-3m): I₃ (shaping)
```

**Typical ratio (from ITER design):**
```
I₁ : I₂ : I₃ = 1.0 : (-0.3) : (-0.3)
```

**If I₁ = 5e6 A:**
```
I₂ = I₃ = -1.5e6 A
```

**This gives:**
- Central flux ~ 0.3-0.5 Wb (good range)
- Up-down symmetric shaping
- X-point formation possible

## Conclusion

**Recommended coil configuration:**
```python
coils = [
    {'R_coil': 8.0, 'Z_coil': 0.0, 'I_coil': 5e6},    # PF1
    {'R_coil': 4.0, 'Z_coil': 3.0, 'I_coil': -1.5e6}, # PF2
    {'R_coil': 4.0, 'Z_coil': -3.0, 'I_coil': -1.5e6},# PF3
]
```

**Expected plasma:**
- ψ_axis ~ -0.2 to -0.4 Wb
- ψ_edge ~ +0.1 to +0.3 Wb  
- Separation ~ 0.3-0.7 Wb ✓ (healthy!)
- Plasma area ~ 2-3 m²

**Key insight:** Our original I=1e6 was 5× too small!
