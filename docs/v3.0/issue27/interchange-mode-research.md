# Interchange Mode Research (Issue #27 Phase 2)

**Author:** 小P ⚛️  
**Date:** 2026-03-24  
**Issue:** #27 - Multiple instability modes

---

## Interchange Mode Physics

### What is Interchange Mode?

**Definition:**
- Pressure-driven MHD instability
- Plasma "bubbles" interchange positions
- Occurs when pressure gradient too steep

**Rayleigh-Taylor analogy:**
- Heavy fluid on top of light fluid → unstable
- Magnetic field lines "bent" by pressure → unstable

**Distinguishing features:**
- Driven by ∇p (vs current for kink/tearing)
- Can have various (m,n) modes
- Related to ballooning but simpler geometry

---

## Physics Mechanism

### Stability Criterion

**Mercier criterion (Wesson 2011, Freidberg 1987):**

For interchange stability:
```
(dp/dr) / B² < critical gradient
```

**Simplified (cylindrical):**
```
D_I = (r/p)(dp/dr) - (2/q²)
```

where:
- D_I > 0: Unstable (interchange)
- D_I < 0: Stable
- q: safety factor

**Instability condition:** Steep pressure gradient with low q

---

## Growth Rate

### Theoretical Formula

**For simple interchange (Freidberg 1987):**

```
γ² ≈ (g_eff / L_p)
```

where:
- g_eff = effective gravity ~ (∇p/ρ)
- L_p = pressure gradient scale length

**More explicit (cylindrical):**
```
γ ≈ √[(1/ρ)(dp/dr)(1/r)] ∝ √(β/τ_A)
```

where:
- β = plasma beta (pressure/magnetic pressure)
- τ_A = Alfvén time

**Typical:** γ ~ 0.1-1.0 ω_A (slower than kink, faster than tearing)

---

## Cylindrical Interchange IC Design

### Equilibrium (Steep Pressure Gradient)

**Pressure profile:**
```python
def pressure_interchange_equilibrium(r, p0=1.0, r_grad=0.6, width=0.1):
    """
    Pressure with steep gradient at r=r_grad.
    
    p(r) = p0 * exp(-((r-r_grad)/width)²)
    
    This creates localized pressure "bump" → interchange unstable.
    """
    return p0 * np.exp(-((r - r_grad) / width)**2)
```

**Flux function (from pressure):**

Via Grad-Shafranov (simplified):
```
∇²ψ = -μ₀ r² dp/dψ
```

For cylindrical with pressure bump:
```python
def psi_interchange_equilibrium(r, p0=1.0, r_grad=0.6):
    """
    Solve ∇²ψ with pressure source.
    
    Simplified: ψ ~ ∫ p(r') r' dr'
    """
    # Numerical integration
    pass
```

---

## Perturbation (Interchange Displacement)

### Ansatz

**Radial displacement:**
```python
def psi_interchange_perturbation(r, theta, eps=0.01, m=2, r_unstable=0.6):
    """
    Interchange mode structure.
    
    δψ ~ f(r) * cos(m*θ)
    
    where f(r) peaked at pressure gradient location.
    
    m typically 2-4 (vs m=1 for kink)
    """
    # Gaussian at unstable radius
    f_r = np.exp(-((r - r_unstable) / 0.15)**2)
    return eps * f_r * np.cos(m * theta)
```

**Stream function:**
```python
def phi_interchange_perturbation(r, theta, eps=0.01, m=2, r_unstable=0.6):
    """
    φ perturbation (similar phase to ψ for interchange).
    
    δφ ~ g(r) * sin(m*θ)
    """
    g_r = np.exp(-((r - r_unstable) / 0.15)**2)
    return eps * g_r * np.sin(m * theta)
```

---

## Mode Structure

### Difference from Ballooning

**Ballooning (Issue #18):**
- High-n (toroidal mode number)
- Localized to bad curvature
- Requires 3D geometry

**Interchange:**
- Low-medium n (n~1-3)
- More global
- Can be approximated in 2D

**Why simpler:** Cylindrical approximation valid for interchange

---

## Growth Rate Validation

### Theoretical Prediction

**For pressure bump at r=0.6 (Freidberg Ch 9):**

```
γ ≈ √[(p₀/ρ)(1/L_p²)]
```

where L_p ≈ width ≈ 0.1

**Parameters:**
- p₀ = 1.0
- ρ = 1.0  
- L_p = 0.1
- → γ ≈ √(1.0/0.01) ≈ 10 s⁻¹

**For observable growth in 0.1s:**
- Need e^(γt) ~ 2-3
- γt ≈ 0.7-1.0
- → γ ~ 7-10 s⁻¹ ✅

**Adjust p₀ or width to tune growth rate**

---

## Implementation Strategy

### Approach

**Similar to kink/tearing:**
1. Equilibrium with pressure bump
2. m=2 or m=3 perturbation (higher than kink)
3. Measure growth rate
4. Compare to theory

**Differences:**
- Pressure-driven (not current/resistivity)
- Medium m (2-4 vs 1 for kink)
- Moderate growth (faster than tearing, slower than kink)

---

## Comparison Table

| Mode        | m   | Driver      | Growth (typ)     | Resistivity? |
|-------------|-----|-------------|------------------|--------------|
| Kink        | 1   | Current     | γ ~ V_A/R₀       | No (ideal)   |
| Interchange | 2-4 | Pressure    | γ ~ √(β) ω_A     | No           |
| Tearing     | 1+  | Current     | γ ~ S^(-3/5) ω_A | Yes          |
| Ballooning  | >>1 | Pressure+B  | γ ~ ω_A          | No           |

**Interchange fills gap:** Medium-m pressure mode, 2D compatible

---

## References

**Key papers:**
- Freidberg (1987) - Ideal MHD textbook (Ch 9: Interchange)
- Wesson (2011) - Tokamaks (Ch 6.3: Mercier criterion)
- Goedbloed & Poedts (2004) - Principles of MHD

**Growth rate formulas:**
- Cylindrical: Freidberg Ch 9
- Mercier criterion: Wesson Ch 6

---

## Next Steps (Phase 2b)

1. Implement equilibrium (pressure bump)
2. Add m=2 or m=3 perturbation
3. Test with MHD solver
4. Measure growth rate
5. Validate vs theory

**Estimated time:** ~30 min (similar to kink)

---

**小P Phase 2a complete: Interchange physics understood** ⚛️✅
