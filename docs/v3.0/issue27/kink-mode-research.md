# Kink Mode Research (Issue #27 Phase 1)

**Author:** 小P ⚛️  
**Date:** 2026-03-24  
**Issue:** #27 - Multiple instability modes

---

## Kink Mode Physics (m=1)

### What is Kink Mode?

**Definition:**
- MHD instability with helical deformation (m=1)
- Current-driven (J×B force imbalance)
- Two types: Internal kink, External kink

**Distinguishing features:**
- m=1 (vs m>1 for ballooning/tearing)
- Global displacement of plasma column
- Safety factor q ≈ 1 resonance

---

## Internal Kink (m=1, n=1)

### Physics Mechanism

**Kadomtsev 1975, Porcelli 1996:**
- Occurs when q₀ < 1 (on-axis safety factor)
- Resonant at q=1 surface
- Ideal MHD (no resistivity needed)

**Growth rate (ideal):**
```
γ² ≈ (m/R₀)² V_A² [β_p - β_p,crit]
```

where:
- V_A = Alfvén velocity
- β_p = poloidal beta
- Criterion: q₀ < 1

**Typical parameters:**
- Growth time: ~τ_A (Alfvén time, µs)
- Fast instability!

---

## External Kink (m=1)

### Physics

**Wesson 2011:**
- Occurs near plasma edge
- Requires q_edge < q_crit
- Can be wall-stabilized

**Growth rate:**
```
γ ≈ V_A / (R₀ √(1 + (r_wall/a)²))
```

**Simplification for cylindrical:**
- Ignore toroidal effects
- Focus on m=1 displacement

---

## Cylindrical Kink IC Design

### Equilibrium

**Current profile (for q ≈ 1):**
```python
def current_kink_equilibrium(r, j0=1.0, a=0.8):
    """
    Current profile with q ≈ 1 near core.
    
    J_z(r) = j0 * (1 - (r/a)²)  # Parabolic
    
    Safety factor:
    q(r) ≈ (r² B_z) / (R₀ ∫J_z dr)
    
    Choose j0 such that q(0) < 1
    """
    return j0 * (1 - (r/a)**2)
```

**Flux function:**
```python
def psi_kink_equilibrium(r, a=0.8):
    """
    ∇²ψ = -μ₀ J_z
    
    Solution (cylindrical):
    ψ(r) = -(μ₀ j₀ a²/4) [(r/a)² - (r/a)⁴/2]
    """
    # Solve Poisson equation
    pass
```

---

## Perturbation (m=1 displacement)

### Ansatz

**Helical displacement:**
```python
def psi_kink_perturbation(r, θ, ε=0.01, r_res=0.5):
    """
    m=1, n=1 kink mode.
    
    δψ ~ f(r) sin(mθ - nφ)
    
    For 2D (ignore φ):
    δψ = ε * f(r) * sin(θ)
    
    where f(r) peaked at q=1 surface (r ≈ r_res)
    """
    f_r = np.exp(-((r - r_res)/0.2)**2)  # Gaussian at resonance
    return ε * f_r * np.sin(θ)
```

**Stream function:**
```python
def phi_kink_perturbation(r, θ, ε=0.01, r_res=0.5):
    """
    Phase relationship with δψ.
    
    For kink: δφ ~ cos(θ) (90° phase)
    """
    f_r = np.exp(-((r - r_res)/0.2)**2)
    return ε * f_r * np.cos(θ)
```

---

## Growth Rate Validation

### Theoretical Prediction

**For cylindrical kink (Freidberg 1987):**
```
γ ≈ 0.3 V_A / R₀  (internal kink, q₀ ≈ 0.9)
```

**Parameters:**
- V_A = B₀/√(μ₀ρ) ≈ 1e6 m/s (typical)
- R₀ = 1.0 m
- → γ ≈ 3e5 s⁻¹ (very fast!)

**For simulation (need to slow down):**
- Use reduced B₀ or increased ρ
- Target: γ ~ 1-10 s⁻¹ (observable in 0.1s)

---

## Implementation Strategy

### Approach

**Similar to Issue #29 (Harris sheet):**
1. Equilibrium with q ≈ 1
2. m=1 perturbation
3. Measure growth rate
4. Compare to theory

**Difference from tearing:**
- No resistivity needed (ideal MHD)
- Faster growth (ideal vs resistive)
- Global mode (vs localized tearing)

---

## References

**Key papers:**
- Kadomtsev (1975) - Internal kink theory
- Porcelli (1996) - Review
- Freidberg (1987) - Ideal MHD book
- Wesson (2011) - Tokamaks textbook

**Growth rate formulas:**
- Cylindrical: Freidberg Ch 8
- Toroidal: Wesson Ch 7

---

## Next Steps (Phase 2)

1. Implement equilibrium (q ≈ 1 current profile)
2. Add m=1 perturbation
3. Test with MHD solver
4. Measure growth rate
5. Validate vs theory

**Estimated time:** Similar to Issue #29 (~1 hour if analytical)

---

**小P Phase 1 complete: Kink mode physics understood** ⚛️✅
