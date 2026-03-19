# Energy Dissipation Theory: First-Principles Derivation

**Author:** 小P ⚛️  
**Date:** 2026-03-19  
**Purpose:** Resolve 30x discrepancy between theory and numerics in dH/dt

---

## Problem Statement

**Observed:**
```
dH/dt_numeric ≈ -0.03 × dH/dt_theory
```

**Test assumes:**
```
dH/dt = -η ∫ J² dV
```

**Goal:** Derive correct formula from first principles.

---

## Starting Equations

### Hamiltonian

```
H[ψ, φ] = ∫ dV [(1/2)|∇φ|² + (1/2)|∇ψ|²]
```

where:
- ψ: poloidal magnetic flux
- φ: electrostatic potential (stream function)
- ∇φ: velocity field (E×B flow)
- ∇ψ: poloidal magnetic field

Volume element in toroidal geometry:
```
dV = R·dr·dθ·dφ  (toroidal)
dV₂D = R·dr·dθ    (poloidal cross-section)
```

For axisymmetric case (∂/∂φ = 0):
```
H = 2π ∫∫ R [(1/2)|∇φ|² + (1/2)|∇ψ|²] dr dθ
```

### Evolution Equations

```
∂ψ/∂t = {ψ, H} - η·J
∂ω/∂t = {ω, H} + S_P - ν·∇²ω
```

where:
- ω = ∇²φ (vorticity)
- J = Δ*ψ/(μ₀R) (toroidal current density)
- Δ* = Grad-Shafranov operator
- {f, g} = Poisson bracket
- S_P = pressure force term

---

## Derivation: dH/dt

### Step 1: Time Derivative of H

```
dH/dt = d/dt ∫ dV [(1/2)|∇φ|² + (1/2)|∇ψ|²]
```

Split into kinetic and magnetic parts:
```
dH/dt = dK/dt + dU/dt
```

where:
```
K = ∫ (1/2)|∇φ|² dV  (kinetic energy)
U = ∫ (1/2)|∇ψ|² dV  (magnetic energy)
```

---

### Step 2: Magnetic Energy Evolution

```
dU/dt = ∫ ∇ψ · ∇(∂ψ/∂t) dV
```

Integration by parts (assuming boundary terms vanish):
```
dU/dt = -∫ ψ·∇²(∂ψ/∂t) dV
```

**But wait!** In toroidal geometry with Grad-Shafranov operator:
```
Δ*ψ = R²∇·(∇ψ/R²) ≠ ∇²ψ
```

**Correct form:**
```
∫ ∇ψ·∇f dV = ∫ R·∇ψ·∇f dr dθ dφ
```

For axisymmetric (∂/∂φ = 0):
```
∫ ∇ψ·∇f dV = 2π ∫∫ R·∇ψ·∇f dr dθ
```

Integration by parts in weighted form:
```
∫∫ R·∇ψ·∇f dr dθ = -∫∫ ψ·∇·(R∇f) dr dθ  (boundary = 0)
                   = -∫∫ ψ·[R∇²f + ∇R·∇f] dr dθ
```

For toroidal geometry:
```
∇R·∇f = (∂R/∂r)(∂f/∂r) + (1/r²)(∂R/∂θ)(∂f/∂θ)
      = cos(θ)·(∂f/∂r)  (since R = R₀ + r·cos(θ))
```

So:
```
∇·(R∇ψ) = R∇²ψ + cos(θ)·∂ψ/∂r
         = R[∂²ψ/∂r² + (1/r²)∂²ψ/∂θ² + (cos(θ)/R)∂ψ/∂r]
         = R·(1/R²)·Δ*ψ  (by definition of Δ*)
```

Therefore:
```
dU/dt = -2π ∫∫ ψ·(1/R)·Δ*(∂ψ/∂t) dr dθ
```

Substitute ∂ψ/∂t = {ψ, H} - ηJ:
```
dU/dt = -2π ∫∫ ψ·(1/R)·Δ*{ψ, H} dr dθ  (ideal part)
        + 2π η ∫∫ ψ·(1/R)·Δ*J dr dθ       (resistive part)
```

**But:** Δ*J is ill-defined! Need to use Δ*ψ directly.

**Correction:** Since J = Δ*ψ/(μ₀R):
```
Δ*J = (1/μ₀R)·Δ*(Δ*ψ)  ← This is NOT what we want!
```

**Re-examine:** The correct dissipative term should come from Ohm's law:
```
E = η·J  (parallel Ohm's law)
```

Power dissipation:
```
P = ∫ J·E dV = η ∫ J² dV
```

So resistive magnetic energy dissipation:
```
dU/dt|_resistive = -η ∫ J² dV
```

**This matches the test assumption!** But we need to verify the integration measure.

---

### Step 3: Volume Element Check

**Test code uses:**
```python
dV = (R0 + r*cos(θ)) * dr * dθ  # 2D poloidal
J2_int = np.sum(J**2 * dV)
dH_theory = -eta * J2_int  # Missing 2π?
```

**Correct 3D volume:**
```
∫ J² dV = 2π ∫∫ J²·R dr dθ
```

**So the test should be:**
```python
dH_theory = -eta * 2*np.pi * J2_int
```

**Hypothesis:** Missing factor of 2π!

---

### Step 4: Kinetic Energy Evolution

```
dK/dt = ∫ ∇φ·∇(∂φ/∂t) dV
```

Using ω = ∇²φ → ∂φ/∂t from ∂ω/∂t:
```
dK/dt = ∫ φ·(∂ω/∂t) dV
```

Substitute ∂ω/∂t = {ω, H} + S_P - ν∇²ω:
```
dK/dt = ∫ φ·{ω, H} dV  (ideal)
      + ∫ φ·S_P dV      (pressure)
      - ν ∫ φ·∇²ω dV    (viscous)
```

Viscous term:
```
∫ φ·∇²ω dV = ∫ ∇φ·∇ω dV  (by parts)
           = -∫ (∇²φ)·ω dV (by parts again)
           = -∫ ω² dV
```

So:
```
dK/dt|_viscous = -ν ∫ ω² dV
```

---

## Summary: Total Energy Dissipation

```
dH/dt = dK/dt + dU/dt
```

**Ideal parts cancel** (Hamiltonian structure).

**Dissipative parts:**
```
dH/dt = -η·(2π) ∫∫ J²·R dr dθ - ν·(2π) ∫∫ ω²·R dr dθ
```

**For test with ν = 0:**
```
dH/dt = -η·(2π) ∫∫ J²·R dr dθ
```

**Test code computes:**
```python
J2_int = np.sum(J**2 * dV)  # dV = R·dr·dθ
dH_theory = -eta * J2_int   # ❌ Missing 2π!
```

**Correct:**
```python
dH_theory = -eta * 2*np.pi * J2_int  # ✅
```

---

## Hypothesis

**Root cause:** Missing factor of **2π** in dissipation formula.

The 30x error suggests:
```
30 ≈ 1/(2π) × (some other factor?)
```

Actually:
```
1/(2π) ≈ 0.159 ≈ 1/6.28
```

But observed error is 0.03, which is:
```
0.03 ≈ 1/33
```

So it's not just 2π. Need to check:
1. Is `compute_hamiltonian` using φ or ω?
2. Is there a normalization issue in H computation?

---

## Next Steps

1. **Test Hypothesis 1:** Add 2π factor and re-run
2. **Code Audit:** Check if `compute_hamiltonian(psi, omega, grid)` is correct
3. **Should be:** `compute_hamiltonian(psi, phi, grid)` where φ = solve Poisson ∇²φ = ω

---

**STATUS:** Derivation complete. Hypothesis: missing 2π + potential ω/φ confusion.
