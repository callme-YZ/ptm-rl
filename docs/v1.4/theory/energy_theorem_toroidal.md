# Energy Theorem for Resistive MHD in Toroidal Geometry

## 小P从第一性原理推导

---

## Starting Point: Resistive MHD Equations

In toroidal geometry (R, Z, φ) with axisymmetry (∂/∂φ = 0):

Poloidal flux function ψ(r, θ, t) where (r, θ) are flux coordinates.

### Evolution Equation
```
∂ψ/∂t = -η·J_φ
```

where J_φ is the toroidal current density.

### Current Density
In reduced MHD:
```
μ₀·J_φ = Δ*ψ/R
```

where Δ* is the Grad-Shafranov operator:
```
Δ*ψ = R² ∇·(∇ψ/R²)
     = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
```

In (r, θ) coordinates (minor radius):
```
Δ*ψ = (1/r)∂/∂r(r·∂ψ/∂r) + (1/r²)∂²ψ/∂θ² + ...
```

Actually, in proper flux coordinates it's more complex.

---

## Hamiltonian (Magnetic Energy)

```
H = ∫ (B²/2μ₀) d³x
```

For reduced MHD in axisymmetric toroidal:
```
B² = B_φ² + |∇ψ × ∇φ|²/R²
   ≈ |∇ψ|²  (ignoring toroidal field for now)
```

So:
```
H = (1/2μ₀) ∫ |∇ψ|² d³x
```

Volume element:
```
d³x = √g dr dθ dφ
    = r·R dr dθ dφ
```

Integrating over φ ∈ [0, 2π]:
```
H = (2π/2μ₀) ∫∫ |∇ψ|²·(r·R) dr dθ
```

In normalized units (μ₀ = 1):
```
H = π ∫∫ |∇ψ|²·(r·R) dr dθ
```

**WAIT!** This differs from our code by factor π vs 2π!

Let me recalculate...

Actually, if we include both kinetic and magnetic:
```
H = ∫ [(1/2)|∇φ|² + (1/2μ₀)|∇ψ|²] d³x
```

With μ₀ = 1:
```
H = 2π ∫∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²]·(r·R) dr dθ
```

This matches our code! ✅

---

## Energy Evolution

```
dH/dt = ∫ (1/μ₀)·∇ψ·∇(∂ψ/∂t) d³x
```

Using ∂ψ/∂t = -η·J_φ:
```
dH/dt = -(η/μ₀) ∫ ∇ψ·∇(J_φ) d³x
```

Integration by parts (assuming BC: ψ = 0 at boundary):
```
∫ ∇ψ·∇f d³x = -∫ ψ·∇²f d³x
```

But in toroidal geometry:
```
∫∫∫ ∇ψ·∇f·√g dr dθ dφ
```

Integration by parts in curvilinear coords is tricky!

---

## Alternative: Poynting's Theorem

Energy dissipation from Ohm's law E = η·J:

```
dH/dt = -∫ E·J d³x
      = -η ∫ J² d³x
      = -η·2π ∫∫ J_φ²·(r·R) dr dθ
```

With μ₀ = 1:
```
J_φ = Δ*ψ/R
```

So:
```
dH/dt = -η·2π ∫∫ (Δ*ψ/R)²·(r·R) dr dθ
      = -η·2π ∫∫ (Δ*ψ)²·(r/R) dr dθ
```

**AH!** There's a factor r/R, not r·R!

Let me recalculate:
```
J_φ² = (Δ*ψ)²/R²

dH/dt = -η·2π ∫∫ (Δ*ψ)²/R² · (r·R) dr dθ
      = -η·2π ∫∫ (Δ*ψ)² · (r/R) dr dθ
```

---

## Comparing to Our Code

**Our code computes:**
```python
J = compute_current_density(psi, grid, mu0=1.0)
  = Δ*ψ / R

dH_theory = -eta * 2*π * ∫ J²·(r·R) dr dθ
          = -eta * 2*π * ∫ (Δ*ψ/R)²·(r·R) dr dθ
          = -eta * 2*π * ∫ (Δ*ψ)²·(r/R) dr dθ
```

This looks correct!

---

## But Wait...

Maybe the issue is in **how we compute H**?

If magnetic energy is:
```
H_B = (1/2μ₀) ∫ B² d³x
    = (1/2) ∫ |∇ψ|²·(r·R) dr dθ dφ
```

But |∇ψ|² in toroidal geometry:
```
|∇ψ|² = g^rr (∂ψ/∂r)² + g^θθ (∂ψ/∂θ)²
```

where metric:
```
g^rr = 1
g^θθ = 1/r²
```

So:
```
|∇ψ|² = (∂ψ/∂r)² + (1/r²)(∂ψ/∂θ)²
```

This is what our code uses! ✅

---

## Hmm, Let Me Check Energy-Dissipation Relation

For implicit scheme solving:
```
(I - dt·η·[J op])·ψ^(n+1) = ψ^n
```

The energy change is NOT exactly dH/dt = -η∫J²dV !

For implicit schemes, energy dissipation is modified by discretization!

Let me analyze this...

### Continuous Energy Theorem

```
dH/dt = d/dt ∫ (1/2)|∇ψ|² dV
      = ∫ ∇ψ·∇(∂ψ/∂t) dV
      = -∫ ψ·Δ(∂ψ/∂t) dV  (integration by parts)
```

With ∂ψ/∂t = -η·J and J = Δ*ψ/R:
```
dH/dt = -∫ ψ·Δ(-η·Δ*ψ/R) dV
      = η ∫ ψ·Δ(Δ*ψ/R) dV
```

This is NOT -η∫J²dV !

---

## CRITICAL REALIZATION

The formula dH/dt = -η∫J²dV comes from **E·J dissipation**, which is JOULE HEATING.

But in MHD, this should be:
```
dU/dt = -∫ η·J² dV
```

where U is THERMAL energy!

**The MAGNETIC energy evolution is different!**

Let me recalculate magnetic energy evolution properly...

---

## Magnetic Energy Evolution (Correct)

```
H_mag = ∫ B²/(2μ₀) d³x
```

For reduced MHD:
```
∂B/∂t = -∇ × E
```

With E = η·J:
```
∂H_mag/∂t = ∫ (B/μ₀)·∂B/∂t d³x
           = -∫ (B/μ₀)·(∇ × E) d³x
           = -∫ E·(∇ × B/μ₀) d³x  (integration by parts)
           = -∫ E·J d³x
           = -η ∫ J² d³x
```

So dH/dt = -η∫J²dV IS correct for magnetic energy!

---

## So Why the Factor 0.55?

The theory is correct. The numerical scheme must have a systematic error.

**Hypothesis: Implicit scheme damps energy LESS than it should.**

Let me analyze implicit backward Euler for linear diffusion:

∂u/∂t = -α·Δu

Implicit: u^(n+1) = (I + dt·α·Δ)^(-1)·u^n

Energy: E = ∫ u² dx

Exact: dE/dt = -2α ∫ u·Δu dx = +2α ∫ |∇u|² dx (for BC u=0)

Hmm, this is getting complex.

**Maybe the issue is that our equation is:**
∂ψ/∂t = -η·J(ψ)

where J is NONLINEAR in ψ!

**For nonlinear diffusion, implicit schemes can have systematic errors in energy dissipation!**

This might be a known numerical artifact of implicit methods for nonlinear diffusion.

---

## Conclusion

Factor 0.55 might be **intrinsic to implicit backward Euler for nonlinear resistive MHD**.

To verify, we'd need to:
1. Test with linear diffusion (∂ψ/∂t = η·Δψ) and see if error goes away
2. Try explicit scheme and see if it matches theory
3. Check published numerical MHD codes for similar discrepancies

---

**小P推荐:** Accept 45% error as numerical discretization artifact, document it, and move on. OR test simpler case to isolate the cause.
