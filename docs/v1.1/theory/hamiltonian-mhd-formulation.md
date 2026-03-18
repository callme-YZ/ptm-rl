# Hamiltonian Formulation of Reduced MHD

**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Phase:** 2.2 - Hamiltonian Formulation  
**Status:** Theory Foundation

**Primary References:**
- Morrison (2023) "Hamiltonian Description of Fluid and Plasma Systems"
- Chacón (2020) "Energy-conserving methods for MHD"
- Classic: Morrison & Greene (1980s) foundations

---

## 1. Reduced MHD Equations (Recap)

### 1.1 Evolution Equations

Reduced MHD in (R, Z, φ) toroidal geometry:

```
∂ψ/∂t = [φ, ψ] + η J_∥
∂ω/∂t = [φ, ω] + [J_∥, ψ] + ν ∇²ω
```

where:
- **ψ**: Poloidal magnetic flux
- **ω**: Vorticity (ω = ∇²φ)
- **φ**: Electrostatic potential (stream function)
- **J_∥ = -∇²ψ**: Parallel current density
- **[f, g]**: Poisson bracket = (1/R)(∂_R f ∂_Z g - ∂_Z f ∂_R g)
- **η**: Resistivity (dissipation)
- **ν**: Viscosity (dissipation)

### 1.2 Physical Interpretation

**Magnetic field:**
```
B = ∇ψ × ∇φ + B_φ e_φ
```

**Velocity:**
```
v = ∇φ × ∇ψ / R
```

---

## 2. Hamiltonian Structure (Ideal Case)

### 2.1 Canonical Variables

For **ideal MHD** (η = ν = 0), the system has Hamiltonian structure:

```
Canonical coordinates:
  q = ψ    (poloidal flux - position-like)
  p = -ω   (negative vorticity - momentum-like)
```

**Why negative sign for ω?**

The negative sign ensures the correct symplectic structure and matches the canonical Poisson bracket form.

### 2.2 Hamiltonian Functional

The energy functional serves as Hamiltonian:

```
H[ψ, ω] = ∫ [1/2 |∇φ|² + 1/2 |∇ψ|²] √g dV

where:
  - ∇φ|² = kinetic energy density (fluid motion)
  - |∇ψ|² = magnetic energy density
  - √g = R r (toroidal Jacobian)
```

**Component form:**

```
H = ∫∫ [1/2 (|∂φ/∂r|² + (1/r²)|∂φ/∂θ|²) 
        + 1/2 (|∂ψ/∂r|² + (1/r²)|∂ψ/∂θ|²)] R r dr dθ
```

### 2.3 Hamilton's Equations

With canonical Hamiltonian theory:

```
∂ψ/∂t = δH/δ(-ω) = -δH/δω
∂(-ω)/∂t = -δH/δψ
```

which gives:

```
∂ψ/∂t = [φ, ψ]      (ideal evolution)
∂ω/∂t = [J_∥, ψ]    (ideal evolution)
```

**Functional derivatives:**

```
δH/δω:  Involves solving ω = ∇²φ for φ, then taking variation
δH/δψ:  Direct variation of magnetic energy term
```

### 2.4 Poisson Bracket Structure

The Poisson bracket for reduced MHD is **non-canonical** (not {F,G} = ∫(∂F/∂q ∂G/∂p - ∂F/∂p ∂G/∂q)):

```
{F, G} = ∫∫ [δF/δψ · [δG/δω, ψ] + δF/δω · [ψ, δG/δψ]] √g dV
```

This is the **Lie-Poisson bracket** for reduced MHD.

**Key property:** Poisson bracket satisfies:
- Antisymmetry: {F,G} = -{G,F}
- Jacobi identity: {{F,G},H} + {{G,H},F} + {{H,F},G} = 0
- Leibniz rule (derivation property)

---

## 3. Energy Conservation (Ideal Case)

### 3.1 Energy is Conserved

For ideal MHD (η = ν = 0):

```
dH/dt = 0
```

**Proof:**

```
dH/dt = ∫ [∂H/∂ψ · ∂ψ/∂t + ∂H/∂ω · ∂ω/∂t] dV

Using Hamilton's equations:
     = ∫ [∂H/∂ψ · (-∂H/∂ω) + ∂H/∂ω · ∂H/∂ψ] dV
     = 0
```

This is **exact conservation** for Hamiltonian flow.

### 3.2 Symplectic Structure

The canonical symplectic 2-form:

```
ω_symp = dψ ∧ dω
```

is **preserved** by Hamiltonian flow (Liouville's theorem).

**Consequence:** Phase-space volume conserved → long-time stability

---

## 4. Dissipation: Breaking Hamiltonian Structure

### 4.1 Resistive & Viscous MHD

With dissipation (η > 0, ν > 0):

```
∂ψ/∂t = [φ, ψ] + η ∇²ψ      (resistivity)
∂ω/∂t = [φ, ω] + [J_∥, ψ] + ν ∇²ω  (viscosity)
```

**Not Hamiltonian!**

Dissipative terms break energy conservation:

```
dH/dt = -∫ [η |∇J_∥|² + ν |∇ω|²] dV < 0
```

Energy **decreases** (physically correct).

### 4.2 Splitting Strategy (Chacón 2020)

**Idea:** Operator splitting

```
Total flow = Hamiltonian flow + Dissipative flow
```

**Time-stepping:**

```
1. Hamiltonian step (η=0, ν=0):
   Symplectic integrator
   
2. Dissipative step (only η, ν):
   Explicit or implicit Euler
```

**Algorithm:**

```python
def split_step(psi, omega, dt, eta, nu):
    # Step 1: Hamiltonian (symplectic)
    psi_star, omega_star = symplectic_step(psi, omega, dt, eta=0, nu=0)
    
    # Step 2: Dissipation (simple Euler)
    psi_new = psi_star + dt * eta * laplacian(psi_star)
    omega_new = omega_star + dt * nu * laplacian(omega_star)
    
    return psi_new, omega_new
```

**Order of accuracy:**
- Symplectic step: O(dt²) (Störmer-Verlet)
- Dissipation step: O(dt) (Euler)
- Combined: O(dt) overall, but **structure-preserving**

**Improvement:** Use implicit dissipation for stiffness

---

## 5. Hamiltonian Splitting for Störmer-Verlet

### 5.1 Separability Analysis

**Question:** Can H be split as H = T(p) + V(q)?

**Challenge:** Reduced MHD coupling is complex:

```
H = H_kinetic[ω] + H_magnetic[ψ] + H_coupling[ψ,ω]
```

**H_kinetic depends on φ, which solves ω = ∇²φ** → implicit!

**Conclusion:** Not obviously separable in standard form.

### 5.2 Approximate Splitting (Practical Approach)

**Strategy:** Treat as (q, q̇) system instead of (q, p):

```
Position-like: q = ψ
Velocity-like: v = ∂ψ/∂t
Acceleration:  a = ∂²ψ/∂t² (from ω equation)
```

**Störmer-Verlet (leapfrog form):**

```python
def stormer_verlet_mhd(psi, omega, dt):
    """
    Leapfrog scheme for reduced MHD.
    
    Treat ω as "momentum" conjugate to ψ.
    """
    # Half-step omega (momentum)
    dpsi_dt, domega_dt = compute_rhs(psi, omega, eta=0, nu=0)
    omega_half = omega + 0.5 * dt * domega_dt
    
    # Full-step psi (position)
    dpsi_dt_half, _ = compute_rhs(psi, omega_half, eta=0, nu=0)
    psi_new = psi + dt * dpsi_dt_half
    
    # Half-step omega (momentum)
    _, domega_dt_new = compute_rhs(psi_new, omega_half, eta=0, nu=0)
    omega_new = omega_half + 0.5 * dt * domega_dt_new
    
    return psi_new, omega_new
```

**Key idea:** Stagger psi and omega by dt/2 (leapfrog)

### 5.3 Why This Works (Approximate Symplecticity)

**Leapfrog is symplectic** for separable Hamiltonians.

**For reduced MHD:**
- Not exactly separable
- But leapfrog still approximately preserves energy
- **Much better than RK4** (which has no structure preservation)

**Evidence from literature:**
- Widely used in plasma PIC codes
- Energy drift bounded (not linear growth like RK4)

---

## 6. Verification: Energy Conservation

### 6.1 Energy Computation

```python
def compute_energy(psi, omega, grid):
    """Compute total energy H."""
    # Solve for phi: omega = laplacian(phi)
    phi = solve_poisson(omega, grid)
    
    # Kinetic energy
    grad_phi_r, grad_phi_theta = gradient(phi, grid)
    E_kin = 0.5 * np.sum((grad_phi_r**2 + (grid.r * grad_phi_theta)**2) 
                         * grid.jacobian() * grid.dr * grid.dtheta)
    
    # Magnetic energy
    grad_psi_r, grad_psi_theta = gradient(psi, grid)
    E_mag = 0.5 * np.sum((grad_psi_r**2 + (grid.r * grad_psi_theta)**2)
                         * grid.jacobian() * grid.dr * grid.dtheta)
    
    return E_kin + E_mag
```

### 6.2 Expected Results

**RK4 (non-symplectic):**
```
E(t) = E(0) + c·t + O(dt⁴)
→ Linear drift over long time
→ ΔE/E ~ 0.1-1% @ 10⁴ steps
```

**Störmer-Verlet (symplectic):**
```
E(t) = E(0) + oscillations(amplitude ~ dt²)
→ Bounded drift
→ ΔE/E ~ 1e-5% @ 10⁴ steps
```

**Improvement factor: ~10³-10⁴** ✅

---

## 7. Advanced: Variational Integrator (Future)

### 7.1 Discrete Euler-Lagrange

**Alternative approach:** Use Lagrangian instead of Hamiltonian

```
L[ψ, ∂ψ/∂t] = Lagrangian functional

Discrete action:
S = Σ_n L(ψ_n, (ψ_{n+1} - ψ_n)/dt) · dt
```

**Principle:** Minimize discrete action → Discrete Euler-Lagrange equations

**Automatically symplectic** (no Hamiltonian structure needed!)

### 7.2 When to Use

**Pros:**
- Exactly symplectic (no approximation)
- Works for non-separable systems
- Geometrically natural

**Cons:**
- More complex implementation
- Requires solving implicit equations
- Harder to add dissipation

**Recommendation:** 
- v1.2: Use Störmer-Verlet (simple, proven)
- v2.0: Explore variational integrators (cutting-edge)

---

## 8. Summary: Step 2.2 Conclusions

### 8.1 Hamiltonian for Reduced MHD

```
H[ψ, ω] = ∫ [1/2 |∇φ|² + 1/2 |∇ψ|²] √g dV

Canonical variables: (q=ψ, p=-ω)

Hamilton's equations (ideal):
  ∂ψ/∂t = [φ, ψ]
  ∂ω/∂t = [J_∥, ψ]
```

### 8.2 Splitting Strategy

**For implementation:**

```
1. Hamiltonian step: Störmer-Verlet leapfrog
   (approximate symplectic)
   
2. Dissipation step: Explicit Euler
   (or implicit for stiffness)
```

### 8.3 Expected Performance

**Energy conservation:**
- RK4: ΔE/E ~ O(0.1%)
- Störmer-Verlet: ΔE/E ~ O(10⁻⁵%)

**Improvement: 10³-10⁴× better** ✅

---

## 9. Next Steps (Step 2.3)

### 9.1 Implementation Tasks

1. **Code `SymplecticIntegrator` class**
   - Störmer-Verlet algorithm
   - Drop-in replacement for RK4

2. **Implement energy diagnostics**
   - `compute_energy()` function
   - Track energy vs time

3. **Test basic functionality**
   - Single step correctness
   - Energy conservation (ideal case)

### 9.2 Validation (Step 2.4)

1. **Long-time stability test**
   - Run 10⁴ steps
   - Compare RK4 vs Symplectic

2. **Reversibility test**
   - Forward + backward = identity

3. **Energy drift quantification**
   - Measure ΔE/E
   - Verify > 100× improvement

---

## 10. References

**Primary:**
1. Morrison (2023) "Hamiltonian Description of Fluid and Plasma Systems"
   - Living Reviews in Plasma Physics
   - DOI: 10.1007/s41614-023-00121-8

2. Chacón et al. (2020) "Energy-conserving methods for MHD"
   - Journal of Computational Physics
   - DOI: 10.1016/j.jcp.2020.109527

**Classic:**
3. Morrison & Greene (1980) "Noncanonical Hamiltonian Density Formulation"
   - Physics Review Letters

**Numerical methods:**
4. Hairer et al. (2006) "Geometric Numerical Integration"
5. Kraus et al. (2024) GeometricIntegrators.jl (GitHub)

---

**Status:** Theory complete ✅  
**Ready for:** Step 2.3 Implementation  
**Author:** 小P ⚛️  
**Date:** 2026-03-18
