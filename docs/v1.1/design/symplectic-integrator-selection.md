# Symplectic Integrator Selection for PTM-RL v1.2

**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Phase:** 2.1 - Symplectic Integrator Selection  
**Status:** Design & Selection

---

## Executive Summary

**Recommendation:** Implement **Störmer-Verlet** (2nd-order symplectic) integrator for PTM-RL v1.2.

**Reasons:**
1. **Symplectic property** preserves phase-space structure → long-time stability
2. **Simple** to implement (3-step algorithm)
3. **2nd-order** sufficient for MHD (dt ~ 1e-4 already small)
4. **Proven** in molecular dynamics and plasma physics

**Expected improvement over RK4:**
- Energy drift: O(1)% → O(10⁻⁵)% over 10⁴ steps
- Phase-space structure preservation
- Reversibility

---

## 1. Theoretical Foundation

### 1.1 What is a Symplectic Integrator?

**Hamiltonian System:**
```
H(q, p) = kinetic energy + potential energy

Hamilton's equations:
  dq/dt = ∂H/∂p
  dp/dt = -∂H/∂q
```

**Symplectic Condition:**

A numerical integrator (qₙ, pₙ) → (qₙ₊₁, pₙ₊₁) is **symplectic** if it preserves the 2-form:

```
ω = dq ∧ dp
```

Geometrically: preserves phase-space volume (Liouville's theorem).

**Why this matters:**

**Non-symplectic (e.g., RK4):**
- Phase-space volume drifts
- Energy drift grows linearly: E(t) - E(0) ~ O(t)
- Long-time instability

**Symplectic (e.g., Störmer-Verlet):**
- Phase-space volume conserved
- Energy oscillates but bounded: |E(t) - E(0)| ~ O(dt²)
- Long-time stability guaranteed

---

### 1.2 Störmer-Verlet Algorithm

**Classic Hamiltonian: H = T(p) + V(q)**

Störmer-Verlet (velocity Verlet form):

```python
def stormer_verlet(q, p, dt, grad_V):
    """
    Symplectic integrator for separable Hamiltonian.
    
    H = p²/(2m) + V(q)
    
    Steps:
    1. Half-step momentum: p_{n+1/2} = p_n - (dt/2) * ∇V(q_n)
    2. Full-step position: q_{n+1} = q_n + dt * p_{n+1/2}/m
    3. Half-step momentum: p_{n+1} = p_{n+1/2} - (dt/2) * ∇V(q_{n+1})
    """
    p_half = p - 0.5 * dt * grad_V(q)
    q_new = q + dt * p_half / m
    p_new = p_half - 0.5 * dt * grad_V(q_new)
    return q_new, p_new
```

**Key properties:**
- **Symplectic:** Preserves dq ∧ dp (can be proven algebraically)
- **2nd-order:** Local error O(dt³), global error O(dt²)
- **Reversible:** Running backwards recovers initial state
- **Energy-preserving:** Hamiltonian oscillates around true value

---

### 1.3 Why RK4 is Not Symplectic

**RK4 Algorithm:**
```python
def rk4_step(y, dy_dt, dt):
    k1 = dy_dt(y)
    k2 = dy_dt(y + 0.5*dt*k1)
    k3 = dy_dt(y + 0.5*dt*k2)
    k4 = dy_dt(y + dt*k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

**Problem:** Generic RK methods (except special cases) do **not** preserve symplectic structure.

**Consequence:**
- Phase-space volume drifts
- Energy drift accumulates: E(t) - E(0) ~ c·t (linear growth)
- For long simulations (t >> 1), energy error becomes O(1)

**Example (molecular dynamics):**
- RK4 @ 10⁴ steps: ΔE/E ~ 1% (noticeable drift)
- Störmer-Verlet @ 10⁴ steps: ΔE/E ~ 10⁻⁵ (bounded oscillation)

---

## 2. Algorithm Comparison

| Feature | RK4 | Störmer-Verlet | 4th-order Symplectic (Forest-Ruth) |
|---------|-----|----------------|------------------------------------|
| **Order** | 4th | 2nd | 4th |
| **Symplectic** | ❌ No | ✅ Yes | ✅ Yes |
| **Energy drift** | O(t) | O(1) | O(1) |
| **Complexity** | Moderate | Simple | Complex |
| **Separability req** | No | Yes (T+V) | Yes (T+V) |
| **Stability** | Good | Excellent | Excellent |
| **Implementation** | Easy | Very easy | Moderate |

**Trade-offs:**

**RK4:**
- ✅ High accuracy per step (4th-order)
- ✅ No separability requirement
- ❌ Energy drift over long time
- ❌ Not structure-preserving

**Störmer-Verlet:**
- ✅ Symplectic (structure-preserving)
- ✅ Simple implementation
- ✅ Long-time stability
- ❌ Requires separable Hamiltonian H = T(p) + V(q)
- ❌ Lower order (2nd vs 4th)

**4th-order Symplectic:**
- ✅ Symplectic + high order
- ❌ Complex (multiple substeps)
- ❌ Slower per step
- ❌ Harder to implement

---

## 3. Application to Reduced MHD

### 3.1 Reduced MHD Equations

```
∂ψ/∂t = [φ, ψ] + η ∇²ψ
∂ω/∂t = [φ, ω] + [J, ψ] + ν ∇²ω
```

where:
- ψ: poloidal flux
- ω: vorticity (ω = ∇²φ)
- [·,·]: Poisson bracket
- J = -∇²ψ: parallel current

### 3.2 Hamiltonian Structure

**Ideal MHD (η=0, ν=0) has approximate Hamiltonian:**

```
H[ψ, ω] = ∫ (1/2 |∇φ|² + 1/2 |∇ψ|²) R dR dZ
         = kinetic energy + magnetic energy

Canonical variables:
  q = ψ
  p = -ω
```

**Hamilton's equations (ideal case):**
```
∂ψ/∂t = δH/δω  (kinetic contribution)
∂ω/∂t = -δH/δψ (magnetic contribution)
```

**With dissipation:**
```
∂ψ/∂t = δH/δω + η ∇²ψ  (resistivity)
∂ω/∂t = -δH/δψ + ν ∇²ω (viscosity)
```

### 3.3 Separability Analysis

**Challenge:** Reduced MHD is **not obviously separable** H ≠ T(ω) + V(ψ)

**Reason:** 
- Poisson brackets couple ψ and ω non-trivially
- [φ, ψ] involves solving ∇²φ = ω (implicit)

**Solutions:**

**Option A: Approximate Splitting**
- Split into "kinetic" (φ-related) and "magnetic" (ψ-related) parts
- Apply Störmer-Verlet to split system
- Accept small loss of exact symplecticity

**Option B: Treat as (q, q̇) system**
- Use ψ as q, ∂ψ/∂t as v (velocity)
- Apply Störmer-Verlet even though not canonical
- Energy conservation will be approximate but better than RK4

**Option C: Variational Integrator**
- Use discrete Euler-Lagrange equations
- Automatically symplectic (no Hamiltonian needed)
- More complex implementation

**Recommendation:** Start with **Option B** (simplest)

---

## 4. Design Decision

### 4.1 Selected Algorithm: Störmer-Verlet

**Reasons:**

1. **Symplectic property** most important for long-time stability
2. **Simple to implement** (3 lines of code)
3. **2nd-order sufficient** given dt ~ 1e-4
4. **Proven in plasma codes** (e.g., PIC simulations)

**Trade-off accepted:**
- Lower order (2nd vs RK4's 4th) → may need smaller dt
- **But:** Energy stability >> accuracy per step for RL

### 4.2 Implementation Strategy

**Störmer-Verlet for MHD:**

```python
def symplectic_step(psi, omega, dt, compute_rhs):
    """
    Störmer-Verlet applied to reduced MHD.
    
    Treat as:
      q = psi
      v = dpsi/dt
      a = d²psi/dt² (from omega equation)
    """
    # Step 1: Half-step omega (momentum-like)
    dpsi_dt, domega_dt = compute_rhs(psi, omega)
    omega_half = omega + 0.5 * dt * domega_dt
    
    # Step 2: Full-step psi (position-like)
    dpsi_dt_half, _ = compute_rhs(psi, omega_half)
    psi_new = psi + dt * dpsi_dt_half
    
    # Step 3: Half-step omega
    _, domega_dt_new = compute_rhs(psi_new, omega_half)
    omega_new = omega_half + 0.5 * dt * domega_dt_new
    
    return psi_new, omega_new
```

**Note:** This is a **leapfrog** scheme (psi and omega staggered by dt/2).

---

## 5. API Design

### 5.1 Interface Specification

```python
class SymplecticIntegrator:
    """
    Symplectic integrator for reduced MHD.
    
    Compatible interface with RK4Integrator for drop-in replacement.
    """
    
    def __init__(self, grid: ToroidalGrid, dt: float, 
                 eta: float = 1e-6, nu: float = 1e-6):
        """
        Parameters
        ----------
        grid : ToroidalGrid
            Spatial grid
        dt : float
            Time step
        eta : float
            Resistivity
        nu : float
            Viscosity
        """
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        
        # State
        self.psi = None
        self.omega = None
        self.t = 0.0
    
    def initialize(self, psi0: np.ndarray, omega0: np.ndarray):
        """Initialize fields."""
        self.psi = psi0.copy()
        self.omega = omega0.copy()
        self.t = 0.0
    
    def step(self) -> None:
        """Take one Störmer-Verlet step."""
        self.psi, self.omega = self._stormer_verlet_step(
            self.psi, self.omega, self.dt
        )
        self.t += self.dt
    
    def _stormer_verlet_step(self, psi, omega, dt):
        """Störmer-Verlet algorithm (internal)."""
        # Implementation as above
        pass
    
    def compute_rhs(self, psi, omega):
        """
        Compute RHS of reduced MHD equations.
        
        Returns
        -------
        dpsi_dt, domega_dt : np.ndarray
        """
        # Same as RK4 implementation
        pass
```

### 5.2 Drop-in Replacement Strategy

**Usage should be identical:**

```python
# Old (RK4)
solver = RK4Integrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)
solver.initialize(psi0, omega0)
solver.step()

# New (Symplectic) - exactly the same API!
solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)
solver.initialize(psi0, omega0)
solver.step()
```

**Interface compatibility:**
- Same constructor arguments
- Same `initialize()` method
- Same `step()` method
- Same state variables (`psi`, `omega`, `t`)

---

## 6. Validation Plan (Step 2.4)

### 6.1 Energy Conservation Test

**Setup:**
```python
# Initial condition: perturbed equilibrium
psi0 = solovev_equilibrium() + perturbation
omega0 = laplacian(psi0)

# Run both integrators
rk4 = RK4Integrator(grid, dt=1e-4)
sym = SymplecticIntegrator(grid, dt=1e-4)

for _ in range(10000):
    rk4.step()
    sym.step()
    
    E_rk4 = compute_energy(rk4.psi, rk4.omega)
    E_sym = compute_energy(sym.psi, sym.omega)
```

**Expected:**
- RK4: ΔE/E ~ 0.1-1% (drift)
- Symplectic: ΔE/E ~ 1e-5 (bounded oscillation)

### 6.2 Reversibility Test

**Symplectic integrators are time-reversible:**

```python
# Forward
psi0 = initial_condition()
solver.initialize(psi0, omega0)
for _ in range(1000):
    solver.step()
psi_forward = solver.psi.copy()

# Backward (negative dt)
solver.dt = -1e-4
for _ in range(1000):
    solver.step()
psi_reversed = solver.psi

# Should recover initial state
error = np.max(np.abs(psi_reversed - psi0))
assert error < 1e-10  # Machine precision
```

---

## 7. Known Limitations

### 7.1 Reduced MHD Not Exactly Hamiltonian

**Issue:** Poisson brackets introduce non-Hamiltonian structure

**Impact:**
- Symplectic integrator is approximate (not exact)
- Energy conservation good but not perfect

**Mitigation:**
- Acceptable for v1.2 (still >> RK4)
- Future: Variational integrator for exact symplecticity

### 7.2 Dissipation Breaks Symplecticity

**Issue:** η, ν terms are dissipative → not Hamiltonian

**Impact:**
- Energy should decrease (physically correct)
- Symplectic integrator approximates dissipative flow

**Mitigation:**
- Use operator splitting: symplectic step + dissipation step
- Or: accept approximate treatment (still better than RK4)

---

## 8. References

**Symplectic integrators:**
1. Hairer et al., "Geometric Numerical Integration" (2006)
2. Leimkuhler & Reich, "Simulating Hamiltonian Dynamics" (2004)

**MHD applications:**
3. Qin & Guan, "Variational symplectic integrator for long-time simulations of the guiding-center motion" (2008)
4. Shadwick et al., "Symplectic integration of extended magnetohydrodynamics" (2014)

**Molecular dynamics (classic application):**
5. Verlet, "Computer experiments on classical fluids" (1967)

---

## 9. Next Steps (Step 2.2)

1. **Hamiltonian formulation review**
   - Document H[ψ, ω] explicitly
   - Identify separability (if exists)

2. **Implementation (Step 2.3)**
   - Code `SymplecticIntegrator` class
   - Test basic functionality

3. **Validation (Step 2.4)**
   - Energy conservation test
   - Reversibility test
   - Comparison with RK4

---

**Sign-off:** 小P ⚛️  
**Status:** Design complete, ready for Step 2.2  
**Date:** 2026-03-18
