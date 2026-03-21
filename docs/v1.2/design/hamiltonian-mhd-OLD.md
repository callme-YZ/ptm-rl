# Hamiltonian MHD Formulation Design

**Version:** v1.2  
**Author:** 小P ⚛️  
**Date:** 2026-03-18  
**Status:** Design Phase

---

## Executive Summary

This document describes the addition of **Hamiltonian physics** to the v1.1 reduced MHD solver, enabling proper use of symplectic time integration and realistic plasma dynamics.

**Key additions:**
1. **Poisson bracket** [ψ, φ] — Nonlinear advection
2. **Stream function solver** ∇²φ = -ω — Elliptic problem
3. **Strang splitting** — Hamiltonian + dissipation
4. **Energy conservation verification** — Ideal limit testing

**Goal:** Transform v1.1's pure diffusion into a true Hamiltonian system with dissipation as perturbation.

---

## 1. Physics Background

### 1.1 Current v1.1 (Pure Diffusion)

**Equations:**
$$
\frac{\partial \psi}{\partial t} = \eta \nabla^2 \psi
$$

$$
\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega
$$

**Limitations:**
- **No dynamics** — Only diffusive decay
- **No wave propagation** — No Alfvén waves, no instabilities
- **Symplectic integrator wasted** — Designed for Hamiltonian, not diffusion
- **Unrealistic** — Real plasma has advection, not just diffusion

### 1.2 Target v1.2 (Hamiltonian + Dissipation)

**Full reduced MHD:**
$$
\frac{\partial \psi}{\partial t} = [\psi, \phi] + \eta \nabla^2 \psi
$$

$$
\frac{\partial \omega}{\partial t} = [\omega, \phi] + \nu \nabla^2 \omega
$$

where:
- $[\cdot, \cdot]$ = Poisson bracket (advection term)
- $\phi$ = stream function (velocity potential)
- $\eta, \nu$ = resistivity, viscosity (dissipation)

**Constraint:**
$$
\nabla^2 \phi = -\omega
$$

(Elliptic problem solved at each time step)

### 1.3 Hamiltonian Structure

**Energy functional:**
$$
H[\psi, \omega] = \int_V \left( \frac{1}{2}|\nabla \psi|^2 + \frac{1}{2}\omega^2 \right) dV
$$

**Interpretation:**
- $|\nabla \psi|^2$ — Magnetic energy
- $\omega^2$ — Kinetic energy (vorticity squared)

**Canonical variables:**
- $(\psi, \omega)$ form a conjugate pair
- Symplectic form: $\{\psi(x), \omega(y)\} = \delta(x - y)$

**Ideal evolution (η=ν=0):**
$$
\frac{\partial \psi}{\partial t} = \frac{\delta H}{\delta \omega}, \quad \frac{\partial \omega}{\partial t} = -\frac{\delta H}{\delta \psi}
$$

**With dissipation:**
$$
\frac{dH}{dt} = -\int_V \left( \eta |\nabla \psi|^2 + \nu |\nabla \omega|^2 \right) dV \leq 0
$$

Energy decays, but Hamiltonian structure preserved.

### 1.4 Why Hamiltonian Matters for v1.2

**Physics:**
- **Realistic dynamics** — Tearing modes, Kelvin-Helmholtz, Alfvén waves
- **Energy conservation** — In ideal limit, numerically exact
- **Long-time accuracy** — No secular drift in phase space

**Numerics:**
- **Symplectic integrator justified** — Störmer-Verlet designed for this!
- **Geometric preservation** — Phase space volume conserved
- **Stability** — Better than RK4 for Hamiltonian systems

**RL implications:**
- **Realistic control problem** — Can stabilize actual instabilities
- **Transferable strategies** — Learned policies physical, not artifact of solver

---

## 2. Poisson Bracket Implementation

### 2.1 Definition

**General form:**
$$
[A, B] = \frac{\partial A}{\partial x} \frac{\partial B}{\partial y} - \frac{\partial A}{\partial y} \frac{\partial B}{\partial x}
$$

**In toroidal coordinates $(r, \theta, \phi)$:**

For reduced MHD (2D in poloidal plane):
$$
[A, B] = \frac{1}{r(R_0 + r\cos\theta)} \left( \frac{\partial A}{\partial r} \frac{\partial B}{\partial \theta} - \frac{\partial A}{\partial \theta} \frac{\partial B}{\partial r} \right)
$$

**Metric factor:** $1/(r R)$ where $R = R_0 + r\cos\theta$

### 2.2 Finite-Difference Schemes

**Option A: Arakawa Scheme (Recommended)**

Conserves energy AND enstrophy discretely:
$$
[A, B]_{i,j} = \frac{1}{12\Delta r \Delta \theta} \left[ 
  (A_{i+1,j} - A_{i-1,j})(B_{i,j+1} - B_{i,j-1}) - (A_{i,j+1} - A_{i,j-1})(B_{i+1,j} - B_{i-1,j})
\right] + \ldots
$$

(Full stencil involves 9 points)

**Pros:**
- Energy/enstrophy conserving
- Stable for long-time integration
- Minimal numerical dissipation

**Cons:**
- Complex implementation (9-point stencil)
- Requires careful boundary handling

**Option B: Simple Centered Differences**

$$
\frac{\partial A}{\partial r} \approx \frac{A_{i+1,j} - A_{i-1,j}}{2\Delta r}
$$

**Pros:**
- Simple to implement
- Fast

**Cons:**
- Does not conserve discrete energy
- May accumulate errors over long time

**小P Recommendation:** Start with Option B for quick implementation, upgrade to Arakawa if energy drift becomes issue.


### 2.3 Implementation Code

```python
def poisson_bracket_simple(A, B, grid):
    """
    Compute [A, B] using centered differences.
    
    Parameters
    ----------
    A, B : np.ndarray (nr, ntheta)
        Scalar fields
    grid : ToroidalGrid
    
    Returns
    -------
    bracket : np.ndarray (nr, ntheta)
        [A, B] field
    """
    dr = grid.dr
    dtheta = grid.dtheta
    r = grid.r_grid
    R = grid.R_grid  # R0 + r*cos(theta)
    
    # Gradients
    dA_dr = np.zeros_like(A)
    dA_dr[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2*dr)
    dA_dr[0, :] = (A[1, :] - A[0, :]) / dr  # One-sided
    dA_dr[-1, :] = (A[-1, :] - A[-2, :]) / dr
    
    dA_dtheta = np.zeros_like(A)
    dA_dtheta[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2*dtheta)
    # Periodic in theta
    dA_dtheta[:, 0] = (A[:, 1] - A[:, -1]) / (2*dtheta)
    dA_dtheta[:, -1] = (A[:, 0] - A[:, -2]) / (2*dtheta)
    
    # Same for B
    dB_dr = np.zeros_like(B)
    dB_dr[1:-1, :] = (B[2:, :] - B[:-2, :]) / (2*dr)
    dB_dr[0, :] = (B[1, :] - B[0, :]) / dr
    dB_dr[-1, :] = (B[-1, :] - B[-2, :]) / dr
    
    dB_dtheta = np.zeros_like(B)
    dB_dtheta[:, 1:-1] = (B[:, 2:] - B[:, :-2]) / (2*dtheta)
    dB_dtheta[:, 0] = (B[:, 1] - B[:, -1]) / (2*dtheta)
    dB_dtheta[:, -1] = (B[:, 0] - B[:, -2]) / (2*dtheta)
    
    # Poisson bracket with metric factor
    bracket = (dA_dr * dB_dtheta - dA_dtheta * dB_dr) / (r * R)
    
    return bracket
```

### 2.4 Validation Tests

**Test 1: Anti-symmetry**
$$
[A, B] = -[B, A]
$$

```python
def test_antisymmetry():
    A = np.random.randn(nr, ntheta)
    B = np.random.randn(nr, ntheta)
    
    AB = poisson_bracket(A, B, grid)
    BA = poisson_bracket(B, A, grid)
    
    assert np.allclose(AB, -BA, atol=1e-10)
```

**Test 2: Analytical Case**
$$
[r^2, \theta] = \frac{2r}{r R} \frac{\partial \theta}{\partial \theta} = \frac{2}{R}
$$

```python
def test_analytical():
    r = grid.r_grid
    theta = grid.theta_grid
    R = grid.R_grid
    
    A = r**2
    B = theta
    
    bracket = poisson_bracket(A, B, grid)
    expected = 2.0 / R
    
    assert np.allclose(bracket, expected, rtol=1e-3)
```

**Test 3: Jacobi Identity**
$$
[A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
$$

```python
def test_jacobi():
    A = np.random.randn(nr, ntheta)
    B = np.random.randn(nr, ntheta)
    C = np.random.randn(nr, ntheta)
    
    term1 = poisson_bracket(A, poisson_bracket(B, C, grid), grid)
    term2 = poisson_bracket(B, poisson_bracket(C, A, grid), grid)
    term3 = poisson_bracket(C, poisson_bracket(A, B, grid), grid)
    
    jacobi = term1 + term2 + term3
    
    assert np.max(np.abs(jacobi)) < 1e-6  # Should be ~zero
```

---

## 3. Stream Function Solver

### 3.1 The Elliptic Problem

At each time step, need to solve:
$$
\nabla^2 \phi = -\omega
$$

in toroidal geometry:
$$
\frac{1}{r}\frac{\partial}{\partial r}\left(r \frac{\partial \phi}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 \phi}{\partial \theta^2} = -\omega
$$

**Boundary conditions:**
- Periodic in $\theta$: $\phi(r, 0) = \phi(r, 2\pi)$
- At $r=0$ (axis): Regularity
- At $r=a$ (edge): $\phi = 0$ (Dirichlet) or $\partial\phi/\partial r = 0$ (Neumann)

### 3.2 Solver Options Comparison

| Method | Complexity | Accuracy | Flexibility | Recommendation |
|--------|------------|----------|-------------|----------------|
| **FFT** | O(N log N) | High | Periodic BC only | ✅ v1.2 |
| **CG** | O(N²) worst | Medium | General BC | v1.3 free-boundary |
| **Multigrid** | O(N) | High | General | v2.0 production |

**小P Choice for v1.2:** **FFT-based direct solver**

**Rationale:**
- Toroidal geometry naturally periodic in $\theta$
- Fast and accurate
- Simple implementation
- No convergence issues (direct method)


### 3.3 FFT Implementation

```python
import numpy as np
from numpy.fft import fft, ifft, fftfreq

def solve_stream_function_fft(omega, grid):
    """
    Solve ∇²φ = -ω using FFT in θ direction, finite-difference in r.
    
    Parameters
    ----------
    omega : np.ndarray (nr, ntheta)
        Vorticity field
    grid : ToroidalGrid
    
    Returns
    -------
    phi : np.ndarray (nr, ntheta)
        Stream function
    """
    nr, ntheta = omega.shape
    dr = grid.dr
    r = grid.r_grid
    
    # FFT in theta direction
    omega_k = fft(omega, axis=1)
    phi_k = np.zeros_like(omega_k, dtype=complex)
    
    # Wavenumbers
    k_theta = fftfreq(ntheta, d=1.0/ntheta) * 2*np.pi
    
    # Solve for each Fourier mode
    for m in range(ntheta):
        k = k_theta[m]
        
        # Radial ODE: (1/r d/dr(r dφ/dr) - k²/r² φ) = -ω_k
        # Rewrite as: d²φ/dr² + (1/r) dφ/dr - (k²/r²) φ = -ω_k
        
        # Build tri-diagonal matrix
        A = np.zeros((nr, nr))
        b = -omega_k[:, m]
        
        for i in range(1, nr-1):
            ri = r[i, 0]  # Radial coordinate
            
            # Second derivative: (φ_{i+1} - 2φ_i + φ_{i-1}) / dr²
            # First derivative: (φ_{i+1} - φ_{i-1}) / (2dr)
            # k² term: -k²/r² φ_i
            
            A[i, i-1] = 1.0/dr**2 - 1.0/(2*ri*dr)
            A[i, i] = -2.0/dr**2 - k**2/ri**2
            A[i, i+1] = 1.0/dr**2 + 1.0/(2*ri*dr)
        
        # Boundary conditions
        A[0, 0] = 1.0  # φ(r=0) = 0 (or regularity condition)
        A[-1, -1] = 1.0  # φ(r=a) = 0 (Dirichlet)
        b[0] = 0.0
        b[-1] = 0.0
        
        # Solve
        phi_k[:, m] = np.linalg.solve(A, b)
    
    # Inverse FFT
    phi = np.real(ifft(phi_k, axis=1))
    
    return phi
```

**Performance:**
- O(N log N) per time step
- For 64×128 grid: ~10 ms/step

### 3.4 Validation Tests

**Test 1: Known Solution**
$$
\omega = \sin(k\theta) \Rightarrow \phi = -\frac{\sin(k\theta)}{k^2}
$$

```python
def test_known_solution():
    k = 3
    theta = grid.theta_grid
    omega = np.sin(k * theta)
    
    phi = solve_stream_function_fft(omega, grid)
    phi_expected = -np.sin(k * theta) / k**2
    
    error = np.max(np.abs(phi - phi_expected))
    assert error < 1e-6
```

**Test 2: Round-trip**
$$
\nabla^2 (\text{solve}(\omega)) = \omega
$$

```python
def test_roundtrip():
    omega = np.random.randn(nr, ntheta)
    phi = solve_stream_function_fft(omega, grid)
    omega_recovered = laplacian_toroidal(phi, grid)
    
    error = np.max(np.abs(omega_recovered + omega))  # Note sign
    assert error < 1e-4
```

**Test 3: Energy Identity**
$$
\int \phi \omega \, dV = -\int |\nabla \phi|^2 \, dV
$$

```python
def test_energy_identity():
    omega = np.random.randn(nr, ntheta)
    phi = solve_stream_function_fft(omega, grid)
    
    # LHS: ∫ φω dV
    lhs = np.sum(phi * omega) * grid.dr * grid.dtheta
    
    # RHS: -∫ |∇φ|² dV
    grad_phi = gradient_toroidal(phi, grid)
    rhs = -np.sum(grad_phi[0]**2 + grad_phi[1]**2) * grid.dr * grid.dtheta
    
    assert np.abs(lhs - rhs) / np.abs(lhs) < 0.01  # 1% tolerance
```

---

## 4. Symplectic Integration with Dissipation

### 4.1 The Challenge

**Ideal part (Hamiltonian):**
$$
\frac{\partial \psi}{\partial t} = [\psi, \phi]
$$

**Dissipative part:**
$$
\frac{\partial \psi}{\partial t} = \eta \nabla^2 \psi
$$

**Problem:** Symplectic integrators preserve phase space volume, which is incompatible with dissipation (volume contracts).

**Solution:** Operator splitting.

### 4.2 Strang Splitting (Recommended)

**Idea:** Alternate half-steps of dissipation and full-step of Hamiltonian.

$$
\psi^{n+1} = e^{\frac{\Delta t}{2} D} \circ e^{\Delta t H} \circ e^{\frac{\Delta t}{2} D} \, \psi^n
$$

where:
- $e^{\Delta t H}$ = Hamiltonian evolution (symplectic)
- $e^{\Delta t D}$ = Dissipation evolution (explicit or implicit)

**Accuracy:** 2nd-order in $\Delta t$

**Implementation:**

```python
def step_strang(psi, omega, dt, grid):
    """One time step with Strang splitting."""
    
    # Half-step dissipation
    psi_half = psi + (dt/2) * eta * laplacian_toroidal(psi, grid)
    omega_half = omega + (dt/2) * nu * laplacian_toroidal(omega, grid)
    
    # Full-step Hamiltonian (symplectic)
    psi_star, omega_star = symplectic_step(psi_half, omega_half, dt, grid)
    
    # Half-step dissipation
    psi_new = psi_star + (dt/2) * eta * laplacian_toroidal(psi_star, grid)
    omega_new = omega_star + (dt/2) * nu * laplacian_toroidal(omega_star, grid)
    
    return psi_new, omega_new
```


### 4.3 Symplectic Substep (Störmer-Verlet)

For Hamiltonian part: $\partial_t \psi = [\psi, \phi]$, $\partial_t \omega = [\omega, \phi]$

**Störmer-Verlet:**
```python
def symplectic_step(psi, omega, dt, grid):
    # Half-step for psi
    phi = solve_stream_function_fft(omega, grid)
    psi_half = psi + (dt/2) * poisson_bracket(psi, phi, grid)
    
    # Full-step for omega
    phi_half = solve_stream_function_fft(omega, grid)
    omega_new = omega + dt * poisson_bracket(omega, phi_half, grid)
    
    # Half-step for psi
    phi_new = solve_stream_function_fft(omega_new, grid)
    psi_new = psi_half + (dt/2) * poisson_bracket(psi_half, phi_new, grid)
    
    return psi_new, omega_new
```

### 4.4 Stability Constraints

**CFL for Poisson bracket:**
$$
\Delta t < \frac{\Delta x}{|\mathbf{v}|_{max}}
$$
where $v \sim \nabla \phi$

**Diffusion stability:**
$$
\Delta t < \frac{\Delta x^2}{2\max(\eta, \nu)}
$$

**Typical values (v1.2):**
- Grid: 64×128, $\Delta r \sim 0.005$
- $\eta = 10^{-5}$, $\nu = 10^{-4}$
- CFL: $\Delta t < 10^{-3}$
- Diffusion: $\Delta t < 10^{-4}$ (more restrictive)

**Recommendation:** $\Delta t = 10^{-4}$ for stability.

---

## 5. Energy Conservation Verification

### 5.1 Ideal Limit Test

**Setup:** $\eta = \nu = 0$, run 10,000 steps

**Expected:** Energy conserved to machine precision:
$$
\left| \frac{E(t) - E(0)}{E(0)} \right| < 10^{-10}
$$

**Test code:**
```python
def test_energy_conservation_ideal():
    solver = HamiltonianMHDSolver(grid, dt=1e-4, eta=0, nu=0)
    solver.initialize(psi0, omega0)
    
    E0 = solver.compute_energy()
    
    for n in range(10000):
        solver.step()
    
    E_final = solver.compute_energy()
    drift = abs(E_final - E0) / E0
    
    assert drift < 1e-10
```

### 5.2 Dissipative Test

**Setup:** $\eta = 10^{-5}$, $\nu = 10^{-4}$

**Expected:** Energy decays monotonically:
$$
E(t+\Delta t) < E(t) \quad \forall t
$$

**Test code:**
```python
def test_energy_decay():
    solver = HamiltonianMHDSolver(grid, dt=1e-4, eta=1e-5, nu=1e-4)
    solver.initialize(psi0, omega0)
    
    E_prev = solver.compute_energy()
    
    for n in range(1000):
        solver.step()
        E_now = solver.compute_energy()
        
        assert E_now <= E_prev  # Monotonic decay
        E_prev = E_now
```

---

## 6. Implementation Timeline

**Week 1-2: Toroidal Solver Fix (concurrent with Doc 1)**
- Prerequisite for all v1.2 work

**Week 3: Poisson Bracket (3 days)**
- Day 1-2: Implement + unit tests
- Day 3: Integrate into RHS

**Week 4: Stream Function Solver (4 days)**
- Day 1-2: FFT Poisson solver
- Day 3: Validation tests
- Day 4: Performance tuning

**Week 5: Symplectic Integration (5 days)**
- Day 1-2: Strang splitting
- Day 3: Störmer-Verlet substep
- Day 4-5: Combined tests

**Week 6: Validation (3 days)**
- Day 1-2: Energy conservation tests
- Day 3: Long-time stability (10k steps)

**Total:** 4 weeks (parallel with solver fix)

---

## 7. Acceptance Criteria

### 7.1 Mandatory

- [ ] **Poisson bracket anti-symmetry:** [A,B] = -[B,A] to machine precision
- [ ] **Stream function accuracy:** $\|\nabla^2 \phi + \omega\| < 10^{-6}$
- [ ] **Energy conservation (ideal):** $|\\Delta E|/E < 10^{-8}$ over 10k steps
- [ ] **Energy decay (dissipative):** $dE/dt < 0$ always
- [ ] **Stability:** 1000 steps without NaN/Inf

### 7.2 Optional

- [ ] **Performance:** < 1.5× slower than v1.1 diffusion
- [ ] **Arakawa scheme:** Discrete energy conservation
- [ ] **Adaptive dt:** CFL-based time-step control

---

## 8. Physics Validation Cases

### 8.1 Kelvin-Helmholtz Instability

**Setup:** Shear flow $v(r) = \tanh(r/L)$

**Expected:** Linear growth rate $\gamma = k v_0$ for small $k$

**Test:** Compare simulation growth vs. theory (within 5%)

### 8.2 Tearing Mode

**Setup:** Current sheet with $q < 1$

**Expected:** Exponential growth → saturation → island formation

**Test:** Island width $w \propto (\Delta')^{-1/2}$ (Rutherford theory)

### 8.3 Alfvén Wave

**Setup:** Sinusoidal perturbation $\delta B$

**Expected:** Phase speed $v_A = B_0/\sqrt{\mu_0 \rho}$

**Test:** Measure dispersion relation $\omega(k)$

---

## 9. Risk Mitigation

### 9.1 If FFT Solver Fails

**Backup:** Iterative CG solver (slower but more general)

### 9.2 If Symplectic Integration Unstable

**Backup:** Implicit-Explicit (IMEX) Runge-Kutta

### 9.3 If Poisson Bracket Too Slow

**Backup:** Simplified bracket (ignore metric factors, accept error)

---

## 10. Summary

**v1.2 transforms v1.1 from diffusion-only to full Hamiltonian MHD:**

| Aspect | v1.1 | v1.2 |
|--------|------|------|
| **Dynamics** | Diffusion only | Hamiltonian + dissipation |
| **Physics** | Unrealistic | Realistic (tearing, KH, Alfvén) |
| **Integrator** | Underutilized | Properly symplectic |
| **Energy** | Decays | Conserved (ideal) or decays (dissipative) |
| **RL** | Framework test | Realistic control problem |

**Key deliverables:**
1. Poisson bracket operator (anti-symmetric, tested)
2. FFT-based Poisson solver (fast, accurate)
3. Strang splitting (2nd-order, stable)
4. Energy conservation verification (ideal + dissipative)

**Timeline:** 4 weeks (parallel with toroidal solver fix)

**小P commitment:** Hamiltonian structure is the physics heart of v1.2 ⚛️

---

**Document Status:** Design Complete  
**Next:** Implementation (after toroidal solver fix)

