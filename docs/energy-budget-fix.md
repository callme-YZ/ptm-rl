# Energy Budget Test Fix

**Author:** 小P ⚛️  
**Date:** 2026-03-19  
**Issue:** 30x discrepancy in dH/dt between theory and numerics

---

## Root Causes Identified

### Issue #1: Wrong Input to `compute_hamiltonian`

**Current (WRONG):**
```python
H = compute_hamiltonian(psi, omega, grid)
```

**Problem:**
- Test passes `omega` (vorticity ω = ∇²φ)
- But `compute_hamiltonian` expects `phi` (potential φ)
- Function computes H = ∫ [(1/2)|∇ω|² + (1/2)|∇ψ|²] dV ❌

**Correct Hamiltonian:**
```
H = ∫ [(1/2)|∇φ|² + (1/2)|∇ψ|²] dV
```

**Why it matters:**
```
∫ |∇φ|² dV = ∫ φ·∇²φ dV = ∫ φ·ω dV  (by integration by parts)
```

But:
```
∫ |∇ω|² dV ≠ ∫ φ·ω dV
```

**Energy error from this:**
```
|∇ω|² involves second derivatives of φ
This scales as O(k⁴) for Fourier mode with wavenumber k
While |∇φ|² scales as O(k²)

For small-scale features → huge energy overestimate!
```

---

### Issue #2: Missing 2π Factor

**Current:**
```python
J2_int = np.sum(J**2 * dV)  # dV = R·dr·dθ (2D)
dH_theory = -eta * J2_int
```

**Problem:**
- `dV` is 2D poloidal cross-section
- Need to integrate over toroidal angle: ∫₀²ᵖ dφ = 2π

**Correct:**
```python
dH_theory = -eta * 2*np.pi * J2_int
```

**But wait:** Does `compute_hamiltonian` already include 2π?

Check `hamiltonian.py` line 121:
```python
# Multiply by 2π for toroidal direction
H = 2 * np.pi * energy_2d
```

**YES!** So `compute_hamiltonian` returns full 3D energy.

**For consistency:** dH/dt should also include 2π:
```
dH/dt = -η·(2π) ∫∫ J²·R dr dθ
```

---

### Issue #3: Energy Conservation Interpretation

**Hamiltonian energy functional:**
```
H[ψ, φ] = ∫ d³x [(1/2)|∇φ|² + (1/2)|∇ψ|²]
```

**Time derivative:**
```
dH/dt = ∫ d³x [∇φ·∇(∂φ/∂t) + ∇ψ·∇(∂ψ/∂t)]
```

Using ∂ψ/∂t = {ψ, H} - ηJ and integration by parts:
```
dH/dt = -η ∫ d³x J·(∂ψ/∂t resistive part)
```

**Key insight:** The resistive dissipation comes from:
```
∂ψ/∂t|_resistive = -η·J
```

So:
```
dU/dt|_resistive = ∫ ∇ψ·∇(∂ψ/∂t|_resistive) dV
                 = -η ∫ ∇ψ·∇J dV
```

**But this is NOT η∫J² dV!**

Need to use Ohm's law properly...

**Re-check:** In resistive MHD, the dissipation is:
```
P_dissipated = ∫ J·E dV = η ∫ J² dV
```

where E = ηJ (Ohm's law).

**Energy balance:**
```
dH/dt + P_dissipated = 0  (energy conservation)
```

So:
```
dH/dt = -P_dissipated = -η ∫ J² dV  ✓
```

**This confirms the formula is correct**, but we need to be careful about:
1. Volume element (2π factor)
2. Using φ not ω in H

---

## Fix Implementation

### Fix #1: Solve Poisson Equation

**Add function to invert Laplacian:**
```python
def solve_poisson_toroidal(omega: np.ndarray, grid: ToroidalGrid) -> np.ndarray:
    """
    Solve ∇²φ = ω for φ given ω.
    
    Uses direct FFT method in toroidal geometry.
    """
    # Fourier transform in θ direction (periodic)
    omega_hat = np.fft.rfft(omega, axis=1)
    phi_hat = np.zeros_like(omega_hat)
    
    nr, ntheta = omega.shape
    nk = omega_hat.shape[1]
    
    for k in range(nk):
        # Solve d²φ/dr² + (1/r)dφ/dr - (k²/r²)φ = ω for each mode
        # Use finite difference with tridiagonal solver
        
        # This is a simplified version - needs proper boundary conditions!
        # For now, use a placeholder
        pass
    
    phi = np.fft.irfft(phi_hat, n=ntheta, axis=1)
    return phi
```

**Alternatively:** Use iterative solver (Jacobi/SOR/Conjugate Gradient).

**Simple fix for testing:** If ω was computed from φ originally, we can store φ:
```python
# In test:
omega = -laplacian_toroidal(psi, grid)
phi = psi.copy()  # WRONG! Need to solve ∇²φ = -ω

# Better: Initialize both φ and ω
phi = 0.1 * np.sin(grid.theta_grid) * grid.r_grid**2
omega = -laplacian_toroidal(phi, grid)
```

---

### Fix #2: Update Test Code

```python
# test_imex_energy_budget.py

# After line 47 (omega initialization), add:
# For energy computation, we need φ not ω
# Option A: If we have initial φ, evolve it
# Option B: Solve Poisson ∇²φ = ω each step (expensive!)
# Option C: Use correct energy formula with ω

# For now, let's use a workaround:
# Since solver evolves (ψ, ω), we need to reconstruct φ

# Quick fix: assume small amplitude → H ≈ ∫(1/2)|∇ψ|² (magnetic dominated)
# Then dH/dt ≈ dU/dt = -η∫J² dV

# Better fix: implement Poisson solver
from pytokmhd.operators import solve_poisson_toroidal  # To be implemented

# Replace line 60:
# H0 = compute_hamiltonian(psi, omega, grid)  # WRONG
phi0 = solve_poisson_toroidal(omega, grid)
H0 = compute_hamiltonian(psi, phi0, grid)  # CORRECT

# Similarly in loop (line 77):
phi = solve_poisson_toroidal(omega, grid)
H = compute_hamiltonian(psi, phi, grid)
```

**And add 2π factor:**
```python
# Line 73:
dH_theory = -eta * 2*np.pi * J2_int  # Added 2π
```

---

## Validation Steps

After fix:
1. Re-run test
2. Check energy monotonically decreases
3. Verify relative error < 5%

Expected improvement:
- From 96% error → <5% error
- Factor of 30x → factor of <1.05x

---

## Alternative: Energy-Enstrophy Formulation

**If Poisson solver is too expensive**, use alternative Hamiltonian:

```
H[ψ, ω] = ∫ [(1/2)φ·ω + (1/2)|∇ψ|²] dV
```

where φ is determined by ∇²φ = ω (not explicitly solved).

Then:
```
dH/dt = ∫ [(1/2)∂φ/∂t·ω + (1/2)φ·∂ω/∂t] dV + dU/dt
```

This avoids solving Poisson, but complicates the formula.

**Recommendation:** Implement fast Poisson solver (FFT-based or multigrid).

---

## Status

- [x] Theory derivation complete
- [x] Root causes identified
- [ ] Poisson solver implementation
- [ ] Test code update
- [ ] Validation

**Next:** Implement Poisson solver and update test.
