# M4 Observation Space Design

**Project:** PTM-RL v1.1 - M4 RL Integration  
**Author:** 小P ⚛️  
**Date:** 2026-03-17

## Executive Summary

Observation space for toroidal MHD RL environment.

**Key Choices:**
- Fourier modes (8D) for ψ field
- Energy + ∇·B diagnostics
- Total: 11D observation (v1.1 minimal)

## 1. Physics Quantities

### 1.1 Primary: Poloidal Flux ψ

**Chosen:**
```python
psi_modes: np.ndarray (8,)  # Fourier coefficients
```

### 1.2 Energy Diagnostics

```python
energy: float
energy_drift: float  # |E - E_eq| / E_eq
```

### 1.3 Constraint: ∇·B

```python
div_B_max: float  # Should be < 1e-6
```

## 2. Observation Specification

**v1.1 Minimal:**
```python
observation = {
    'psi_modes': (8,),
    'energy': (1,),
    'energy_drift': (1,),
    'div_B_max': (1,),
}
# Total: 11D
```

## 3. Normalization

- ψ modes: [-1, 1]
- Energy: relative to E_eq
- ∇·B: normalized by 1e-6 threshold

## 4. v1.1 Limitations

**Can observe:**
- ✅ Equilibrium evolution (slow diffusion)
- ✅ Energy changes
- ✅ Constraint satisfaction

**Cannot observe:**
- ❌ Realistic tearing mode (no island growth)
- ❌ Non-linear dynamics

**Impact:** Framework validation only, defer realistic control to v1.2

## 5. Implementation

```python
def get_observation(solver):
    psi_modes = fourier_decompose(solver.psi)
    E = compute_energy(solver.psi, solver.omega, solver.grid)
    div_B = compute_div_B(solver.psi, solver.grid)
    
    return {
        'psi_modes': normalize(psi_modes),
        'energy': (E - E_eq) / E_eq,
        'energy_drift': abs((E - E_eq) / E_eq),
        'div_B_max': np.max(np.abs(div_B)) / 1e-6,
    }
```

## 6. Validation

**✅ Completeness:** Captures essential equilibrium physics  
**✅ Tractability:** 11D very manageable for RL  
**✅ Normalization:** All quantities in reasonable ranges  
**✅ RL Compatible:** Works with PPO/SAC/TD3

## 7. Recommendations for 小A

**Priority 1:**
1. Implement 8-mode Fourier decomposition
2. Add energy computation
3. Add ∇·B diagnostic

**Total: ~200 lines of code**

---

**Status:** Complete ✅  
**Next:** Reward function design
