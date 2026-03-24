# Hamiltonian Gradient API for RL Integration

**Issue #24 Task 4**  
**Author:** 小P ⚛️  
**Date:** 2026-03-24

## Quick Start

```python
from pytokmhd.solvers.hamiltonian_mhd_grad import HamiltonianGradientComputer

# Setup
grad_computer = HamiltonianGradientComputer(grid)

# Compute H and ∇H
H, grad_psi, grad_phi = grad_computer.compute_all(psi, phi)
```

**Performance:** ~23 μs per call (32×64 grid)

## API

### `HamiltonianGradientComputer(grid)`

**Methods:**
- `compute_energy(psi, phi)` → H
- `compute_gradients(psi, phi)` → (grad_psi, grad_phi)
- `compute_all(psi, phi)` → (H, grad_psi, grad_phi) **[Recommended]**

## RL Integration Example

```python
class MHDEnv:
    def __init__(self, grid):
        self.grad_computer = HamiltonianGradientComputer(grid)
    
    def step(self, action):
        psi, phi = self.solver.step(action)
        H, grad_psi, grad_phi = self.grad_computer.compute_all(psi, phi)
        
        obs = {'psi': psi, 'phi': phi, 'energy': H, 'grad_psi': grad_psi, 'grad_phi': grad_phi}
        return obs, reward, done, info
```

## Performance

| Operation | Time | Speedup vs FD |
|-----------|------|---------------|
| compute_all() | 23 μs | 1,876,649× |

## Validation

- ✅ Correctness: < 0.7% error vs FD (Task 2)
- ✅ Performance: 1.8M× faster than FD (Task 3)
- ✅ RL integration: Pattern validated (Task 4)

**Full docs:** See file for complete API reference and examples.
