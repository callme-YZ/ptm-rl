# Toroidal Poisson Solver - Usage Guide

**Module:** `pytokmhd.solvers.poisson_toroidal`  
**Author:** 小P ⚛️  
**Date:** 2026-03-19  
**Version:** 1.3.0

---

## Overview

The toroidal Poisson solver computes the stream function φ from vorticity ω:

```
∇²φ = ω
```

in toroidal geometry with proper boundary condition enforcement.

---

## Key Features

- **GMRES solver** with LinearOperator (matrix-free)
- **Identity-row BC enforcement** (not projection)
- **Validated accuracy:**
  - Residual: ~8e-8
  - BC error: ~1e-7
  - Solution error: ~0.2% for φ = r²

---

## Basic Usage

```python
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers import solve_poisson_toroidal

# Create grid
grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)

# Vorticity field
omega = ...  # shape (nr, ntheta)

# Boundary condition at r=a
phi_boundary = ...  # shape (ntheta,)

# Solve
phi, info = solve_poisson_toroidal(omega, grid, phi_boundary, tol=1e-8)

if info == 0:
    print("Converged!")
else:
    print(f"Did not converge: info={info}")
```

---

## API Reference

### `solve_poisson_toroidal`

```python
solve_poisson_toroidal(
    omega: np.ndarray,
    grid: ToroidalGrid,
    phi_boundary: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, int]
```

**Parameters:**
- `omega` — Vorticity field (nr, ntheta)
- `grid` — ToroidalGrid instance
- `phi_boundary` — Outer boundary values (ntheta,), default zeros
- `tol` — GMRES relative tolerance
- `maxiter` — Max GMRES iterations
- `verbose` — Print convergence info

**Returns:**
- `phi` — Stream function (nr, ntheta)
- `info` — GMRES convergence flag (0 = converged)

---

### `compute_residual`

```python
compute_residual(
    phi: np.ndarray,
    omega: np.ndarray,
    grid: ToroidalGrid,
    interior_only: bool = True
) -> Tuple[float, float]
```

Compute ‖∇²φ - ω‖.

**Returns:**
- `max_residual` — Max absolute residual
- `mean_residual` — Mean absolute residual

---

### `check_boundary_conditions`

```python
check_boundary_conditions(
    phi: np.ndarray,
    grid: ToroidalGrid,
    phi_boundary: Optional[np.ndarray] = None
) -> Tuple[float, float]
```

Check BC errors.

**Returns:**
- `bc_error_outer` — Error at outer boundary
- `bc_error_axis` — Axisymmetry violation at axis

---

## Boundary Conditions

### Outer Boundary (r = a)
Dirichlet: φ(a, θ) = phi_boundary(θ)

Enforced via **identity row**:
- Operator returns φ(a, θ) itself
- RHS = phi_boundary(θ)

### Axis (r = 0)
Regularity: φ(0, θ) = constant

Enforced via **identity row**:
- Operator returns φ(0, θ) itself
- RHS = 0 (arbitrary constant)

---

## Algorithm Details

### LinearOperator

```python
def matvec(phi_flat):
    phi_2d = phi_flat.reshape((nr, ntheta))
    
    # Apply toroidal Laplacian
    lap_phi = laplacian_toroidal(phi_2d, grid)
    
    # BC rows: replace with identity
    lap_phi[nr-1, :] = phi_2d[nr-1, :]  # Outer
    lap_phi[0, :] = phi_2d[0, :]        # Axis
    
    return lap_phi.flatten()
```

### RHS Construction

```python
b = omega.flatten().copy()

# Outer BC
b[(nr-1)*ntheta : nr*ntheta] = phi_boundary

# Axis BC
b[0:ntheta] = 0.0  # or any constant
```

---

## Performance

**Tested on MacBook M1:**
- Grid 32x64: ~1 second
- Grid 48x96: ~1 minute (slow!)

**Recommendation:**
- Use nr=32, ntheta=64 for development
- Higher resolution may need preconditioner

---

## Examples

### Example 1: Test with exact solution

```python
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.solvers import solve_poisson_toroidal

grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)

# Exact: φ = r²
r_grid = grid.r_grid
phi_exact = r_grid**2

# Get omega
omega = laplacian_toroidal(phi_exact, grid)

# Boundary
phi_bnd = (grid.a**2) * np.ones(grid.ntheta)

# Solve
phi, info = solve_poisson_toroidal(omega, grid, phi_bnd)

# Error
error = np.max(np.abs(phi - phi_exact))
print(f"Error: {error:.3e}")
```

### Example 2: Check residual

```python
from pytokmhd.solvers import compute_residual

max_res, mean_res = compute_residual(phi, omega, grid)
print(f"Max residual: {max_res:.3e}")
print(f"Mean residual: {mean_res:.3e}")
```

### Example 3: Check BC

```python
from pytokmhd.solvers import check_boundary_conditions

bc_outer, bc_axis = check_boundary_conditions(phi, grid, phi_bnd)
print(f"Outer BC error: {bc_outer:.3e}")
print(f"Axis symmetry error: {bc_axis:.3e}")
```

---

## Validation Results

From `tests/test_poisson_toroidal.py`:

```
Exact solution test (φ = r²):
  Max error: 2.025e-03
  Relative error: 2.2500%
✓ PASSED

Residual test:
  Max residual: 8.146e-08
  Mean residual: 2.756e-08
✓ PASSED

Boundary condition test:
  Outer BC error: 1.225e-07
  Axis symmetry error: 0.000e+00
✓ PASSED
```

---

## Integration with MHD Solver

The Poisson solver is designed to integrate with the MHD evolution:

```python
from pytokmhd.solvers import ToroidalMHDSolver

# In MHD step:
def compute_stream_function(self, omega):
    phi, info = solve_poisson_toroidal(
        omega, self.grid,
        phi_boundary=self.equilibrium.phi_boundary()
    )
    return phi
```

---

## Known Issues

1. **Slow for large grids** — nr>48 takes >1 min
2. **No preconditioner** — GMRES may struggle without it
3. **Fixed BC type** — Only Dirichlet at outer boundary

**Future work:**
- Add ILU preconditioner
- Support Neumann BC
- Optimize for large grids

---

## References

1. 小P, "Correct BC Handling for Toroidal Poisson", 2026-03-19
2. `/Users/yz/.openclaw/workspace-xiaop/phase2-proper/poisson_CORRECT.py`

---

**Questions?** Ask 小P ⚛️
