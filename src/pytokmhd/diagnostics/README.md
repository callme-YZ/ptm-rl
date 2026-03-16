# PyTokMHD Diagnostics API

## Overview

Tearing mode diagnostics module for MHD simulations. Provides real-time monitoring of magnetic island formation, growth rates, and rational surface locations.

---

## Quick Start

```python
from pytokmhd.diagnostics import TearingModeMonitor

# Create monitor
monitor = TearingModeMonitor(m=2, n=1, track_every=10)

# Evolution loop
for step in range(n_steps):
    psi, omega = mhd_step(psi, omega, dt)
    
    # Update diagnostics
    diag = monitor.update(psi, omega, t, r, z, q_profile)
    
    # Check for warnings
    if diag and diag['w'] > w_threshold:
        print(f"Warning: Island width = {diag['w']:.4f}")

# Plot results
from pytokmhd.diagnostics.visualization import plot_island_evolution
plot_island_evolution(monitor, save_path='island_evolution.png')
```

---

## Core Functions

### Magnetic Island Detection

#### `compute_island_width(psi, r, z, q_profile, m=2, n=1)`

Compute magnetic island width from Poincaré section.

**Parameters:**
- `psi` (ndarray): Poloidal flux (Nr, Nz)
- `r` (ndarray): Radial coordinates (Nr,)
- `z` (ndarray): Vertical coordinates (Nz,)
- `q_profile` (ndarray): Safety factor profile (Nr,)
- `m` (int): Poloidal mode number
- `n` (int): Toroidal mode number

**Returns:**
- `w` (float): Island width
- `r_s` (float): Rational surface radius
- `phase` (float): Island phase (radians)

**Method:**
1. Find rational surface q(r_s) = m/n
2. Extract flux along circle at r_s
3. Identify O-points and X-points
4. Compute width from separatrix

**Example:**
```python
w, r_s, phase = compute_island_width(psi, r, z, q_profile, m=2, n=1)
print(f"Island width: {w:.4f} at r_s = {r_s:.3f}")
```

---

#### `compute_helical_flux(psi, r, z, m=2, n=1)`

Compute helical flux amplitude using Fourier decomposition.

**Parameters:**
- `psi` (ndarray): Poloidal flux
- `r, z` (ndarray): Grid coordinates
- `m, n` (int): Mode numbers

**Returns:**
- `delta_psi` (complex): Helical flux amplitude δψ_mn

**Method:**
```
δψ_mn = ∫ ψ(r,θ) exp(-i m θ) dθ
```

**Example:**
```python
delta_psi = compute_helical_flux(psi, r, z, m=2, n=1)
print(f"Helical flux amplitude: {abs(delta_psi):.6f}")
```

---

### Growth Rate Measurement

#### `compute_growth_rate(w_history, t_history, transient_fraction=0.2)`

Compute growth rate from island width time series.

**Parameters:**
- `w_history` (array_like): Island width time series
- `t_history` (array_like): Time stamps
- `transient_fraction` (float): Fraction of initial data to skip

**Returns:**
- `gamma` (float): Growth rate (1/time)
- `sigma_gamma` (float): Uncertainty estimate

**Method:**
Fits exponential growth: w(t) = w_0 exp(γ t)

Uses log-linear regression: log(w) = log(w_0) + γ t

**Example:**
```python
gamma, sigma = compute_growth_rate(w_history, t_history)
print(f"Growth rate: γ = {gamma:.4f} ± {sigma:.4f}")
```

---

#### `energy_growth_rate(psi, omega, dr, dz, r_grid)`

Compute growth rate from energy evolution.

**Parameters:**
- `psi` (ndarray): Poloidal flux
- `omega` (ndarray): Vorticity
- `dr, dz` (float): Grid spacing
- `r_grid` (ndarray): Radial coordinates

**Returns:**
- `gamma` (float): Growth rate γ = (1/2E) dE/dt

**Example:**
```python
gamma = energy_growth_rate(psi, omega, dr, dz, r_grid)
```

---

### Rational Surface Location

#### `find_rational_surface(q_profile, r_grid, q_target, method='spline')`

Find radius where q(r) = q_target.

**Parameters:**
- `q_profile` (array_like): Safety factor profile
- `r_grid` (array_like): Radial grid points
- `q_target` (float): Target q value (e.g., 2.0 for m=2, n=1)
- `method` (str): Interpolation method ('linear', 'spline', or 'newton')

**Returns:**
- `r_s` (float): Rational surface radius
- `accuracy` (float): Relative accuracy |q(r_s) - q_target| / q_target

**Methods:**
- `'linear'`: Linear interpolation (fast)
- `'spline'`: Cubic spline (recommended)
- `'newton'`: Newton iteration (high accuracy)

**Example:**
```python
r_s, acc = find_rational_surface(q_profile, r_grid, q_target=2.0, method='spline')
print(f"q=2 surface at r = {r_s:.4f} (accuracy: {acc:.2e})")
```

---

#### `find_all_rational_surfaces(q_profile, r_grid, m_max=5, n_max=3)`

Find all rational surfaces q = m/n in the domain.

**Returns:**
- `surfaces` (list of dict): Each entry contains {'m', 'n', 'q', 'r_s', 'accuracy'}

**Example:**
```python
surfaces = find_all_rational_surfaces(q_profile, r_grid, m_max=5, n_max=3)
for surf in surfaces:
    print(f"m={surf['m']}, n={surf['n']}: r_s = {surf['r_s']:.3f}")
```

---

## TearingModeMonitor Class

Real-time tearing mode diagnostics during MHD evolution.

### Initialization

```python
monitor = TearingModeMonitor(m=2, n=1, track_every=10, gamma_window=50)
```

**Parameters:**
- `m` (int): Poloidal mode number
- `n` (int): Toroidal mode number
- `track_every` (int): Track diagnostics every N steps
- `gamma_window` (int): Number of points for growth rate calculation

### Methods

#### `update(psi, omega, t, r, z, q_profile)`

Update diagnostics at current timestep.

**Returns:**
- `diagnostics` (dict or None): Contains {'w', 'r_s', 'phase', 'gamma', 'sigma', 't', 'step'}

**Example:**
```python
diag = monitor.update(psi, omega, t, r, z, q_profile)
if diag:
    print(f"t={diag['t']:.2f}: w={diag['w']:.4f}, γ={diag['gamma']:.4f}")
```

---

#### `get_latest_gamma(n_avg=10)`

Get average growth rate over last n_avg measurements.

**Returns:**
- `gamma_avg` (float): Average growth rate
- `sigma_avg` (float): Average uncertainty

---

#### `is_unstable(threshold=0.0)`

Check if mode is unstable (γ > threshold).

**Returns:**
- `unstable` (bool): True if mode is unstable

---

#### `get_summary()`

Get summary statistics.

**Returns:**
- `summary` (dict): Contains {'n_samples', 'w_current', 'w_max', 'gamma_avg', 't_final', 'mode', 'r_s'}

---

#### `reset()`

Reset all history.

---

## Visualization

### `plot_island_evolution(monitor, save_path=None, figsize=(10, 8))`

Plot island width and growth rate evolution.

**Example:**
```python
from pytokmhd.diagnostics.visualization import plot_island_evolution

fig = plot_island_evolution(monitor, save_path='evolution.png')
```

Features:
- Island width vs time
- Exponential growth fit
- Growth rate evolution
- Uncertainty bands

---

### `plot_poincare_section(psi, r, z, r_s=None, levels=20)`

Plot Poincaré section (flux contours).

---

### `plot_diagnostics_summary(monitor, psi=None, r=None, z=None)`

Comprehensive diagnostics summary plot.

Includes:
- Island width evolution
- Growth rate evolution
- Current Poincaré section (if psi provided)

---

## Performance

**Benchmarks (64×128 grid):**
- `compute_island_width`: ~0.05s
- `find_rational_surface`: ~0.001s (spline)
- `TearingModeMonitor` overhead: <2% (track_every=10)

---

## Validation

All algorithms validated against:
1. Analytical solutions (Solovev equilibrium)
2. Synthetic exponential growth data
3. Integration tests with MHD evolution

Test coverage: 100%

See `tests/test_diagnostics.py` for details.

---

## References

1. Furth, Killeen, Rosenbluth (1963) - "Finite-Resistivity Instabilities"
2. Wesson (2011) - Tokamaks, Chapter 10
3. White (2001) - "Theory of Toroidally Confined Plasmas"

---

## Version

v0.1.0 (2026-03-16)
