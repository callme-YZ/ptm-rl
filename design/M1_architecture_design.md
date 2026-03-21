# M1 Architecture Design: Toroidal Geometry Module

**Version:** 1.0  
**Date:** 2026-03-17  
**Status:** Draft

---

## 1. Directory Structure

### New Module Layout

```
ptm_rl/
├── geometry/                   # NEW: Geometry module (Layer 1)
│   ├── __init__.py            # Exports BaseGrid, CylindricalGrid, ToroidalGrid
│   ├── base.py                # BaseGrid abstract class
│   ├── cylindrical.py         # CylindricalGrid (refactored from v1.0)
│   └── toroidal.py            # ToroidalGrid (NEW)
│
├── operators/                  # NEW: Differential operators (Layer 2)
│   ├── __init__.py            # Exports gradient, divergence, laplacian
│   ├── metric.py              # Metric tensor utilities
│   └── differential.py        # Coordinate-free differential operators
│
├── physics/                    # EXISTING: Physics models (Layer 3)
│   ├── mhd.py                 # MHD equations (uses operators)
│   └── ...
│
└── env/                        # EXISTING: RL environment
    └── mhd_env.py             # Updated to support geometry='toroidal'

tests/
├── geometry/                   # NEW: Geometry tests
│   ├── test_cylindrical_grid.py
│   └── test_toroidal_grid.py
│
├── operators/                  # NEW: Operator tests
│   └── test_differential.py
│
└── integration/                # NEW: Integration tests
    ├── test_toroidal_mhd.py   # End-to-end toroidal MHD
    └── test_toroidal_performance.py
```

### File Responsibilities

**ptm_rl/geometry/base.py**
- `BaseGrid` abstract class
- Interface: `metric_tensor()`, `gradient()`, `divergence()`, `laplacian()`
- Coordinate transformations (abstract methods)

**ptm_rl/geometry/cylindrical.py**
- `CylindricalGrid` (refactored from v1.0 `Grid` class)
- Implements BaseGrid interface
- Backward compatible with v1.0

**ptm_rl/geometry/toroidal.py**
- `ToroidalGrid` (NEW)
- Toroidal coordinates (R, θ, φ)
- Parameters: R₀ (major radius), a (minor radius)
- Grid resolution: Nr, Nθ, Nφ

**ptm_rl/operators/metric.py**
- Metric tensor g_ij calculation
- Christoffel symbols Γⁱⱼₖ
- Jacobian √g

**ptm_rl/operators/differential.py**
- `gradient(f, grid)` - ∇f in curvilinear coordinates
- `divergence(v, grid)` - ∇·v using metric
- `laplacian(f, grid)` - ∇²f = ∇·∇f

**tests/geometry/**
- Unit tests for grid construction
- Metric tensor correctness
- Coordinate transformations

**tests/integration/**
- End-to-end MHD simulations
- Performance benchmarks
- Comparison with analytical solutions

---

## 2. API Design

### 2.1 BaseGrid Abstract Class

```python
# ptm_rl/geometry/base.py
from abc import ABC, abstractmethod
import torch
from typing import Tuple

class BaseGrid(ABC):
    """Abstract base class for coordinate grids."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    @abstractmethod
    def metric_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor g_ij at given coordinates.
        
        Args:
            coords: Tensor of shape (..., ndim) with coordinate values
        
        Returns:
            Tensor of shape (..., ndim, ndim) with metric components
        """
        pass
    
    @abstractmethod
    def jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute √g (square root of metric determinant).
        
        Args:
            coords: Tensor of shape (..., ndim)
        
        Returns:
            Tensor of shape (...,) with Jacobian values
        """
        pass
    
    @abstractmethod
    def to_cartesian(self, coords: torch.Tensor) -> torch.Tensor:
        """Transform coordinates to Cartesian (x, y, z).
        
        Args:
            coords: Tensor of shape (..., ndim) in native coordinates
        
        Returns:
            Tensor of shape (..., 3) in Cartesian coordinates
        """
        pass
    
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        pass
```

### 2.2 ToroidalGrid Implementation

```python
# ptm_rl/geometry/toroidal.py
import torch
from .base import BaseGrid

class ToroidalGrid(BaseGrid):
    """Toroidal coordinate grid (R, θ, φ).
    
    Coordinates:
        R: radial coordinate (distance from axis)
        θ: poloidal angle (0 to 2π)
        φ: toroidal angle (0 to 2π)
    
    Parameters:
        R0: major radius (center of torus)
        a: minor radius (tube radius)
        Nr: number of radial points
        Ntheta: number of poloidal points
        Nphi: number of toroidal points
    
    Metric (in lowest order):
        ds² = dR² + R²dθ² + (R₀ + R cos θ)²dφ²
    """
    
    def __init__(
        self,
        R0: float,
        a: float,
        Nr: int,
        Ntheta: int,
        Nphi: int,
        device: str = 'cpu'
    ):
        super().__init__(device)
        self.R0 = R0
        self.a = a
        self.Nr = Nr
        self.Ntheta = Ntheta
        self.Nphi = Nphi
        
        # Create grid
        self.R = torch.linspace(0, a, Nr, device=device)
        self.theta = torch.linspace(0, 2*torch.pi, Ntheta, device=device)
        self.phi = torch.linspace(0, 2*torch.pi, Nphi, device=device)
    
    def metric_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Metric tensor in toroidal coordinates.
        
        Args:
            coords: shape (..., 3) with [R, θ, φ]
        
        Returns:
            shape (..., 3, 3) with g_ij components
        """
        R = coords[..., 0]
        theta = coords[..., 1]
        
        g = torch.zeros(*coords.shape[:-1], 3, 3, device=self.device)
        g[..., 0, 0] = 1.0                              # g_RR
        g[..., 1, 1] = R**2                             # g_θθ
        g[..., 2, 2] = (self.R0 + R * torch.cos(theta))**2  # g_φφ
        
        return g
    
    def jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """√g = R(R₀ + R cos θ)"""
        R = coords[..., 0]
        theta = coords[..., 1]
        return R * (self.R0 + R * torch.cos(theta))
    
    def to_cartesian(self, coords: torch.Tensor) -> torch.Tensor:
        """Transform (R, θ, φ) → (x, y, z).
        
        x = (R₀ + R cos θ) cos φ
        y = (R₀ + R cos θ) sin φ
        z = R sin θ
        """
        R = coords[..., 0]
        theta = coords[..., 1]
        phi = coords[..., 2]
        
        r_major = self.R0 + R * torch.cos(theta)
        
        x = r_major * torch.cos(phi)
        y = r_major * torch.sin(phi)
        z = R * torch.sin(theta)
        
        return torch.stack([x, y, z], dim=-1)
    
    @property
    def ndim(self) -> int:
        return 3
```

### 2.3 CylindricalGrid Refactor

```python
# ptm_rl/geometry/cylindrical.py
import torch
from .base import BaseGrid

class CylindricalGrid(BaseGrid):
    """Cylindrical coordinate grid (r, z) - v1.0 compatible.
    
    This is a refactor of the original Grid class from v1.0.
    Maintains full backward compatibility.
    """
    
    def __init__(self, nr: int, nz: int, dr: float, dz: float, device: str = 'cpu'):
        super().__init__(device)
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        
        # Grid arrays (v1.0 compatible)
        self.r = torch.linspace(dr/2, nr*dr - dr/2, nr, device=device)
        self.z = torch.linspace(dz/2, nz*dz - dz/2, nz, device=device)
    
    def metric_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Metric in cylindrical coordinates: ds² = dr² + r²dθ² + dz²"""
        r = coords[..., 0]
        g = torch.zeros(*coords.shape[:-1], 2, 2, device=self.device)
        g[..., 0, 0] = 1.0      # g_rr
        g[..., 1, 1] = 1.0      # g_zz (θ integrated out in 2D)
        return g
    
    def jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """√g = r (from r dr dθ dz)"""
        return coords[..., 0]
    
    def to_cartesian(self, coords: torch.Tensor) -> torch.Tensor:
        """(r, z) → (x, y, z) assuming θ=0"""
        r = coords[..., 0]
        z = coords[..., 1]
        x = r
        y = torch.zeros_like(r)
        return torch.stack([x, y, z], dim=-1)
    
    @property
    def ndim(self) -> int:
        return 2
```

### 2.4 Differential Operators

```python
# ptm_rl/operators/differential.py
import torch
from ptm_rl.geometry.base import BaseGrid

def gradient(f: torch.Tensor, grid: BaseGrid, coords: torch.Tensor) -> torch.Tensor:
    """Compute ∇f in curvilinear coordinates.
    
    ∇ⁱf = gⁱʲ ∂f/∂xʲ
    
    Args:
        f: scalar field, shape (...,)
        grid: BaseGrid instance
        coords: coordinates, shape (..., ndim)
    
    Returns:
        Contravariant gradient, shape (..., ndim)
    """
    # Finite difference ∂f/∂xⁱ
    df = torch.gradient(f, dim=tuple(range(f.ndim)))
    
    # Raise index with inverse metric
    g = grid.metric_tensor(coords)
    g_inv = torch.linalg.inv(g)
    
    grad_f = torch.einsum('...ij,...j->...i', g_inv, torch.stack(df, dim=-1))
    return grad_f

def divergence(v: torch.Tensor, grid: BaseGrid, coords: torch.Tensor) -> torch.Tensor:
    """Compute ∇·v in curvilinear coordinates.
    
    ∇·v = (1/√g) ∂ᵢ(√g vⁱ)
    
    Args:
        v: contravariant vector field, shape (..., ndim)
        grid: BaseGrid instance
        coords: coordinates, shape (..., ndim)
    
    Returns:
        Divergence, shape (...,)
    """
    sqrt_g = grid.jacobian(coords)
    
    # √g vⁱ
    sqrt_g_v = sqrt_g[..., None] * v
    
    # ∂ᵢ(√g vⁱ)
    div_terms = [torch.gradient(sqrt_g_v[..., i], dim=i)[i] 
                 for i in range(v.shape[-1])]
    
    div_v = sum(div_terms) / sqrt_g
    return div_v

def laplacian(f: torch.Tensor, grid: BaseGrid, coords: torch.Tensor) -> torch.Tensor:
    """Compute ∇²f = ∇·∇f.
    
    Args:
        f: scalar field, shape (...,)
        grid: BaseGrid instance
        coords: coordinates, shape (..., ndim)
    
    Returns:
        Laplacian, shape (...,)
    """
    grad_f = gradient(f, grid, coords)
    lap_f = divergence(grad_f, grid, coords)
    return lap_f
```

---

## 3. Module Dependencies

### 3.1 Dependency Layers

```
Layer 3: Physics & Environment
├── ptm_rl/physics/mhd.py          (uses operators + geometry)
├── ptm_rl/env/mhd_env.py          (uses physics)
│
Layer 2: Operators
├── ptm_rl/operators/differential.py   (uses geometry)
├── ptm_rl/operators/metric.py         (uses geometry)
│
Layer 1: Geometry (pure, no dependencies)
├── ptm_rl/geometry/base.py
├── ptm_rl/geometry/cylindrical.py
└── ptm_rl/geometry/toroidal.py
```

### 3.2 Import Rules

**Allowed:**
```python
# Layer 2 imports Layer 1
from ptm_rl.geometry import BaseGrid, ToroidalGrid

# Layer 3 imports Layer 1-2
from ptm_rl.geometry import ToroidalGrid
from ptm_rl.operators import gradient, divergence
```

**Forbidden:**
```python
# ❌ Layer 1 cannot import Layer 2/3
from ptm_rl.operators import gradient  # in geometry/toroidal.py

# ❌ Layer 2 cannot import Layer 3
from ptm_rl.physics import MHD  # in operators/differential.py
```

### 3.3 Circular Dependency Prevention

**Problem:** If `ToroidalGrid` needs differential operators internally, and operators need grid → circular import.

**Solution:** Keep geometry **pure**. Differential operators are **external utilities** that operate on grids.

```python
# ✅ Correct usage
grid = ToroidalGrid(R0=1.0, a=0.3, ...)
psi = torch.randn(grid.Nr, grid.Ntheta, grid.Nphi)
grad_psi = gradient(psi, grid, coords)  # operator uses grid

# ❌ Wrong (if ToroidalGrid.gradient() existed)
grad_psi = grid.gradient(psi)  # would require operators inside geometry
```

### 3.4 Dependency Graph

```
┌─────────────────────────────────────┐
│  ptm_rl/env/mhd_env.py              │  Layer 3
│  - MHDEnv                            │
│  - Reward function                   │
└────────────┬────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────┐
│  ptm_rl/physics/mhd.py              │  Layer 3
│  - MHD equations                     │
│  - ∇·B=0, ∇×B=μ₀J, etc.             │
└────────────┬────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────┐
│  ptm_rl/operators/differential.py   │  Layer 2
│  - gradient, divergence, curl        │
└────────────┬────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────┐
│  ptm_rl/geometry/base.py            │  Layer 1
│  - BaseGrid (abstract)               │
├─────────────────────────────────────┤
│  ptm_rl/geometry/toroidal.py        │
│  - ToroidalGrid (concrete)           │
├─────────────────────────────────────┤
│  ptm_rl/geometry/cylindrical.py     │
│  - CylindricalGrid (v1.0 compat)     │
└─────────────────────────────────────┘
```

### 3.5 External Dependencies

- **PyTorch** (all layers)
- **NumPy** (optional, for I/O)
- **Gymnasium** (env layer only)

No circular dependencies within `ptm_rl`.

---

## 4. Testing Framework

### 4.1 Unit Tests

#### tests/geometry/test_toroidal_grid.py

```python
import torch
import pytest
from ptm_rl.geometry import ToroidalGrid

def test_toroidal_grid_construction():
    """Test grid initialization."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    assert grid.R0 == 1.0
    assert grid.a == 0.3
    assert grid.Nr == 32

def test_metric_tensor_shape():
    """Test metric tensor dimensions."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    coords = torch.rand(10, 3)  # 10 points, (R, θ, φ)
    g = grid.metric_tensor(coords)
    assert g.shape == (10, 3, 3)

def test_metric_positive_definite():
    """Test that metric is positive definite."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    coords = torch.tensor([[0.1, 0.5, 1.0]])
    g = grid.metric_tensor(coords)
    eigenvalues = torch.linalg.eigvalsh(g)
    assert torch.all(eigenvalues > 0)

def test_jacobian_positivity():
    """Test √g > 0."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    coords = torch.rand(100, 3)
    coords[:, 0] = coords[:, 0] * 0.3  # R ∈ [0, 0.3]
    sqrt_g = grid.jacobian(coords)
    assert torch.all(sqrt_g >= 0)

def test_coordinate_transformation():
    """Test (R,θ,φ) → (x,y,z)."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    
    # Point on major radius at θ=0, φ=0
    coords = torch.tensor([[0.0, 0.0, 0.0]])
    xyz = grid.to_cartesian(coords)
    
    # Should be (R₀, 0, 0)
    assert torch.allclose(xyz, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)

def test_analytical_metric():
    """Compare with analytical metric formula."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
    
    R, theta, phi = 0.2, torch.pi/4, torch.pi/3
    coords = torch.tensor([[R, theta, phi]])
    g = grid.metric_tensor(coords)
    
    # Analytical values
    g_RR = 1.0
    g_theta_theta = R**2
    g_phi_phi = (grid.R0 + R * torch.cos(theta))**2
    
    assert torch.isclose(g[0, 0, 0], torch.tensor(g_RR))
    assert torch.isclose(g[0, 1, 1], g_theta_theta)
    assert torch.isclose(g[0, 2, 2], g_phi_phi)
```

#### tests/operators/test_differential.py

```python
import torch
from ptm_rl.geometry import ToroidalGrid
from ptm_rl.operators import gradient, divergence, laplacian

def test_gradient_constant_field():
    """∇(const) = 0."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=16, Ntheta=32, Nphi=32)
    f = torch.ones(16, 32, 32)
    coords = torch.stack(torch.meshgrid(grid.R, grid.theta, grid.phi, indexing='ij'), dim=-1)
    
    grad_f = gradient(f, grid, coords)
    assert torch.allclose(grad_f, torch.zeros_like(grad_f), atol=1e-5)

def test_divergence_constant_field():
    """∇·(const vector) ≠ 0 in curvilinear (due to Christoffel symbols)."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=16, Ntheta=32, Nphi=32)
    v = torch.ones(16, 32, 32, 3)
    coords = torch.stack(torch.meshgrid(grid.R, grid.theta, grid.phi, indexing='ij'), dim=-1)
    
    div_v = divergence(v, grid, coords)
    # Not necessarily zero, but should be finite
    assert torch.all(torch.isfinite(div_v))

def test_laplacian_quadratic():
    """∇²(r²) in cylindrical → should give constant."""
    from ptm_rl.geometry import CylindricalGrid
    grid = CylindricalGrid(nr=32, nz=32, dr=0.1, dz=0.1)
    r_grid, z_grid = torch.meshgrid(grid.r, grid.z, indexing='ij')
    f = r_grid**2
    coords = torch.stack([r_grid, z_grid], dim=-1)
    
    lap_f = laplacian(f, grid, coords)
    # ∇²(r²) = 4 in 2D cylindrical
    # (approximate test due to finite differences)
    assert torch.allclose(lap_f, torch.full_like(lap_f, 4.0), atol=0.5)
```

### 4.2 Integration Tests

#### tests/integration/test_toroidal_mhd.py

```python
import torch
from ptm_rl.geometry import ToroidalGrid
from ptm_rl.physics.mhd import MHD
from ptm_rl.env.mhd_env import MHDEnv

def test_toroidal_mhd_initialization():
    """Test MHD system in toroidal geometry."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=16, Ntheta=32, Nphi=32)
    mhd = MHD(grid=grid, nu=1e-3, eta=1e-3)
    
    # Initial condition: constant pressure + toroidal B-field
    state = mhd.initialize_state()
    assert state.shape == (16, 32, 32, 8)  # (ρ, vR, vθ, vφ, BR, Bθ, Bφ, p)

def test_mhd_env_step():
    """Test RL environment step with toroidal geometry."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=16, Ntheta=32, Nphi=32)
    env = MHDEnv(grid=grid, geometry='toroidal')
    
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)

def test_divergence_free_magnetic_field():
    """Test ∇·B = 0 preservation."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=16, Ntheta=32, Nphi=32)
    mhd = MHD(grid=grid)
    
    state = mhd.initialize_state()
    B = state[..., 4:7]  # Magnetic field components
    
    from ptm_rl.operators import divergence
    coords = torch.stack(torch.meshgrid(grid.R, grid.theta, grid.phi, indexing='ij'), dim=-1)
    div_B = divergence(B, grid, coords)
    
    # Should be ~ 0 (within numerical error)
    assert torch.allclose(div_B, torch.zeros_like(div_B), atol=1e-4)
```

### 4.3 Performance Benchmarks

#### tests/integration/test_toroidal_performance.py

```python
import torch
import time
from ptm_rl.geometry import ToroidalGrid, CylindricalGrid
from ptm_rl.operators import gradient

def benchmark_gradient_toroidal():
    """Benchmark gradient computation in toroidal geometry."""
    grid = ToroidalGrid(R0=1.0, a=0.3, Nr=64, Ntheta=128, Nphi=128)
    f = torch.randn(64, 128, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
    coords = torch.stack(torch.meshgrid(grid.R, grid.theta, grid.phi, indexing='ij'), dim=-1)
    
    start = time.time()
    for _ in range(100):
        grad_f = gradient(f, grid, coords)
    elapsed = time.time() - start
    
    print(f"Toroidal gradient: {elapsed/100*1000:.2f} ms/call")
    assert elapsed < 10.0  # Should be < 100ms per call

def benchmark_cylindrical_vs_toroidal():
    """Compare computational cost."""
    grid_cyl = CylindricalGrid(nr=64, nz=128, dr=0.01, dz=0.01)
    grid_tor = ToroidalGrid(R0=1.0, a=0.3, Nr=64, Ntheta=128, Nphi=64)
    
    # Same total grid points: 64×128 vs 64×128×64 (latter is larger)
    # Expect toroidal to be ~64x slower (one more dimension)
    
    # (Benchmark code here)
```

### 4.4 Coverage Target

- **Unit tests:** >90% line coverage
- **Integration tests:** All major workflows
- **Performance tests:** Regression detection

**Run all tests:**
```bash
pytest tests/ --cov=ptm_rl --cov-report=html
```

---

## 5. Backward Compatibility

### 5.1 v1.0 Code Remains Unchanged

**Guarantee:** All existing v1.0 code continues to work without modification.

**v1.0 usage (still valid):**
```python
from ptm_rl.env import MHDEnv

env = MHDEnv(nr=32, nz=64, dr=0.1, dz=0.1)  # Cylindrical (default)
obs, info = env.reset()
```

**Internal changes:**
- `Grid` class → renamed to `CylindricalGrid` (aliased for backward compat)
- `MHDEnv` internally uses `CylindricalGrid` when `geometry='cylindrical'` (default)

### 5.2 New Toroidal Geometry API

**v1.1 usage (new):**
```python
from ptm_rl.env import MHDEnv

env = MHDEnv(
    geometry='toroidal',
    R0=1.0,        # major radius
    a=0.3,         # minor radius
    Nr=32,         # radial points
    Ntheta=64,     # poloidal points
    Nphi=64        # toroidal points
)
```

**Direct grid usage (advanced):**
```python
from ptm_rl.geometry import ToroidalGrid
from ptm_rl.operators import gradient

grid = ToroidalGrid(R0=1.0, a=0.3, Nr=32, Ntheta=64, Nphi=64)
psi = torch.randn(32, 64, 64)  # Poloidal flux
coords = torch.stack(torch.meshgrid(grid.R, grid.theta, grid.phi, indexing='ij'), dim=-1)
grad_psi = gradient(psi, grid, coords)
```

### 5.3 Migration Guide

#### For Existing v1.0 Users

**No action required.** Your code will continue to work.

**Optional upgrade:**
```python
# Old (still works)
from ptm_rl.env import MHDEnv
env = MHDEnv(nr=32, nz=64, dr=0.1, dz=0.1)

# New (explicit, recommended)
from ptm_rl.env import MHDEnv
env = MHDEnv(geometry='cylindrical', nr=32, nz=64, dr=0.1, dz=0.1)
```

#### For New Users (Toroidal)

```python
# Step 1: Import
from ptm_rl.env import MHDEnv

# Step 2: Create toroidal environment
env = MHDEnv(
    geometry='toroidal',
    R0=1.65,       # ITER-like: major radius 1.65 m
    a=0.5,         # minor radius 0.5 m
    Nr=64,
    Ntheta=128,
    Nphi=128
)

# Step 3: Train as usual
from stable_baselines3 import PPO
model = PPO('MlpPolicy', env)
model.learn(total_timesteps=100000)
```

### 5.4 Deprecated Features

**None.** This is an additive release (v1.0 → v1.1).

**Future deprecations (v2.0):**
- May remove `nr, nz, dr, dz` kwargs from `MHDEnv` (require explicit `geometry` param)
- `Grid` alias for `CylindricalGrid` may be removed

### 5.5 Configuration File Compatibility

**v1.0 config (still works):**
```yaml
env:
  nr: 32
  nz: 64
  dr: 0.1
  dz: 0.1
```

**v1.1 config (new):**
```yaml
env:
  geometry: toroidal
  R0: 1.0
  a: 0.3
  Nr: 32
  Ntheta: 64
  Nphi: 64
```

**Auto-detection logic in `MHDEnv.__init__()`:**
```python
def __init__(self, geometry='cylindrical', **kwargs):
    if geometry == 'cylindrical':
        # v1.0 path
        self.grid = CylindricalGrid(nr=kwargs['nr'], nz=kwargs['nz'], ...)
    elif geometry == 'toroidal':
        # v1.1 path
        self.grid = ToroidalGrid(R0=kwargs['R0'], a=kwargs['a'], ...)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
```

### 5.6 Testing Backward Compatibility

**Add regression test:**
```python
# tests/test_backward_compatibility.py
def test_v1_0_api():
    """Ensure v1.0 code still works."""
    from ptm_rl.env import MHDEnv
    
    # Old API (no geometry param)
    env = MHDEnv(nr=16, nz=32, dr=0.1, dz=0.1)
    obs, info = env.reset()
    
    assert obs.shape == env.observation_space.shape
```

**Run on every commit:**
```bash
pytest tests/test_backward_compatibility.py
```

---

## 6. Implementation Checklist

### 6.1 M1 Execution Steps

#### Phase 1: Geometry Module (Week 1)

- [ ] **Task 1.1:** Create `ptm_rl/geometry/` directory
  - [ ] `__init__.py` (exports)
  - [ ] `base.py` (BaseGrid abstract class)
  
- [ ] **Task 1.2:** Implement `ToroidalGrid`
  - [ ] `toroidal.py` (class definition)
  - [ ] `metric_tensor()` method
  - [ ] `jacobian()` method
  - [ ] `to_cartesian()` method
  
- [ ] **Task 1.3:** Refactor `CylindricalGrid`
  - [ ] Move from `env/grid.py` to `geometry/cylindrical.py`
  - [ ] Inherit from `BaseGrid`
  - [ ] Add alias `Grid = CylindricalGrid` for backward compat
  
- [ ] **Task 1.4:** Unit tests
  - [ ] `tests/geometry/test_toroidal_grid.py` (6 tests)
  - [ ] `tests/geometry/test_cylindrical_grid.py` (refactor existing)

#### Phase 2: Operators Module (Week 2)

- [ ] **Task 2.1:** Create `ptm_rl/operators/` directory
  - [ ] `__init__.py` (exports)
  - [ ] `metric.py` (Christoffel symbols, Jacobian utils)
  
- [ ] **Task 2.2:** Implement differential operators
  - [ ] `differential.py` (gradient, divergence, laplacian)
  - [ ] Coordinate-free implementation using `BaseGrid.metric_tensor()`
  
- [ ] **Task 2.3:** Unit tests
  - [ ] `tests/operators/test_differential.py` (3 tests)
  - [ ] Analytical validation (constant fields, quadratic)

#### Phase 3: Physics Integration (Week 3)

- [ ] **Task 3.1:** Update `MHD` class
  - [ ] Accept `grid: BaseGrid` instead of `nr, nz`
  - [ ] Use `operators.gradient` instead of manual finite diff
  - [ ] Toroidal coordinate support
  
- [ ] **Task 3.2:** Update `MHDEnv`
  - [ ] Add `geometry='cylindrical'|'toroidal'` parameter
  - [ ] Instantiate correct grid based on geometry
  - [ ] Observation space handling (2D vs 3D)
  
- [ ] **Task 3.3:** Integration tests
  - [ ] `tests/integration/test_toroidal_mhd.py` (3 tests)
  - [ ] ∇·B = 0 validation
  - [ ] Full episode run

#### Phase 4: Documentation & Validation (Week 4)

- [ ] **Task 4.1:** API documentation
  - [ ] Docstrings (Google style)
  - [ ] Sphinx docs generation
  - [ ] Migration guide (v1.0 → v1.1)
  
- [ ] **Task 4.2:** Performance benchmarks
  - [ ] `tests/integration/test_toroidal_performance.py`
  - [ ] CPU vs GPU comparison
  - [ ] Cylindrical vs Toroidal cost
  
- [ ] **Task 4.3:** Backward compatibility validation
  - [ ] Run all v1.0 examples
  - [ ] Regression test suite
  
- [ ] **Task 4.4:** Code review
  - [ ] Type hints (mypy clean)
  - [ ] Test coverage >90%
  - [ ] No circular dependencies

### 6.2 Acceptance Criteria

#### Functional Requirements

- ✅ `ToroidalGrid` passes all unit tests
- ✅ Differential operators validated against analytical solutions
- ✅ ∇·B = 0 maintained in toroidal simulations (error < 1e-4)
- ✅ v1.0 code runs without modification
- ✅ `MHDEnv(geometry='toroidal', ...)` trains successfully

#### Non-Functional Requirements

- ✅ Test coverage >90% (measured by pytest-cov)
- ✅ Type hints complete (mypy passes)
- ✅ No circular imports (enforced by pytest-importcheck)
- ✅ Documentation complete (Sphinx HTML builds)
- ✅ Performance acceptable (gradient < 100ms on 64³ grid)

#### Deliverables

1. **Code:**
   - `ptm_rl/geometry/` module
   - `ptm_rl/operators/` module
   - Updated `ptm_rl/physics/` and `ptm_rl/env/`

2. **Tests:**
   - Unit tests (geometry, operators)
   - Integration tests (toroidal MHD)
   - Performance benchmarks
   - Backward compatibility tests

3. **Documentation:**
   - API reference (Sphinx)
   - Migration guide (Markdown)
   - Example notebooks (Jupyter)

4. **Validation:**
   - Coverage report (HTML)
   - Performance report (CSV/plot)
   - Type check report (mypy)

### 6.3 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Circular dependencies | High | Enforce layered architecture, use pytest-importcheck |
| Numerical instability | High | Analytical validation, multiple test cases |
| Breaking v1.0 | Critical | Regression test suite, CI enforcement |
| Performance degradation | Medium | Benchmarks, profiling, GPU optimization |
| Incomplete testing | Medium | Coverage target >90%, code review |

### 6.4 Success Metrics

- [ ] All tests pass (pytest)
- [ ] Coverage ≥90% (pytest-cov)
- [ ] Type checks pass (mypy)
- [ ] No regressions (backward compat tests)
- [ ] Documentation builds (Sphinx)
- [ ] Example runs successfully:
  ```bash
  python examples/train_toroidal_mhd.py --steps 10000
  ```

---

## Appendix: References

- **Toroidal coordinates:** `docs/v1.1/derivations/toroidal-coordinates.md`
- **Overall design:** `docs/v1.1/design/v1.1-toroidal-symplectic-design-v2.1.md`
- **Differential geometry:** Misner, Thorne, Wheeler - "Gravitation" (Chapter 8)
- **MHD equations:** Goedbloed, Keppens, Poedts - "Magnetohydrodynamics of Laboratory and Astrophysical Plasmas"

---

**End of M1 Architecture Design**
