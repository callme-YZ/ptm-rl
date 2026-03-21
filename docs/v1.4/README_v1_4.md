# MHD Gym Environment v1.4 - Quick Start

## Installation

```bash
cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl
pip install gymnasium numpy scipy matplotlib
```

## Basic Usage

```python
from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D

# Create environment
env = MHDEnv3D(
    grid_size=(32, 64, 32),  # (nr, nθ, nζ)
    dt=0.01,                 # Time step
    max_steps=50,            # Episode length
    I_max=1.0,               # Max coil current
    n_coils=5                # Number of coils
)

# Reset environment
obs, info = env.reset(seed=42)
print(f"Initial energy: {info['E0']:.3e}")

# Run episode
for step in range(50):
    # Random action: 5 coil currents in [-1, 1]
    action = env.action_space.sample()
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Monitor energy
    if (step + 1) % 10 == 0:
        print(f"Step {step+1}: E/E₀ = {info['energy']/info['E0']:.6f}, "
              f"drift = {info['energy_drift']:.4e}")
    
    if terminated or truncated:
        break

# Close environment
env.close()
```

## Observation Space

```python
obs = {
    'psi': np.ndarray (32, 64, 32),    # Normalized stream function
    'omega': np.ndarray (32, 64, 32),  # Normalized vorticity
    'energy': float32,                  # E/E₀
    'max_psi': float32,                 # max|ψ|/ψ_max
    'max_omega': float32                # max|ω|/ω_max
}
```

## Action Space

```python
action = np.array([I₁, I₂, I₃, I₄, I₅])  # Shape: (5,)
# Each Iᵢ ∈ [-1, 1], scaled to physical current [-I_max, I_max]
```

## Reward

```python
reward = -|E(t) - E(t-Δt)| / E₀
```

Negative → energy changed (bad)  
Zero → perfect energy conservation (good)

## Running Tests

```bash
cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl

# Run all tests
PYTHONPATH=$PWD python3 -m pytest tests/rl/test_mhd_env_v1_4.py -v

# Run specific test
PYTHONPATH=$PWD python3 -m pytest tests/rl/test_mhd_env_v1_4.py::TestMHDEnv3D::test_reset -v

# Run manual tests
python3 tests/rl/test_mhd_env_v1_4.py
```

## Example Scripts

```bash
# Demo with random policy
python3 examples/demo_mhd_env_v1_4.py

# Stability test (zero action)
python3 examples/test_stability.py
```

## Physics Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | (32, 64, 32) | (nr, nθ, nζ) resolution |
| `eta` | 1e-4 | Resistivity |
| `dt` | 0.01 | Time step [s] |
| `max_steps` | 50 | Episode length |
| `I_max` | 1.0 | Max coil current |
| `n_coils` | 5 | Number of coils |
| `coil_sigma` | 0.05 | Radial width of coil profile |

## Initial Condition

- **Equilibrium:** ψ₀(r) = (r/a)²(1 - r/a)
- **Perturbation:** Ballooning mode (n=5, m₀=2, ε=0.0001)
- **Safety factor:** Linear q(r) from 1.0 to 3.0

## Typical Performance

**Zero action (no control):**
- Energy drift after 50 steps: ~0.4%
- Reward per step: ~-1e-4

**Random action:**
- Energy drift: 0.4% to 10% (depends on luck)
- May destabilize after 20-30 steps if unlucky

**Goal for RL agent:**
- Keep energy drift < 1% over 50 steps
- Total reward > -0.01

## Troubleshooting

### Instability (energy → inf)

**Cause:** External currents too strong or dt too large

**Solution:**
1. Reduce `I_max` (default 1.0 → 0.5)
2. Reduce `dt` (0.01 → 0.005)
3. Use smaller actions (clip policy output)

### Slow episodes

**Cause:** Large grid or many steps

**Solution:**
1. Reduce grid: (32, 64, 32) → (16, 32, 16) for prototyping
2. Reduce `max_steps`: 50 → 20
3. Use `store_interval` in solver (not implemented yet)

### Tests fail

**Cause:** Missing dependencies or path issues

**Solution:**
```bash
# Check dependencies
pip list | grep -E "gymnasium|numpy|scipy"

# Set PYTHONPATH
export PYTHONPATH=/Users/yz/.openclaw/workspace-xiaoa/ptm-rl:$PYTHONPATH

# Run pytest with verbose output
pytest tests/rl/test_mhd_env_v1_4.py -v -s
```

## API Reference

See docstrings in `src/pytokmhd/rl/mhd_env_v1_4.py` for detailed API documentation.

## Citation

```
MHDEnv3D v1.4 - 3D MHD Gym Environment for Tokamak Control
Author: 小A 🤖
Date: 2026-03-20
Project: Plasma Tearing Mode RL Control
```
