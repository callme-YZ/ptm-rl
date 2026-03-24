# v2.0 Physics Unit Tests

**Issue #17 deliverables**

## Test Coverage

**Total: 46 tests, 100% passing** ✅

### Test Files (8 modules)

1. **test_elsasser_bracket_simple.py** (5 tests)
   - Poisson bracket properties
   - Antisymmetry, linearity, self-bracket

2. **test_toroidal_hamiltonian.py** (7 tests)
   - Hamiltonian energy computation
   - Cylindrical limit, curvature energy, scaling

3. **test_complete_solver.py** (7 tests)
   - RK2 time integration
   - Energy decay, stability, zero step

4. **test_resistive_dynamics.py** (8 tests)
   - Resistive diffusion
   - Pressure gradient force
   - Antisymmetry and symmetry

5. **test_ballooning_ic_simple.py** (3 tests)
   - Module imports
   - BOUT metric, field-aligned coordinates

6. **test_toroidal_bracket.py** (7 tests)
   - Toroidal Morrison bracket
   - ε scaling, toroidal derivatives

7. **test_modules_import.py** (5 tests)
   - Module import smoke tests
   - equilibrium, rmp, bout_metric, field_aligned

8. **test_integration.py** (4 tests)
   - Multi-step stability
   - Energy monotonic decrease
   - Physical value bounds

## Physics Coverage

**Core operators:**
- ✅ Poisson bracket (Morrison bracket)
- ✅ Hamiltonian energy
- ✅ Toroidal coupling

**Solvers:**
- ✅ RK2 time integration
- ✅ Resistive MHD dynamics

**Physics properties:**
- ✅ Energy conservation (resistive decay)
- ✅ Antisymmetry/symmetry checks
- ✅ Numerical stability

## Running Tests

```bash
# All v2.0 tests
cd tests/v2_physics
python3 -m pytest -v

# Specific module
python3 -m pytest test_complete_solver.py -v

# With output
python3 -m pytest test_integration.py -v -s
```

## Test Philosophy

- **Unit tests**: Individual module functions
- **Integration tests**: Full solver over timesteps
- **Smoke tests**: Import and basic functionality
- **Numerical tolerances**: Set for finite difference errors

## Author

小P ⚛️

**Date:** 2026-03-24

**Issue:** #17
