# Deprecation Plan: operators/poisson_solver.py

**Author:** 小P ⚛️  
**Date:** 2026-03-24  
**Issue:** Duplicate Poisson solver implementations with one broken

---

## Problem

**Two implementations exist:**

| Location | Method | Status | Tests |
|----------|--------|--------|-------|
| `operators/poisson_solver.py` | FFT-based | ❌ Broken | FAIL (100% error) |
| `solvers/poisson_toroidal.py` | GMRES-based | ✅ Validated | 10/10 PASS |

**Consequences:**
- Confusing API (two `solve_poisson_toroidal` functions)
- Broken implementation still exported
- Risk of using wrong one

---

## Usage Analysis

**operators version (broken ❌):**
```python
from pytokmhd.operators import solve_poisson_toroidal  # FFT, broken
```

**Imported by:**
- `operators/__init__.py` (exports it)
- No production code uses it! ✅

**solvers version (validated ✅):**
```python
from pytokmhd.solvers import solve_poisson_toroidal  # GMRES, validated
```

**Used by:**
- `solvers/hamiltonian_mhd.py` ✅
- `solvers/hamiltonian_mhd_imex.py` ✅
- All tests import from here ✅

**Verdict:** Safe to deprecate operators version

---

## Deprecation Strategy

### Phase 1: Deprecation Warning (Today)

**Action:** Add deprecation warning to `operators/poisson_solver.py`

```python
import warnings

def solve_poisson_toroidal(*args, **kwargs):
    warnings.warn(
        "operators.poisson_solver.solve_poisson_toroidal is deprecated and broken. "
        "Use solvers.poisson_toroidal.solve_poisson_toroidal instead. "
        "This function will be removed in v3.1.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "This solver is broken. Use pytokmhd.solvers.solve_poisson_toroidal"
    )
```

**Update `operators/__init__.py`:**
- Remove `solve_poisson_toroidal` from exports
- Remove `laplacian_toroidal_check` (only used in broken solver)

### Phase 2: Removal (v3.1, future)

**Action:** Delete `operators/poisson_solver.py` completely

---

## Testing Plan

### Before Deprecation

**Run all tests:**
```bash
pytest tests/ -v
```

**Expected:** All passing (operators version not used)

### After Deprecation

**1. Check no imports broken:**
```bash
python3 -c "from pytokmhd.operators import laplacian_toroidal"  # Should work
python3 -c "from pytokmhd.solvers import solve_poisson_toroidal"  # Should work
```

**2. Check deprecation warning works:**
```bash
python3 -c "from pytokmhd.operators import solve_poisson_toroidal; solve_poisson_toroidal(None, None)"
# Should raise DeprecationWarning + NotImplementedError
```

**3. Re-run all tests:**
```bash
pytest tests/ -v
```

**Expected:** All still passing

**4. Specific tests:**
```bash
pytest tests/test_poisson_toroidal.py -v  # Should use solvers version
pytest tests/test_poisson_solver_validation.py -v  # Should use solvers version
```

---

## Files to Modify

### Deprecate (Phase 1)

1. `src/pytokmhd/operators/poisson_solver.py`
   - Add deprecation warning
   - Make `solve_poisson_toroidal` raise NotImplementedError

2. `src/pytokmhd/operators/__init__.py`
   - Remove `solve_poisson_toroidal` from `__all__`
   - Remove `laplacian_toroidal_check` from `__all__`
   - (Keep imports but don't export)

### Remove (Phase 2, future v3.1)

1. Delete `src/pytokmhd/operators/poisson_solver.py`
2. Remove imports from `operators/__init__.py`

---

## Risk Assessment

**Breakage risk:** ✅ **VERY LOW**

**Reasons:**
1. No production code imports from `operators.poisson_solver`
2. All solvers use `solvers.poisson_toroidal` ✅
3. Tests import from `solvers` ✅

**Only potential issue:**
- External code importing `from pytokmhd.operators import solve_poisson_toroidal`
- **Mitigation:** Deprecation warning guides to correct import

---

## Validation Checklist

Before merge:
- [ ] All tests pass before changes
- [ ] Deprecation warning added
- [ ] `operators/__init__.py` updated
- [ ] All tests pass after changes
- [ ] Deprecation warning fires correctly
- [ ] No production code broken
- [ ] Documentation updated

---

## Implementation

Execute in order:
1. Run baseline tests
2. Add deprecation to `operators/poisson_solver.py`
3. Update `operators/__init__.py`
4. Re-run tests
5. Verify deprecation warning
6. Commit with clear message

**Estimated time:** 15 minutes

**小P签字:** Ready to execute ⚛️
