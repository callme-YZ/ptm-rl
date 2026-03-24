"""
Poisson Solver for Toroidal Geometry (DEPRECATED)

⚠️ DEPRECATED: This implementation is broken and no longer maintained.

Use pytokmhd.solvers.solve_poisson_toroidal instead.

This file will be removed in v3.1.

Author: 小P ⚛️
Created: 2026-03-19
Deprecated: 2026-03-24
"""

import warnings


def solve_poisson_toroidal(*args, **kwargs):
    """
    DEPRECATED: Use pytokmhd.solvers.solve_poisson_toroidal instead.
    
    This FFT-based implementation has bugs and fails validation tests.
    The GMRES-based solver in pytokmhd.solvers is validated and production-ready.
    """
    warnings.warn(
        "operators.poisson_solver.solve_poisson_toroidal is DEPRECATED and broken. "
        "Use pytokmhd.solvers.solve_poisson_toroidal instead. "
        "This function will be removed in v3.1.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "This Poisson solver is broken (100% error in round-trip test). "
        "Use pytokmhd.solvers.solve_poisson_toroidal (GMRES-based, 10/10 tests passing)."
    )


def laplacian_toroidal_check(*args, **kwargs):
    """
    DEPRECATED: This function is only used by the broken Poisson solver.
    
    Use pytokmhd.operators.laplacian_toroidal instead.
    """
    warnings.warn(
        "operators.poisson_solver.laplacian_toroidal_check is DEPRECATED. "
        "Use pytokmhd.operators.laplacian_toroidal instead. "
        "This function will be removed in v3.1.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "This function is deprecated. Use pytokmhd.operators.laplacian_toroidal."
    )


# Keep test function for historical reference
def test_poisson_solver():
    """
    Historical test (FAILS - this is why solver was deprecated).
    
    DO NOT USE. Use pytokmhd.solvers.solve_poisson_toroidal instead.
    """
    warnings.warn(
        "This test is for the deprecated broken solver. "
        "Use tests/test_poisson_toroidal.py for validated solver tests.",
        DeprecationWarning
    )
    print("❌ This solver is deprecated and broken.")
    print("✅ Use pytokmhd.solvers.solve_poisson_toroidal instead.")


if __name__ == "__main__":
    test_poisson_solver()
