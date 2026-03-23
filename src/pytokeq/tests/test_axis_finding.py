"""
Test Plasma Axis Finding - Critical Validation

Purpose: Test find_psi_axis() including boundary rejection
Morning lesson (2026-03-12): Vacuum field can have ψ_max at wall!

Date: 2026-03-12
Status: TEST ONLY (will fail until Step 3 implementation)
"""

import numpy as np
import pytest

# Import axis finding utilities
from pytokeq.equilibrium.solver.picard_gs_solver import find_psi_axis, Grid


def test_axis_normal_case():
    """
    Test 1: Normal case - axis in center
    
    Expected: Pass, returns interior indices
    """
    # Create simple flux with maximum in center
    nr, nz = 65, 65
    R = np.linspace(1.0, 2.0, nr)
    Z = np.linspace(-0.5, 0.5, nz)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing='ij')
    
    # Gaussian-like flux (max at center)
    R0 = 1.5
    Z0 = 0.0
    psi = np.exp(-((R_grid - R0)**2 + (Z_grid - Z0)**2) / 0.1)
    
    grid = Grid(
        R=R_grid, Z=Z_grid,
        dR=R[1]-R[0], dZ=Z[1]-Z[0],
        nr=nr, nz=nz
    )
    
    # Find axis
    i_axis, j_axis, psi_axis = find_psi_axis(psi, grid)
    
    # Verify in interior (NOT on boundary)
    assert 0 < i_axis < nr-1, f"Axis on R boundary: i={i_axis}"
    assert 0 < j_axis < nz-1, f"Axis on Z boundary: j={j_axis}"
    
    # Verify near expected location
    R_axis = R_grid[i_axis, j_axis]
    Z_axis = Z_grid[i_axis, j_axis]
    assert np.abs(R_axis - R0) < 0.05, f"Axis R={R_axis:.2f}, expected {R0:.2f}"
    assert np.abs(Z_axis - Z0) < 0.05, f"Axis Z={Z_axis:.2f}, expected {Z0:.2f}"
    
    # Verify is local maximum
    assert psi_axis == np.max(psi[1:-1, 1:-1]), "Axis not maximum in interior"


def test_axis_on_boundary_R():
    """
    Test 2: Axis on R boundary - should raise RuntimeError
    
    This is the MORNING LESSON case!
    Vacuum field with coils at inner boundary
    
    Expected: RuntimeError with helpful message
    """
    nr, nz = 65, 65
    R = np.linspace(1.0, 2.0, nr)
    Z = np.linspace(-0.5, 0.5, nz)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing='ij')
    
    # Create flux with maximum at R boundary (R=1.0)
    # This simulates vacuum field from coil at wall
    psi = np.exp(-((R_grid - 1.0)**2 + Z_grid**2) / 0.1)
    
    grid = Grid(
        R=R_grid, Z=Z_grid,
        dR=R[1]-R[0], dZ=Z[1]-Z[0],
        nr=nr, nz=nz
    )
    
    # Should raise RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        find_psi_axis(psi, grid)
    
    # Verify error message is helpful
    assert "boundary" in str(excinfo.value).lower(), "Error message should mention boundary"
    assert "initial guess" in str(excinfo.value).lower() or "BC" in str(excinfo.value), \
        "Error message should suggest fix"


def test_axis_on_boundary_Z():
    """
    Test 3: Axis on Z boundary - should raise RuntimeError
    
    Similar to R boundary test
    
    Expected: RuntimeError
    """
    nr, nz = 65, 65
    R = np.linspace(1.0, 2.0, nr)
    Z = np.linspace(-0.5, 0.5, nz)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing='ij')
    
    # Maximum at Z boundary (Z=0.5)
    psi = np.exp(-(R_grid - 1.5)**2 - (Z_grid - 0.5)**2 / 0.1)
    
    grid = Grid(
        R=R_grid, Z=Z_grid,
        dR=R[1]-R[0], dZ=Z[1]-Z[0],
        nr=nr, nz=nz
    )
    
    with pytest.raises(RuntimeError) as excinfo:
        find_psi_axis(psi, grid)
    
    assert "boundary" in str(excinfo.value).lower()


def test_axis_edge_case():
    """
    Test 4: Axis at i=1 or i=nr-2 (just inside boundary)
    
    Expected: Pass (these are interior!)
    """
    nr, nz = 65, 65
    R = np.linspace(1.0, 2.0, nr)
    Z = np.linspace(-0.5, 0.5, nz)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing='ij')
    
    # Place maximum at i=1 (first interior point)
    psi = np.zeros((nr, nz))
    psi[1, nz//2] = 1.0  # Max at i=1, j=middle
    
    grid = Grid(
        R=R_grid, Z=Z_grid,
        dR=R[1]-R[0], dZ=Z[1]-Z[0],
        nr=nr, nz=nz
    )
    
    # Should pass (i=1 is interior)
    i_axis, j_axis, psi_axis = find_psi_axis(psi, grid)
    
    assert i_axis == 1, "Should find axis at i=1"
    assert psi_axis == 1.0, "Should return correct psi value"


# ============================================================================
# Expected Behavior Summary
# ============================================================================

AXIS_FINDING_EXPECTED = {
    'normal_case': {
        'interior': True,  # 0 < i < nr-1, 0 < j < nz-1
        'local_max': True,  # ψ_axis = max in neighborhood
    },
    'boundary_R': {
        'raises': RuntimeError,
        'message_contains': ['boundary', 'initial guess'],
    },
    'boundary_Z': {
        'raises': RuntimeError,
        'message_contains': ['boundary'],
    },
    'edge_case': {
        'i=1': 'allowed',  # First interior point OK
        'i=nr-2': 'allowed',  # Last interior point OK
    }
}

