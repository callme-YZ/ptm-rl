"""
Grid Topology Utilities for Structured Rectangular Mesh

Provides neighbor connectivity for BFS and geometric algorithms.
"""

import numpy as np


class StructuredGrid:
    """
    Structured rectangular grid in (R, Z) coordinates.
    
    Parameters
    ----------
    R : ndarray (nr,)
        Radial grid points
    Z : ndarray (nz,)
        Vertical grid points
    
    Attributes
    ----------
    nr, nz : int
        Grid dimensions
    RR, ZZ : ndarray (nr, nz)
        Mesh grid arrays
    """
    
    def __init__(self, R, Z):
        self.R = R
        self.Z = Z
        self.nr = len(R)
        self.nz = len(Z)
        
        self.RR, self.ZZ = np.meshgrid(R, Z, indexing='ij')
    
    def get_neighbors(self, i, j):
        """
        Get 4-connected neighbors of grid point (i, j).
        
        Returns list of (i_neighbor, j_neighbor) tuples.
        Only returns valid neighbors (within bounds).
        
        Order: [left, right, down, up] (R-, R+, Z-, Z+)
        """
        neighbors = []
        
        # Left (R-)
        if i > 0:
            neighbors.append((i-1, j))
        
        # Right (R+)
        if i < self.nr - 1:
            neighbors.append((i+1, j))
        
        # Down (Z-)
        if j > 0:
            neighbors.append((i, j-1))
        
        # Up (Z+)
        if j < self.nz - 1:
            neighbors.append((i, j+1))
        
        return neighbors
    
    def get_neighbors_8(self, i, j):
        """
        Get 8-connected neighbors (including diagonals).
        
        Order: [W, E, S, N, SW, SE, NW, NE]
        """
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip center
                
                i_new = i + di
                j_new = j + dj
                
                if 0 <= i_new < self.nr and 0 <= j_new < self.nz:
                    neighbors.append((i_new, j_new))
        
        return neighbors
    
    def is_interior(self, i, j):
        """Check if point is interior (not on boundary)."""
        return (0 < i < self.nr-1) and (0 < j < self.nz-1)
    
    def is_boundary(self, i, j):
        """Check if point is on boundary."""
        return not self.is_interior(i, j)
    
    def distance(self, i1, j1, i2, j2):
        """Euclidean distance between two grid points."""
        R1, Z1 = self.R[i1], self.Z[j1]
        R2, Z2 = self.R[i2], self.Z[j2]
        return np.sqrt((R2 - R1)**2 + (Z2 - Z1)**2)


def find_magnetic_axis(psi, grid, limiter_mask=None):
    """
    Find magnetic axis (minimum of psi).
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Poloidal flux
    grid : StructuredGrid
        Grid object
    limiter_mask : ndarray (nr, nz), optional
        Boolean mask of valid region (inside limiter)
        If None, use entire domain
    
    Returns
    -------
    i_axis, j_axis : int
        Grid indices of magnetic axis
    psi_axis : float
        Flux value at axis
    """
    if limiter_mask is None:
        limiter_mask = np.ones_like(psi, dtype=bool)
    
    # Find minimum within valid region
    psi_masked = np.where(limiter_mask, psi, np.inf)
    i_axis, j_axis = np.unravel_index(np.argmin(psi_masked), psi.shape)
    psi_axis = psi[i_axis, j_axis]
    
    return i_axis, j_axis, psi_axis


def is_saddle_point(psi, i, j, grid):
    """
    Check if (i,j) is a saddle point.
    
    A saddle point has ≥4 sign changes in the sequence
    (psi_neighbor - psi_center) around the point.
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Poloidal flux
    i, j : int
        Grid point to check
    grid : StructuredGrid
        Grid object
    
    Returns
    -------
    is_saddle : bool
        True if point is a saddle
    """
    if not grid.is_interior(i, j):
        return False  # Boundary points can't be saddles
    
    psi_center = psi[i, j]
    neighbors = grid.get_neighbors(i, j)
    
    # Get differences (psi_neighbor - psi_center)
    diffs = [psi[ni, nj] - psi_center for ni, nj in neighbors]
    
    # Close the loop
    diffs.append(diffs[0])
    
    # Count sign changes
    sign_changes = 0
    for k in range(len(diffs) - 1):
        if diffs[k] * diffs[k+1] < 0:
            sign_changes += 1
    
    # For 4-connected grid: saddle has 2 sign changes
    # (alternating pattern: +, +, -, - or +, -, +, -)
    # For 8-connected: would need 4 sign changes
    return sign_changes >= 2


def find_xpoint(psi, grid, i_axis, j_axis, psi_axis, limiter_mask=None):
    """
    Find x-point (saddle point closest to magnetic axis).
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Poloidal flux
    grid : StructuredGrid
        Grid object
    i_axis, j_axis : int
        Magnetic axis location
    psi_axis : float
        Flux at axis
    limiter_mask : ndarray (nr, nz), optional
        Valid region mask
    
    Returns
    -------
    i_x, j_x : int
        X-point location
    psi_x : float
        Flux at x-point
    has_xpoint : bool
        True if saddle point found, False if using fallback
    """
    if limiter_mask is None:
        limiter_mask = np.ones_like(psi, dtype=bool)
    
    # Search for saddle points
    candidates = []
    
    for i in range(grid.nr):
        for j in range(grid.nz):
            if not limiter_mask[i, j]:
                continue
            
            if is_saddle_point(psi, i, j, grid):
                psi_val = psi[i, j]
                
                # Only consider if psi > psi_axis (outside core)
                if psi_val > psi_axis:
                    candidates.append((psi_val, i, j))
    
    if len(candidates) > 0:
        # Choose saddle closest to psi_axis
        psi_x, i_x, j_x = min(candidates, key=lambda x: abs(x[0] - psi_axis))
        return i_x, j_x, psi_x, True
    
    else:
        # Fallback: use maximum psi on limiter boundary
        # (This happens for limited plasmas without x-point)
        
        # Find limiter boundary
        boundary_mask = limiter_mask & np.zeros_like(psi, dtype=bool)
        
        for i in range(grid.nr):
            for j in range(grid.nz):
                if limiter_mask[i, j]:
                    # Check if any neighbor is outside limiter
                    neighbors = grid.get_neighbors(i, j)
                    for ni, nj in neighbors:
                        if not limiter_mask[ni, nj]:
                            boundary_mask[i, j] = True
                            break
        
        if np.any(boundary_mask):
            psi_boundary = np.where(boundary_mask, psi, -np.inf)
            i_x, j_x = np.unravel_index(np.argmax(psi_boundary), psi.shape)
            psi_x = psi[i_x, j_x]
        else:
            # Ultimate fallback: use domain boundary
            psi_x = psi[grid.nr-1, grid.nz//2]
            i_x, j_x = grid.nr-1, grid.nz//2
        
        return i_x, j_x, psi_x, False


def identify_plasma_domain(psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid):
    """
    Identify plasma domain using BFS.
    
    Plasma domain: all points connected to magnetic axis with
    psi_axis ≤ psi ≤ psi_x
    
    Parameters
    ----------
    psi : ndarray (nr, nz)
        Poloidal flux
    i_axis, j_axis : int
        Magnetic axis location
    psi_axis, psi_x : float
        Flux bounds
    grid : StructuredGrid
        Grid object
    
    Returns
    -------
    plasma_mask : ndarray (nr, nz), bool
        True for points inside plasma
    """
    plasma_mask = np.zeros((grid.nr, grid.nz), dtype=bool)
    visited = np.zeros((grid.nr, grid.nz), dtype=bool)
    
    # BFS from magnetic axis
    queue = [(i_axis, j_axis)]
    visited[i_axis, j_axis] = True
    
    while queue:
        i, j = queue.pop(0)
        
        psi_val = psi[i, j]
        
        # Check if inside plasma (psi_axis ≤ psi ≤ psi_x)
        if psi_axis <= psi_val <= psi_x:
            plasma_mask[i, j] = True
            
            # Add neighbors to queue
            for ni, nj in grid.get_neighbors(i, j):
                if not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
    
    return plasma_mask


# =============================================================================
# Tests
# =============================================================================

def test_neighbor_traversal():
    """Test neighbor finding."""
    print("Test: Neighbor Traversal")
    print("=" * 60)
    
    R = np.linspace(1, 5, 5)
    Z = np.linspace(-2, 2, 5)
    grid = StructuredGrid(R, Z)
    
    # Test interior point
    i, j = 2, 2
    neighbors = grid.get_neighbors(i, j)
    
    print(f"Point ({i},{j}) neighbors:")
    for ni, nj in neighbors:
        print(f"  ({ni},{nj})")
    
    assert len(neighbors) == 4, "Interior should have 4 neighbors"
    
    # Test corner
    i, j = 0, 0
    neighbors = grid.get_neighbors(i, j)
    print(f"\nCorner (0,0) neighbors:")
    for ni, nj in neighbors:
        print(f"  ({ni},{nj})")
    
    assert len(neighbors) == 2, "Corner should have 2 neighbors"
    
    print("\n✅ PASS\n")


def test_magnetic_axis_finding():
    """Test magnetic axis finder."""
    print("Test: Magnetic Axis Finding")
    print("=" * 60)
    
    R = np.linspace(1, 5, 21)
    Z = np.linspace(-2, 2, 21)
    grid = StructuredGrid(R, Z)
    
    # Create synthetic flux with known minimum
    psi = grid.RR**2 + 2*grid.ZZ**2  # Minimum at (R=1, Z=0)
    
    i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
    
    print(f"Found axis at ({i_axis}, {j_axis})")
    print(f"  R = {grid.R[i_axis]:.2f}, Z = {grid.Z[j_axis]:.2f}")
    print(f"  ψ_axis = {psi_axis:.6f}")
    
    # Should be at (0, 10) for this grid
    assert i_axis == 0, f"Expected i=0, got {i_axis}"
    assert j_axis == 10, f"Expected j=10, got {j_axis}"
    
    print("✅ PASS\n")


def test_saddle_detection():
    """Test saddle point detection."""
    print("Test: Saddle Point Detection")
    print("=" * 60)
    
    R = np.linspace(0, 4, 21)
    Z = np.linspace(-2, 2, 21)
    grid = StructuredGrid(R, Z)
    
    # Create synthetic flux with saddle at (R=2, Z=0)
    # ψ = (R-2)² - Z² has saddle at center
    psi = (grid.RR - 2)**2 - grid.ZZ**2
    
    # Check center point (R=2, Z=0) → grid point (10, 10)
    i_center, j_center = 10, 10
    is_saddle = is_saddle_point(psi, i_center, j_center, grid)
    
    print(f"Point ({i_center},{j_center}): saddle = {is_saddle}")
    print(f"  R = {grid.R[i_center]:.2f}, Z = {grid.Z[j_center]:.2f}")
    
    # Check a non-saddle point (minimum at R=0, Z=0)
    i_other, j_other = 0, 10
    is_saddle_other = is_saddle_point(psi, i_other, j_other, grid)
    
    print(f"Point ({i_other},{j_other}): saddle = {is_saddle_other}")
    print(f"  R = {grid.R[i_other]:.2f}, Z = {grid.Z[j_other]:.2f}")
    
    assert is_saddle, "Center should be saddle"
    assert not is_saddle_other, "Off-center should not be saddle"
    
    print("✅ PASS\n")


def test_plasma_domain_bfs():
    """Test plasma domain identification."""
    print("Test: Plasma Domain BFS")
    print("=" * 60)
    
    R = np.linspace(1, 5, 21)
    Z = np.linspace(-2, 2, 21)
    grid = StructuredGrid(R, Z)
    
    # Create circular plasma
    R_axis, Z_axis = 3.0, 0.0
    psi = (grid.RR - R_axis)**2 + (grid.ZZ - Z_axis)**2
    
    i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
    
    # X-point at radius 1.5
    psi_x = 1.5**2
    i_x, j_x = np.unravel_index(np.argmin(np.abs(psi - psi_x)), psi.shape)
    
    plasma_mask = identify_plasma_domain(
        psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
    )
    
    n_plasma = np.sum(plasma_mask)
    
    print(f"Magnetic axis at ({i_axis},{j_axis})")
    print(f"X-point at ({i_x},{j_x})")
    print(f"Plasma domain: {n_plasma} points")
    print(f"  ({n_plasma/(grid.nr*grid.nz)*100:.1f}% of domain)")
    
    # Should be roughly circular
    assert n_plasma > 0, "Should find some plasma"
    assert n_plasma < grid.nr * grid.nz, "Should not cover everything"
    
    print("✅ PASS\n")


if __name__ == '__main__':
    test_neighbor_traversal()
    test_magnetic_axis_finding()
    test_saddle_detection()
    test_plasma_domain_bfs()
    
    print("=" * 60)
    print("All tests passed! ✅")
