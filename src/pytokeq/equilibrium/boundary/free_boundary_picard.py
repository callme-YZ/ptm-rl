"""
Free-Boundary Picard Iteration Solver

Integrates X-point detection, coil optimization, and G-S solving
into a unified free-boundary equilibrium solver.

Reference: Design doc Section 3.3, Appendix B.3
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..diagnostics.xpoint_finder import find_xpoints, select_primary_xpoint, is_xpoint_valid, XPoint
from .free_boundary_constraints import (
    CoilSet, IsofluxPair, build_constraint_matrix, 
    build_target_vector, optimize_coil_currents, check_constraint_matrix
)
from ..utils.greens_function import greens_psi


@dataclass
class FreeBoundaryResult:
    """Result from free-boundary solver"""
    psi: np.ndarray  # Flux solution
    xpoint: Optional[XPoint]  # Detected X-point
    I_coils: np.ndarray  # Optimized coil currents
    converged: bool  # Convergence flag
    n_iterations: int  # Number of iterations
    residuals: dict  # Convergence history


def solve_free_boundary_picard(
    profile,  # Profile with p', FF'
    grid,  # Computational grid
    coils: CoilSet,  # Coil configuration
    isoflux_pairs: list,  # List[IsofluxPair]
    solve_gs_interior,  # G-S solver function
    max_iter: int = 50,
    tol_psi: float = 1e-6,
    tol_I: float = 1e-3,
    tol_xpoint: float = None,  # Auto-set from grid
    damping: float = 0.5,
    gamma: float = 1e-12,
    verbose: bool = True
) -> FreeBoundaryResult:
    """
    Solve free-boundary equilibrium with decoupled Picard iteration
    
    Algorithm (decoupled to avoid circular dependency):
    
    Iteration k:
      1. Solve G-S with I_coils^(k-1) → ψ^(k)
      2. Detect X-point in ψ^(k) → X^(k)
      3. Optimize I based on ψ^(k), X^(k) → I^(k)
      4. Check convergence
    
    Parameters
    ----------
    profile : Profile
        Plasma profiles (p', FF')
    grid : Grid
        Computational grid
    coils : CoilSet
        Coil locations and initial currents
    isoflux_pairs : List[IsofluxPair]
        Isoflux constraint pairs
    solve_gs_interior : callable
        G-S solver: psi = solve_gs_interior(profile, grid, psi_boundary, ...)
    max_iter : int
        Maximum Picard iterations
    tol_psi : float
        Flux convergence tolerance (absolute)
    tol_I : float
        Coil current convergence tolerance (relative)
    tol_xpoint : float or None
        X-point position tolerance (m). If None, set to 1×grid cell
    damping : float
        Damping factor for coil currents (0.5 = 50% old, 50% new)
    gamma : float
        Tikhonov regularization parameter
    verbose : bool
        Print iteration info
        
    Returns
    -------
    result : FreeBoundaryResult
        Converged solution
    """
    
    # Auto-set X-point tolerance
    if tol_xpoint is None:
        grid_scale = np.sqrt(grid.dR**2 + grid.dZ**2)
        tol_xpoint = 1.0 * grid_scale
    
    # Initialize
    I_coils = coils.I.copy()
    psi = initialize_flux(grid, coils, I_coils)
    
    xpoint = None
    xpoint_old = None
    
    residuals = {
        'psi': [],
        'I': [],
        'xpoint': []
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Free-Boundary Picard Iteration")
        print("="*60)
        print(f"Coils: {len(I_coils)}, Constraints: {2 + len(isoflux_pairs)}")
        print(f"Tol: Δψ={tol_psi:.1e}, ΔI={tol_I:.1e}, ΔX={tol_xpoint:.3f}m")
        print("="*60)
    
    for iteration in range(max_iter):
        psi_old = psi.copy()
        I_old = I_coils.copy()
        
        # ================================================================
        # STEP 1: Solve G-S with current coil currents (FROZEN)
        # ================================================================
        psi_boundary = compute_coil_flux(grid, coils.R, coils.Z, I_coils)
        psi = solve_gs_interior(profile, grid, psi_boundary)
        
        # ================================================================
        # STEP 2: Detect X-point in CURRENT solution (DIAGNOSTIC)
        # ================================================================
        xpoints = find_xpoints(psi, grid)
        
        if len(xpoints) == 0:
            if verbose:
                print(f"  Iter {iteration+1}: No X-point found, continuing...")
            xpoint = None
        else:
            xpoint = select_primary_xpoint(xpoints)
            
            # Validate
            valid, msg = is_xpoint_valid(xpoint, grid)
            if not valid:
                if verbose:
                    print(f"  Iter {iteration+1}: X-point invalid ({msg})")
                xpoint = None
        
        # ================================================================
        # STEP 3: Optimize coil currents (INDEPENDENT)
        # ================================================================
        if xpoint is not None:
            # Build constraint system
            A = build_constraint_matrix(coils, xpoint, isoflux_pairs)
            b = build_target_vector(psi, grid, xpoint, isoflux_pairs)
            
            # Check matrix
            valid, msg = check_constraint_matrix(A, gamma)
            if not valid:
                if verbose:
                    print(f"  Iter {iteration+1}: Constraint matrix {msg}")
                # Continue with current currents
                I_new = I_coils.copy()
            else:
                # Optimize
                I_new = optimize_coil_currents(A, b, gamma)
                
                # Damping to prevent oscillation
                I_coils = (1 - damping) * I_old + damping * I_new
        else:
            # No X-point: keep currents (or could optimize for isoflux only)
            I_new = I_coils.copy()
        
        # ================================================================
        # STEP 4: Check convergence
        # ================================================================
        delta_psi = np.linalg.norm(psi - psi_old)
        delta_I = np.linalg.norm(I_coils - I_old) / (np.linalg.norm(I_old) + 1e-10)
        
        if xpoint and xpoint_old:
            delta_xpoint = np.sqrt((xpoint.R - xpoint_old.R)**2 + 
                                   (xpoint.Z - xpoint_old.Z)**2)
        else:
            delta_xpoint = np.inf
        
        residuals['psi'].append(delta_psi)
        residuals['I'].append(delta_I)
        residuals['xpoint'].append(delta_xpoint)
        
        if verbose:
            xp_str = f"X=({xpoint.R:.2f},{xpoint.Z:.2f})" if xpoint else "No X-point"
            print(f"  Iter {iteration+1:2d}: Δψ={delta_psi:.2e}, ΔI={delta_I:.2e}, "
                  f"ΔX={delta_xpoint:.3f}m, {xp_str}")
        
        # Convergence check
        psi_converged = delta_psi < tol_psi
        I_converged = delta_I < tol_I
        xpoint_converged = delta_xpoint < tol_xpoint
        
        converged = psi_converged and I_converged and xpoint_converged
        
        if converged:
            if verbose:
                print(f"\n✅ Converged in {iteration+1} iterations!")
            break
        
        # Store for next iteration
        xpoint_old = xpoint
    
    else:
        # Max iterations reached
        if verbose:
            print(f"\n⚠️ Max iterations ({max_iter}) reached without convergence")
        converged = False
    
    return FreeBoundaryResult(
        psi=psi,
        xpoint=xpoint,
        I_coils=I_coils,
        converged=converged,
        n_iterations=iteration+1,
        residuals=residuals
    )


def initialize_flux(grid, coils, I_coils: np.ndarray) -> np.ndarray:
    """
    Initialize flux for free-boundary iteration
    
    Simple initialization: just vacuum field from coils
    (Could use 3-stage FreeGS method from Appendix A.1)
    """
    psi = compute_coil_flux(grid, coils.R, coils.Z, I_coils)
    return psi


def compute_coil_flux(grid, R_coils: np.ndarray, Z_coils: np.ndarray, 
                      I_coils: np.ndarray) -> np.ndarray:
    """
    Compute vacuum flux from coils
    
    ψ(R,Z) = Σ I_i · G(R_i, Z_i; R, Z)
    """
    psi = np.zeros_like(grid.R)
    
    for R_c, Z_c, I_c in zip(R_coils, Z_coils, I_coils):
        psi += I_c * greens_psi(R_c, Z_c, grid.R, grid.Z)
    
    return psi


if __name__ == "__main__":
    print("Free-Boundary Picard Solver")
    print("="*60)
    
    # Mock grid
    class MockGrid:
        def __init__(self):
            R = np.linspace(0.5, 2.0, 65)
            Z = np.linspace(-1.0, 1.0, 65)
            self.R, self.Z = np.meshgrid(R, Z, indexing='ij')
            self.dR = R[1] - R[0]
            self.dZ = Z[1] - Z[0]
    
    grid = MockGrid()
    
    # Mock profile
    class MockProfile:
        def pprime(self, psi):
            return -0.1 * np.ones_like(psi)
        def ffprime(self, psi):
            return -0.5 * np.ones_like(psi)
    
    profile = MockProfile()
    
    # Coils (4-coil simple configuration)
    coils = CoilSet(
        R=np.array([0.6, 1.9, 0.6, 1.9]),
        Z=np.array([0.8, 0.8, -0.8, -0.8]),
        I=np.array([1000.0, 1000.0, -1000.0, -1000.0])  # Initial guess
    )
    
    # Isoflux (simple vertical symmetry)
    isoflux = [
        IsofluxPair(R1=1.2, Z1=0.4, R2=1.2, Z2=-0.4),
    ]
    
    # Mock G-S solver (just returns boundary for now)
    def mock_gs_solver(profile, grid, psi_boundary):
        # Real solver would solve interior Δ*ψ = RHS
        # For test: just smooth boundary values inward
        psi = psi_boundary.copy()
        # Simple Laplacian smoothing
        for _ in range(10):
            psi[1:-1, 1:-1] = 0.25 * (psi[:-2, 1:-1] + psi[2:, 1:-1] + 
                                       psi[1:-1, :-2] + psi[1:-1, 2:])
        return psi
    
    # Solve
    print("\nRunning free-boundary solver (mock G-S)...")
    print("This tests iteration structure, not physics correctness")
    print()
    
    result = solve_free_boundary_picard(
        profile=profile,
        grid=grid,
        coils=coils,
        isoflux_pairs=isoflux,
        solve_gs_interior=mock_gs_solver,
        max_iter=10,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Result Summary")
    print("="*60)
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.n_iterations}")
    
    if result.xpoint:
        print(f"\nX-point:")
        print(f"  R = {result.xpoint.R:.3f} m")
        print(f"  Z = {result.xpoint.Z:.3f} m")
    else:
        print("\nNo X-point detected")
    
    print(f"\nCoil currents:")
    for i, I in enumerate(result.I_coils):
        print(f"  Coil {i}: {I/1000:.2f} kA")
    
    print("\n✅ Free-boundary Picard structure complete!")
    print("(Physics validation requires real G-S solver)")
