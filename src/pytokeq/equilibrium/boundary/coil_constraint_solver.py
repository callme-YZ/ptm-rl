"""
Coil Constraint Solver for Free-Boundary Equilibrium

Adjusts coil currents to satisfy plasma constraints (X-points, isoflux).
Uses linearized sensitivity matrix + Tikhonov regularization (FreeGS-style).
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


class CoilConstraintSolver:
    """
    Solve for coil current adjustments to satisfy constraints.
    
    Method: Linearized sensitivity + Tikhonov regularization
    
    minimize ||A·ΔI - b||² + γ²||ΔI||²
    
    where:
      - A: constraint sensitivity matrix (K×M)
      - b: constraint violations (K,)
      - ΔI: coil current adjustments (M,)
      - γ: regularization parameter
    
    Parameters
    ----------
    vacuum_field : VacuumField
        Vacuum field object with sensitivity methods
    gamma : float, optional
        Tikhonov regularization parameter (default: 1e-12)
    """
    
    def __init__(self, vacuum_field, gamma=1e-12):
        self.vacuum_field = vacuum_field
        self.gamma = gamma
        self.n_coils = len(vacuum_field.coils)
    
    def adjust_coils(self, R, Z, psi, constraints):
        """
        Compute coil current adjustments to satisfy constraints.
        
        Parameters
        ----------
        R, Z : ndarray
            Grid coordinates (1D arrays)
        psi : ndarray
            Current flux solution (2D array, shape (len(R), len(Z)))
        constraints : dict
            Constraint specification:
            {
                'xpoints': [(R_x, Z_x), ...],      # X-point locations
                'isoflux': [(R1,Z1,R2,Z2), ...],   # Isoflux constraints
            }
        
        Returns
        -------
        ΔI : ndarray
            Coil current adjustments (A), shape (n_coils,)
        info : dict
            Diagnostic information:
            {
                'constraint_matrix': A,
                'constraint_rhs': b,
                'constraint_error_before': float,
                'n_constraints': int,
            }
        """
        # Build constraint matrix and RHS
        A, b = self._build_constraint_matrix(R, Z, psi, constraints)
        
        if len(b) == 0:
            raise ValueError("No constraints specified")
        
        # Solve Tikhonov problem
        ΔI = self._tikhonov_solve(A, b)
        
        # Compute error before adjustment
        constraint_error = np.linalg.norm(b)
        
        info = {
            'constraint_matrix': A,
            'constraint_rhs': b,
            'constraint_error_before': constraint_error,
            'n_constraints': len(b),
        }
        
        return ΔI, info
    
    def _build_constraint_matrix(self, R, Z, psi, constraints):
        """
        Build linearized constraint matrix A and RHS b.
        
        Returns
        -------
        A : ndarray
            Sensitivity matrix, shape (n_constraints, n_coils)
        b : ndarray
            Constraint violations, shape (n_constraints,)
        """
        A_rows = []
        b_rows = []
        
        # Create interpolator for psi
        psi_interp = RectBivariateSpline(R, Z, psi, kx=3, ky=3)
        
        # X-point constraints: Br=0, Bz=0
        if 'xpoints' in constraints:
            for (R_x, Z_x) in constraints['xpoints']:
                # Current Br, Bz at X-point
                Br_current = self._compute_Br(R, Z, psi, R_x, Z_x, psi_interp)
                Bz_current = self._compute_Bz(R, Z, psi, R_x, Z_x, psi_interp)
                
                # Sensitivity: how each coil affects Br, Bz
                dBr_dI = np.array([
                    self.vacuum_field.dBr_dI(R_x, Z_x, i) 
                    for i in range(self.n_coils)
                ])
                dBz_dI = np.array([
                    self.vacuum_field.dBz_dI(R_x, Z_x, i) 
                    for i in range(self.n_coils)
                ])
                
                # Constraint: Br_new = Br_current + Σ(dBr/dI_i)·ΔI_i = 0
                # Rearrange: Σ(dBr/dI_i)·ΔI_i = -Br_current
                A_rows.append(dBr_dI)
                b_rows.append(-Br_current)
                
                A_rows.append(dBz_dI)
                b_rows.append(-Bz_current)
        
        # Isoflux constraints: ψ(R1,Z1) = ψ(R2,Z2)
        if 'isoflux' in constraints:
            for (R1, Z1, R2, Z2) in constraints['isoflux']:
                # Current ψ values
                psi1 = float(psi_interp(R1, Z1, grid=False))
                psi2 = float(psi_interp(R2, Z2, grid=False))
                
                # Sensitivity: how each coil affects ψ at these points
                dpsi1_dI = np.array([
                    self.vacuum_field.dpsi_dI(R1, Z1, i) 
                    for i in range(self.n_coils)
                ])
                dpsi2_dI = np.array([
                    self.vacuum_field.dpsi_dI(R2, Z2, i) 
                    for i in range(self.n_coils)
                ])
                
                # Constraint: ψ1_new - ψ2_new = 0
                # (psi1 + Σ dpsi1/dI·ΔI) - (psi2 + Σ dpsi2/dI·ΔI) = 0
                # Σ(dpsi1/dI - dpsi2/dI)·ΔI = psi2 - psi1
                A_rows.append(dpsi1_dI - dpsi2_dI)
                b_rows.append(psi2 - psi1)
        
        if len(b_rows) == 0:
            return np.zeros((0, self.n_coils)), np.array([])
        
        A = np.array(A_rows)
        b = np.array(b_rows)
        
        return A, b
    
    def _compute_Br(self, R, Z, psi, R_pt, Z_pt, psi_interp):
        """
        Compute Br = -1/R · ∂ψ/∂Z at (R_pt, Z_pt).
        
        Uses spline interpolator for derivatives.
        """
        # dpsi/dZ using spline
        dpsi_dZ = float(psi_interp(R_pt, Z_pt, dy=1, grid=False))
        Br = -dpsi_dZ / R_pt
        return Br
    
    def _compute_Bz(self, R, Z, psi, R_pt, Z_pt, psi_interp):
        """
        Compute Bz = 1/R · ∂ψ/∂R at (R_pt, Z_pt).
        
        Uses spline interpolator for derivatives.
        """
        # dpsi/dR using spline
        dpsi_dR = float(psi_interp(R_pt, Z_pt, dx=1, grid=False))
        Bz = dpsi_dR / R_pt
        return Bz
    
    def _tikhonov_solve(self, A, b):
        """
        Solve Tikhonov regularized least squares:
        
        minimize ||A·ΔI - b||² + γ²||ΔI||²
        
        Solution: ΔI = (AᵀA + γ²I)⁻¹ · Aᵀb
        
        Parameters
        ----------
        A : ndarray, shape (K, M)
            Constraint matrix
        b : ndarray, shape (K,)
            Constraint RHS
        
        Returns
        -------
        ΔI : ndarray, shape (M,)
            Coil current adjustments
        """
        ATA = A.T @ A
        ATb = A.T @ b
        
        # Regularized normal equations
        n = ATA.shape[0]
        regularized = ATA + self.gamma**2 * np.eye(n)
        
        # Solve
        ΔI = np.linalg.solve(regularized, ATb)
        
        return ΔI
    
    def compute_constraint_error(self, R, Z, psi, constraints):
        """
        Compute current constraint violation (for convergence check).
        
        Returns
        -------
        error : float
            ||constraint violations||₂
        """
        psi_interp = RectBivariateSpline(R, Z, psi, kx=3, ky=3)
        
        errors_sq = []
        
        # X-point errors
        if 'xpoints' in constraints:
            for (R_x, Z_x) in constraints['xpoints']:
                Br = self._compute_Br(R, Z, psi, R_x, Z_x, psi_interp)
                Bz = self._compute_Bz(R, Z, psi, R_x, Z_x, psi_interp)
                errors_sq.append(Br**2 + Bz**2)
        
        # Isoflux errors
        if 'isoflux' in constraints:
            for (R1, Z1, R2, Z2) in constraints['isoflux']:
                psi1 = float(psi_interp(R1, Z1, grid=False))
                psi2 = float(psi_interp(R2, Z2, grid=False))
                errors_sq.append((psi1 - psi2)**2)
        
        return np.sqrt(np.sum(errors_sq))


# =============================================================================
# Unit Tests
# =============================================================================

def test_constraint_matrix_xpoint():
    """Test constraint matrix for single X-point."""
    from vacuum_field import make_standard_tokamak_vacuum
    
    print("Test 1: X-point Constraint Matrix")
    print("=" * 70)
    
    # Create vacuum field with 3 coils
    vacuum = make_standard_tokamak_vacuum(R0=1.25, a=0.5, I_coil=1e5)
    
    # Simple grid
    R = np.linspace(0.5, 2.0, 31)
    Z = np.linspace(-1.0, 1.0, 31)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    # Vacuum field as initial psi
    psi = vacuum.psi(RR, ZZ)
    
    # Single X-point constraint
    R_x, Z_x = 1.5, 0.5
    constraints = {'xpoints': [(R_x, Z_x)]}
    
    # Build matrix
    solver = CoilConstraintSolver(vacuum)
    A, b = solver._build_constraint_matrix(R, Z, psi, constraints)
    
    print(f"Constraint matrix shape: {A.shape}")
    print(f"Expected: (2, 3)  [2 constraints (Br, Bz), 3 coils]")
    print()
    
    print(f"Constraint RHS (b): {b}")
    print(f"  b[0] = -Br(X-point)")
    print(f"  b[1] = -Bz(X-point)")
    print()
    
    print("Sensitivity matrix A:")
    print(A)
    print()
    
    # Check properties
    checks = [
        ("Shape correct", A.shape == (2, 3)),
        ("RHS finite", np.all(np.isfinite(b))),
        ("Matrix finite", np.all(np.isfinite(A))),
        ("Matrix non-zero", np.any(A != 0)),
    ]
    
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print()
    
    if all(r for _, r in checks):
        print("✅ X-point constraint matrix correct")
        return True
    else:
        print("❌ Issues found")
        return False


def test_tikhonov_solve():
    """Test Tikhonov solver."""
    from vacuum_field import make_standard_tokamak_vacuum
    
    print("Test 2: Tikhonov Solve")
    print("=" * 70)
    
    vacuum = make_standard_tokamak_vacuum(R0=1.25, a=0.5, I_coil=1e5)
    
    # Test problem: A·x = b
    # 2 constraints, 3 coils (underdetermined)
    A = np.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
    ])
    b = np.array([1.0, 0.5])
    
    solver = CoilConstraintSolver(vacuum, gamma=1e-3)
    x = solver._tikhonov_solve(A, b)
    
    print(f"Problem: A·x = b")
    print(f"A shape: {A.shape}")
    print(f"b: {b}")
    print()
    print(f"Solution x: {x}")
    print()
    
    # Check residual
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    
    print(f"Residual ||A·x - b||: {residual_norm:.3e}")
    print(f"Solution norm ||x||:  {np.linalg.norm(x):.3e}")
    print()
    
    # Tikhonov should give small but finite solution
    checks = [
        ("Solution finite", np.all(np.isfinite(x))),
        ("Residual small", residual_norm < 0.1),
        ("Solution small", np.linalg.norm(x) < 10.0),  # Regularized
    ]
    
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print()
    
    if all(r for _, r in checks):
        print("✅ Tikhonov solver works")
        return True
    else:
        print("❌ Solver issues")
        return False


def test_full_adjustment():
    """Test full coil adjustment pipeline."""
    from vacuum_field import make_standard_tokamak_vacuum
    
    print("Test 3: Full Coil Adjustment")
    print("=" * 70)
    
    vacuum = make_standard_tokamak_vacuum(R0=1.25, a=0.5, I_coil=1e5)
    
    R = np.linspace(0.5, 2.0, 41)
    Z = np.linspace(-1.0, 1.0, 41)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    psi = vacuum.psi(RR, ZZ)
    
    # X-point constraint
    R_x, Z_x = 1.5, 0.5
    constraints = {'xpoints': [(R_x, Z_x)]}
    
    solver = CoilConstraintSolver(vacuum)
    
    # Compute error before
    error_before = solver.compute_constraint_error(R, Z, psi, constraints)
    
    print(f"Constraint error before: {error_before:.3e}")
    print()
    
    # Adjust coils
    ΔI, info = solver.adjust_coils(R, Z, psi, constraints)
    
    print(f"Coil current adjustments (ΔI):")
    for i, dI in enumerate(ΔI):
        print(f"  Coil {i}: {dI:+.3e} A")
    print()
    
    print(f"Constraint matrix rank: {np.linalg.matrix_rank(info['constraint_matrix'])}")
    print(f"Number of constraints: {info['n_constraints']}")
    print()
    
    # Check if adjustment makes sense
    checks = [
        ("ΔI finite", np.all(np.isfinite(ΔI))),
        ("ΔI non-zero", np.any(ΔI != 0)),
        ("ΔI reasonable", np.all(np.abs(ΔI) < 1e6)),  # < 1 MA
        ("Matrix full rank", np.linalg.matrix_rank(info['constraint_matrix']) == min(info['constraint_matrix'].shape)),
    ]
    
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print()
    
    if all(r for _, r in checks):
        print("✅ Full adjustment pipeline works")
        return True
    else:
        print("❌ Issues in pipeline")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Coil Constraint Solver Tests")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("X-point matrix", test_constraint_matrix_xpoint()))
    print()
    results.append(("Tikhonov solve", test_tikhonov_solve()))
    print()
    results.append(("Full adjustment", test_full_adjustment()))
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, status in results:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}")
    
    print()
    
    if all(status for _, status in results):
        print("🎉 Step 6.1 Complete: CoilConstraintSolver ✅")
    else:
        print("⚠️  Some tests failed")
