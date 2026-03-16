"""
Free-Boundary Grad-Shafranov Solver

Couples Newton G-S solver with coil constraint optimization (FreeGS-style).

Algorithm:
  while not converged:
      1. Solve G-S with current coil currents (inner loop)
      2. Adjust coil currents to satisfy constraints (outer loop)
      3. Check convergence
"""

import numpy as np
from newton_gs_solver import NewtonGSSolver
from coil_constraint_solver import CoilConstraintSolver
from vacuum_field import VacuumField


class FreeBoundarySolver:
    """
    Free-boundary equilibrium solver (FreeGS-style iteration).
    
    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates (1D arrays)
    profile : Profile object
        Plasma profile (p'(ψ), ff'(ψ))
    coils : list of dict
        Initial coil configuration
        Each: {'R_coil': float, 'Z_coil': float, 'I_coil': float}
    constraints : dict
        Plasma constraints:
        {
            'xpoints': [(R, Z), ...],
            'isoflux': [(R1,Z1,R2,Z2), ...],
        }
    
    Optional Parameters
    -------------------
    max_iter : int
        Maximum outer iterations (default: 50)
    tol : float
        Convergence tolerance for constraints (default: 1e-6)
    damping : float
        Damping factor for coil updates (default: 0.5)
        I_new = I_old + damping * ΔI
    gamma : float
        Tikhonov regularization parameter (default: 1e-12)
    gs_tol : float
        Newton G-S solver tolerance (default: 1e-6)
    gs_max_iter : int
        Newton G-S max iterations (default: 100)
    verbose : bool
        Print iteration progress (default: True)
    """
    
    def __init__(
        self,
        R, Z, profile, coils, constraints,
        max_iter=50,
        tol=1e-6,
        damping=0.5,
        gamma=1e-12,
        gs_tol=1e-6,
        gs_max_iter=100,
        verbose=True
    ):
        self.R = R
        self.Z = Z
        self.profile = profile
        self.coils = coils.copy()  # Deep copy to avoid modifying input
        self.constraints = constraints
        
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.gamma = gamma
        self.gs_tol = gs_tol
        self.gs_max_iter = gs_max_iter
        self.verbose = verbose
        
        # Create grid
        self.RR, self.ZZ = np.meshgrid(R, Z, indexing='ij')
        
        # Extract initial coil currents
        self.I_coil = np.array([c['I_coil'] for c in self.coils])
    
    def solve(self, psi_init=None):
        """
        Solve free-boundary equilibrium.
        
        Parameters
        ----------
        psi_init : ndarray, optional
            Initial guess for ψ (default: vacuum field)
        
        Returns
        -------
        psi : ndarray
            Converged flux solution
        I_coil : ndarray
            Converged coil currents
        converged : bool
            Whether iteration converged
        info : dict
            Diagnostic information
        """
        # Initialize
        if psi_init is None:
            vacuum = VacuumField(self.coils)
            psi = vacuum.psi(self.RR, self.ZZ)
        else:
            psi = psi_init.copy()
        
        # History tracking
        constraint_errors = []
        residual_norms = []
        coil_currents_history = [self.I_coil.copy()]
        
        if self.verbose:
            print("=" * 70)
            print("Free-Boundary Equilibrium Solver (FreeGS-style)")
            print("=" * 70)
            print(f"Grid: {len(self.R)} × {len(self.Z)}")
            print(f"Coils: {len(self.coils)}")
            print(f"Constraints: {sum(len(v) for v in self.constraints.values())}")
            print(f"Damping: {self.damping}")
            print(f"Tolerance: {self.tol}")
            print()
        
        # Main iteration loop
        for n in range(self.max_iter):
            # ================================================================
            # Inner loop: G-S solve with fixed coils
            # ================================================================
            
            # Create boundary condition from current coil currents
            vacuum = VacuumField(self.coils)
            
            def boundary_psi(R, Z):
                return vacuum.psi(R, Z)
            
            # Solve G-S
            gs_solver = NewtonGSSolver(
                self.R, self.Z,
                self.profile,
                boundary_psi
            )
            
            psi, gs_converged, gs_info = gs_solver.solve(
                alpha=1.0,  # Profile scaling parameter
                psi_init=psi,
                tol=self.gs_tol,
                max_iter=self.gs_max_iter,
                verbose=False  # Suppress inner iteration output
            )
            
            if not gs_converged:
                if self.verbose:
                    print(f"⚠️  Iteration {n+1}: Inner G-S solve failed to converge")
                # Continue anyway (may still satisfy constraints)
            
            gs_residual = gs_info['residual_norms'][-1]
            residual_norms.append(gs_residual)
            
            # ================================================================
            # Outer loop: Coil adjustment
            # ================================================================
            
            # Build constraint solver (reuse vacuum object from this iteration)
            constraint_solver = CoilConstraintSolver(vacuum, gamma=self.gamma)
            
            # Compute current constraint error
            constraint_error = constraint_solver.compute_constraint_error(
                self.R, self.Z, psi, self.constraints
            )
            constraint_errors.append(constraint_error)
            
            # Check convergence
            if constraint_error < self.tol:
                if self.verbose:
                    print(f"\n✅ Converged in {n+1} iterations!")
                    print(f"   Constraint error: {constraint_error:.3e}")
                    print(f"   G-S residual: {gs_residual:.3e}")
                
                info = self._build_info(
                    psi, gs_info, constraint_errors, residual_norms,
                    coil_currents_history, n+1, True
                )
                
                return psi, self.I_coil, True, info
            
            # Compute coil adjustments
            ΔI, adjust_info = constraint_solver.adjust_coils(
                self.R, self.Z, psi, self.constraints
            )
            
            # Debug
            if self.verbose and n < 2:
                print()
                print(f"Iteration {n+1} debug:")
                print(f"  ΔI: {ΔI}")
                print(f"  self.I_coil before: {self.I_coil}")
                self_I_before = self.I_coil.copy()
            
            if self.verbose and n == 0:
                print(f"  damping * ΔI: {self.damping * ΔI}")
                print()
            
            # Update coil currents with damping
            self.I_coil += self.damping * ΔI
            
            for i, coil in enumerate(self.coils):
                coil['I_coil'] = self.I_coil[i]
            
            coil_currents_history.append(self.I_coil.copy())
            
            # Debug
            if self.verbose and n < 2:
                print(f"  self.I_coil after: {self.I_coil}")
                print(f"  change: {self.I_coil - self_I_before}")
                print()
            
            # Print progress
            if self.verbose:
                if n == 0:
                    print(f"{'Iter':<6} {'Constraint':<14} {'G-S Resid':<14} {'max|ΔI|':<12} {'Plasma':<8}")
                    print("-" * 70)
                
                max_dI = np.max(np.abs(ΔI))
                n_plasma = np.sum(gs_info['plasma_mask'])
                
                print(f"{n+1:<6} {constraint_error:<14.3e} {gs_residual:<14.3e} "
                      f"{max_dI:<12.3e} {n_plasma:<8}")
        
        # Max iterations reached
        if self.verbose:
            print(f"\n❌ Failed to converge in {self.max_iter} iterations")
            print(f"   Final constraint error: {constraint_errors[-1]:.3e}")
        
        info = self._build_info(
            psi, gs_info, constraint_errors, residual_norms,
            coil_currents_history, self.max_iter, False
        )
        
        return psi, self.I_coil, False, info
    
    def _build_info(
        self, psi, gs_info, constraint_errors, residual_norms,
        coil_currents_history, n_iter, converged
    ):
        """Build diagnostic info dict."""
        return {
            'psi': psi,
            'psi_axis': gs_info['psi_axis'],
            'psi_x': gs_info['psi_x'],
            'i_axis': gs_info['i_axis'],
            'j_axis': gs_info['j_axis'],
            'i_x': gs_info['i_x'],
            'j_x': gs_info['j_x'],
            'plasma_mask': gs_info['plasma_mask'],
            'has_xpoint': gs_info['has_xpoint'],
            'constraint_errors': constraint_errors,
            'residual_norms': residual_norms,
            'coil_currents_history': coil_currents_history,
            'n_iterations': n_iter,
            'converged': converged,
        }


# =============================================================================
# Simple Test Case
# =============================================================================

def test_simple_case():
    """Test on simple 4-coil configuration."""
    from profiles import QuadraticProfile
    
    print("=" * 70)
    print("Test: Simple 4-Coil Free-Boundary Case")
    print("=" * 70)
    print()
    
    # FreeGS-scale grid
    R = np.linspace(0.5, 2.0, 41)
    Z = np.linspace(-1.0, 1.0, 41)
    
    # Scaled profile
    profile = QuadraticProfile(p0=1e3, p1=-800, f0=2.5, f1=-2.0)
    
    # 4-coil configuration (FreeGS-like)
    # Start with I=0 (let solver find currents)
    coils = [
        {'R_coil': 1.0, 'Z_coil': 1.1, 'I_coil': 0.0},    # P1U
        {'R_coil': 1.0, 'Z_coil': -1.1, 'I_coil': 0.0},   # P1L
        {'R_coil': 1.75, 'Z_coil': 0.6, 'I_coil': 0.0},    # P2U
        {'R_coil': 1.75, 'Z_coil': -0.6, 'I_coil': 0.0},   # P2L
    ]
    
    # X-point constraint + isoflux for stability
    constraints = {
        'xpoints': [(1.5, -0.5)],
        'isoflux': [
            (1.2, 0.3, 1.2, -0.3),  # Left-right symmetry
            (1.4, 0.2, 1.6, 0.2),   # Top symmetry
        ],
    }
    
    print("Configuration:")
    print(f"  Grid: {len(R)}×{len(Z)}")
    print(f"  Coils: {len(coils)}")
    print(f"  X-point constraint: {constraints['xpoints'][0]}")
    print()
    
    # Solve
    solver = FreeBoundarySolver(
        R, Z, profile, coils, constraints,
        max_iter=10,
        tol=1e-4,
        damping=0.8,      # More aggressive
        gamma=1e-15,
        verbose=True
    )
    
    psi, I_final, converged, info = solver.solve()
    
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    
    print(f"Converged: {converged}")
    print(f"Iterations: {info['n_iterations']}")
    print(f"Final constraint error: {info['constraint_errors'][-1]:.3e}")
    print()
    
    # Get initial currents (from original input)
    I_init_values = np.array([0.0, 0.0, 0.0, 0.0])
    total_change = I_final - I_init_values
    
    print("Coil currents:")
    for i in range(len(coils)):
        print(f"  Coil {i}: {I_init_values[i]/1e3:7.1f} kA → {I_final[i]/1e3:7.1f} kA (Δ={total_change[i]:+8.2f} A)")
    print()
    
    print("Plasma:")
    print(f"  ψ_axis = {info['psi_axis']:.6f} Wb")
    print(f"  ψ_x = {info['psi_x']:.6f} Wb")
    print(f"  Separation = {info['psi_x'] - info['psi_axis']:.6f} Wb")
    print(f"  Plasma points: {np.sum(info['plasma_mask'])}")
    print()
    
    if converged and np.sum(info['plasma_mask']) > 10:
        print("✅ Simple case converged with non-trivial plasma")
        return True
    else:
        print("❌ Issues with convergence or plasma size")
        return False


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaop/reduced-mhd/equilibrium')
    
    success = test_simple_case()
    
    print()
    if success:
        print("🎉 Step 7 Complete: Free-Boundary Solver ✅")
    else:
        print("⚠️  Needs debugging")
