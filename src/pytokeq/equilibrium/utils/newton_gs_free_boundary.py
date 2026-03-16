"""
Newton-Raphson G-S Solver with Free-Boundary (Green's BC)

Key difference from newton_gs_solver.py:
    Boundary condition updated every iteration from plasma Jtor
    (not fixed from vacuum field)
    
Algorithm:
    for iteration:
        1. Compute Jtor from current ψ
        2. Update ψ_boundary from Green's integral of Jtor
        3. Solve Newton step for interior
        4. Check convergence
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from newton_gs_solver import NewtonGSSolver
from free_boundary_condition import free_boundary_greens


class NewtonGSFreeBoundary(NewtonGSSolver):
    """
    Free-boundary Newton G-S solver with Green's function BC.
    
    Inherits from NewtonGSSolver but overrides solve() to update
    boundary condition each iteration.
    
    Parameters
    ----------
    R, Z : ndarray
        Grid coordinates
    profile_model : ProfileModel
        Plasma profiles
    """
    
    def __init__(self, R, Z, profile_model):
        # Initialize with dummy BC (will be updated each iteration)
        def dummy_bc(R_pts, Z_pts):
            return np.zeros_like(R_pts)
        
        super().__init__(R, Z, profile_model, boundary_psi=dummy_bc)
        
    def solve_free_boundary(self, psi_init=None, alpha=0.5, 
                           tol=1e-6, max_iter=50, verbose=True):
        """
        Solve free-boundary G-S with self-consistent boundary.
        
        Parameters
        ----------
        psi_init : ndarray, optional
            Initial guess (if None, use concentric circles)
        alpha : float
            Damping parameter
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        verbose : bool
            Print diagnostics
            
        Returns
        -------
        psi : ndarray
            Converged solution
        converged : bool
            True if converged
        info : dict
            Diagnostic information
        """
        
        if verbose:
            print("=" * 70)
            print("Newton G-S Free-Boundary Solver (Green's BC)")
            print("=" * 70)
            print(f"Grid: {self.nr}×{self.nz}")
            print(f"α = {alpha:.6f}")
            print()
        
        # Initial guess
        if psi_init is None:
            # Concentric circles centered at domain center
            R0 = 0.5 * (self.R[0] + self.R[-1])
            Z0 = 0.5 * (self.Z[0] + self.Z[-1])
            psi = -0.5 * ((self.RR - R0)**2 + (self.ZZ - Z0)**2)
        else:
            psi = psi_init.copy()
        
        # Iteration history
        residual_norms = []
        
        for iteration in range(max_iter):
            
            # === Step 1: Compute Jtor from current ψ ===
            
            # Find plasma region and axis
            psi_axis = psi.max()
            psi_bndry = psi[self.boundary_mask].max()
            
            plasma_mask = (psi > psi_bndry) & (psi < psi_axis)
            
            # Normalized ψ for profile
            psi_norm = np.zeros_like(psi)
            plasma_points = plasma_mask & (~self.boundary_mask)
            
            if np.sum(plasma_points) > 0:
                psi_norm[plasma_points] = (
                    (psi[plasma_points] - psi_bndry) / 
                    (psi_axis - psi_bndry)
                )
            
            # Compute Jtor using profile model
            Jtor = self.profile_model.compute_current_density(
                psi, psi_axis, psi_bndry, alpha=1.0, R=self.RR
            )
            
            # Mask outside plasma
            Jtor[~plasma_mask] = 0.0
            
            # === Step 2: Update boundary from Green's integral ===
            
            if iteration > 0:  # Skip first iteration (no plasma yet)
                if verbose and iteration == 1:
                    print("Updating boundary from Green's function...")
                    
                # Compute boundary ψ from Jtor
                free_boundary_greens(self.RR, self.ZZ, Jtor, psi)
                
                # Update internal BC storage
                self.psi_boundary = psi.copy()
            
            # === Step 3: Compute residual ===
            
            residual = self._compute_residual_free(psi, Jtor)
            res_norm = np.linalg.norm(residual)
            residual_norms.append(res_norm)
            
            # === Step 4: Diagnostics ===
            
            n_plasma = np.sum(plasma_mask)
            i_axis, j_axis = np.unravel_index(psi.argmax(), psi.shape)
            R_axis = self.R[i_axis]
            Z_axis = self.Z[j_axis]
            
            if verbose:
                print(f"Iter {iteration+1:3d}: "
                      f"|R|={res_norm:.6e}, "
                      f"plasma={n_plasma:4d} pts, "
                      f"ψ_axis={psi_axis:.4e}, "
                      f"axis @ R={R_axis:.3f}, Z={Z_axis:.3f}")
            
            # === Step 5: Check convergence ===
            
            if res_norm < tol:
                if verbose:
                    print()
                    print(f"✅ CONVERGED in {iteration+1} iterations")
                    print(f"   Final residual: {res_norm:.6e}")
                    
                    # Sanity checks
                    on_boundary = (i_axis == 0 or i_axis == self.nr-1 or 
                                  j_axis == 0 or j_axis == self.nz-1)
                    
                    if on_boundary:
                        print(f"   ⚠️  Axis on boundary")
                    else:
                        print(f"   ✅ Axis in interior")
                    
                    if n_plasma < 50:
                        print(f"   ⚠️  Few plasma points ({n_plasma})")
                    else:
                        print(f"   ✅ Plasma points: {n_plasma}")
                
                info = {
                    'residual_norms': residual_norms,
                    'plasma_mask': plasma_mask,
                    'psi_axis': psi_axis,
                    'psi_bndry': psi_bndry,
                    'i_axis': i_axis,
                    'j_axis': j_axis,
                    'Jtor': Jtor
                }
                
                return psi, True, info
            
            # === Step 6: Newton step ===
            
            # Build Jacobian
            jacobian = self._build_jacobian_free(psi, Jtor)
            
            # Solve J·δψ = -R
            try:
                delta_psi_flat = spla.spsolve(jacobian, -residual)
                delta_psi = delta_psi_flat.reshape((self.nr, self.nz))
            except:
                if verbose:
                    print(f"\n❌ Linear solve failed at iteration {iteration+1}")
                
                info = {
                    'residual_norms': residual_norms,
                    'plasma_mask': plasma_mask,
                }
                return psi, False, info
            
            # Update with damping
            psi = psi + alpha * delta_psi
        
        # Did not converge
        if verbose:
            print()
            print(f"❌ Did NOT converge in {max_iter} iterations")
            print(f"   Final residual: {res_norm:.6e}")
        
        info = {
            'residual_norms': residual_norms,
            'plasma_mask': plasma_mask,
        }
        
        return psi, False, info
    
    def _compute_residual_free(self, psi, Jtor):
        """
        Compute residual for free-boundary.
        
        Interior: R = Δ*ψ - (-μ₀·R·Jtor)
        Boundary: R = ψ - ψ_boundary (from Green's)
        """
        
        # RHS = -μ₀·R·Jtor
        mu0 = 4e-7 * np.pi
        rhs = -mu0 * self.RR * Jtor
        
        # Boundary: set RHS to boundary value
        rhs[self.boundary_mask] = psi[self.boundary_mask]
        
        # Δ*ψ
        delta_star_psi = self.L_base @ psi.flatten()
        delta_star_psi = delta_star_psi.reshape((self.nr, self.nz))
        
        # Residual
        residual = delta_star_psi - rhs
        
        return residual.flatten()
    
    def _build_jacobian_free(self, psi, Jtor):
        """
        Build Jacobian for free-boundary.
        
        J = ∂R/∂ψ
        
        For interior: J = Δ* - ∂(μ₀RJtor)/∂ψ
        For boundary: J = I (ψ held fixed during Newton step)
        
        Note: This is approximate - full Jacobian would include
        ∂Jtor/∂ψ terms (nonlinear coupling). For now use simplified.
        """
        
        # Start with Δ* operator
        J = self.L_base.copy().tolil()
        
        # Add nonlinear terms (simplified: ignore ∂Jtor/∂ψ)
        # This makes Jacobian = Δ* (linear)
        # Good enough for testing
        
        return J.tocsr()


def test_free_boundary_solver():
    """
    Test free-boundary solver on simple case.
    """
    
    print("=" * 70)
    print("TEST: Free-Boundary Newton Solver")
    print("=" * 70)
    print()
    
    # Grid (use 2^n+1 for Romberg)
    R = np.linspace(0.1, 2.0, 33)
    Z = np.linspace(-1.0, 1.0, 33)
    
    # Profile
    from profiles import QuadraticProfile
    profile = QuadraticProfile(p0=5000, p1=-4000, f0=1.7, f1=-0.5)
    
    print(f"Setup:")
    print(f"  Grid: {len(R)}×{len(Z)}")
    print(f"  Profile: QuadraticProfile(p0={profile.p0}, f0={profile.f0})")
    print()
    
    # Solver
    solver = NewtonGSFreeBoundary(R, Z, profile)
    
    # Solve
    psi, converged, info = solver.solve_free_boundary(
        alpha=0.3,  # Conservative damping
        tol=1e-6,
        max_iter=50,
        verbose=True
    )
    
    print()
    print("=" * 70)
    
    if converged:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
        print(f"   (But this is first integration - debugging expected)")
    
    return psi, converged, info


if __name__ == "__main__":
    psi, converged, info = test_free_boundary_solver()
