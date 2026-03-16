"""
Fixed-Boundary Picard Solver for Grad-Shafranov Equation

Solves:
    Δ*ψ = -μ₀R²p'(ψ) - f(ψ)f'(ψ)
    
where Δ* = R∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²

Method: Picard iteration
    1. Freeze p'(ψ), ff'(ψ) at current ψ
    2. Solve linear Poisson problem for ψ_new
    3. Update ψ ← ψ_new
    4. Repeat until convergence

Boundary conditions: ψ prescribed on boundary (fixed-boundary)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FixedBoundaryPicardSolver:
    """
    Fixed-boundary Picard solver for G-S equation.
    
    Parameters
    ----------
    R : ndarray (nr,)
        Radial grid points
    Z : ndarray (nz,)
        Vertical grid points
    profile_model : callable
        Function with signature rhs = profile_model(psi, R, Z)
        Returns RHS of G-S equation: -μ₀R²p'(ψ) - ff'(ψ)
    boundary_psi : callable
        Function with signature psi = boundary_psi(R, Z)
        Returns ψ on boundary
    """
    
    def __init__(self, R, Z, profile_model, boundary_psi):
        self.R = R
        self.Z = Z
        self.nr = len(R)
        self.nz = len(Z)
        self.dR = R[1] - R[0]  # Assume uniform grid
        self.dZ = Z[1] - Z[0]
        
        self.profile_model = profile_model
        self.boundary_psi = boundary_psi
        
        # Create mesh grid
        self.RR, self.ZZ = np.meshgrid(R, Z, indexing='ij')
        
        # Build Δ* operator matrix (constant, build once)
        self.L = self._build_delta_star_operator()
        
        # Identify boundary nodes
        self.boundary_mask = self._identify_boundary()
        self.interior_mask = ~self.boundary_mask
        
        # Get boundary values
        self.psi_boundary = boundary_psi(self.RR, self.ZZ)
        
    def _build_delta_star_operator(self):
        """
        Build finite difference matrix for Δ* operator.
        
        Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
        
        Using 2nd order centered differences on uniform grid.
        """
        n = self.nr * self.nz
        L = sp.lil_matrix((n, n))
        
        dR = self.dR
        dZ = self.dZ
        
        for i in range(self.nr):
            R = self.R[i]
            
            for j in range(self.nz):
                idx = self._index(i, j)
                
                # Interior point (5-point stencil)
                if 0 < i < self.nr-1 and 0 < j < self.nz-1:
                    # ∂²ψ/∂R²
                    L[idx, self._index(i-1, j)] += 1.0 / dR**2
                    L[idx, idx] -= 2.0 / dR**2
                    L[idx, self._index(i+1, j)] += 1.0 / dR**2
                    
                    # -(1/R)∂ψ/∂R
                    L[idx, self._index(i-1, j)] -= 1.0 / (2*R*dR)
                    L[idx, self._index(i+1, j)] += 1.0 / (2*R*dR)
                    
                    # ∂²ψ/∂Z²
                    L[idx, self._index(i, j-1)] += 1.0 / dZ**2
                    L[idx, idx] -= 2.0 / dZ**2
                    L[idx, self._index(i, j+1)] += 1.0 / dZ**2
                
                else:
                    # Boundary point (identity)
                    L[idx, idx] = 1.0
        
        return L.tocsr()
    
    def _index(self, i, j):
        """Convert 2D index (i,j) to 1D index."""
        return i * self.nz + j
    
    def _identify_boundary(self):
        """Identify boundary nodes."""
        mask = np.zeros((self.nr, self.nz), dtype=bool)
        mask[0, :] = True   # Left boundary (R_min)
        mask[-1, :] = True  # Right boundary (R_max)
        mask[:, 0] = True   # Bottom boundary (Z_min)
        mask[:, -1] = True  # Top boundary (Z_max)
        return mask
    
    def solve(self, psi_init=None, tol=1e-6, max_iter=100, omega=1.0, verbose=True):
        """
        Solve G-S equation using Picard iteration.
        
        Parameters
        ----------
        psi_init : ndarray (nr, nz), optional
            Initial guess. If None, use boundary values.
        tol : float
            Convergence tolerance (relative change in ψ)
        max_iter : int
            Maximum number of iterations
        omega : float
            Relaxation parameter (1.0 = no relaxation)
        verbose : bool
            Print iteration info
        
        Returns
        -------
        psi : ndarray (nr, nz)
            Converged solution
        converged : bool
            Whether iteration converged
        iterations : int
            Number of iterations performed
        residuals : list of float
            Residual history
        """
        # Initialize
        if psi_init is None:
            psi = self.psi_boundary.copy()
        else:
            psi = psi_init.copy()
            # Enforce boundary conditions
            psi[self.boundary_mask] = self.psi_boundary[self.boundary_mask]
        
        residuals = []
        
        if verbose:
            print("Fixed-Boundary Picard Iteration")
            print("=" * 60)
            print(f"Grid: {self.nr} × {self.nz} = {self.nr*self.nz} points")
            print(f"Tolerance: {tol:.2e}")
            print(f"Max iterations: {max_iter}")
            print()
        
        for iteration in range(max_iter):
            # 1. Compute RHS from current ψ
            rhs_field = self.profile_model(psi, self.RR, self.ZZ)
            rhs = rhs_field.flatten()
            
            # 2. Modify RHS for boundary conditions
            #    L·ψ = rhs  →  ψ_boundary = prescribed
            rhs_bc = rhs.copy()
            for i in range(self.nr):
                for j in range(self.nz):
                    idx = self._index(i, j)
                    if self.boundary_mask[i, j]:
                        rhs_bc[idx] = self.psi_boundary[i, j]
            
            # 3. Solve linear system
            psi_new_flat = spla.spsolve(self.L, rhs_bc)
            psi_new = psi_new_flat.reshape(self.nr, self.nz)
            
            # 4. Apply relaxation
            psi_relaxed = omega * psi_new + (1 - omega) * psi
            
            # 5. Check convergence
            delta = psi_relaxed - psi
            residual = np.linalg.norm(delta) / np.linalg.norm(psi)
            residuals.append(residual)
            
            if verbose:
                print(f"Iter {iteration+1:3d}: residual = {residual:.6e}")
            
            # 6. Update
            psi = psi_relaxed
            
            # 7. Check convergence
            if residual < tol:
                if verbose:
                    print()
                    print(f"Converged in {iteration+1} iterations!")
                return psi, True, iteration+1, residuals
        
        # Did not converge
        if verbose:
            print()
            print(f"Did not converge in {max_iter} iterations.")
            print(f"Final residual: {residuals[-1]:.6e}")
        
        return psi, False, max_iter, residuals


class SolovevProfileModel:
    """
    Profile model for Solov'ev equilibrium.
    
    Linear profiles:
        p(ψ) = p0 + p1·ψ  →  p'(ψ) = p1
        f²(ψ) = f0² + f1·ψ  →  f·f'(ψ) = f1/2
    
    RHS = -μ₀R²·p1 - f1/2
    """
    
    def __init__(self, p1, f1, mu0=4*np.pi*1e-7):
        self.p1 = p1
        self.f1 = f1
        self.mu0 = mu0
    
    def __call__(self, psi, R, Z):
        """Compute RHS of G-S equation."""
        return -self.mu0 * R**2 * self.p1 - self.f1 / 2


def test_solovev_recovery():
    """
    Test: Recover Solov'ev analytical solution using Picard.
    
    This is a perfect test case because:
    - Analytical solution is known exactly
    - RHS is constant (linear profiles)
    - Should converge in 1 iteration!
    """
    try:
        from .solovev import make_standard_solovev
    except ImportError:
        from solovev import make_standard_solovev
    
    print("Test: Solov'ev Equilibrium Recovery")
    print("=" * 70)
    print()
    
    # Create analytical solution
    sol = make_standard_solovev()
    
    print("Analytical Solov'ev parameters:")
    print(f"  A = {sol.A:.6f}")
    print(f"  C = {sol.C:.6f}")
    print(f"  p1 = {sol.p1:.6e}")
    print(f"  f1 = {sol.f1:.6f}")
    print()
    
    # Create grid
    R = np.linspace(1, 8, 50)
    Z = np.linspace(-3, 3, 50)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    
    # Exact solution on grid
    psi_exact = sol.psi(RR, ZZ)
    
    # Boundary condition: exact ψ on boundary
    def boundary_psi(R, Z):
        return sol.psi(R, Z)
    
    # Profile model
    profile = SolovevProfileModel(sol.p1, sol.f1, sol.mu0)
    
    # Create solver
    solver = FixedBoundaryPicardSolver(R, Z, profile, boundary_psi)
    
    # Solve
    print("Solving with Picard iteration...")
    print()
    psi_numerical, converged, iters, residuals = solver.solve(
        tol=1e-10, max_iter=10, verbose=True
    )
    
    # Compare with analytical
    error = np.linalg.norm(psi_numerical - psi_exact) / np.linalg.norm(psi_exact)
    max_error = np.max(np.abs(psi_numerical - psi_exact))
    
    print()
    print("Comparison with Analytical Solution")
    print("=" * 70)
    print(f"Relative L2 error: {error:.6e}")
    print(f"Max absolute error: {max_error:.6e}")
    print()
    
    if error < 1e-6:
        print("✅ SUCCESS: Numerical solution matches analytical!")
    else:
        print("❌ FAILURE: Large error detected")
    
    return psi_numerical, psi_exact, error


if __name__ == '__main__':
    # Run test
    psi_num, psi_exact, error = test_solovev_recovery()
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Analytical
        im0 = axes[0].contourf(psi_exact.T, levels=20, cmap='RdBu_r')
        axes[0].set_title('Analytical Solov\'ev')
        axes[0].set_xlabel('R index')
        axes[0].set_ylabel('Z index')
        plt.colorbar(im0, ax=axes[0])
        
        # Numerical
        im1 = axes[1].contourf(psi_num.T, levels=20, cmap='RdBu_r')
        axes[1].set_title('Numerical (Picard)')
        axes[1].set_xlabel('R index')
        axes[1].set_ylabel('Z index')
        plt.colorbar(im1, ax=axes[1])
        
        # Error
        error_field = np.abs(psi_num - psi_exact)
        im2 = axes[2].contourf(error_field.T, levels=20, cmap='Reds')
        axes[2].set_title(f'Absolute Error (max={error_field.max():.2e})')
        axes[2].set_xlabel('R index')
        axes[2].set_ylabel('Z index')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('solovev_validation.png', dpi=150)
        print("Plot saved to: solovev_validation.png")
        
    except ImportError:
        print("(matplotlib not available, skipping plot)")
