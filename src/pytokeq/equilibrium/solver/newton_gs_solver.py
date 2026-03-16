"""
Newton-Raphson Solver for Grad-Shafranov Equation

Solves nonlinear G-S equation:
    Δ*ψ = -μ₀R²·p'(ψ) - f·f'(ψ)

Using Newton iteration:
    J(ψ^k)·δψ = -R(ψ^k)
    ψ^(k+1) = ψ^k + δψ

where J is the Jacobian matrix and R is the residual.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class NewtonGSSolver:
    """
    Newton solver for nonlinear Grad-Shafranov equation.
    
    Parameters
    ----------
    R : ndarray (nr,)
        Radial grid
    Z : ndarray (nz,)
        Vertical grid
    profile_model : ProfileModel
        Pressure/field profile model
    boundary_psi : callable
        ψ on boundary
    """
    
    def __init__(self, R, Z, profile_model, boundary_psi):
        self.R = R
        self.Z = Z
        self.nr = len(R)
        self.nz = len(Z)
        
        self.dR = R[1] - R[0]
        self.dZ = Z[1] - Z[0]
        
        self.profile_model = profile_model
        self.boundary_psi_func = boundary_psi
        
        # Mesh grid
        self.RR, self.ZZ = np.meshgrid(R, Z, indexing='ij')
        
        # Boundary mask and values
        self.boundary_mask = np.zeros((self.nr, self.nz), dtype=bool)
        self.boundary_mask[0, :] = True   # Left
        self.boundary_mask[-1, :] = True  # Right
        self.boundary_mask[:, 0] = True   # Bottom
        self.boundary_mask[:, -1] = True  # Top
        
        self.psi_boundary = boundary_psi(self.RR, self.ZZ)
        
        # Build base Δ* operator (linear part)
        self._build_delta_star_operator()
    
    def _index(self, i, j):
        """Convert 2D index to flat index (column-major)."""
        return i + j * self.nr
    
    def _build_delta_star_operator(self):
        """
        Build linear Δ* operator (without profile terms).
        
        Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
        """
        n = self.nr * self.nz
        L = sp.lil_matrix((n, n))
        
        for i in range(self.nr):
            for j in range(self.nz):
                k = self._index(i, j)
                
                # Boundary: identity
                if self.boundary_mask[i, j]:
                    L[k, k] = 1.0
                    continue
                
                # Interior: 5-point stencil for Δ*
                R_val = self.R[i]
                
                # ∂²/∂R² terms
                L[k, self._index(i-1, j)] += 1.0 / self.dR**2
                L[k, k] += -2.0 / self.dR**2
                L[k, self._index(i+1, j)] += 1.0 / self.dR**2
                
                # -(1/R)∂/∂R terms (central difference)
                # ∂ψ/∂R ≈ (ψ[i+1] - ψ[i-1]) / (2dR)
                # -(1/R)∂ψ/∂R = -(1/R)·(ψ[i+1] - ψ[i-1])/(2dR)
                L[k, self._index(i-1, j)] -= -1.0 / (2*R_val*self.dR)  # +1/(2R*dR)
                L[k, self._index(i+1, j)] -= +1.0 / (2*R_val*self.dR)  # -1/(2R*dR)
                
                # ∂²/∂Z² terms
                L[k, self._index(i, j-1)] += 1.0 / self.dZ**2
                L[k, k] += -2.0 / self.dZ**2
                L[k, self._index(i, j+1)] += 1.0 / self.dZ**2
        
        self.L_base = L.tocsr()
    
    def _compute_residual(self, psi, alpha, psi_ma, psi_x, plasma_mask):
        """
        Compute residual R(ψ) = Δ*ψ - RHS.
        
        Returns flat residual vector.
        """
        # Apply linear operator
        psi_flat = psi.flatten(order='F')
        L_psi = self.L_base.dot(psi_flat)
        L_psi_2d = L_psi.reshape((self.nr, self.nz), order='F')
        
        # Compute RHS (only in plasma)
        rhs = np.zeros_like(psi)
        
        if np.any(plasma_mask):
            rhs[plasma_mask] = self.profile_model.compute_rhs(
                psi[plasma_mask],
                psi_ma,
                psi_x,
                alpha,
                self.RR[plasma_mask]
            )
        
        # Residual = L*ψ - RHS (interior only)
        residual_2d = L_psi_2d - rhs
        
        # Boundary: enforce ψ = ψ_boundary
        residual_2d[self.boundary_mask] = (
            psi[self.boundary_mask] - self.psi_boundary[self.boundary_mask]
        )
        
        return residual_2d.flatten(order='F')
    
    def _compute_jacobian(self, psi, alpha, psi_ma, psi_x, plasma_mask):
        """
        Compute analytical Jacobian J = ∂R/∂ψ.
        
        Residual: R(ψ) = Δ*ψ - RHS(ψ)
                       = Δ*ψ + μ₀R²·p'(ψ) + f·f'(ψ)
        
        Jacobian: J = ∂R/∂ψ = Δ* + ∂RHS/∂ψ
        
        where:
            ∂RHS/∂ψ = μ₀R²·∂p'/∂ψ + ∂(f·f')/∂ψ
        
        Chain rule:
            ∂p'/∂ψ = (∂p'/∂ψ_N)·(∂ψ_N/∂ψ)
                    = p''(ψ_N) / (ψ_x - ψ_ma)
            
            ∂(f·f')/∂ψ = ff''(ψ_N) / (ψ_x - ψ_ma)
        
        Returns
        -------
        J : scipy.sparse.csr_matrix
            Jacobian matrix (nr*nz × nr*nz)
        """
        n = self.nr * self.nz
        J = self.L_base.tolil()  # Start with Δ* operator
        
        # Compute normalized flux in plasma
        if not np.any(plasma_mask):
            # No plasma → return pure Δ*
            return J.tocsr()
        
        # Denominator for chain rule
        delta_psi = psi_x - psi_ma
        
        if abs(delta_psi) < 1e-10:
            # Degenerate case: axis = separatrix
            # No plasma current → return pure Δ*
            return J.tocsr()
        
        # Compute ψ_N in plasma
        psi_N_plasma = (psi[plasma_mask] - psi_ma) / delta_psi
        psi_N_plasma = np.clip(psi_N_plasma, 0, 1)
        
        # Get profile second derivatives
        p_dd = self.profile_model.p_double_prime(psi_N_plasma, alpha)
        ff_dd = self.profile_model.ff_double_prime(psi_N_plasma, alpha)
        
        # Add diagonal terms for plasma points
        mu0 = 4 * np.pi * 1e-7
        plasma_index = 0
        
        for i in range(self.nr):
            for j in range(self.nz):
                if self.boundary_mask[i, j]:
                    continue  # Boundary already identity
                
                if not plasma_mask[i, j]:
                    continue  # Outside plasma
                
                # This is a plasma point
                R_val = self.RR[i, j]
                
                # ∂RHS/∂ψ = μ₀R²·p''/(Δψ) + ff''/(Δψ)
                d_rhs_d_psi = (
                    mu0 * R_val**2 * p_dd[plasma_index] / delta_psi
                    + ff_dd[plasma_index] / delta_psi
                )
                
                # J = Δ* + ∂RHS/∂ψ
                k = self._index(i, j)
                J[k, k] += d_rhs_d_psi
                
                plasma_index += 1
        
        return J.tocsr()
    
    def solve(self, alpha, psi_init=None, tol=1e-6, max_iter=20, verbose=True):
        """
        Solve G-S equation using Newton-Raphson.
        
        Parameters
        ----------
        alpha : float
            Profile scaling parameter
        psi_init : ndarray (nr, nz), optional
            Initial guess
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum Newton iterations
        verbose : bool
            Print iteration info
        
        Returns
        -------
        psi : ndarray (nr, nz)
            Converged flux
        converged : bool
            Convergence flag
        info : dict
            Diagnostic info
        """
        # Initialize
        if psi_init is None:
            psi = self.psi_boundary.copy()
        else:
            psi = psi_init.copy()
        
        # Import grid topology for diagnostics
        from grid_topology import (
            StructuredGrid, find_magnetic_axis, find_xpoint,
            identify_plasma_domain
        )
        
        grid = StructuredGrid(self.R, self.Z)
        
        if verbose:
            print("Newton-Raphson G-S Solver")
            print("=" * 70)
            print(f"Grid: {self.nr} × {self.nz}")
            print(f"α = {alpha:.6f}")
            print()
        
        residual_norms = []
        
        for iteration in range(max_iter):
            # 1. Find plasma geometry
            i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
            i_x, j_x, psi_x, has_xpoint = find_xpoint(
                psi, grid, i_axis, j_axis, psi_axis
            )
            plasma_mask = identify_plasma_domain(
                psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
            )
            
            # 2. Compute residual
            residual = self._compute_residual(psi, alpha, psi_axis, psi_x, plasma_mask)
            residual_norm = np.linalg.norm(residual)
            residual_norms.append(residual_norm)
            
            if verbose:
                n_plasma = np.sum(plasma_mask)
                print(f"Iter {iteration+1:2d}: |R|={residual_norm:.6e}, "
                      f"plasma={n_plasma:4d} pts, "
                      f"ψ_axis={psi_axis:+.4f}, ψ_x={psi_x:+.4f}")
            
            # 3. Check convergence
            if residual_norm < tol:
                if verbose:
                    print(f"\n✅ Converged in {iteration+1} iterations!")
                
                info = {
                    'i_axis': i_axis,
                    'j_axis': j_axis,
                    'psi_axis': psi_axis,
                    'i_x': i_x,
                    'j_x': j_x,
                    'psi_x': psi_x,
                    'has_xpoint': has_xpoint,
                    'plasma_mask': plasma_mask,
                    'residual_norms': residual_norms
                }
                
                return psi, True, info
            
            # 4. Compute Jacobian
            J = self._compute_jacobian(psi, alpha, psi_axis, psi_x, plasma_mask)
            
            # 5. Solve linear system: J·δψ = -R
            try:
                delta_psi_flat = spla.spsolve(J, -residual)
            except Exception as e:
                if verbose:
                    print(f"\n❌ Linear solve failed: {e}")
                break
            
            delta_psi = delta_psi_flat.reshape((self.nr, self.nz), order='F')
            
            # 6. Line search (simple damping for robustness)
            lambda_damp = 0.3  # Conservative damping to prevent overshoot
            psi_new = psi + lambda_damp * delta_psi
            
            # 7. Update
            psi = psi_new
            
            # Enforce boundary
            psi[self.boundary_mask] = self.psi_boundary[self.boundary_mask]
        
        # Did not converge
        if verbose:
            print(f"\n❌ Did not converge in {max_iter} iterations")
            print(f"Final residual: {residual_norms[-1]:.6e}")
        
        # Return best result
        i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
        i_x, j_x, psi_x, has_xpoint = find_xpoint(
            psi, grid, i_axis, j_axis, psi_axis
        )
        plasma_mask = identify_plasma_domain(
            psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
        )
        
        info = {
            'i_axis': i_axis,
            'j_axis': j_axis,
            'psi_axis': psi_axis,
            'i_x': i_x,
            'j_x': j_x,
            'psi_x': psi_x,
            'has_xpoint': has_xpoint,
            'plasma_mask': plasma_mask,
            'residual_norms': residual_norms
        }
        
        return psi, False, info


# =============================================================================
# Test
# =============================================================================


    def solve_trust_region(self, alpha, psi_init=None, 
                           tol=1e-6, max_iter=50, verbose=True):
        """
        Solve G-S using Trust Region method (scipy).
        
        This is the production-grade solver with automatic step size 
        adaptation and global convergence guarantees.
        
        Parameters
        ----------
        alpha : float
            Profile scaling parameter
        psi_init : ndarray (nr, nz), optional
            Initial guess
        tol : float
            Convergence tolerance (function tolerance)
        max_iter : int
            Maximum iterations
        verbose : bool
            Print iteration diagnostics
        
        Returns
        -------
        psi : ndarray (nr, nz)
            Converged flux solution
        converged : bool
            True if converged
        info : dict
            Diagnostic information including:
            - scipy_result: full scipy optimization result
            - iterations: number of function evaluations
            - final_cost: ‖R‖²/2
            - plasma_mask, psi_axis, psi_x, etc.
        """
        from scipy.optimize import least_squares
        from functools import partial
        from grid_topology import (
            StructuredGrid, find_magnetic_axis, find_xpoint,
            identify_plasma_domain
        )
        
        # Initial guess
        if psi_init is None:
            psi_init = self.psi_boundary.copy()
        
        psi_flat_init = psi_init.ravel(order='F')
        
        if verbose:
            print("Trust Region G-S Solver (scipy)")
            print("=" * 70)
            print(f"Grid: {self.nr} × {self.nz}")
            print(f"α = {alpha:.6f}")
            print(f"Method: Trust Region Reflective (TRF)")
            print(f"Tolerance: ftol={tol:.1e}")
            print()
        
        # Nested functions with closure over self and alpha
        def residual_vector(psi_flat):
            """Residual function for scipy."""
            psi = psi_flat.reshape((self.nr, self.nz), order='F')
            
            # Find plasma domain
            grid = StructuredGrid(self.R, self.Z)
            i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
            i_x, j_x, psi_x, _ = find_xpoint(
                psi, grid, i_axis, j_axis, psi_axis
            )
            plasma_mask = identify_plasma_domain(
                psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
            )
            
            # Compute residual
            residual_2d = self._compute_residual(
                psi, alpha, psi_axis, psi_x, plasma_mask
            )
            
            return residual_2d.ravel(order='F')
        
        def jacobian_matrix(psi_flat):
            """Jacobian function for scipy."""
            psi = psi_flat.reshape((self.nr, self.nz), order='F')
            
            # Find plasma domain
            grid = StructuredGrid(self.R, self.Z)
            i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
            i_x, j_x, psi_x, _ = find_xpoint(
                psi, grid, i_axis, j_axis, psi_axis
            )
            plasma_mask = identify_plasma_domain(
                psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
            )
            
            # Compute analytical Jacobian
            J = self._compute_jacobian(
                psi, alpha, psi_axis, psi_x, plasma_mask
            )
            
            return J  # Already scipy.sparse.csr_matrix
        
        # Solve using Trust Region
        result = least_squares(
            residual_vector,
            psi_flat_init,
            jac=jacobian_matrix,
            method='trf',           # Trust Region Reflective
            ftol=tol,               # Function tolerance
            xtol=1e-10,             # Parameter tolerance (tight)
            gtol=1e-10,             # Gradient tolerance (tight)
            max_nfev=max_iter * 5,  # Max function evals
            verbose=2 if verbose else 0,
            x_scale='jac',          # Scale by Jacobian diagonal
            tr_solver='lsmr'        # Use LSMR for large sparse
        )
        
        # Extract solution
        psi = result.x.reshape((self.nr, self.nz), order='F')
        
        # Enforce boundary conditions
        psi[self.boundary_mask] = self.psi_boundary[self.boundary_mask]
        
        # Final diagnostics
        converged = result.success
        
        grid = StructuredGrid(self.R, self.Z)
        i_axis, j_axis, psi_axis = find_magnetic_axis(psi, grid)
        i_x, j_x, psi_x, has_xpoint = find_xpoint(
            psi, grid, i_axis, j_axis, psi_axis
        )
        plasma_mask = identify_plasma_domain(
            psi, i_axis, j_axis, psi_axis, i_x, j_x, psi_x, grid
        )
        
        if verbose:
            print()
            print("=" * 70)
            print("Convergence Summary")
            print("=" * 70)
            print(f"Status: {result.message}")
            print(f"Converged: {converged}")
            print(f"Function evals: {result.nfev}")
            print(f"Jacobian evals: {result.njev}")
            print(f"Final cost (‖R‖²/2): {result.cost:.6e}")
            print(f"Optimality (‖∇f‖): {result.optimality:.6e}")
            print(f"Plasma points: {np.sum(plasma_mask)}")
            print(f"ψ_axis: {psi_axis:.6f}")
            print(f"ψ_x: {psi_x:.6f}")
            print()
        
        info = {
            'i_axis': i_axis,
            'j_axis': j_axis,
            'psi_axis': psi_axis,
            'i_x': i_x,
            'j_x': j_x,
            'psi_x': psi_x,
            'has_xpoint': has_xpoint,
            'plasma_mask': plasma_mask,
            'scipy_result': result,
            'iterations': result.nfev,
            'jacobian_evals': result.njev,
            'final_cost': result.cost,
            'optimality': result.optimality,
            'message': result.message,
            'converged': converged
        }
        
        return psi, converged, info


def test_newton_quadratic():
    """Test Newton solver with quadratic profile."""
    from profiles import QuadraticProfile
    
    print("Test: Newton-Raphson with Quadratic Profile")
    print("=" * 70)
    print()
    
    # Grid
    R = np.linspace(1, 8, 41)
    Z = np.linspace(-3, 3, 41)
    
    # Simple boundary
    def boundary_psi(R, Z):
        return np.zeros_like(R)
    
    # Quadratic profile
    profile = QuadraticProfile(p0=1e4, p1=-5e3, f0=0.1, f1=-0.05)
    
    # Solver
    solver = NewtonGSSolver(R, Z, profile, boundary_psi)
    
    # Initial guess (circular)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    psi_init = -(RR - 4.5)**2 - 2*ZZ**2 + 10
    
    # Solve
    alpha = 1.0
    psi, converged, info = solver.solve(alpha, psi_init=psi_init, tol=1e-6, max_iter=20)
    
    print()
    print("Results:")
    print("=" * 70)
    print(f"Converged: {converged}")
    print(f"Iterations: {len(info['residual_norms'])}")
    print(f"Final residual: {info['residual_norms'][-1]:.6e}")
    print(f"Plasma points: {np.sum(info['plasma_mask'])}")
    print()
    
    if converged:
        print("✅ PASS")
    else:
        print("⚠️ Did not converge in max iterations")
    
    return psi, info


if __name__ == '__main__':
    psi, info = test_newton_quadratic()


    # =========================================================================
    # Trust Region Method (Production-grade solver)
    # =========================================================================
    
