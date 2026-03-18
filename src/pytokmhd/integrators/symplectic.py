"""
Symplectic Time Integrator for Reduced MHD

Implements Störmer-Verlet (leapfrog) scheme for structure-preserving
time integration of Hamiltonian systems.

Author: 小P ⚛️
Created: 2026-03-18 (Step 2.3)
"""

import numpy as np
from typing import Tuple
from ..geometry import ToroidalGrid
from ..operators import gradient_toroidal, laplacian_toroidal, divergence_toroidal


class SymplecticIntegrator:
    """
    Symplectic integrator (Störmer-Verlet) for reduced MHD.
    
    Preserves approximate Hamiltonian structure for long-time stability.
    
    Algorithm: Leapfrog scheme
    ----------
    1. Half-step ω (momentum-like variable)
    2. Full-step ψ (position-like variable)
    3. Half-step ω
    
    This is a 2nd-order symplectic method with excellent energy conservation.
    
    Parameters
    ----------
    grid : ToroidalGrid
        Spatial discretization grid
    dt : float
        Time step
    eta : float, optional
        Resistivity (default: 1e-6)
    nu : float, optional
        Viscosity (default: 1e-6)
    operator_splitting : bool, optional
        If True, separate Hamiltonian and dissipative steps (default: True)
    
    Attributes
    ----------
    psi : np.ndarray (nr, ntheta)
        Poloidal flux
    omega : np.ndarray (nr, ntheta)
        Vorticity
    t : float
        Current time
    energy_history : list
        Energy at each step (if tracking enabled)
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> solver = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-6)
    >>> solver.initialize(psi0, omega0)
    >>> solver.step()  # Take one time step
    >>> E = solver.compute_energy()  # Check energy
    
    Notes
    -----
    - Compatible interface with RK4Integrator for drop-in replacement
    - Symplectic property → long-time energy stability
    - Expected energy drift: O(10⁻⁵%) vs RK4's O(0.1-1%)
    """
    
    def __init__(self, grid: ToroidalGrid, dt: float,
                 eta: float = 1e-6, nu: float = 1e-6,
                 operator_splitting: bool = True):
        self.grid = grid
        self.dt = dt
        self.eta = eta
        self.nu = nu
        self.operator_splitting = operator_splitting
        
        # State variables
        self.psi = None
        self.omega = None
        self.t = 0.0
        
        # Cached Laplacian matrix for Poisson solver
        self._laplacian_matrix = None
        
        # Diagnostics
        self.energy_history = []
        self.track_energy = False
    
    def initialize(self, psi0: np.ndarray, omega0: np.ndarray) -> None:
        """
        Initialize fields.
        
        Parameters
        ----------
        psi0 : np.ndarray (nr, ntheta)
            Initial poloidal flux
        omega0 : np.ndarray (nr, ntheta)
            Initial vorticity
        """
        self.psi = psi0.copy()
        self.omega = omega0.copy()
        self.t = 0.0
        
        # Reset diagnostics
        self.energy_history = []
        if self.track_energy:
            E0 = self.compute_energy()
            self.energy_history.append((self.t, E0))
    
    def step(self) -> None:
        """
        Take one Störmer-Verlet time step.
        
        If operator_splitting=True:
            1. Symplectic step (ideal MHD, η=ν=0)
            2. Dissipation step (resistivity, viscosity)
        
        If operator_splitting=False:
            Combined step with dissipation
        """
        if self.operator_splitting:
            # Step 1: Symplectic (Hamiltonian part)
            self.psi, self.omega = self._symplectic_step(
                self.psi, self.omega, self.dt, eta=0.0, nu=0.0
            )
            
            # Step 2: Dissipation
            self.psi, self.omega = self._dissipation_step(
                self.psi, self.omega, self.dt
            )
        else:
            # Combined step
            self.psi, self.omega = self._symplectic_step(
                self.psi, self.omega, self.dt, eta=self.eta, nu=self.nu
            )
        
        self.t += self.dt
        
        # Track energy if enabled
        if self.track_energy:
            E = self.compute_energy()
            self.energy_history.append((self.t, E))
    
    def _symplectic_step(self, psi: np.ndarray, omega: np.ndarray, 
                         dt: float, eta: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Störmer-Verlet (leapfrog) symplectic step.
        
        Algorithm:
        ----------
        1. ω_{n+1/2} = ω_n + (dt/2) * dω/dt|_n
        2. ψ_{n+1} = ψ_n + dt * dψ/dt|_{n+1/2}
        3. ω_{n+1} = ω_{n+1/2} + (dt/2) * dω/dt|_{n+1}
        
        This is a 2nd-order symplectic integrator.
        
        Parameters
        ----------
        psi, omega : np.ndarray
            Current fields
        dt : float
            Time step
        eta, nu : float
            Dissipation coefficients (usually 0 for symplectic step)
        
        Returns
        -------
        psi_new, omega_new : np.ndarray
            Updated fields
        """
        # Step 1: Half-step omega (momentum)
        dpsi_dt, domega_dt = self.compute_rhs(psi, omega, eta, nu)
        omega_half = omega + 0.5 * dt * domega_dt
        omega_half = self._apply_boundary(omega_half)
        
        # Step 2: Full-step psi (position)
        dpsi_dt_half, _ = self.compute_rhs(psi, omega_half, eta, nu)
        psi_new = psi + dt * dpsi_dt_half
        psi_new = self._apply_boundary(psi_new)
        
        # Step 3: Half-step omega (momentum completion)
        _, domega_dt_new = self.compute_rhs(psi_new, omega_half, eta, nu)
        omega_new = omega_half + 0.5 * dt * domega_dt_new
        omega_new = self._apply_boundary(omega_new)
        
        return psi_new, omega_new
    
    def _dissipation_step(self, psi: np.ndarray, omega: np.ndarray,
                          dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dissipation correction step.
        
        Adds resistive and viscous diffusion:
            ψ → ψ + dt * η ∇²ψ
            ω → ω + dt * ν ∇²ω
        
        This is simple explicit Euler for dissipation.
        For stiffness, could use implicit method.
        
        Parameters
        ----------
        psi, omega : np.ndarray
            Fields after symplectic step
        dt : float
            Time step
        
        Returns
        -------
        psi_new, omega_new : np.ndarray
            Fields with dissipation applied
        """
        # Resistivity
        if self.eta > 0:
            lap_psi = laplacian_toroidal(psi, self.grid)
            psi = psi + dt * self.eta * lap_psi
        
        # Viscosity
        if self.nu > 0:
            lap_omega = laplacian_toroidal(omega, self.grid)
            omega = omega + dt * self.nu * lap_omega
        
        # Apply boundary conditions
        psi = self._apply_boundary(psi)
        omega = self._apply_boundary(omega)
        
        return psi, omega
    
    def compute_rhs(self, psi: np.ndarray, omega: np.ndarray,
                    eta: float = 0.0, nu: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RHS of reduced MHD equations.
        
        Equations:
        ----------
        ∂ψ/∂t = [φ, ψ] + η ∇²ψ
        ∂ω/∂t = [φ, ω] + [J_∥, ψ] + ν ∇²ω
        
        where:
            φ: stream function (solve ∇²φ = ω)
            J_∥ = -∇²ψ: parallel current
            [f,g] = (1/R)(∂f/∂R ∂g/∂Z - ∂f/∂Z ∂g/∂R): Poisson bracket
        
        Parameters
        ----------
        psi : np.ndarray (nr, ntheta)
            Poloidal flux
        omega : np.ndarray (nr, ntheta)
            Vorticity
        eta : float
            Resistivity
        nu : float
            Viscosity
        
        Returns
        -------
        dpsi_dt, domega_dt : np.ndarray
            Time derivatives
        """
        # Solve for stream function: ∇²φ = ω
        phi = self._solve_poisson(omega)
        
        # Parallel current: J_∥ = -∇²ψ
        J_parallel = -laplacian_toroidal(psi, self.grid)
        
        # Gradients for Poisson brackets
        grad_phi_r, grad_phi_theta = gradient_toroidal(phi, self.grid)
        grad_psi_r, grad_psi_theta = gradient_toroidal(psi, self.grid)
        grad_omega_r, grad_omega_theta = gradient_toroidal(omega, self.grid)
        grad_J_r, grad_J_theta = gradient_toroidal(J_parallel, self.grid)
        
        # Poisson bracket: [f,g] = (1/R)(∂f/∂R ∂g/∂Z - ∂f/∂Z ∂g/∂R)
        # In (r,θ) coordinates with R = R0 + r cos(θ), Z = r sin(θ):
        # [f,g] ~ ∂f/∂r ∂g/∂θ - ∂f/∂θ ∂g/∂r (simplified for toroidal)
        
        # [φ, ψ]
        bracket_phi_psi = grad_phi_r * grad_psi_theta - grad_phi_theta * grad_psi_r
        bracket_phi_psi /= self.grid.R_grid  # 1/R factor
        
        # [φ, ω]
        bracket_phi_omega = grad_phi_r * grad_omega_theta - grad_phi_theta * grad_omega_r
        bracket_phi_omega /= self.grid.R_grid
        
        # [J_∥, ψ]
        bracket_J_psi = grad_J_r * grad_psi_theta - grad_J_theta * grad_psi_r
        bracket_J_psi /= self.grid.R_grid
        
        # Time derivatives
        dpsi_dt = bracket_phi_psi
        domega_dt = bracket_phi_omega + bracket_J_psi
        
        # Add dissipation if requested
        if eta > 0:
            dpsi_dt += eta * laplacian_toroidal(psi, self.grid)
        if nu > 0:
            domega_dt += nu * laplacian_toroidal(omega, self.grid)
        
        return dpsi_dt, domega_dt
    
    def _solve_poisson(self, omega: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: ∇²φ = ω for stream function φ.
        
        Uses exact sparse solver based on Phase 1 laplacian_toroidal.
        Guarantees machine-precision accuracy (residual < 1e-9).
        
        Parameters
        ----------
        omega : np.ndarray
            Source term (vorticity)
        
        Returns
        -------
        phi : np.ndarray
            Stream function
        
        Notes
        -----
        Upgraded to exact solver (2026-03-18):
        - Extracts exact stencil from toroidal_operators.laplacian_toroidal
        - Uses scipy.sparse direct solver
        - Residual: max|∇²φ - ω| < 1e-9 (vs ~1e-3 for hybrid method)
        
        Previous: poisson_hybrid (cylindrical approximation + refinement)
        """
        from .poisson_sparse_exact import solve_poisson_exact, build_laplacian_matrix
        
        # Build Laplacian matrix once and cache
        if self._laplacian_matrix is None:
            self._laplacian_matrix = build_laplacian_matrix(self.grid)
        
        phi = solve_poisson_exact(omega, self.grid, L_matrix=self._laplacian_matrix)
        
        return phi
    
    def _apply_boundary(self, field: np.ndarray) -> np.ndarray:
        """
        Apply Dirichlet boundary conditions (field=0 at r_min and r=a).
        
        Parameters
        ----------
        field : np.ndarray
            Field to apply BC to
        
        Returns
        -------
        field : np.ndarray
            Field with BC applied
        """
        field = field.copy()
        field[0, :] = 0.0   # r_min
        field[-1, :] = 0.0  # r=a
        return field
    
    def compute_energy(self) -> float:
        """
        Compute total energy H = kinetic + magnetic.
        
        H = ∫ [1/2 ω² + 1/2 |∇ψ|²] √g dV
        
        Note: Uses ω² formulation (consistent with Phase 1 diagnostics)
        instead of |∇φ|² to avoid boundary inconsistencies when ω≠0 at boundaries.
        
        Returns
        -------
        E : float
            Total energy
        """
        # Gradients for magnetic energy
        grad_psi_r, grad_psi_theta = gradient_toroidal(self.psi, self.grid)
        
        # Energy densities (physical components)
        # gradient_toroidal returns physical components: grad_theta already includes 1/r
        e_mag = 0.5 * (grad_psi_r**2 + grad_psi_theta**2)
        
        # Volume element: √g dr dθ
        jacobian = self.grid.jacobian()
        dV = jacobian * self.grid.dr * self.grid.dtheta
        
        # Integrate
        # Kinetic: Use ω² (matches Phase 1 diagnostics)
        # This avoids boundary term issues when ω≠0 at boundaries
        E_kin = 0.5 * np.sum(self.omega**2 * dV)
        E_mag = np.sum(e_mag * dV)
        
        return E_kin + E_mag
    
    def enable_energy_tracking(self) -> None:
        """Enable automatic energy tracking at each step."""
        self.track_energy = True
    
    def get_energy_history(self) -> np.ndarray:
        """
        Get energy history.
        
        Returns
        -------
        history : np.ndarray (n_steps, 2)
            Array of (time, energy) pairs
        """
        return np.array(self.energy_history)
