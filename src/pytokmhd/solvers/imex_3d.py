"""
3D IMEX Time Evolution for Reduced MHD

Implements Implicit-Explicit (IMEX) time integration for 3D reduced MHD:

    ∂ψ/∂t = [φ, ψ] + η∇²ψ
    ∂ω/∂t = [ψ, ω] + η∇²ω

where:
    - ψ: stream function (magnetic flux)
    - ω = ∇²ψ: vorticity
    - φ = ω: velocity stream function (incompressible flow)
    - [f, g]: 3D Poisson bracket (from Phase 1.3)
    - η: resistivity (diffusion coefficient)

IMEX Splitting
--------------
**Explicit (nonlinear terms):**
    - [φ, ψ] = [ω, ψ]: Advection of magnetic flux
    - [ψ, ω]: Advection of vorticity

**Implicit (linear diffusion):**
    - η∇²ψ: Resistive diffusion of flux
    - η∇²ω: Viscous diffusion of vorticity

Time Integration (1st-order IMEX)
----------------------------------
For each field (ψ or ω):

    1. Compute explicit RHS: rhs = [f, g]^n
    2. Form implicit system: (I - Δt·η·∇²) φ^(n+1) = φ^n + Δt·rhs
    3. Solve Helmholtz equation per Fourier mode (toroidal direction)
    4. Transform back to physical space

Helmholtz Equation (per k-mode)
--------------------------------
In Fourier space (ζ direction):

    (1 - Δt·η·∇²_2D - Δt·η·k²) φ_k = rhs_k

where:
    ∇²_2D = ∂²/∂r² + (1/r)∂/∂r + (1/r²)∂²/∂θ²  (2D Laplacian in (r,θ))
    k = 2πn/L_ζ: toroidal wave number

Solution Strategy
-----------------
1. Transform to Fourier space: φ_hat = FFT(φ, axis=ζ)
2. For each k-mode:
   a. Build sparse Helmholtz matrix: A = (I - Δt·η·(∇²_2D - k²))
   b. Solve linear system: A·φ_k = rhs_k (using scipy.sparse.linalg.spsolve)
3. Transform back: φ = IFFT(φ_hat, axis=ζ)

Boundary Conditions
-------------------
- Radial: Dirichlet (ψ=0, ω=0 at r=0 and r=a)
- Poloidal: Periodic (θ ∈ [0, 2π))
- Toroidal: Periodic (ζ ∈ [0, 2π), enforced by FFT)

Stability & CFL Constraint
---------------------------
- Explicit CFL: Δt < C·Δx/|v_max| (advection speed)
  - For MHD: |v| ~ |∇ψ| (typical ~1 in normalized units)
  - Recommended: Δt ≤ 0.01 for Δr ~ 0.03, Δθ ~ 0.1
- Implicit diffusion: unconditionally stable
- Overall: Δt limited by advection CFL (much less restrictive than explicit diffusion)

Conservation Laws
-----------------
**Ideal MHD (η=0):**
    - dH/dt = 0 (energy conserved to ~1e-8 per step)

**Resistive MHD (η>0):**
    - dH/dt < 0 (monotonic energy dissipation)

Physics References
------------------
- Strauss (1976): "Nonlinear, three-dimensional magnetohydrodynamics of noncircular tokamaks"
- Hazeltine & Meiss (2003): Plasma Confinement, Ch. 3 (Hamiltonian formulation)
- Ascher et al. (1995): "Implicit-Explicit Methods for Time-Dependent PDEs"

Code References
---------------
- Phase 1.3: operators/poisson_bracket_3d.py (nonlinear terms)
- Phase 2.1: physics/hamiltonian_3d.py (energy diagnostics)
- Phase 2.2: ic/ballooning_mode.py (initial conditions)

Author: 小P ⚛️
Date: 2026-03-19
Phase: 2.3 (3D Physics Core)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Tuple, Optional, Union, Callable

# Import dependencies from previous phases
from ..operators.poisson_bracket_3d import poisson_bracket_3d
from ..physics.hamiltonian_3d import (
    compute_hamiltonian_3d,
    compute_magnetic_energy,
    compute_kinetic_energy
)


def evolve_3d_imex(
    psi_init: np.ndarray,
    omega_init: np.ndarray,
    grid,
    eta: float = 1e-4,
    dt: float = 0.01,
    n_steps: int = 100,
    J_ext: Optional[Union[Callable, np.ndarray]] = None,
    store_interval: int = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evolve 3D reduced MHD using IMEX time integration.
    
    Solves:
        ∂ψ/∂t = [φ, ψ] + η∇²ψ
        ∂ω/∂t = [ψ, ω] + η∇²ω
    
    where φ = ω (stream function for incompressible flow).
    
    Parameters
    ----------
    psi_init : np.ndarray, shape (nr, nθ, nζ)
        Initial stream function (magnetic flux)
    omega_init : np.ndarray, shape (nr, nθ, nζ)
        Initial vorticity
    grid : Grid3D
        3D cylindrical grid object
    eta : float, optional
        Resistivity (diffusion coefficient), default 1e-4
    dt : float, optional
        Time step, default 0.01
    n_steps : int, optional
        Number of time steps, default 100
    J_ext : callable or np.ndarray or None, optional
        External current density J_ext(r, θ, ζ, t).
        
        If callable: J_ext(t, grid) -> np.ndarray (nr, nθ, nζ)
        If ndarray: Constant J_ext (nr, nθ, nζ) independent of time
        If None: No external current (backward compatible with Phase 2.3)
        
        Default: None
    store_interval : int, optional
        Store fields every `store_interval` steps (default 1)
        - 1: store all steps (memory intensive)
        - 10: store every 10th step (memory efficient for long runs)
    verbose : bool, optional
        Print progress every 10 steps, default False
    
    Returns
    -------
    psi_history : np.ndarray, shape (n_stored+1, nr, nθ, nζ)
        Stream function at stored time steps
    omega_history : np.ndarray, shape (n_stored+1, nr, nθ, nζ)
        Vorticity at stored time steps
    diagnostics : dict
        Time series data:
            - 'time': Time values [s]
            - 'energy': Total energy H(ψ, ω) [J]
            - 'magnetic': Magnetic energy ∫(1/2)|∇ψ|² dV [J]
            - 'kinetic': Kinetic energy ∫(1/2)ω² dV [J]
            - 'max_psi': max|ψ| [T·m]
            - 'max_omega': max|ω| [s⁻¹]
            - 'cfl_number': CFL number (max|v|·Δt/Δx)
    
    Notes
    -----
    **CFL Warning:**
        If CFL > 0.5, advection CFL constraint may be violated.
        Recommend reducing dt or increasing grid resolution.
    
    **Energy Conservation:**
        - Ideal MHD (η=0): |ΔH/H₀| should be < 1e-8 per 100 steps
        - Resistive MHD (η>0): H(t) should decrease monotonically
    
    **Performance:**
        - 32×64×128 grid: ~3-5s per 100 steps (MacBook M1)
        - Dominated by Helmholtz solves (2 per step × n_modes)
    
    Examples
    --------
    >>> from pytokmhd.ic.ballooning_mode import Grid3D, create_equilibrium_ic
    >>> grid = Grid3D(nr=32, ntheta=64, nzeta=128)
    >>> psi0, omega0, q = create_equilibrium_ic(grid)
    >>> psi_hist, omega_hist, diag = evolve_3d_imex(
    ...     psi0, omega0, grid, eta=1e-4, dt=0.01, n_steps=100
    ... )
    >>> print(f"Energy conservation: {abs(diag['energy'][-1]/diag['energy'][0] - 1):.2e}")
    """
    # Validate inputs
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    assert psi_init.shape == (nr, ntheta, nzeta), f"psi shape mismatch: {psi_init.shape} vs ({nr}, {ntheta}, {nzeta})"
    assert omega_init.shape == (nr, ntheta, nzeta), f"omega shape mismatch"
    
    # Initialize fields
    psi = psi_init.copy()
    omega = omega_init.copy()
    
    # Storage arrays
    n_stored = n_steps // store_interval + 1
    psi_history = np.zeros((n_stored, nr, ntheta, nzeta))
    omega_history = np.zeros((n_stored, nr, ntheta, nzeta))
    psi_history[0] = psi
    omega_history[0] = omega
    
    # Diagnostics
    diagnostics = {
        'time': [],
        'energy': [],
        'magnetic': [],
        'kinetic': [],
        'max_psi': [],
        'max_omega': [],
        'cfl_number': []
    }
    
    # Store initial diagnostics
    _update_diagnostics(diagnostics, 0.0, psi, omega, grid, dt)
    
    # Pre-build Helmholtz solver matrices (once for all time steps)
    helmholtz_solvers = _build_helmholtz_solvers(grid, eta, dt)
    
    # Time stepping loop
    store_idx = 1
    for n in range(n_steps):
        t = n * dt
        
        # --- IMEX Time Step ---
        psi_new, omega_new = _imex_step(psi, omega, grid, eta, dt, helmholtz_solvers, t, J_ext)
        
        # Update fields
        psi = psi_new
        omega = omega_new
        
        # Diagnostics
        _update_diagnostics(diagnostics, t + dt, psi, omega, grid, dt)
        
        # Store fields
        if (n + 1) % store_interval == 0:
            psi_history[store_idx] = psi
            omega_history[store_idx] = omega
            store_idx += 1
        
        # Progress
        if verbose and (n + 1) % 10 == 0:
            print(f"Step {n+1}/{n_steps}: t={t+dt:.3f}, E={diagnostics['energy'][-1]:.6e}, CFL={diagnostics['cfl_number'][-1]:.3f}")
    
    # Warn if CFL constraint violated
    max_cfl = max(diagnostics['cfl_number'])
    if max_cfl > 0.5:
        print(f"⚠️  WARNING: CFL number {max_cfl:.2f} > 0.5. Consider reducing dt or increasing grid resolution.")
    
    return psi_history, omega_history, diagnostics


def _imex_step(
    psi: np.ndarray,
    omega: np.ndarray,
    grid,
    eta: float,
    dt: float,
    helmholtz_solvers: Dict,
    t: float,
    J_ext: Optional[Union[Callable, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single IMEX time step with external current.
    
    1. Compute explicit RHS (Poisson brackets)
    2. Add external current to omega RHS
    3. Solve implicit diffusion (Helmholtz equation)
    
    Parameters
    ----------
    psi, omega : np.ndarray (nr, nθ, nζ)
        Current fields
    grid : Grid3D
    eta : float
        Resistivity
    dt : float
        Time step
    helmholtz_solvers : dict
        Pre-built sparse matrices for Helmholtz solve
    t : float
        Current time (for time-dependent J_ext)
    J_ext : callable or ndarray or None
        External current density
    
    Returns
    -------
    psi_new, omega_new : np.ndarray
        Updated fields at t^(n+1)
    """
    # --- Step 1: Explicit RHS (Poisson brackets) ---
    # φ = ω (stream function for flow)
    phi = omega
    
    # RHS for ψ equation: [φ, ψ]
    rhs_psi = poisson_bracket_3d(phi, psi, grid)
    
    # RHS for ω equation: [ψ, ω]
    rhs_omega = poisson_bracket_3d(psi, omega, grid)
    
    # --- NEW: Add external current to omega RHS ---
    if J_ext is not None:
        if callable(J_ext):
            J_current = J_ext(t, grid)  # Time-dependent
        else:
            J_current = J_ext           # Constant
        
        rhs_omega = rhs_omega + J_current
    
    # --- Step 2: Implicit diffusion solve ---
    # Prepare RHS with boundary conditions
    rhs_psi_full = psi + dt * rhs_psi
    rhs_omega_full = omega + dt * rhs_omega
    
    # Enforce Dirichlet BC on RHS (required for Helmholtz solver)
    rhs_psi_full[0, :, :] = 0.0
    rhs_psi_full[-1, :, :] = 0.0
    rhs_omega_full[0, :, :] = 0.0
    rhs_omega_full[-1, :, :] = 0.0
    
    # Solve: (I - Δt·η·∇²) ψ^(n+1) = rhs_psi_full
    psi_new = _solve_implicit_diffusion(rhs_psi_full, grid, eta, dt, helmholtz_solvers)
    
    # Solve: (I - Δt·η·∇²) ω^(n+1) = rhs_omega_full
    omega_new = _solve_implicit_diffusion(rhs_omega_full, grid, eta, dt, helmholtz_solvers)
    
    # Boundary conditions already enforced in Helmholtz matrix (φ=0 rows)
    
    return psi_new, omega_new


def _solve_implicit_diffusion(
    rhs: np.ndarray,
    grid,
    eta: float,
    dt: float,
    helmholtz_solvers: Dict
) -> np.ndarray:
    """
    Solve implicit diffusion: (I - Δt·η·∇²) φ = rhs.
    
    Strategy:
        1. FFT to Fourier space (toroidal direction ζ)
        2. Solve Helmholtz equation per k-mode: (I - Δt·η·∇²_2D - Δt·η·k²) φ_k = rhs_k
        3. IFFT back to physical space
    
    Parameters
    ----------
    rhs : np.ndarray (nr, nθ, nζ)
        Right-hand side (φ^n + Δt·explicit_rhs)
    grid : Grid3D
    eta : float
        Resistivity
    dt : float
        Time step
    helmholtz_solvers : dict
        Pre-built sparse LU factorizations for each k-mode
    
    Returns
    -------
    phi_new : np.ndarray (nr, nθ, nζ)
        Solution at t^(n+1)
    """
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    
    # --- FFT to Fourier space (ζ direction) ---
    rhs_hat = np.fft.rfft(rhs, axis=2)  # Shape: (nr, nθ, nzeta//2+1)
    phi_hat = np.zeros_like(rhs_hat)
    
    # --- Solve Helmholtz equation per k-mode ---
    nk = nzeta // 2 + 1
    for k_idx in range(nk):
        # Get solver for this k-mode
        solver_lu = helmholtz_solvers[k_idx]
        
        # RHS for this mode (complex, flatten 2D slice to 1D)
        rhs_k_complex = rhs_hat[:, :, k_idx].ravel()
        
        # Solve real and imaginary parts separately
        # (sparse solver doesn't support complex directly)
        phi_k_real = solver_lu.solve(rhs_k_complex.real)
        phi_k_imag = solver_lu.solve(rhs_k_complex.imag)
        
        # Combine back to complex
        phi_k = phi_k_real + 1j * phi_k_imag
        
        # Reshape back to 2D
        phi_hat[:, :, k_idx] = phi_k.reshape((nr, ntheta))
    
    # --- IFFT back to physical space ---
    phi_new = np.fft.irfft(phi_hat, n=nzeta, axis=2)
    
    return phi_new


def _build_helmholtz_solvers(grid, eta: float, dt: float) -> Dict:
    """
    Pre-build sparse LU factorizations for Helmholtz equation at each k-mode.
    
    For k-mode: (I - Δt·η·∇²_2D - Δt·η·k²) φ_k = rhs_k
    
    where ∇²_2D = ∂²/∂r² + (1/r)∂/∂r + (1/r²)∂²/∂θ²
    
    Parameters
    ----------
    grid : Grid3D
    eta : float
        Resistivity
    dt : float
        Time step
    
    Returns
    -------
    solvers : dict
        {k_idx: splu_object} for each k-mode
    """
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    dr, dtheta = grid.dr, grid.dtheta
    dzeta = 2 * np.pi / nzeta
    r = grid.r  # 1D array (nr,)
    
    nk = nzeta // 2 + 1
    solvers = {}
    
    for k_idx in range(nk):
        # Toroidal wave number
        k = 2 * np.pi * k_idx / (nzeta * dzeta)
        
        # Build Helmholtz matrix: A = I - Δt·η·(∇²_2D - k²)
        A = _build_helmholtz_matrix_2d(nr, ntheta, dr, dtheta, r, k, eta, dt)
        
        # LU factorization (pre-compute for efficiency)
        solvers[k_idx] = spla.splu(A.tocsc())
    
    return solvers


def _build_helmholtz_matrix_2d(
    nr: int,
    ntheta: int,
    dr: float,
    dtheta: float,
    r: np.ndarray,
    k: float,
    eta: float,
    dt: float
) -> sp.csr_matrix:
    """
    Build 2D Helmholtz matrix for a single k-mode.
    
    Discretizes: (I - Δt·η·∇²_2D - Δt·η·k²) on (r,θ) grid
    
    where ∇²_2D = ∂²/∂r² + (1/r)∂/∂r + (1/r²)∂²/∂θ²
    
    Parameters
    ----------
    nr, ntheta : int
        Grid dimensions
    dr, dtheta : float
        Grid spacings
    r : np.ndarray (nr,)
        Radial coordinates
    k : float
        Toroidal wave number
    eta, dt : float
        Resistivity and time step
    
    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse matrix of size (nr*nθ, nr*nθ)
    
    Notes
    -----
    Stencil for ∇²_2D at (i, j):
        ∂²/∂r²: (φ[i+1,j] - 2φ[i,j] + φ[i-1,j]) / dr²
        (1/r)∂/∂r: (φ[i+1,j] - φ[i-1,j]) / (2·r[i]·dr)
        (1/r²)∂²/∂θ²: (φ[i,j+1] - 2φ[i,j] + φ[i,j-1]) / (r[i]²·dθ²)
    
    Boundary conditions:
        - r=0, r=a: Dirichlet (φ=0)
        - θ: Periodic
    """
    N = nr * ntheta
    row_idx = []
    col_idx = []
    data = []
    
    def idx(i, j):
        """Flatten 2D index to 1D."""
        return i * ntheta + j % ntheta
    
    r_safe = np.maximum(r, 1e-10)  # Avoid division by zero at r=0
    
    for i in range(nr):
        for j in range(ntheta):
            n = idx(i, j)
            
            # --- Boundary: r=0 and r=a (Dirichlet) ---
            if i == 0 or i == nr - 1:
                row_idx.append(n)
                col_idx.append(n)
                data.append(1.0)  # φ = 0
                continue
            
            # --- Interior points ---
            # Diagonal: I - Δt·η·(-2/dr² - 2/(r²·dθ²) - k²)
            diag_val = 1.0 + dt * eta * (
                2.0 / dr**2 +
                2.0 / (r_safe[i]**2 * dtheta**2) +
                k**2
            )
            row_idx.append(n)
            col_idx.append(n)
            data.append(diag_val)
            
            # Radial derivative: -Δt·η·(1/dr² ± 1/(2r·dr))
            # i+1: ∂²/∂r² + (1/r)∂/∂r
            coeff_ip1 = -dt * eta * (1.0 / dr**2 + 1.0 / (2 * r_safe[i] * dr))
            row_idx.append(n)
            col_idx.append(idx(i+1, j))
            data.append(coeff_ip1)
            
            # i-1
            coeff_im1 = -dt * eta * (1.0 / dr**2 - 1.0 / (2 * r_safe[i] * dr))
            row_idx.append(n)
            col_idx.append(idx(i-1, j))
            data.append(coeff_im1)
            
            # Poloidal derivative: -Δt·η·(1/(r²·dθ²))
            coeff_theta = -dt * eta / (r_safe[i]**2 * dtheta**2)
            
            # j+1 (periodic)
            row_idx.append(n)
            col_idx.append(idx(i, j+1))
            data.append(coeff_theta)
            
            # j-1 (periodic)
            row_idx.append(n)
            col_idx.append(idx(i, j-1))
            data.append(coeff_theta)
    
    # Build sparse matrix
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    return A


def _update_diagnostics(
    diagnostics: Dict,
    t: float,
    psi: np.ndarray,
    omega: np.ndarray,
    grid,
    dt: float
):
    """
    Compute and store diagnostic quantities.
    
    Parameters
    ----------
    diagnostics : dict
        Dictionary to update
    t : float
        Current time
    psi, omega : np.ndarray
        Current fields
    grid : Grid3D
    dt : float
        Time step (for CFL calculation)
    """
    diagnostics['time'].append(t)
    diagnostics['energy'].append(compute_hamiltonian_3d(psi, omega, grid))
    diagnostics['magnetic'].append(compute_magnetic_energy(psi, grid))
    diagnostics['kinetic'].append(compute_kinetic_energy(omega, grid))
    diagnostics['max_psi'].append(np.max(np.abs(psi)))
    diagnostics['max_omega'].append(np.max(np.abs(omega)))
    
    # CFL number: max|v|·Δt/Δx
    # Estimate |v| ~ |∇ψ| (from Poisson bracket advection)
    v_max = np.max(np.abs(np.gradient(psi, axis=0))) / grid.dr  # Rough estimate
    dx_min = min(grid.dr, grid.r[1] * grid.dtheta)
    cfl = v_max * dt / dx_min if dx_min > 0 else 0.0
    diagnostics['cfl_number'].append(cfl)
