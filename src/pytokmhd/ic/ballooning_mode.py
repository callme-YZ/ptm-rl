"""
3D Ballooning Mode Initial Conditions

Implements initial conditions for ballooning mode instabilities in 3D tokamak geometry.

Physics Background
------------------
Ballooning modes are high-n (toroidal mode number) MHD instabilities that:
    1. Localize at bad curvature regions (outboard side, θ₀=0)
    2. Couple multiple poloidal modes m via toroidal geometry
    3. Dominate in high-β tokamaks (β > βc ≈ 1%)

The perturbation structure is:
    ψ₁(r, θ, ζ) = A(r) · Y(θ₀) · exp(i·n·ζ)

where:
    - A(r): Radial profile (Gaussian centered at rational surface r_s)
    - Y(θ₀): Ballooning envelope (sum over coupled m modes)
    - θ₀ = θ - n·q(r)·ζ: Extended poloidal angle (field-line coordinate)
    - n: Toroidal mode number (typical: n=5-10)

Ballooning Envelope:
    Y(θ₀) = Σ_m a_m · exp(i·m·θ₀)
    
where m modes are coupled within range m₀±Δm (typically Δm=2).

Radial Profile:
    A(r) = exp(-(r - r_s)²/Δr²)
    
localized at rational surface r_s where q(r_s) = m₀/n.

References
----------
- Learning notes: notes/v1.4/1.3-ballooning-modes.md
- Connor et al. (1978): "Shear, periodicity, and plasma ballooning modes"
- Cowley et al. (1991): "The effect of curvature on toroidal MHD"

Author: 小P ⚛️
Created: 2026-03-19
Phase: 2.2 (3D Initial Conditions)
"""

import numpy as np
from typing import Tuple, Optional


class Grid3D:
    """
    3D toroidal grid (r, θ, ζ) for ballooning mode ICs.
    
    Parameters
    ----------
    nr : int
        Radial grid points
    ntheta : int
        Poloidal grid points
    nzeta : int
        Toroidal grid points
    r_max : float, optional
        Minor radius (default: 1.0)
    R0 : float, optional
        Major radius (default: 3.0)
    r_min : float, optional
        Inner boundary (default: 0.1*r_max to avoid r=0 singularity)
    
    Attributes
    ----------
    nr, ntheta, nzeta : int
        Grid resolution
    r, theta, zeta : np.ndarray (1D)
        Coordinate arrays
    dr, dtheta, dzeta : float
        Grid spacing
    r_max, R0 : float
        Geometry parameters
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, ntheta=64, nzeta=128)
    >>> print(f"Grid: {grid.nr} x {grid.ntheta} x {grid.nzeta}")
    >>> print(f"Spacing: dr={grid.dr:.4f}, dθ={grid.dtheta:.4f}, dζ={grid.dzeta:.4f}")
    """
    
    def __init__(
        self,
        nr: int,
        ntheta: int,
        nzeta: int,
        r_max: float = 1.0,
        R0: float = 3.0,
        r_min: Optional[float] = None,
    ):
        """Initialize 3D toroidal grid."""
        # Validation
        if nr < 8 or ntheta < 16 or nzeta < 16:
            raise ValueError(
                f"Grid too coarse: nr={nr}, ntheta={ntheta}, nzeta={nzeta}. "
                f"Minimum: nr>=8, ntheta>=16, nzeta>=16"
            )
        
        self.nr = nr
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.r_max = r_max
        self.R0 = R0
        
        # Radial: [r_min, r_max], avoid r=0 singularity
        if r_min is None:
            r_min = 0.1 * r_max
        self.r = np.linspace(r_min, r_max, nr)
        
        # Poloidal: [0, 2π), periodic
        self.theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        
        # Toroidal: [0, 2π), periodic
        self.zeta = np.linspace(0, 2*np.pi, nzeta, endpoint=False)
        
        # Grid spacing
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]
        self.dzeta = self.zeta[1] - self.zeta[0]
        
        # Aliases for compatibility
        self.nθ = ntheta
        self.nζ = nzeta
        self.dθ = self.dtheta
        self.dζ = self.dzeta
        
        # Geometry attributes for Poisson bracket (Phase 1.3 compatibility)
        # R(r, θ) = R₀ + r·cos(θ) (major radius in toroidal geometry)
        r_2d = self.r[:, np.newaxis]  # (nr, 1)
        theta_2d = self.theta[np.newaxis, :]  # (1, ntheta)
        self.R_grid = self.R0 + r_2d * np.cos(theta_2d)  # (nr, ntheta)
        
        # Toroidal magnetic field (default: B₀ = 1.0 T for normalized units)
        self.B0 = 1.0


def create_q_profile(
    r: np.ndarray,
    q0: float = 1.0,
    qa: float = 3.0,
    profile_type: str = "linear",
) -> np.ndarray:
    """
    Create safety factor q-profile.
    
    The safety factor q(r) determines the pitch of magnetic field lines
    and controls rational surfaces where q(r_s) = m/n.
    
    Parameters
    ----------
    r : np.ndarray (nr,)
        Radial coordinate array
    q0 : float, optional
        Safety factor at axis (default: 1.0)
    qa : float, optional
        Safety factor at edge (default: 3.0)
    profile_type : str, optional
        Profile shape: 'linear' or 'parabolic' (default: 'linear')
    
    Returns
    -------
    q : np.ndarray (nr,)
        Safety factor profile
    
    Notes
    -----
    - Linear: q(r) = q₀ + (qa - q₀) * (r/a)
    - Parabolic: q(r) = q₀ + (qa - q₀) * (r/a)²
    
    Physical constraints:
        - q₀ >= 1 (avoids q=1 rational surface near axis)
        - qa > q₀ (monotonic increasing, ensures magnetic shear)
        - Typical values: q₀ ∈ [1, 2], qa ∈ [3, 5]
    
    Examples
    --------
    >>> r = np.linspace(0.1, 1.0, 32)
    >>> q = create_q_profile(r, q0=1.0, qa=3.0, profile_type='linear')
    >>> assert q[0] > 1.0 and q[-1] < 3.5  # Monotonic increasing
    """
    if q0 < 0.5:
        raise ValueError(f"q0 must be >= 0.5 to avoid low-q instabilities, got {q0}")
    if qa <= q0:
        raise ValueError(f"qa={qa} must be > q0={q0} for magnetic shear")
    
    a = r[-1]  # Edge radius (assume r starts near 0)
    r_norm = r / a  # Normalized radius [0, 1]
    
    if profile_type == "linear":
        q = q0 + (qa - q0) * r_norm
    elif profile_type == "parabolic":
        q = q0 + (qa - q0) * r_norm**2
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")
    
    return q


def create_equilibrium_ic(
    grid: Grid3D,
    q_profile_type: str = "linear",
    psi0_type: str = "polynomial",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create axisymmetric equilibrium ψ₀(r, θ).
    
    The equilibrium is independent of toroidal angle ζ (∂ψ₀/∂ζ = 0).
    
    Parameters
    ----------
    grid : Grid3D
        3D toroidal grid
    q_profile_type : str, optional
        Safety factor profile: 'linear' or 'parabolic' (default: 'linear')
    psi0_type : str, optional
        Equilibrium choice: 'zero' or 'polynomial' (default: 'polynomial')
    
    Returns
    -------
    psi0 : np.ndarray (nr, nθ, nζ)
        Equilibrium stream function (axisymmetric: ∂/∂ζ = 0)
    omega0 : np.ndarray (nr, nθ, nζ)
        Equilibrium vorticity ω₀ = ∇²ψ₀
    q_profile : np.ndarray (nr,)
        Safety factor q(r)
    
    Notes
    -----
    Equilibrium choices:
        - 'zero': ψ₀ = 0 (force-free equilibrium, ∇²ψ₀ = 0)
        - 'polynomial': ψ₀ = (r/a)² * (1 - r/a) (satisfies BC at r=0, r=a)
    
    Boundary conditions:
        - Radial: ψ₀(0, θ) = ψ₀(a, θ) = 0 (Dirichlet)
        - Poloidal: periodic
        - Toroidal: independent (∂/∂ζ = 0)
    
    Physical interpretation:
        - ψ₀: Magnetic flux function (B = ∇ψ × ẑ)
        - ω₀: Current density J_z ∝ ∇²ψ₀
        - q(r): Rotational transform (field line pitch)
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, ntheta=64, nzeta=128)
    >>> psi0, omega0, q = create_equilibrium_ic(grid)
    >>> assert psi0.shape == (32, 64, 128)
    >>> assert np.allclose(psi0[:, :, 0], psi0[:, :, 1])  # Axisymmetric
    """
    # Create q-profile
    q_profile = create_q_profile(
        grid.r,
        q0=1.0,
        qa=3.0,
        profile_type=q_profile_type,
    )
    
    # Create axisymmetric equilibrium
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    a = grid.r_max
    
    # 2D equilibrium (r, θ)
    r_2d, theta_2d = np.meshgrid(grid.r, grid.theta, indexing='ij')
    
    if psi0_type == "zero":
        psi0_2d = np.zeros((nr, ntheta))
        omega0_2d = np.zeros((nr, ntheta))
    elif psi0_type == "polynomial":
        # ψ₀(r) = (r/a)² * (1 - r/a)
        # Satisfies BC: ψ₀(0) = 0, ψ₀(a) = 0
        r_norm = r_2d / a
        psi0_2d = r_norm**2 * (1 - r_norm)
        
        # Compute ∇²ψ₀ in cylindrical coordinates
        # ∇²ψ = (1/r) d/dr(r dψ/dr) + (1/r²) d²ψ/dθ²
        # For axisymmetric: ∇²ψ = (1/r) d/dr(r dψ/dr)
        
        # dψ/dr = d/dr[(r/a)²(1-r/a)] = (r/a²)[2(1-r/a) - (r/a)]
        #       = (r/a²)[2 - 3r/a]
        dpsi_dr = (r_2d / a**2) * (2 - 3*r_2d/a)
        
        # d²ψ/dr² = d/dr[(r/a²)(2 - 3r/a)]
        #         = (1/a²)[2 - 6r/a]
        d2psi_dr2 = (1/a**2) * (2 - 6*r_2d/a)
        
        # ∇²ψ = d²ψ/dr² + (1/r) dψ/dr
        # Handle r=0 singularity
        r_safe = np.maximum(r_2d, 1e-10)
        omega0_2d = d2psi_dr2 + dpsi_dr / r_safe
    else:
        raise ValueError(f"Unknown psi0_type: {psi0_type}")
    
    # Broadcast to 3D (replicate along ζ axis)
    psi0 = np.repeat(psi0_2d[:, :, np.newaxis], nzeta, axis=2)
    omega0 = np.repeat(omega0_2d[:, :, np.newaxis], nzeta, axis=2)
    
    # Enforce Dirichlet boundary conditions (required for IMEX solver)
    # Grid boundaries (r_min, r_max) approximate physical boundaries (0, a)
    psi0[0, :, :] = 0.0  # Inner boundary
    psi0[-1, :, :] = 0.0  # Outer boundary
    omega0[0, :, :] = 0.0
    omega0[-1, :, :] = 0.0
    
    return psi0, omega0, q_profile


def create_ballooning_mode_ic(
    grid: Grid3D,
    n: int = 5,
    m0: int = 2,
    epsilon: float = 0.01,
    r_s: float = 0.5,
    Delta_r: float = 0.1,
    q_profile: Optional[np.ndarray] = None,
    num_m_modes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D ballooning mode perturbation ψ₁(r, θ, ζ).
    
    The perturbation structure is:
        ψ₁(r, θ, ζ) = ε · A(r) · Y(θ₀) · exp(i·n·ζ)
    
    where θ₀ = θ - n·q(r)·ζ is the extended poloidal angle (field-line coordinate).
    
    Parameters
    ----------
    grid : Grid3D
        3D toroidal grid
    n : int, optional
        Toroidal mode number (default: 5, typical range: 5-10)
    m0 : int, optional
        Central poloidal mode (default: 2)
    epsilon : float, optional
        Perturbation amplitude (default: 0.01, typical range: 0.01-0.1)
    r_s : float, optional
        Rational surface radius (default: 0.5)
    Delta_r : float, optional
        Radial width of mode (default: 0.1)
    q_profile : np.ndarray (nr,), optional
        Safety factor profile (if None, create linear q-profile)
    num_m_modes : int, optional
        Number of coupled m modes (default: 5, i.e., m0±2)
    
    Returns
    -------
    psi1 : np.ndarray (nr, nθ, nζ)
        Ballooning mode perturbation
    omega1 : np.ndarray (nr, nθ, nζ)
        Perturbation vorticity ω₁ = ∇²ψ₁
    
    Notes
    -----
    Algorithm:
        1. Radial profile: A(r) = exp(-(r - r_s)²/Δr²)
        2. Extended angle: θ₀ = θ - n·q(r)·ζ
        3. Ballooning envelope: Y(θ₀) = Σ_m a_m · exp(i·m·θ₀)
        4. Full perturbation: ψ₁ = ε · A(r) · Y(θ₀) · exp(i·n·ζ)
        5. Take real part (physical field)
    
    Rational surface:
        - Located at r = r_s where q(r_s) ≈ m₀/n
        - Mode is radially localized around r_s with width Δr
    
    Mode coupling:
        - m modes coupled within range [m0 - Δm, m0 + Δm]
        - Default: Δm = (num_m_modes - 1) / 2 = 2 (i.e., m0±2)
        - Amplitudes: Gaussian-like (peak at m0)
    
    Physical interpretation:
        - Localized at bad curvature region (outboard side, θ₀ ≈ 0)
        - High-n modes couple multiple poloidal modes m
        - Dominant instability in high-β tokamaks
    
    Examples
    --------
    >>> grid = Grid3D(nr=32, ntheta=64, nzeta=128)
    >>> psi1, omega1 = create_ballooning_mode_ic(grid, n=5, m0=2, epsilon=0.01)
    >>> assert psi1.shape == (32, 64, 128)
    >>> assert np.max(np.abs(psi1)) < 0.02  # Small perturbation
    """
    # Validate inputs
    if n < 1:
        raise ValueError(f"Toroidal mode number n must be >= 1, got {n}")
    if m0 < 1:
        raise ValueError(f"Poloidal mode number m0 must be >= 1, got {m0}")
    if epsilon <= 0 or epsilon > 1:
        raise ValueError(f"Perturbation amplitude epsilon must be in (0, 1], got {epsilon}")
    if r_s <= 0 or r_s > grid.r_max:
        raise ValueError(f"Rational surface r_s must be in (0, {grid.r_max}], got {r_s}")
    if Delta_r <= 0 or Delta_r > grid.r_max:
        raise ValueError(f"Radial width Delta_r must be in (0, {grid.r_max}], got {Delta_r}")
    
    # Create q-profile if not provided
    if q_profile is None:
        q_profile = create_q_profile(grid.r, q0=1.0, qa=3.0, profile_type='linear')
    
    # === Step 1: Radial profile A(r) ===
    # Gaussian centered at rational surface r_s
    A_r = np.exp(-(grid.r - r_s)**2 / Delta_r**2)  # (nr,)
    A_r_3d = A_r[:, np.newaxis, np.newaxis]  # Broadcast to (nr, 1, 1)
    
    # === Step 2: Extended poloidal angle θ₀ ===
    # Create 3D meshgrid
    r_3d, theta_3d, zeta_3d = np.meshgrid(
        grid.r, grid.theta, grid.zeta, indexing='ij'
    )
    
    # q-profile in 3D
    q_3d = q_profile[:, np.newaxis, np.newaxis]  # (nr, 1, 1)
    q_3d = np.broadcast_to(q_3d, r_3d.shape)  # (nr, ntheta, nzeta)
    
    # Extended angle: θ₀ = θ - n·q(r)·ζ
    theta_0 = theta_3d - n * q_3d * zeta_3d  # (nr, ntheta, nzeta)
    
    # === Step 3: Ballooning envelope Y(θ₀) ===
    # Couple m modes: m0-Δm, ..., m0, ..., m0+Δm
    Delta_m = (num_m_modes - 1) // 2
    m_modes = np.arange(m0 - Delta_m, m0 + Delta_m + 1)
    
    # Amplitudes: Gaussian-like (peak at m0)
    # a_m ∝ exp(-(m - m0)²/σ²), σ = Δm/2
    sigma_m = Delta_m / 2 if Delta_m > 0 else 1.0
    a_m = np.exp(-((m_modes - m0)**2) / (2 * sigma_m**2))
    a_m = a_m / np.max(a_m)  # Normalize to max = 1
    
    # Sum over m modes: Y(θ₀) = Σ_m a_m · exp(i·m·θ₀)
    Y = np.zeros_like(theta_0, dtype=complex)
    for m, amp in zip(m_modes, a_m):
        # Simplified phase (no WKB correction in v1.4)
        Y += amp * np.exp(1j * m * theta_0)
    
    # === Step 4: Full perturbation ===
    # ψ₁ = ε · A(r) · Y(θ₀) · exp(i·n·ζ)
    psi1_complex = epsilon * A_r_3d * Y * np.exp(1j * n * zeta_3d)
    
    # Take real part (physical field)
    psi1 = psi1_complex.real
    
    # === Step 5: Compute ω₁ = ∇²ψ₁ ===
    # Use finite difference Laplacian
    omega1 = _compute_laplacian_3d_simple(psi1, grid)
    
    return psi1, omega1


def _compute_laplacian_3d_simple(
    psi: np.ndarray,
    grid: Grid3D,
) -> np.ndarray:
    """
    Compute 3D Laplacian ∇²ψ using 2nd-order finite differences.
    
    In cylindrical coordinates:
        ∇²ψ = ∂²ψ/∂r² + (1/r) ∂ψ/∂r + (1/r²) ∂²ψ/∂θ² + ∂²ψ/∂ζ²
    
    Parameters
    ----------
    psi : np.ndarray (nr, nθ, nζ)
        Field to differentiate
    grid : Grid3D
        Grid object
    
    Returns
    -------
    laplacian : np.ndarray (nr, nθ, nζ)
        Laplacian ∇²ψ
    
    Notes
    -----
    - Radial: 2nd-order centered FD (Dirichlet BC)
    - Poloidal: 2nd-order centered FD (periodic BC)
    - Toroidal: 2nd-order centered FD (periodic BC)
    - Metric factor 1/r² handled carefully at small r
    """
    nr, ntheta, nzeta = grid.nr, grid.ntheta, grid.nzeta
    dr, dtheta, dzeta = grid.dr, grid.dtheta, grid.dzeta
    
    omega = np.zeros_like(psi)
    
    # Create 3D r array for metric factors
    r_3d = grid.r[:, np.newaxis, np.newaxis]  # (nr, 1, 1)
    r_safe = np.maximum(r_3d, 1e-10)  # Avoid division by zero
    
    # === Radial derivatives ===
    # ∂²ψ/∂r²: 2nd-order centered FD
    d2psi_dr2 = np.zeros_like(psi)
    d2psi_dr2[1:-1, :, :] = (
        psi[2:, :, :] - 2*psi[1:-1, :, :] + psi[:-2, :, :]
    ) / dr**2
    # Boundaries: assume ψ = 0 at r=0 and r=r_max (Dirichlet BC)
    d2psi_dr2[0, :, :] = (psi[1, :, :] - 2*psi[0, :, :]) / dr**2
    d2psi_dr2[-1, :, :] = (-2*psi[-1, :, :] + psi[-2, :, :]) / dr**2
    
    # ∂ψ/∂r: 2nd-order centered FD
    dpsi_dr = np.zeros_like(psi)
    dpsi_dr[1:-1, :, :] = (psi[2:, :, :] - psi[:-2, :, :]) / (2*dr)
    dpsi_dr[0, :, :] = (psi[1, :, :] - psi[0, :, :]) / dr
    dpsi_dr[-1, :, :] = (psi[-1, :, :] - psi[-2, :, :]) / dr
    
    # === Poloidal derivatives ===
    # ∂²ψ/∂θ²: 2nd-order centered FD (periodic)
    d2psi_dtheta2 = np.zeros_like(psi)
    d2psi_dtheta2[:, 1:-1, :] = (
        psi[:, 2:, :] - 2*psi[:, 1:-1, :] + psi[:, :-2, :]
    ) / dtheta**2
    # Periodic BC
    d2psi_dtheta2[:, 0, :] = (
        psi[:, 1, :] - 2*psi[:, 0, :] + psi[:, -1, :]
    ) / dtheta**2
    d2psi_dtheta2[:, -1, :] = (
        psi[:, 0, :] - 2*psi[:, -1, :] + psi[:, -2, :]
    ) / dtheta**2
    
    # === Toroidal derivatives ===
    # ∂²ψ/∂ζ²: 2nd-order centered FD (periodic)
    d2psi_dzeta2 = np.zeros_like(psi)
    d2psi_dzeta2[:, :, 1:-1] = (
        psi[:, :, 2:] - 2*psi[:, :, 1:-1] + psi[:, :, :-2]
    ) / dzeta**2
    # Periodic BC
    d2psi_dzeta2[:, :, 0] = (
        psi[:, :, 1] - 2*psi[:, :, 0] + psi[:, :, -1]
    ) / dzeta**2
    d2psi_dzeta2[:, :, -1] = (
        psi[:, :, 0] - 2*psi[:, :, -1] + psi[:, :, -2]
    ) / dzeta**2
    
    # === Combine: ∇²ψ = ∂²ψ/∂r² + (1/r) ∂ψ/∂r + (1/r²) ∂²ψ/∂θ² + ∂²ψ/∂ζ² ===
    omega = (
        d2psi_dr2
        + dpsi_dr / r_safe
        + d2psi_dtheta2 / r_safe**2
        + d2psi_dzeta2
    )
    
    return omega
