"""
Toroidal Coordinate System

Implements toroidal coordinate system (r, θ, φ) for tokamak geometry.

Coordinates:
    r: minor radius (flux surface label) [m]
    θ: poloidal angle [rad]
    φ: toroidal angle [rad] (axisymmetric: ∂/∂φ = 0)

Metric tensor (orthogonal in (r,θ) plane):
    g_rr = 1
    g_θθ = r²
    g_φφ = R² = (R₀ + r*cos(θ))²
    g_rθ = 0 (orthogonal)
    
Jacobian:
    √g = r*R

Coordinate transformation:
    R = R₀ + r*cos(θ)  (major radius)
    Z = r*sin(θ)       (vertical)
    φ = φ              (toroidal angle)

References:
    - Design doc: v1.1-toroidal-symplectic-design.md
    - Pyrokinetics toroidal study: notes/pyrokinetics-toroidal-study.md
    - D'haeseleer et al. "Flux Coordinates and Magnetic Field Structure"

Author: 小P ⚛️
Created: 2026-03-17
"""

import numpy as np
from typing import Tuple


class ToroidalGrid:
    """
    Toroidal coordinate grid for tokamak MHD simulations.
    
    This class provides the geometric foundation for toroidal MHD solvers,
    implementing the metric tensor and coordinate transformations.
    
    Parameters
    ----------
    R0 : float
        Major radius [m]
    a : float
        Minor radius [m] (plasma edge)
    nr : int
        Number of radial grid points
    ntheta : int
        Number of poloidal grid points
    
    Attributes
    ----------
    R0 : float
        Major radius
    a : float
        Minor radius
    nr, ntheta : int
        Grid resolution
    r : np.ndarray (nr,)
        Radial coordinate array [0, a]
    theta : np.ndarray (ntheta,)
        Poloidal angle array [0, 2π]
    r_grid : np.ndarray (nr, ntheta)
        2D radial coordinate mesh
    theta_grid : np.ndarray (nr, ntheta)
        2D poloidal angle mesh
    R_grid : np.ndarray (nr, ntheta)
        Major radius R = R₀ + r*cos(θ)
    Z_grid : np.ndarray (nr, ntheta)
        Vertical coordinate Z = r*sin(θ)
    dr : float
        Radial grid spacing
    dtheta : float
        Poloidal grid spacing
    
    Examples
    --------
    >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
    >>> g_rr, g_tt, g_pp = grid.metric_tensor()
    >>> J = grid.jacobian()
    >>> print(f"Jacobian range: [{J.min():.3f}, {J.max():.3f}]")
    
    Notes
    -----
    - Assumes axisymmetric equilibrium (∂/∂φ = 0)
    - Coordinates are orthogonal in (r, θ) plane
    - Periodic boundary in θ direction
    """
    
    def __init__(self, R0: float, a: float, nr: int, ntheta: int):
        """
        Initialize toroidal grid.
        
        Parameters
        ----------
        R0 : float
            Major radius [m], must be > 0
        a : float
            Minor radius [m], must be > 0 and < R0
        nr : int
            Radial resolution, must be >= 32
        ntheta : int
            Poloidal resolution, must be >= 64
        """
        # Validation
        if R0 <= 0:
            raise ValueError(f"Major radius R0 must be positive, got {R0}")
        if a <= 0:
            raise ValueError(f"Minor radius a must be positive, got {a}")
        if a >= R0:
            raise ValueError(f"Minor radius a={a} must be < R0={R0}")
        if nr < 32:
            raise ValueError(f"Radial resolution nr must be >= 32, got {nr}")
        if ntheta < 64:
            raise ValueError(f"Poloidal resolution ntheta must be >= 64, got {ntheta}")
        
        self.R0 = R0
        self.a = a
        self.nr = nr
        self.ntheta = ntheta
        
        # 1D coordinate arrays
        # r: [0, a] with small offset from r=0 to avoid singularity
        self.r = np.linspace(1e-6, a, nr)
        # theta: [0, 2π] periodic
        self.theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        
        # Grid spacing
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]
        
        # 2D meshgrid
        self.r_grid, self.theta_grid = np.meshgrid(self.r, self.theta, indexing='ij')
        
        # Cartesian coordinates (R, Z)
        self.R_grid = self.R0 + self.r_grid * np.cos(self.theta_grid)
        self.Z_grid = self.r_grid * np.sin(self.theta_grid)
    
    def metric_tensor(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute covariant metric tensor components.
        
        For orthogonal toroidal coordinates:
            g_rr = 1                    (radial)
            g_θθ = r²                   (poloidal)
            g_φφ = R² = (R₀+r*cos(θ))²  (toroidal)
            g_rθ = g_rφ = g_θφ = 0      (orthogonality)
        
        Returns
        -------
        g_rr : np.ndarray (nr, ntheta)
            Radial metric component (= 1 everywhere)
        g_tt : np.ndarray (nr, ntheta)
            Poloidal metric component (= r²)
        g_pp : np.ndarray (nr, ntheta)
            Toroidal metric component (= R²)
        
        Notes
        -----
        - All off-diagonal components are zero (orthogonal system)
        - These are used in differential operators (gradient, divergence, Laplacian)
        
        Examples
        --------
        >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        >>> g_rr, g_tt, g_pp = grid.metric_tensor()
        >>> assert np.allclose(g_rr, 1.0)
        >>> assert np.allclose(g_tt, grid.r_grid**2)
        """
        g_rr = np.ones_like(self.r_grid)
        g_tt = self.r_grid**2
        g_pp = self.R_grid**2
        
        return g_rr, g_tt, g_pp
    
    def jacobian(self) -> np.ndarray:
        """
        Compute Jacobian √g = r*R.
        
        The Jacobian is the volume element:
            dV = √g dr dθ dφ = r*R dr dθ dφ
        
        Returns
        -------
        J : np.ndarray (nr, ntheta)
            Jacobian √g = r*(R₀ + r*cos(θ))
        
        Notes
        -----
        - Must be positive everywhere (required for valid coordinate system)
        - Used in divergence and volume integration
        
        Examples
        --------
        >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        >>> J = grid.jacobian()
        >>> assert np.all(J > 0)  # Must be positive
        """
        J = self.r_grid * self.R_grid
        return J
    
    def to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """
        Convert toroidal (r, θ) to Cartesian (R, Z).
        
        Transformation:
            R = R₀ + r*cos(θ)
            Z = r*sin(θ)
        
        Parameters
        ----------
        r : float
            Minor radius [m], 0 <= r <= a
        theta : float
            Poloidal angle [rad], typically [0, 2π]
        
        Returns
        -------
        R : float
            Major radius [m]
        Z : float
            Vertical coordinate [m]
        
        Examples
        --------
        >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        >>> R, Z = grid.to_cartesian(r=0.2, theta=np.pi/4)
        >>> print(f"R={R:.4f}, Z={Z:.4f}")
        R=1.1414, Z=0.1414
        """
        R = self.R0 + r * np.cos(theta)
        Z = r * np.sin(theta)
        return R, Z
    
    def from_cartesian(self, R: float, Z: float) -> Tuple[float, float]:
        """
        Convert Cartesian (R, Z) to toroidal (r, θ).
        
        Inverse transformation:
            r = √[(R-R₀)² + Z²]
            θ = atan2(Z, R-R₀)
        
        Parameters
        ----------
        R : float
            Major radius [m]
        Z : float
            Vertical coordinate [m]
        
        Returns
        -------
        r : float
            Minor radius [m]
        theta : float
            Poloidal angle [rad], in range [0, 2π]
        
        Notes
        -----
        - Returns theta in [0, 2π] (not [-π, π])
        - Inverse of to_cartesian()
        
        Examples
        --------
        >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        >>> r_in, theta_in = 0.2, np.pi/4
        >>> R, Z = grid.to_cartesian(r_in, theta_in)
        >>> r_out, theta_out = grid.from_cartesian(R, Z)
        >>> assert abs(r_out - r_in) < 1e-12
        >>> assert abs(theta_out - theta_in) < 1e-12
        """
        r = np.sqrt((R - self.R0)**2 + Z**2)
        theta = np.arctan2(Z, R - self.R0)
        
        # Ensure theta in [0, 2π]
        if theta < 0:
            theta += 2 * np.pi
        
        return r, theta
    
    def get_rational_surface(self, m: int, n: int) -> dict:
        """
        Get radial location of (m, n) rational surface.
        
        For a given safety factor profile q(r), the rational surface
        is where q(r_s) = m/n.
        
        Parameters
        ----------
        m : int
            Poloidal mode number
        n : int
            Toroidal mode number
        
        Returns
        -------
        info : dict
            {'r_s': float, 'q_s': float} if rational surface exists,
            otherwise {'r_s': None, 'q_s': None}
        
        Notes
        -----
        - Requires q-profile to be set externally (not part of grid)
        - For M1 implementation, this is a placeholder
        - Will be implemented in M2 when equilibrium is added
        
        Examples
        --------
        >>> grid = ToroidalGrid(R0=1.0, a=0.3, nr=64, ntheta=128)
        >>> info = grid.get_rational_surface(m=2, n=1)
        >>> # Currently returns placeholder
        """
        # Placeholder for M1
        # Will be implemented when equilibrium module is added
        return {'r_s': None, 'q_s': None}
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ToroidalGrid(R0={self.R0:.2f}m, a={self.a:.2f}m, "
                f"nr={self.nr}, ntheta={self.ntheta})")
