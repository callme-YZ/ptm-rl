#!/usr/bin/env python3
"""
Flux Surface Tracer

Locate flux surfaces on (R,Z) computational grid.
Used for accurate q-profile calculation.

Reference: FreeGS critical.py::find_psisurface
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Optional
import warnings


class FluxSurfaceTracer:
    """
    Locate flux surfaces ψ = const on (R,Z) grid
    
    Uses ray-shooting method from magnetic axis:
    1. Shoot rays at different poloidal angles θ
    2. Find where ψ(r) crosses target value
    3. Linear interpolation for exact location
    
    Parameters
    ----------
    psi : array (nr, nz)
        Poloidal flux on grid
    R_1d : array (nr,)
        Major radius grid
    Z_1d : array (nz,)
        Height grid
    
    Attributes
    ----------
    R_axis, Z_axis : float
        Magnetic axis location
    psi_axis : float
        Flux at magnetic axis (extremum)
    psi_edge : float
        Flux at edge (boundary value)
    """
    
    def __init__(self, psi: np.ndarray, R_1d: np.ndarray, Z_1d: np.ndarray):
        self.psi = psi
        self.R_1d = R_1d
        self.Z_1d = Z_1d
        
        # Grid spacing
        self.dR = R_1d[1] - R_1d[0]
        self.dZ = Z_1d[1] - Z_1d[0]
        
        # Setup interpolator for off-grid evaluation
        self.psi_interp = RectBivariateSpline(R_1d, Z_1d, psi)
        
        # Find magnetic axis
        self.R_axis, self.Z_axis, self.psi_axis = self._find_axis()
        
        # Edge value (assume boundary is at edge)
        # Tokamak sign convention: psi should increase from axis to edge
        # If solver produces psi_axis > psi_edge, we need to normalize correctly
        boundary_vals = np.concatenate([
            psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]
        ])
        psi_boundary_min = np.min(boundary_vals)
        psi_boundary_max = np.max(boundary_vals)
        
        # Find which boundary value is farther from axis
        if abs(self.psi_axis - psi_boundary_min) > abs(self.psi_axis - psi_boundary_max):
            self.psi_edge = psi_boundary_min
        else:
            self.psi_edge = psi_boundary_max
        
        # Ensure psi_axis and psi_edge are ordered for normalization
        # After this: psi_norm = (psi - psi_axis) / (psi_edge - psi_axis)
        # should give psi_norm=0 at axis, psi_norm=1 at edge
        # This works regardless of sign convention
    
    def _find_axis(self) -> Tuple[float, float, float]:
        """
        Find magnetic axis (O-point)
        
        Returns
        -------
        R_axis, Z_axis, psi_axis : float
            Location and flux value at magnetic axis
        """
        # Find extremum of psi
        # For standard tokamak: axis is maximum
        # For some codes: axis is minimum
        # Use absolute extremum
        
        psi_abs = np.abs(self.psi)
        idx = np.unravel_index(np.argmax(psi_abs), self.psi.shape)
        
        i_axis, j_axis = idx
        
        R_axis = self.R_1d[i_axis]
        Z_axis = self.Z_1d[j_axis]
        psi_axis = self.psi[i_axis, j_axis]
        
        # Refine with parabolic fit (improve accuracy)
        if 1 <= i_axis < len(self.R_1d) - 1:
            # Fit parabola in R direction
            p0 = self.psi[i_axis-1, j_axis]
            p1 = self.psi[i_axis, j_axis]
            p2 = self.psi[i_axis+1, j_axis]
            
            # Parabola vertex offset
            denom = 2 * (p0 - 2*p1 + p2)
            if abs(denom) > 1e-12:
                dr = -self.dR * (p2 - p0) / denom
                R_axis = R_axis + dr
        
        if 1 <= j_axis < len(self.Z_1d) - 1:
            # Fit parabola in Z direction
            p0 = self.psi[i_axis, j_axis-1]
            p1 = self.psi[i_axis, j_axis]
            p2 = self.psi[i_axis, j_axis+1]
            
            denom = 2 * (p0 - 2*p1 + p2)
            if abs(denom) > 1e-12:
                dz = -self.dZ * (p2 - p0) / denom
                Z_axis = Z_axis + dz
        
        # Re-evaluate psi at refined location
        psi_axis = self.psi_interp(R_axis, Z_axis, grid=False).item()
        
        return R_axis, Z_axis, psi_axis
    
    def find_surface_points(
        self, 
        psi_target: float, 
        ntheta: int = 128,
        xpoint: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find points on flux surface ψ = psi_target
        
        Parameters
        ----------
        psi_target : float
            Target flux value (absolute, not normalized)
        ntheta : int
            Number of poloidal angle samples (default: 128)
        xpoint : tuple (R_x, Z_x), optional
            X-point location to avoid in sampling
        
        Returns
        -------
        R_surf : array (ntheta,)
            R coordinates on surface
        Z_surf : array (ntheta,)
            Z coordinates on surface
            
        Notes
        -----
        Points are sorted by poloidal angle θ = arctan2(R-R_axis, Z-Z_axis)
        """
        # Normalize target psi for comparison
        psi_norm_target = (psi_target - self.psi_axis) / (self.psi_edge - self.psi_axis)
        
        # Create theta grid
        theta_grid = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        
        # Avoid X-point if provided
        if xpoint is not None:
            R_x, Z_x = xpoint
            xpoint_theta = np.arctan2(R_x - self.R_axis, Z_x - self.Z_axis)
            
            # Ensure xpoint_theta in [0, 2π)
            if xpoint_theta < 0:
                xpoint_theta += 2*np.pi
            
            # Check if any grid point is too close
            TOLERANCE = 1e-3  # 0.001 radians ≈ 0.06 degrees
            if np.any(np.abs(theta_grid - xpoint_theta) < TOLERANCE):
                warnings.warn(
                    "Theta grid too close to X-point, shifting by half-step",
                    stacklevel=2
                )
                dtheta = theta_grid[1] - theta_grid[0]
                theta_grid += dtheta / 2
        
        # Shoot rays at each angle
        R_surf = []
        Z_surf = []
        
        for theta in theta_grid:
            try:
                R, Z = self._ray_search(theta, psi_norm_target)
                R_surf.append(R)
                Z_surf.append(Z)
            except ValueError:
                # Ray didn't find crossing (e.g., psi_target outside domain)
                # Skip this point
                continue
        
        if len(R_surf) == 0:
            warnings.warn(
                f"No flux surface found for psi={psi_target:.4e}",
                stacklevel=2
            )
            return np.array([]), np.array([])
        
        return np.array(R_surf), np.array(Z_surf)
    
    def _ray_search(
        self, 
        theta: float, 
        psi_norm_target: float,
        n_ray: int = 100
    ) -> Tuple[float, float]:
        """
        Find flux surface point along one ray
        
        Parameters
        ----------
        theta : float
            Poloidal angle (radians) from magnetic axis
        psi_norm_target : float
            Normalized flux target (0 to 1)
        n_ray : int
            Number of sample points along ray
        
        Returns
        -------
        R, Z : float
            Location where ray crosses psi = target
            
        Raises
        ------
        ValueError
            If ray doesn't find a crossing
        """
        # Ray endpoint (extend beyond likely separatrix)
        r_max = 2 * max(
            self.R_1d.max() - self.R_axis,
            self.R_axis - self.R_1d.min(),
            self.Z_1d.max() - self.Z_axis,
            self.Z_axis - self.Z_1d.min()
        )
        
        R_end = self.R_axis + r_max * np.sin(theta)
        Z_end = self.Z_axis + r_max * np.cos(theta)
        
        # Clip to domain (maintain direction)
        if abs(R_end - self.R_axis) > 1e-6:
            R_clip = np.clip(R_end, self.R_1d[0], self.R_1d[-1])
            scale = abs((R_clip - self.R_axis) / (R_end - self.R_axis))
            Z_end = self.Z_axis + (Z_end - self.Z_axis) * scale
            R_end = R_clip
        
        if abs(Z_end - self.Z_axis) > 1e-6:
            Z_clip = np.clip(Z_end, self.Z_1d[0], self.Z_1d[-1])
            scale = abs((Z_clip - self.Z_axis) / (Z_end - self.Z_axis))
            R_end = self.R_axis + (R_end - self.R_axis) * scale
            Z_end = Z_clip
        
        # Sample along ray
        R_ray = np.linspace(self.R_axis, R_end, n_ray)
        Z_ray = np.linspace(self.Z_axis, Z_end, n_ray)
        
        # Evaluate psi along ray
        psi_ray = self.psi_interp(R_ray, Z_ray, grid=False)
        
        # Normalize
        psi_norm_ray = (psi_ray - self.psi_axis) / (self.psi_edge - self.psi_axis)
        
        # Find first point where psi_norm > target
        # (assumes psi increases from axis to edge)
        ind = np.argmax(psi_norm_ray > psi_norm_target)
        
        if ind == 0:
            # Check if we're at axis or didn't find crossing
            if psi_norm_ray[0] > psi_norm_target:
                # Target is inside innermost sampled point
                # Return axis (very close to target surface)
                return self.R_axis, self.Z_axis
            else:
                # Didn't find crossing (target beyond domain)
                raise ValueError(f"Ray at theta={theta:.3f} didn't cross target psi")
        
        # Linear interpolation between ind-1 and ind
        f = (psi_norm_ray[ind] - psi_norm_target) / \
            (psi_norm_ray[ind] - psi_norm_ray[ind-1])
        
        # Clamp f to [0, 1] (should be automatic, but be safe)
        f = np.clip(f, 0, 1)
        
        R = (1 - f) * R_ray[ind] + f * R_ray[ind-1]
        Z = (1 - f) * Z_ray[ind] + f * Z_ray[ind-1]
        
        # Warn if extrapolating (f > 1 shouldn't happen after clip)
        if f > 0.999:
            warnings.warn(
                f"Ray search near extrapolation (f={f:.3f}) at theta={theta:.3f}",
                stacklevel=3
            )
        
        return R, Z
    
    def normalize_psi(self, psi: float) -> float:
        """
        Normalize flux: 0 at axis, 1 at edge
        
        Parameters
        ----------
        psi : float
            Absolute flux value
        
        Returns
        -------
        psi_norm : float
            Normalized flux
        """
        return (psi - self.psi_axis) / (self.psi_edge - self.psi_axis)
    
    def denormalize_psi(self, psi_norm: float) -> float:
        """
        Convert normalized flux to absolute value
        
        Parameters
        ----------
        psi_norm : float
            Normalized flux (0 to 1)
        
        Returns
        -------
        psi : float
            Absolute flux value
        """
        return self.psi_axis + psi_norm * (self.psi_edge - self.psi_axis)
