#!/usr/bin/env python3
"""
Safety Factor (q-profile) Calculator

Accurate q-profile calculation using flux surface averaging.
Fixes cylindrical approximation error (PHYS-01).

Reference: FreeGS critical.py::find_safety
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Union
import warnings

from .flux_surface_tracer import FluxSurfaceTracer


class QCalculator:
    """
    Calculate safety factor q(ψ) from equilibrium
    
    Uses flux surface integration:
        q(ψ) = (1/2π) ∮ [F(ψ) / (R² Bθ)] dl
    
    where:
    - F(ψ) = R·Bφ: toroidal field function
    - Bθ = |∇ψ|/R: poloidal field magnitude
    - dl: poloidal arc length element
    
    Parameters
    ----------
    psi : array (nr, nz)
        Poloidal flux on grid
    R_1d : array (nr,)
        Major radius grid
    Z_1d : array (nz,)
        Height grid
    fpol : callable
        F(psi_norm) function (toroidal field)
    Br_func : callable
        Br(R, Z) function (radial field)
    Bz_func : callable
        Bz(R, Z) function (vertical field)
    
    Attributes
    ----------
    tracer : FluxSurfaceTracer
        Flux surface locator
    """
    
    def __init__(
        self, 
        psi: np.ndarray,
        R_1d: np.ndarray,
        Z_1d: np.ndarray,
        fpol: callable,
        Br_func: callable,
        Bz_func: callable
    ):
        self.psi = psi
        self.R_1d = R_1d
        self.Z_1d = Z_1d
        self.fpol = fpol
        self.Br_func = Br_func
        self.Bz_func = Bz_func
        
        # Initialize flux surface tracer
        self.tracer = FluxSurfaceTracer(psi, R_1d, Z_1d)
    
    def compute_q_single(
        self, 
        psi_norm: float,
        ntheta: int = 128,
        xpoint: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Compute q at one flux surface
        
        Parameters
        ----------
        psi_norm : float
            Normalized flux (0 = axis, 1 = edge)
        ntheta : int
            Number of poloidal samples (default: 128)
        xpoint : tuple (R_x, Z_x), optional
            X-point location to avoid
        
        Returns
        -------
        q : float
            Safety factor
            
        Notes
        -----
        Returns NaN if flux surface not found
        """
        # Convert to absolute psi
        psi_target = self.tracer.denormalize_psi(psi_norm)
        
        # Get surface points
        R_surf, Z_surf = self.tracer.find_surface_points(
            psi_target, ntheta=ntheta, xpoint=xpoint
        )
        
        if len(R_surf) == 0:
            warnings.warn(
                f"No flux surface found at psi_norm={psi_norm:.3f}",
                stacklevel=2
            )
            return np.nan
        
        # Evaluate toroidal field function
        try:
            F = self.fpol(psi_norm)
        except Exception as e:
            warnings.warn(
                f"Failed to evaluate F(psi_norm={psi_norm:.3f}): {e}",
                stacklevel=2
            )
            return np.nan
        
        # Evaluate poloidal field at surface points
        Br = self.Br_func(R_surf, Z_surf)
        Bz = self.Bz_func(R_surf, Z_surf)
        
        # Poloidal field magnitude
        Btheta = np.sqrt(Br**2 + Bz**2)
        
        # Check for very small Btheta (numerical issue)
        if np.any(Btheta < 1e-12):
            warnings.warn(
                f"Very small Btheta detected at psi_norm={psi_norm:.3f}",
                stacklevel=2
            )
            # Set minimum to avoid division by zero
            Btheta = np.maximum(Btheta, 1e-12)
        
        # Integrand: F / (R² Bθ)
        qint = F / (R_surf**2 * Btheta)
        
        # Compute arc length elements
        # Use central difference with periodic boundary
        dR = np.roll(R_surf, -1) - np.roll(R_surf, 1)
        dZ = np.roll(Z_surf, -1) - np.roll(Z_surf, 1)
        dR /= 2.0
        dZ /= 2.0
        
        dl = np.sqrt(dR**2 + dZ**2)
        
        # Integrate: q = (1/2π) ∮ qint·dl
        q = np.sum(qint * dl) / (2 * np.pi)
        
        return q
    
    def compute_q_profile(
        self,
        psi_norm: Optional[Union[float, np.ndarray]] = None,
        npsi: int = 100,
        ntheta: int = 128,
        extrapolate: bool = True
    ) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute q profile with optional extrapolation
        
        Parameters
        ----------
        psi_norm : float, array, or None
            Normalized flux values to compute q at
            If None, use default grid [0.01, 0.99]
        npsi : int
            Number of points if psi_norm is None (default: 100)
        ntheta : int
            Number of poloidal samples per surface (default: 128)
        extrapolate : bool
            Use quadratic extrapolation for psi_norm < 0.01 or > 0.99
            
        Returns
        -------
        q : float or array
            If psi_norm was provided: return q values only
        psi_norm, q : tuple of arrays
            If psi_norm was None: return both psi_norm grid and q
            
        Notes
        -----
        Extrapolation avoids numerical issues at magnetic axis (Bθ → 0)
        and at the edge. Uses quadratic interpolation on safe interior
        range [0.01, 0.99].
        """
        if psi_norm is None:
            # Default: safe interior range
            psi_norm_calc = np.linspace(0.01, 0.99, npsi)
            return_both = True
        else:
            psi_norm_calc = np.atleast_1d(psi_norm)
            return_both = False
        
        # Check if extrapolation needed
        need_extrap = extrapolate and np.any(
            (psi_norm_calc < 0.01) | (psi_norm_calc > 0.99)
        )
        
        if need_extrap:
            # Compute on safe interior range
            psi_inner = np.linspace(0.01, 0.99, npsi)
            q_inner = np.array([
                self.compute_q_single(p, ntheta=ntheta) 
                for p in psi_inner
            ])
            
            # Check for NaN values
            if np.any(np.isnan(q_inner)):
                warnings.warn(
                    "NaN values in interior q calculation, "
                    "extrapolation may be unreliable",
                    stacklevel=2
                )
                # Remove NaN for interpolation
                valid = ~np.isnan(q_inner)
                psi_inner = psi_inner[valid]
                q_inner = q_inner[valid]
            
            # Quadratic interpolation + extrapolation
            try:
                interp = interp1d(
                    psi_inner, q_inner,
                    kind='quadratic',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                q_result = interp(psi_norm_calc)
            except Exception as e:
                warnings.warn(
                    f"Extrapolation failed: {e}. "
                    "Using linear extrapolation instead.",
                    stacklevel=2
                )
                # Fallback to linear
                interp = interp1d(
                    psi_inner, q_inner,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                q_result = interp(psi_norm_calc)
        else:
            # Direct calculation (no extrapolation needed)
            q_result = np.array([
                self.compute_q_single(p, ntheta=ntheta) 
                for p in psi_norm_calc
            ])
        
        # Return format
        if return_both:
            return psi_norm_calc, q_result
        else:
            if len(q_result) == 1:
                return float(q_result[0])
            return q_result
    
    def q_cylindrical_approx(self, psi_norm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        OLD METHOD: Cylindrical approximation for q
        
        q ≈ r·Bφ / (R·Bθ)
        
        This is kept for comparison and validation.
        Known to give ~30% error at magnetic axis.
        
        Parameters
        ----------
        psi_norm : float or array
            Normalized flux
        
        Returns
        -------
        q : float or array
            Approximate safety factor
        """
        # This would require (r, θ) coordinates
        # Not implemented here - just placeholder
        # Use actual implementation from existing code if needed
        raise NotImplementedError(
            "Cylindrical approximation kept for reference only. "
            "Use compute_q_profile() for accurate calculation."
        )


def integrate_along_surface(
    f_values: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray
) -> float:
    """
    Generic surface integral: ∮ f·dl
    
    Parameters
    ----------
    f_values : array (n,)
        Function values at surface points
    R_surf : array (n,)
        R coordinates of surface points
    Z_surf : array (n,)
        Z coordinates of surface points
    
    Returns
    -------
    integral : float
        Line integral ∮ f·dl
        
    Notes
    -----
    Uses trapezoidal rule with periodic boundary
    """
    # Arc length elements (central difference with periodic BC)
    dR = np.roll(R_surf, -1) - np.roll(R_surf, 1)
    dZ = np.roll(Z_surf, -1) - np.roll(Z_surf, 1)
    dR /= 2.0
    dZ /= 2.0
    
    dl = np.sqrt(dR**2 + dZ**2)
    
    # Integrate
    integral = np.sum(f_values * dl)
    
    return integral


def surface_average(
    f_values: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray
) -> float:
    """
    Surface average: ⟨f⟩ = ∮ f·dl / ∮ dl
    
    Parameters
    ----------
    f_values : array (n,)
        Function values at surface points
    R_surf : array (n,)
        R coordinates
    Z_surf : array (n,)
        Z coordinates
    
    Returns
    -------
    f_avg : float
        Surface-averaged value
    """
    # Arc length elements
    dR = np.roll(R_surf, -1) - np.roll(R_surf, 1)
    dZ = np.roll(Z_surf, -1) - np.roll(Z_surf, 1)
    dR /= 2.0
    dZ /= 2.0
    
    dl = np.sqrt(dR**2 + dZ**2)
    
    # Average
    numerator = np.sum(f_values * dl)
    denominator = np.sum(dl)
    
    if denominator < 1e-12:
        warnings.warn("Surface length near zero in averaging", stacklevel=2)
        return 0.0
    
    return numerator / denominator
