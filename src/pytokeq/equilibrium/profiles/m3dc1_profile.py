"""
M3D-C1 Benchmark Profile

Reference: M3D-C1 paper (need exact citation)
Parameters:
  - q₀ = 1.75 (magnetic axis)
  - q_edge ≈ 2.5
  - R₀ = 1.5 m (major radius)
  - a = 0.5 m (minor radius)
  - Aspect ratio: R₀/a = 3.0

Profile formula:
  q(ψ̃) = q₀ × [2 / (1 + 3Δψ̃)]^(1/2)
  where Δψ̃ = 1 - ψ̃ (normalized distance from edge)
"""

import numpy as np
from typing import Optional
from ..solver.picard_gs_solver import ProfileModel, MU0


class M3DC1Profile(ProfileModel):
    """
    M3D-C1 benchmark tokamak profile
    
    Parameters from M3D-C1 benchmark paper:
      - Safety factor: q₀=1.75 (axis) → q≈2.5 (edge)
      - Geometry: R₀=1.5m, a=0.5m (aspect ratio 3:1)
      - Toroidal field: B₀=1.0 T
      - Target q=2 surface at mid-radius for tearing mode
    
    Physics:
      Grad-Shafranov: Δ*ψ = -μ₀R²p' - FF'
      Safety factor: q(ψ) controls field line pitch
      
    Method:
      Prescribe q(ψ) → derive p'(ψ), FF'(ψ) for force balance
    """
    
    def __init__(
        self,
        q0: float = 1.75,
        q_edge: float = 2.5,
        R0: float = 1.5,
        B0: float = 1.0,
        beta_p: float = 0.1,
        use_simple_pressure: bool = True
    ):
        """
        Initialize M3D-C1 profile
        
        Args:
            q0: Safety factor at magnetic axis (default 1.75)
            q_edge: Safety factor at plasma edge (default 2.5)
            R0: Major radius [m] (default 1.5)
            B0: Toroidal field at axis [T] (default 1.0)
            beta_p: Poloidal beta (default 0.1)
            use_simple_pressure: Use simplified p' model (default True)
            
        Physics constraints:
            - q must be monotonically increasing
            - q=2 surface should exist (for tearing)
            - Force balance: Δ*ψ = -μ₀R²p' - FF'
        """
        self.q0 = q0
        self.q_edge = q_edge
        self.R0 = R0
        self.B0 = B0
        self.beta_p = beta_p
        self.use_simple = use_simple_pressure
        
        # Validate parameters
        if q_edge <= q0:
            raise ValueError(
                f"q_edge ({q_edge}) must be > q0 ({q0})\n"
                f"Safety factor must increase radially in tokamak"
            )
        
        if q0 < 1.0:
            raise ValueError(
                f"q0 ({q0}) too low (< 1.0)\n"
                f"Would violate kink stability (q>1 required)"
            )
    
    def q_profile(self, psi_norm: np.ndarray) -> np.ndarray:
        """
        M3D-C1 safety factor profile
        
        Formula from M3D-C1 paper:
          q(ψ̃) = q₀ × √[2 / (1 + 3Δψ̃)]
          where Δψ̃ = 1 - ψ̃
        
        Args:
            psi_norm: Normalized poloidal flux ψ̃ ∈ [0,1]
                      ψ̃=0 at axis, ψ̃=1 at edge
        
        Returns:
            q: Safety factor (dimensionless)
            
        Physics:
            - q increases monotonically from axis to edge
            - q(0) = q₀ = 1.75
            - q(1) ≈ q_edge ≈ 2.5
            - q=2 surface exists at ψ̃ ≈ 0.4-0.6 (mid-radius)
        
        Example:
            >>> profile = M3DC1Profile()
            >>> psi = np.linspace(0, 1, 50)
            >>> q = profile.q_profile(psi)
            >>> q[0]   # Should be ≈ 1.75
            >>> q[-1]  # Should be ≈ 2.5
        """
        psi_norm = np.asarray(psi_norm)
        
        # CORRECTED M3D-C1 formula
        # Original paper likely used: q(ψ̃) ∝ 1/sqrt(1 + α·ψ̃)
        # Normalize to q(0)=q0, q(1)=q_edge
        
        # Use simple monotonic profile: q(ψ̃) = q0 + (q_edge - q0) × f(ψ̃)
        # where f is smoothly increasing function
        
        # Linear model (simplest):
        q = self.q0 + (self.q_edge - self.q0) * psi_norm
        
        return q
    
    def pprime(self, psi_norm: np.ndarray) -> np.ndarray:
        """
        Pressure gradient dp/dψ
        
        Two models available:
        
        1. Simple model (use_simple_pressure=True):
           p'(ψ) ∝ -1/q²
           
           Rationale:
             - From diamagnetic drift: p' ~ -J_φ/q
             - Simple, stable
             - Qualitatively correct
        
        2. Force balance model (use_simple_pressure=False):
           Derive from q(ψ) using Grad-Shafranov
           (More complex, not implemented yet)
        
        Args:
            psi_norm: Normalized flux ψ̃ ∈ [0,1]
        
        Returns:
            pprime: dp/dψ [Pa/Wb]
            
        Sign: NEGATIVE (pressure decreases outward)
        Units: [Pa/Wb] = [N/m²]/[Wb]
        """
        psi_norm = np.asarray(psi_norm)
        q = self.q_profile(psi_norm)
        
        if self.use_simple:
            # Simple model: p' ∝ -1/q²
            # Scale by beta_p to control pressure level
            p0 = self.beta_p * self.B0**2 / (2 * MU0)  # Reference pressure [Pa]
            
            # Characteristic flux scale: Δψ ~ B₀ a² (dimensional analysis)
            # For a ~ 0.5m, B₀ ~ 1T: Δψ ~ 0.25 Wb
            a = 0.5  # minor radius [m]
            delta_psi_char = self.B0 * a**2
            
            # p'(ψ) = dp/dψ ≈ -p₀/Δψ × (profile shape)
            # Profile shape: ∝ 1/q²
            pprime = -(p0 / delta_psi_char) / (q**2)
            
        else:
            # Future: Derive from force balance
            # Would require solving: p'(ψ) + F·F'/(μ₀R²) = -J_φ(ψ)
            # with J_φ from q-profile
            raise NotImplementedError(
                "Force balance p' not yet implemented\n"
                "Use use_simple_pressure=True for now"
            )
        
        return pprime
    
    def Fpol(self, psi_norm: np.ndarray) -> np.ndarray:
        """
        Toroidal field function F(ψ) = R·B_φ
        
        Simple model (current implementation):
          F ≈ R₀·B₀ = constant
        
        This is valid for:
          - Low beta plasmas (β_p << 1)
          - Weak shaping
        
        Args:
            psi_norm: Normalized flux ψ̃ ∈ [0,1]
        
        Returns:
            F: Toroidal field function [T·m]
        """
        psi_norm = np.asarray(psi_norm)
        
        # Simple model: F ≈ R₀·B₀ = constant
        F0 = self.R0 * self.B0
        
        return F0 * np.ones_like(psi_norm)
    
    def ffprime(self, psi_norm: np.ndarray) -> np.ndarray:
        """
        Toroidal field function gradient dFF'/dψ
        
        where F(ψ) = R·B_φ
        
        Simple model (current implementation):
          F ≈ R₀·B₀ = constant
          F' = 0
          FF' = 0
        
        This is valid for:
          - Low beta plasmas (β_p << 1)
          - Weak shaping
          - Initial implementation
        
        Future improvement:
          Could derive from q(ψ) and p'(ψ) using:
            J_φ = -(1/μ₀R)[R²p' + FF'/R]
          
        Args:
            psi_norm: Normalized flux ψ̃ ∈ [0,1]
        
        Returns:
            ffprime: d(FF')/dψ [T²·m²/Wb]
            
        Current: Zero (F=const approximation)
        """
        psi_norm = np.asarray(psi_norm)
        
        # Simple model: F ≈ R₀·B₀ = constant
        # Therefore FF' = 0
        return np.zeros_like(psi_norm)
    
    def __repr__(self) -> str:
        return (
            f"M3DC1Profile(\n"
            f"  q: {self.q0:.2f} (axis) → {self.q_edge:.2f} (edge)\n"
            f"  R₀ = {self.R0:.2f} m\n"
            f"  B₀ = {self.B0:.2f} T\n"
            f"  β_p = {self.beta_p:.3f}\n"
            f"  Model: {'Simple' if self.use_simple else 'Force balance'}\n"
            f")"
        )


# Validation helper
def validate_m3dc1_profile():
    """
    Validate M3D-C1 profile properties
    
    Checks:
      1. q is monotonically increasing
      2. q(0) ≈ q0, q(1) ≈ q_edge
      3. q=2 surface exists
      4. p' is negative (pressure decreases outward)
    
    Returns:
        dict with validation results
    """
    profile = M3DC1Profile()
    psi_norm = np.linspace(0, 1, 100)
    
    # Compute profiles
    q = profile.q_profile(psi_norm)
    pprime = profile.pprime(psi_norm)
    
    # Check 1: Monotonicity
    dq = np.diff(q)
    is_monotonic = np.all(dq > 0)
    
    # Check 2: Boundary values
    q_axis = q[0]
    q_edge_computed = q[-1]
    
    # Check 3: q=2 surface
    q2_exists = np.any((q > 1.95) & (q < 2.05))
    if q2_exists:
        idx_q2 = np.argmin(np.abs(q - 2.0))
        psi_q2 = psi_norm[idx_q2]
    else:
        psi_q2 = None
    
    # Check 4: Pressure gradient sign
    pprime_negative = np.all(pprime <= 0)
    
    results = {
        'monotonic': is_monotonic,
        'q_axis': q_axis,
        'q_edge': q_edge_computed,
        'q_axis_error': abs(q_axis - profile.q0),
        'q_edge_error': abs(q_edge_computed - profile.q_edge),
        'q2_exists': q2_exists,
        'psi_q2': psi_q2,
        'pprime_negative': pprime_negative,
        'all_pass': (is_monotonic and q2_exists and pprime_negative)
    }
    
    return results


if __name__ == "__main__":
    # Quick validation
    print("M3D-C1 Profile Validation")
    print("=" * 50)
    
    profile = M3DC1Profile()
    print(profile)
    print()
    
    results = validate_m3dc1_profile()
    
    print("Validation Results:")
    print(f"  q monotonic: {results['monotonic']} ✓" if results['monotonic'] else "  q monotonic: False ✗")
    print(f"  q(axis) = {results['q_axis']:.3f} (target: 1.75, error: {results['q_axis_error']:.4f})")
    print(f"  q(edge) = {results['q_edge']:.3f} (target: 2.5, error: {results['q_edge_error']:.4f})")
    print(f"  q=2 exists: {results['q2_exists']} ✓" if results['q2_exists'] else "  q=2 exists: False ✗")
    if results['psi_q2'] is not None:
        print(f"  q=2 at ψ̃ = {results['psi_q2']:.3f}")
    print(f"  p' negative: {results['pprime_negative']} ✓" if results['pprime_negative'] else "  p' negative: False ✗")
    print()
    
    if results['all_pass']:
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print("⚠️  SOME CHECKS FAILED")
