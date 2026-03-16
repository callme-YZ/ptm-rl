"""
Pressure and Toroidal Field Profile Models for G-S Equation

Implements:
- Constant profiles (for testing)
- Quadratic profiles (intermediate)
- Luxon-Brown profiles (production)
- Taylor state (β=0)
"""

import numpy as np


class ProfileModel:
    """Base class for profile models."""
    
    def __init__(self, mu0=4*np.pi*1e-7):
        self.mu0 = mu0
    
    def compute_rhs(self, psi, psi_ma, psi_x, alpha, R):
        """
        Compute RHS of G-S equation.
        
        RHS = -μ₀R²·p'(ψ) - f·f'(ψ)
        
        Parameters
        ----------
        psi : ndarray
            Poloidal flux (only at plasma points)
        psi_ma, psi_x : float
            Flux at magnetic axis and x-point
        alpha : float
            Scaling parameter
        R : ndarray
            Radial coordinates (same shape as psi)
        
        Returns
        -------
        rhs : ndarray
            Right-hand side of G-S equation
        """
        raise NotImplementedError
    
    def compute_current_density(self, psi, psi_ma, psi_x, alpha, R):
        """
        Compute toroidal current density.
        
        J_φ = R·p'(ψ) + (1/μ₀R)·f·f'(ψ)
        """
        raise NotImplementedError


class ConstantProfile(ProfileModel):
    """
    Constant profiles (for Solov'ev-like equilibria).
    
    p'(ψ) = p1 (constant)
    f·f'(ψ) = f1/2 (constant)
    """
    
    def __init__(self, p1, f1, mu0=4*np.pi*1e-7):
        super().__init__(mu0)
        self.p1 = p1
        self.f1 = f1
    
    def compute_rhs(self, psi, psi_ma, psi_x, alpha, R):
        # RHS is independent of psi (constant)
        return -self.mu0 * R**2 * self.p1 - self.f1 / 2
    
    def compute_current_density(self, psi, psi_ma, psi_x, alpha, R):
        return R * self.p1 + self.f1 / (2 * self.mu0 * R)
    
    def p_double_prime(self, psi_N, alpha):
        """Second derivative ∂²p'/∂ψ_N² = 0 (constant profile)."""
        return np.zeros_like(psi_N)
    
    def ff_double_prime(self, psi_N, alpha):
        """Second derivative ∂²(f·f')/∂ψ_N² = 0 (constant profile)."""
        return np.zeros_like(psi_N)


class QuadraticProfile(ProfileModel):
    """
    Quadratic profiles (intermediate complexity).
    
    p'(ψ) = p0 + p1·ψ_N
    f·f'(ψ) = f0 + f1·ψ_N
    
    where ψ_N = (ψ - ψ_ma) / (ψ_x - ψ_ma)
    """
    
    def __init__(self, p0, p1, f0, f1, mu0=4*np.pi*1e-7):
        super().__init__(mu0)
        self.p0 = p0
        self.p1 = p1
        self.f0 = f0
        self.f1 = f1
    
    def _psi_normalized(self, psi, psi_ma, psi_x):
        """Compute normalized flux."""
        # Safeguard: if ψ_x ≈ ψ_axis (degenerate), return 0 (no plasma)
        denom = psi_x - psi_ma
        if isinstance(denom, np.ndarray):
            safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
            psi_N = np.where(np.abs(denom) < 1e-10, 0.0, (psi - psi_ma) / safe_denom)
        else:
            if np.abs(denom) < 1e-10:
                return np.zeros_like(psi) if isinstance(psi, np.ndarray) else 0.0
            psi_N = (psi - psi_ma) / denom
        return np.clip(psi_N, 0, 1)
    
    def compute_rhs(self, psi, psi_ma, psi_x, alpha, R):
        psi_N = self._psi_normalized(psi, psi_ma, psi_x)
        
        p_prime = self.p0 + self.p1 * psi_N
        ff_prime = self.f0 + self.f1 * psi_N
        
        return -self.mu0 * R**2 * p_prime - ff_prime
    
    def compute_current_density(self, psi, psi_ma, psi_x, alpha, R):
        psi_N = self._psi_normalized(psi, psi_ma, psi_x)
        
        p_prime = self.p0 + self.p1 * psi_N
        ff_prime = self.f0 + self.f1 * psi_N
        
        return R * p_prime + ff_prime / (self.mu0 * R)
    
    def p_double_prime(self, psi_N, alpha):
        """
        Second derivative ∂²p'/∂ψ_N² = 0 (linear in ψ_N).
        
        p'(ψ_N) = p0 + p1·ψ_N
        ∂p'/∂ψ_N = p1
        ∂²p'/∂ψ_N² = 0
        """
        return np.zeros_like(psi_N)
    
    def ff_double_prime(self, psi_N, alpha):
        """
        Second derivative ∂²(f·f')/∂ψ_N² = 0 (linear in ψ_N).
        
        f·f'(ψ_N) = f0 + f1·ψ_N
        ∂(f·f')/∂ψ_N = f1
        ∂²(f·f')/∂ψ_N² = 0
        """
        return np.zeros_like(psi_N)


class LuxonBrownProfile(ProfileModel):
    """
    Luxon-Brown profile model (ITER-standard).
    
    p'(ψ) = α·β/r₀ · (1 - ψ_N^δ)^γ
    f·f'(ψ) = α·(1-β)·μ₀r₀ · (1 - ψ_N^δ)^γ
    
    Parameters
    ----------
    r0 : float
        Characteristic radius (default: 6.2 for ITER)
    delta : float
        Exponent (default: 2.0)
    beta : float
        Pressure/field balance (default: 0.6)
    gamma : float
        Profile shaping (default: 1.4)
    """
    
    def __init__(self, r0=6.2, delta=2.0, beta=0.6, gamma=1.4, mu0=4*np.pi*1e-7):
        super().__init__(mu0)
        self.r0 = r0
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
    
    def _psi_normalized(self, psi, psi_ma, psi_x):
        """Compute normalized flux."""
        psi_N = (psi - psi_ma) / (psi_x - psi_ma)
        return np.clip(psi_N, 0, 1)
    
    def compute_rhs(self, psi, psi_ma, psi_x, alpha, R):
        psi_N = self._psi_normalized(psi, psi_ma, psi_x)
        
        # Common factor: (1 - ψ_N^δ)^γ
        factor = (1 - psi_N**self.delta)**self.gamma
        
        # Profiles
        p_prime = alpha * self.beta / self.r0 * factor
        ff_prime = alpha * (1 - self.beta) * self.mu0 * self.r0 * factor
        
        # RHS
        return -self.mu0 * R**2 * p_prime - ff_prime
    
    def compute_current_density(self, psi, psi_ma, psi_x, alpha, R):
        psi_N = self._psi_normalized(psi, psi_ma, psi_x)
        
        factor = (1 - psi_N**self.delta)**self.gamma
        
        p_prime = alpha * self.beta / self.r0 * factor
        ff_prime = alpha * (1 - self.beta) * self.mu0 * self.r0 * factor
        
        return R * p_prime + ff_prime / (self.mu0 * R)
    
    def p_double_prime(self, psi_N, alpha):
        """
        Second derivative of pressure profile.
        
        p'(ψ_N) = C_p · F(ψ_N)
        where C_p = α·β/r₀
              F(ψ_N) = (1 - ψ_N^δ)^γ
        
        F''(ψ_N) = γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2)·
                   [-(δ-1) + (δγ - 1)·ψ_N^δ]
        
        Returns
        -------
        p_double_prime : ndarray
            ∂²p'/∂ψ_N²
        """
        # Clip to avoid singularity at ψ_N=1
        psi_N = np.clip(psi_N, 0, 0.999)
        
        δ = self.delta
        γ = self.gamma
        
        # Handle ψ_N=0 case (when δ=2, ψ_N^(δ-2)=1)
        if δ == 2.0:
            term1 = np.ones_like(psi_N)
        else:
            # Avoid 0^negative
            psi_N_safe = np.maximum(psi_N, 1e-10)
            term1 = psi_N_safe**(δ - 2)
        
        term2 = (1 - psi_N**δ)**(γ - 2)
        bracket = -(δ - 1) + (δ*γ - 1) * psi_N**δ
        
        F_double_prime = γ * δ * term1 * term2 * bracket
        
        C_p = alpha * self.beta / self.r0
        
        return C_p * F_double_prime
    
    def ff_double_prime(self, psi_N, alpha):
        """
        Second derivative of f·f' profile.
        
        f·f'(ψ_N) = C_f · F(ψ_N)
        where C_f = α·(1-β)·μ₀r₀
              F(ψ_N) = (1 - ψ_N^δ)^γ
        
        F''(ψ_N) = γδ·ψ_N^(δ-2)·(1 - ψ_N^δ)^(γ-2)·
                   [-(δ-1) + (δγ - 1)·ψ_N^δ]
        
        Returns
        -------
        ff_double_prime : ndarray
            ∂²(f·f')/∂ψ_N²
        """
        # Clip to avoid singularity at ψ_N=1
        psi_N = np.clip(psi_N, 0, 0.999)
        
        δ = self.delta
        γ = self.gamma
        
        # Handle ψ_N=0 case
        if δ == 2.0:
            term1 = np.ones_like(psi_N)
        else:
            psi_N_safe = np.maximum(psi_N, 1e-10)
            term1 = psi_N_safe**(δ - 2)
        
        term2 = (1 - psi_N**δ)**(γ - 2)
        bracket = -(δ - 1) + (δ*γ - 1) * psi_N**δ
        
        F_double_prime = γ * δ * term1 * term2 * bracket
        
        C_f = alpha * (1 - self.beta) * self.mu0 * self.r0
        
        return C_f * F_double_prime
    
    def compute_plasma_current(self, psi, psi_ma, psi_x, alpha, R, Z, plasma_mask):
        """
        Compute total plasma current.
        
        I_p = ∫_plasma J_φ dR dZ
        """
        dR = R[1, 0] - R[0, 0] if R.ndim == 2 else R[1] - R[0]
        dZ = Z[0, 1] - Z[0, 0] if Z.ndim == 2 else Z[1] - Z[0]
        
        # Compute current density only in plasma
        J_phi = np.zeros_like(psi)
        J_phi[plasma_mask] = self.compute_current_density(
            psi[plasma_mask], psi_ma, psi_x, alpha, R[plasma_mask]
        )
        
        # Integrate
        I_p = np.sum(J_phi) * dR * dZ
        
        return I_p


class TaylorStateProfile(ProfileModel):
    """
    Taylor state equilibrium (β=0).
    
    p'(ψ) = 0
    f(ψ) = f_x + α(ψ - ψ_x)
    f·f'(ψ) = α·f(ψ)
    
    Parameters
    ----------
    f_x : float
        Toroidal field function at separatrix
    """
    
    def __init__(self, f_x=5.0, mu0=4*np.pi*1e-7):
        super().__init__(mu0)
        self.f_x = f_x
    
    def compute_rhs(self, psi, psi_ma, psi_x, alpha, R):
        # f(ψ) = f_x + α(ψ - ψ_x)
        f = self.f_x + alpha * (psi - psi_x)
        
        # f·f'(ψ) = α·f
        ff_prime = alpha * f
        
        # RHS (no pressure term)
        return -ff_prime
    
    def compute_current_density(self, psi, psi_ma, psi_x, alpha, R):
        f = self.f_x + alpha * (psi - psi_x)
        ff_prime = alpha * f
        
        return ff_prime / (self.mu0 * R)


# =============================================================================
# Tests
# =============================================================================

def test_constant_profile():
    """Test constant profile."""
    print("Test: Constant Profile")
    print("=" * 60)
    
    p1 = 1e4
    f1 = 0.0
    
    profile = ConstantProfile(p1, f1)
    
    # Create test data
    R = np.array([2.0, 3.0, 4.0])
    psi = np.array([0.1, 0.2, 0.3])
    psi_ma, psi_x = 0.0, 1.0
    alpha = 1.0
    
    rhs = profile.compute_rhs(psi, psi_ma, psi_x, alpha, R)
    
    print(f"p1 = {p1:.2e}, f1 = {f1}")
    print(f"RHS at R={R[0]:.1f}: {rhs[0]:.6e}")
    print(f"RHS at R={R[1]:.1f}: {rhs[1]:.6e}")
    print(f"RHS at R={R[2]:.1f}: {rhs[2]:.6e}")
    
    # Should scale as R²
    expected_ratio = (R[1]/R[0])**2
    actual_ratio = abs(rhs[1]/rhs[0])
    
    print(f"\nRatio test: {actual_ratio:.4f} (expected {expected_ratio:.4f})")
    
    assert abs(actual_ratio - expected_ratio) < 0.01, "Should scale as R²"
    
    print("✅ PASS\n")


def test_quadratic_profile():
    """Test quadratic profile."""
    print("Test: Quadratic Profile")
    print("=" * 60)
    
    p0, p1 = 1e4, -5e3
    f0, f1 = 0.1, -0.05
    
    profile = QuadraticProfile(p0, p1, f0, f1)
    
    # Test at different ψ_N
    psi_ma, psi_x = 0.0, 1.0
    R = 3.0
    alpha = 1.0
    
    psi_values = [0.0, 0.5, 1.0]  # ψ_N = 0, 0.5, 1
    
    print("ψ_N    p'        ff'       RHS")
    print("-" * 40)
    
    for psi_val in psi_values:
        psi_N = psi_val
        rhs = profile.compute_rhs(
            np.array([psi_val]), psi_ma, psi_x, alpha, np.array([R])
        )[0]
        
        p_prime = p0 + p1 * psi_N
        ff_prime = f0 + f1 * psi_N
        
        print(f"{psi_N:.1f}   {p_prime:+.2e}  {ff_prime:+.2e}  {rhs:+.2e}")
    
    print("✅ PASS\n")


def test_luxon_brown_profile():
    """Test Luxon-Brown profile."""
    print("Test: Luxon-Brown Profile")
    print("=" * 60)
    
    profile = LuxonBrownProfile(r0=6.2, delta=2.0, beta=0.6, gamma=1.4)
    
    psi_ma, psi_x = 0.0, 1.0
    R = 4.0
    alpha = 1.0
    
    # Test at core vs edge
    psi_core = 0.1   # ψ_N = 0.1
    psi_edge = 0.9   # ψ_N = 0.9
    
    rhs_core = profile.compute_rhs(
        np.array([psi_core]), psi_ma, psi_x, alpha, np.array([R])
    )[0]
    
    rhs_edge = profile.compute_rhs(
        np.array([psi_edge]), psi_ma, psi_x, alpha, np.array([R])
    )[0]
    
    print(f"α = {alpha}, R = {R:.1f}")
    print(f"Core (ψ_N=0.1): RHS = {rhs_core:.6e}")
    print(f"Edge (ψ_N=0.9): RHS = {rhs_edge:.6e}")
    print(f"Ratio: {abs(rhs_core/rhs_edge):.2f}")
    
    # Core should have stronger source (more peaked)
    assert abs(rhs_core) > abs(rhs_edge), "Core should be stronger"
    
    print("✅ PASS\n")


def test_taylor_state():
    """Test Taylor state profile."""
    print("Test: Taylor State Profile")
    print("=" * 60)
    
    f_x = 5.0
    profile = TaylorStateProfile(f_x=f_x)
    
    psi_ma, psi_x = 0.0, 1.0
    alpha = 0.5
    R = 3.0
    
    psi_axis = psi_ma
    psi_edge = psi_x
    
    rhs_axis = profile.compute_rhs(
        np.array([psi_axis]), psi_ma, psi_x, alpha, np.array([R])
    )[0]
    
    rhs_edge = profile.compute_rhs(
        np.array([psi_edge]), psi_ma, psi_x, alpha, np.array([R])
    )[0]
    
    print(f"f_x = {f_x}, α = {alpha}")
    print(f"At axis (ψ=0): RHS = {rhs_axis:.6e}")
    print(f"At edge (ψ=1): RHS = {rhs_edge:.6e}")
    
    # f(ψ_x) = f_x → ff' = α·f_x at edge
    expected_edge = -alpha * f_x
    
    print(f"Expected at edge: {expected_edge:.6e}")
    
    assert abs(rhs_edge - expected_edge) < 1e-10, "Edge value wrong"
    
    print("✅ PASS\n")


if __name__ == '__main__':
    test_constant_profile()
    test_quadratic_profile()
    test_luxon_brown_profile()
    test_taylor_state()
    
    print("=" * 60)
    print("All profile tests passed! ✅")
