"""
Unit Tests for Toroidal Hamiltonian (v2.0 Physics)

Issue #17: Add unit tests for v2.0 physics modules

Tests:
1. Hamiltonian computation (cylindrical vs toroidal)
2. Curvature energy contribution
3. Energy positivity
4. Scaling with epsilon

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import pytest
import jax.numpy as jnp

from pim_rl.physics.v2.elsasser_bracket import ElsasserState, MorrisonBracket
from pim_rl.physics.v2.toroidal_hamiltonian import (
    toroidal_hamiltonian,
    compute_curvature_vector
)


class TestToroidalHamiltonian:
    """Test toroidal Hamiltonian energy computation"""
    
    @pytest.fixture
    def grid(self):
        """Standard test grid"""
        return MorrisonBracket((16, 16, 8), dr=0.1, dtheta=0.1, dz=0.1)
    
    @pytest.fixture
    def test_state(self, grid):
        """Generate test Elsasser state"""
        Nr, Ntheta, Nz = grid.Nr, grid.Ntheta, grid.Nz
        
        # Simple smooth fields
        r = jnp.linspace(0, 1, Nr)[:, None, None]
        theta = jnp.linspace(0, 2*jnp.pi, Ntheta)[None, :, None]
        z = jnp.linspace(0, 1, Nz)[None, None, :]
        
        z_plus = jnp.sin(jnp.pi * r) * jnp.cos(theta) * jnp.sin(jnp.pi * z)
        z_minus = jnp.cos(jnp.pi * r) * jnp.sin(2*theta) * jnp.cos(jnp.pi * z) * 0.5
        P = jnp.ones((Nr, Ntheta, Nz)) * 0.1
        
        return ElsasserState(z_plus=z_plus, z_minus=z_minus, P=P)
    
    def test_cylindrical_limit(self, grid, test_state):
        """
        Test ε=0 recovers cylindrical Hamiltonian.
        
        H(ε=0) should equal cylindrical energy only.
        """
        H_cyl = toroidal_hamiltonian(test_state, grid, epsilon=0.0)
        
        # Cylindrical energy: ∫ (z⁺² + z⁻²)/4 dV
        energy_density = (test_state.z_plus**2 + test_state.z_minus**2) / 4
        expected = jnp.sum(energy_density) * grid.dV
        
        rel_error = abs(H_cyl - expected) / expected
        
        assert rel_error < 1e-10, f"Cylindrical limit: rel error = {rel_error:.3e}"
        print(f"✅ Cylindrical limit (ε=0): H = {H_cyl:.6e}")
    
    def test_curvature_energy_contribution(self, grid, test_state):
        """
        Test that curvature energy changes H.
        
        H(ε>0) ≠ H(ε=0) due to -∫ h·P dV term.
        """
        H_cyl = toroidal_hamiltonian(test_state, grid, epsilon=0.0)
        H_tor = toroidal_hamiltonian(test_state, grid, epsilon=0.3)
        
        # Should be different
        delta_H = H_tor - H_cyl
        
        assert abs(delta_H) > 1e-6, f"Curvature energy too small: ΔH = {delta_H:.3e}"
        
        # With positive pressure, curvature energy is typically negative
        # (check sign makes sense)
        print(f"✅ Curvature contribution: ΔH = {delta_H:.6e}")
        print(f"   Relative change: {delta_H/H_cyl:.2%}")
    
    def test_energy_positivity_cylindrical(self, grid, test_state):
        """
        Test H > 0 for cylindrical case.
        
        Energy should be positive (kinetic + magnetic).
        """
        H = toroidal_hamiltonian(test_state, grid, epsilon=0.0)
        
        assert H > 0, f"Energy negative: H = {H:.6e}"
        print(f"✅ Energy positivity (ε=0): H = {H:.6e} > 0")
    
    def test_epsilon_scaling(self, grid, test_state):
        """
        Test curvature energy scales linearly with ε.
        
        ΔH ∝ ε for small ε.
        """
        eps_values = [0.0, 0.1, 0.2, 0.3]
        H_values = [toroidal_hamiltonian(test_state, grid, epsilon=eps) for eps in eps_values]
        
        # Curvature energies
        H0 = H_values[0]
        delta_H = [H - H0 for H in H_values]
        
        # Check linear scaling: ΔH(2ε) ≈ 2·ΔH(ε)
        ratio_1 = delta_H[2] / delta_H[1] if abs(delta_H[1]) > 1e-10 else 0
        ratio_2 = delta_H[3] / delta_H[1] if abs(delta_H[1]) > 1e-10 else 0
        
        # Should be close to 2 and 3 respectively
        assert abs(ratio_1 - 2.0) < 0.1, f"Scaling violation: ΔH(0.2)/ΔH(0.1) = {ratio_1:.2f} (expect 2)"
        assert abs(ratio_2 - 3.0) < 0.1, f"Scaling violation: ΔH(0.3)/ΔH(0.1) = {ratio_2:.2f} (expect 3)"
        
        print(f"✅ Epsilon scaling:")
        for eps, dH in zip(eps_values, delta_H):
            print(f"   ε={eps:.1f}: ΔH = {dH:.6e}")
    
    def test_curvature_vector(self, grid):
        """
        Test curvature vector h = εx computation.
        
        Should have correct shape and magnitude.
        """
        epsilon = 0.3
        h = compute_curvature_vector((grid.Nr, grid.Ntheta, grid.Nz), 
                                     grid.dr, grid.dtheta, epsilon)
        
        # Check shape
        assert h.shape == (grid.Nr, grid.Ntheta, grid.Nz, 3), f"Shape: {h.shape}"
        
        # Check magnitude ~ ε
        h_mag = jnp.sqrt(h[:,:,:,0]**2 + h[:,:,:,1]**2 + h[:,:,:,2]**2)
        expected_mag = epsilon  # |εx| = ε for unit x
        
        max_diff = jnp.max(jnp.abs(h_mag - expected_mag))
        
        assert max_diff < 1e-7, f"Magnitude error: {max_diff:.3e}"
        print(f"✅ Curvature vector: |h| = {epsilon} ± {max_diff:.3e}")
    
    def test_zero_state_energy(self, grid):
        """
        Test H = 0 for zero state.
        
        No fields → no energy.
        """
        Nr, Ntheta, Nz = grid.Nr, grid.Ntheta, grid.Nz
        
        zero_state = ElsasserState(
            z_plus=jnp.zeros((Nr, Ntheta, Nz)),
            z_minus=jnp.zeros((Nr, Ntheta, Nz)),
            P=jnp.zeros((Nr, Ntheta, Nz))
        )
        
        H = toroidal_hamiltonian(zero_state, grid, epsilon=0.3)
        
        assert abs(H) < 1e-12, f"Zero state energy: H = {H:.3e}"
        print(f"✅ Zero state: H = {H:.3e}")


def test_hamiltonian_callable():
    """Test that toroidal_hamiltonian is callable"""
    grid = MorrisonBracket((8, 8, 4), 0.1, 0.1, 0.1)
    
    state = ElsasserState(
        z_plus=jnp.ones((8, 8, 4)),
        z_minus=jnp.ones((8, 8, 4)) * 0.5,
        P=jnp.ones((8, 8, 4)) * 0.1
    )
    
    H = toroidal_hamiltonian(state, grid, epsilon=0.2)
    
    assert isinstance(float(H), float), f"H not scalar: {type(H)}"
    print(f"✅ Hamiltonian callable: H = {H:.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
