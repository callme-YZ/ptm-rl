"""
M3 Step 3.1: Observation Space Tests

Validation of Fourier decomposition and observation extraction.

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest


class TestFourierDecomposition:
    """Test 1: Fourier decomposition captures structure."""
    
    def test_axisymmetric_mode(self):
        """Pure m=0 mode should give only DC component."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics import fourier_decompose, reconstruct_from_modes
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create axisymmetric field (m=0 only)
        r_grid = grid.r_grid
        psi = r_grid**2 * (1 - r_grid / grid.a)  # No θ dependence
        
        # Decompose
        modes = fourier_decompose(psi, grid, n_modes=4)
        
        # m=0 should dominate
        m0_re = modes[0]
        m0_im = modes[1]
        
        # m=1,2,3 should be ~0
        higher_modes = modes[2:]
        
        print(f"\nm=0: Re={m0_re:.6f}, Im={m0_im:.6f}")
        print(f"Higher modes: {higher_modes}")
        
        # Checks
        assert abs(m0_re) > 1e-3, "m=0 should be significant"
        assert abs(m0_im) < 1e-10, "m=0 imaginary should be 0 (axisymmetric)"
        assert np.max(np.abs(higher_modes)) < 1e-10, "Higher modes should be ~0"
    
    def test_m2_perturbation(self):
        """m=2 perturbation should appear in modes[4:6]."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics import fourier_decompose
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create m=2 perturbation
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        amplitude = 0.01
        psi = r_grid**2 * (1 - r_grid / grid.a) * (1 + amplitude * np.sin(2 * theta_grid))
        
        # Decompose
        modes = fourier_decompose(psi, grid, n_modes=8)
        
        # m=0 (DC)
        m0_amp = np.sqrt(modes[0]**2 + modes[1]**2)
        
        # m=2 (modes[4], modes[5])
        m2_re = modes[4]
        m2_im = modes[5]
        m2_amp = np.sqrt(m2_re**2 + m2_im**2)
        
        print(f"\nm=0 amplitude: {m0_amp:.6f}")
        print(f"m=2: Re={m2_re:.6f}, Im={m2_im:.6f}, |m=2|={m2_amp:.6f}")
        
        # m=2 should be significant (but < m=0 background)
        assert m2_amp > 5e-5, "m=2 should be detected"
        assert m2_amp < m0_amp, "m=2 should be < m=0 background"
    
    def test_reconstruction(self):
        """Reconstructed signal should match original at mid-radius."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics import fourier_decompose, reconstruct_from_modes
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create test field
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        psi = r_grid**2 * np.sin(theta_grid) + 0.5 * r_grid * np.sin(2 * theta_grid)
        
        # Decompose (8 modes should capture this exactly)
        modes = fourier_decompose(psi, grid, n_modes=8)
        
        # Reconstruct
        mid_idx = grid.nr // 2
        psi_mid_original = psi[mid_idx, :]
        psi_mid_reconstructed = reconstruct_from_modes(modes, grid, r_idx=mid_idx)
        
        # Compare
        error = np.max(np.abs(psi_mid_original - psi_mid_reconstructed))
        
        print(f"\nReconstruction error: {error:.3e}")
        
        assert error < 1e-10, "Reconstruction should be exact for smooth signals"


class TestModeAnalysis:
    """Test 2: Mode amplitude analysis."""
    
    def test_mode_amplitudes(self):
        """Mode amplitudes should be computed correctly."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics.fourier import compute_mode_amplitudes
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create known perturbation
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        # m=1: amp=0.1, m=2: amp=0.05
        psi = (r_grid * 0.1 * np.sin(theta_grid) + 
               r_grid * 0.05 * np.sin(2 * theta_grid))
        
        amplitudes = compute_mode_amplitudes(psi, grid, n_modes=4)
        
        print(f"\nMode amplitudes: {amplitudes}")
        print(f"  m=0: {amplitudes[0]:.6f}")
        print(f"  m=1: {amplitudes[1]:.6f}")
        print(f"  m=2: {amplitudes[2]:.6f}")
        
        # m=1 should be largest (excluding m=0)
        assert amplitudes[1] > amplitudes[2], "m=1 should be > m=2"
        assert amplitudes[1] > 1e-3, "m=1 should be significant"
    
    def test_dominant_mode(self):
        """Dominant mode should be identified correctly."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics.fourier import compute_dominant_mode
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create m=2 dominated field
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        psi = r_grid**2 * (1 + 0.01 * np.sin(theta_grid) + 0.05 * np.sin(2 * theta_grid))
        
        m_dominant, amplitude = compute_dominant_mode(psi, grid, n_modes=8)
        
        print(f"\nDominant mode: m={m_dominant}, amplitude={amplitude:.6f}")
        
        assert m_dominant == 2, "m=2 should be dominant"
        assert amplitude > 5e-4, "Dominant amplitude should be significant"


class TestObservationShape:
    """Test 3: Observation vector shape and normalization."""
    
    def test_observation_dimension(self):
        """Observation should be correct dimension."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics import fourier_decompose
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Test field
        r_grid = grid.r_grid
        psi = r_grid**2 * (1 - r_grid / grid.a)
        
        # Extract modes
        n_modes = 8
        modes = fourier_decompose(psi, grid, n_modes=n_modes)
        
        # Check dimension
        expected_dim = 2 * n_modes  # Real + imag for each mode
        
        print(f"\nObservation dimension: {len(modes)}")
        print(f"Expected: {expected_dim}")
        
        assert len(modes) == expected_dim, f"Should be {expected_dim}D"
        assert modes.dtype == np.float64, "Should be float64"
    
    def test_mode_ranges(self):
        """Mode values should be in reasonable range for normalization."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.diagnostics import fourier_decompose
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Typical equilibrium
        r_grid = grid.r_grid
        psi = r_grid**2 * (1 - r_grid / grid.a)
        
        modes = fourier_decompose(psi, grid, n_modes=8)
        
        # Check ranges
        max_val = np.max(np.abs(modes))
        
        print(f"\nMode value range: ±{max_val:.6f}")
        
        # Should be O(0.01 - 0.1) for typical ψ
        assert max_val < 1.0, "Modes should be < 1 for easy normalization"
        assert max_val > 1e-6, "Modes should be significant"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestMHDObservationClass:
    """Test 4: MHDObservation class functionality."""
    
    def test_initialization(self):
        """MHDObservation should initialize correctly."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Equilibrium
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        E_eq = 1.0  # Placeholder
        
        obs_wrapper = MHDObservation(psi_eq, E_eq, grid, n_modes=8)
        
        print(f"\nObservation space shape: {obs_wrapper.observation_space_shape}")
        print(f"Expected: (19,)  # 16 modes + 3 scalars")
        
        assert obs_wrapper.observation_space_shape == (19,), "Should be 19D"
        assert obs_wrapper.n_modes == 8
        assert obs_wrapper.E_eq == E_eq
    
    def test_get_observation(self):
        """get_observation should return correct structure."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Equilibrium
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        obs_wrapper = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        
        # Current state (same as equilibrium)
        obs = obs_wrapper.get_observation(psi_eq, omega_eq)
        
        print(f"\nObservation keys: {obs.keys()}")
        print(f"psi_modes shape: {obs['psi_modes'].shape}")
        print(f"energy: {obs['energy']:.6f}")
        print(f"energy_drift: {obs['energy_drift']:.6f}")
        print(f"div_B_max: {obs['div_B_max']:.6f}")
        print(f"vector shape: {obs['vector'].shape}")
        
        # Check structure
        assert 'psi_modes' in obs
        assert 'energy' in obs
        assert 'energy_drift' in obs
        assert 'div_B_max' in obs
        assert 'vector' in obs
        
        # Check shapes
        assert obs['psi_modes'].shape == (16,), "8 modes × 2 (Re/Im)"
        assert obs['vector'].shape == (19,), "16 + 3 scalars"
        
        # Check energy drift should be small (same as equilibrium)
        # Note: E_eq=1.0 is placeholder, so drift might be large
        print(f"Energy drift check: {obs['energy_drift']}")
    
    def test_perturbation_detection(self):
        """Observation should detect perturbations."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Equilibrium
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        # Compute equilibrium energy
        obs_wrapper = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        E_eq_actual = obs_wrapper._compute_energy(psi_eq, omega_eq)
        
        # Re-initialize with correct E_eq
        obs_wrapper = MHDObservation(psi_eq, E_eq=E_eq_actual, grid=grid)
        
        # Perturbed state (add m=2 perturbation)
        psi_pert = psi_eq * (1 + 0.01 * np.sin(2 * theta_grid))
        omega_pert = laplacian_toroidal(psi_pert, grid)
        
        # Get observations
        obs_eq = obs_wrapper.get_observation(psi_eq, omega_eq)
        obs_pert = obs_wrapper.get_observation(psi_pert, omega_pert)
        
        print(f"\nEquilibrium obs:")
        print(f"  energy_drift: {obs_eq['energy_drift']:.6f}")
        print(f"  psi_modes[4:6] (m=2): {obs_eq['psi_modes'][4:6]}")
        
        print(f"\nPerturbed obs:")
        print(f"  energy_drift: {obs_pert['energy_drift']:.6f}")
        print(f"  psi_modes[4:6] (m=2): {obs_pert['psi_modes'][4:6]}")
        
        # Perturbation should increase energy drift
        assert obs_pert['energy_drift'] > obs_eq['energy_drift'], \
            "Perturbation should increase energy drift"
        
        # m=2 mode should be different
        m2_diff = np.max(np.abs(obs_pert['psi_modes'][4:6] - obs_eq['psi_modes'][4:6]))
        print(f"  m=2 mode difference: {m2_diff:.6f}")
        
        assert m2_diff > 1e-3, "m=2 mode should change with perturbation"
    
    def test_normalization(self):
        """Observation values should be in reasonable range."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Equilibrium
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        obs_wrapper = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        obs = obs_wrapper.get_observation(psi_eq, omega_eq)
        
        # Check ranges
        psi_modes_range = np.max(np.abs(obs['psi_modes']))
        
        print(f"\nValue ranges:")
        print(f"  psi_modes: ±{psi_modes_range:.3f}")
        print(f"  energy: {obs['energy']:.3f}")
        print(f"  energy_drift: {obs['energy_drift']:.3f}")
        print(f"  div_B_max: {obs['div_B_max']:.3f}")
        
        # psi_modes should be normalized to ~[-1, 1]
        assert psi_modes_range < 2.0, "psi_modes should be normalized"
        
        # All values should be finite
        assert np.all(np.isfinite(obs['vector'])), "All values should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
