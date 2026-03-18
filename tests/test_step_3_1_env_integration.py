"""
M3 Step 3.1.3: Environment Integration Tests

Test integration of MHDObservation into gym environment.

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest


class TestEnvironmentIntegration:
    """Test 1: MHDObservation integrates with environment."""
    
    def test_create_observation_wrapper(self):
        """Create observation wrapper from environment state."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        # Create grid
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        # Create equilibrium
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        # Create observation wrapper
        obs_wrapper = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        
        # Compute equilibrium energy for proper initialization
        E_eq = obs_wrapper._compute_energy(psi_eq, omega_eq)
        
        # Re-create with correct energy
        obs_wrapper = MHDObservation(psi_eq, E_eq=E_eq, grid=grid)
        
        # Get observation
        obs = obs_wrapper.get_observation(psi_eq, omega_eq)
        
        print(f"\nObservation wrapper created:")
        print(f"  Observation space: {obs_wrapper.observation_space_shape}")
        print(f"  Energy drift: {obs['energy_drift']:.6f}")
        
        assert obs['energy_drift'] < 0.01, "Equilibrium should have small drift"
    
    def test_observation_from_symplectic_solver(self):
        """Extract observation from symplectic integrator state."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators.symplectic import SymplecticIntegrator
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        # Create grid and integrator
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        integrator = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-5)
        
        # Initialize with equilibrium
        r_grid = grid.r_grid
        psi0 = r_grid**2 * (1 - r_grid / grid.a)
        omega0 = laplacian_toroidal(psi0, grid)
        
        integrator.initialize(psi0, omega0)
        
        # Create observation wrapper
        E_eq = integrator.compute_energy()
        obs_wrapper = MHDObservation(psi0, E_eq=E_eq, grid=grid)
        
        # Take a few steps
        for _ in range(10):
            integrator.step()
        
        # Extract observation
        obs = obs_wrapper.get_observation(integrator.psi, integrator.omega)
        
        print(f"\nAfter 10 steps:")
        print(f"  Energy drift: {obs['energy_drift']:.6f}")
        print(f"  div_B_max: {obs['div_B_max']:.6f}")
        print(f"  psi_modes[0:4]: {obs['psi_modes'][0:4]}")
        
        # Should still be stable (15% is expected toroidal transient from Phase 2)
        assert obs['energy_drift'] < 0.2, "Energy drift should be reasonable"
        assert np.all(np.isfinite(obs['vector'])), "All values should be finite"
    
    def test_observation_sequence(self):
        """Observation should track evolution correctly."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators.symplectic import SymplecticIntegrator
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        # Setup
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        integrator = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-5)
        
        r_grid = grid.r_grid
        theta_grid = grid.theta_grid
        
        # Start with small perturbation
        psi0 = r_grid**2 * (1 - r_grid / grid.a) * (1 + 0.001 * np.sin(theta_grid))
        omega0 = laplacian_toroidal(psi0, grid)
        
        integrator.initialize(psi0, omega0)
        
        # Create observation wrapper
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        obs_wrapper_temp = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        E_eq = obs_wrapper_temp._compute_energy(psi_eq, omega_eq)
        
        obs_wrapper = MHDObservation(psi_eq, E_eq=E_eq, grid=grid)
        
        # Track observations
        obs_history = []
        
        for step in range(100):
            obs = obs_wrapper.get_observation(integrator.psi, integrator.omega)
            obs_history.append(obs['energy_drift'])
            integrator.step()
        
        # Plot would go here
        print(f"\nObservation sequence (100 steps):")
        print(f"  Initial drift: {obs_history[0]:.6f}")
        print(f"  Final drift: {obs_history[-1]:.6f}")
        print(f"  Max drift: {np.max(obs_history):.6f}")
        
        # All should be reasonable
        assert np.all(np.array(obs_history) < 1.0), "All drifts should be < 100%"


class TestObservationSpace:
    """Test 2: Gym observation space compatibility."""
    
    def test_observation_space_box(self):
        """Observation should fit into gym.spaces.Box."""
        import gymnasium as gym
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        obs_wrapper = MHDObservation(psi_eq, E_eq=1.0, grid=grid)
        
        # Create gym space
        obs_shape = obs_wrapper.observation_space_shape
        observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        print(f"\nGym observation space: {observation_space}")
        
        # Get observation
        obs = obs_wrapper.get_observation(psi_eq, omega_eq)
        obs_vector = obs['vector'].astype(np.float32)
        
        # Check containment
        assert observation_space.contains(obs_vector), \
            "Observation should fit in Box space"
        
        print(f"Observation vector shape: {obs_vector.shape}")
        print(f"Observation space shape: {observation_space.shape}")
        
        assert obs_vector.shape == observation_space.shape
    
    def test_normalize_clipping(self):
        """Normalization should clip extreme values."""
        from pytokmhd.rl.observations import normalize_observation
        
        # Create mock observation with extreme values
        obs = {
            'psi_modes': np.random.randn(16),
            'energy': 100.0,  # Very large
            'energy_drift': 50.0,  # Large
            'div_B_max': 1000.0,  # Huge
            'vector': np.zeros(19),
        }
        
        # Normalize
        obs_norm = normalize_observation(obs, clip=10.0)
        
        print(f"\nBefore normalization:")
        print(f"  energy: {obs['energy']:.1f}")
        print(f"  energy_drift: {obs['energy_drift']:.1f}")
        print(f"  div_B_max: {obs['div_B_max']:.1f}")
        
        print(f"\nAfter normalization (clip=10):")
        print(f"  energy: {obs_norm['energy']:.1f}")
        print(f"  energy_drift: {obs_norm['energy_drift']:.1f}")
        print(f"  div_B_max: {obs_norm['div_B_max']:.1f}")
        
        # Check clipping
        assert obs_norm['energy'] == 10.0, "energy should be clipped to 10"
        assert obs_norm['energy_drift'] == 10.0, "drift should be clipped"
        assert obs_norm['div_B_max'] == 10.0, "div_B should be clipped"


class TestEnvironmentReset:
    """Test 3: Environment reset with new observation."""
    
    def test_reset_creates_observation(self):
        """Reset should create consistent observation."""
        from pytokmhd.geometry import ToroidalGrid
        from pytokmhd.integrators.symplectic import SymplecticIntegrator
        from pytokmhd.rl.observations import MHDObservation
        from pytokmhd.operators import laplacian_toroidal
        
        # Simulate environment reset
        grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
        integrator = SymplecticIntegrator(grid, dt=1e-4, eta=1e-6, nu=1e-5)
        
        # Initialize
        r_grid = grid.r_grid
        psi_eq = r_grid**2 * (1 - r_grid / grid.a)
        omega_eq = laplacian_toroidal(psi_eq, grid)
        
        integrator.initialize(psi_eq, omega_eq)
        
        # Create observation wrapper
        E_eq = integrator.compute_energy()
        obs_wrapper = MHDObservation(psi_eq, E_eq=E_eq, grid=grid)
        
        # Get initial observation
        obs_init = obs_wrapper.get_observation(integrator.psi, integrator.omega)
        
        print(f"\nInitial observation:")
        print(f"  energy_drift: {obs_init['energy_drift']:.6f}")
        print(f"  Shape: {obs_init['vector'].shape}")
        
        # Reset by re-initializing
        integrator.initialize(psi_eq, omega_eq)
        obs_reset = obs_wrapper.get_observation(integrator.psi, integrator.omega)
        
        # Should be identical
        error = np.max(np.abs(obs_init['vector'] - obs_reset['vector']))
        
        print(f"Reset error: {error:.3e}")
        
        assert error < 1e-10, "Reset should give identical observation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
