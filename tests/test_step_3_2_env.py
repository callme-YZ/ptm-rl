"""
M3 Step 3.2.3: Environment Integration Tests

Validation of MHDEnv with action/observation integration.

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest


class TestEnvironmentBasics:
    """Test 1: Basic environment functionality."""
    
    def test_environment_creation(self):
        """Environment should initialize correctly."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64, dt=1e-4, max_steps=100)
        
        print(f"\nEnvironment created:")
        print(f"  observation_space: {env.observation_space}")
        print(f"  action_space: {env.action_space}")
        print(f"  max_steps: {env.max_steps}")
        
        assert env.observation_space.shape == (19,)
        assert env.action_space.shape == (2,)
        assert env.max_steps == 100
    
    def test_reset(self):
        """Reset should return valid observation."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64)
        
        obs, info = env.reset(seed=42)
        
        print(f"\nReset observation:")
        print(f"  shape: {obs.shape}")
        print(f"  dtype: {obs.dtype}")
        print(f"  E_eq: {info['E_eq']:.6e}")
        print(f"  perturbation_amplitude: {info['perturbation_amplitude']}")
        print(f"  perturbation_mode: {info['perturbation_mode']}")
        
        assert obs.shape == (19,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert 'E_eq' in info


class TestEnvironmentStep:
    """Test 2: Step functionality."""
    
    def test_step_with_action(self):
        """Step should accept action and return valid outputs."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=10)
        obs, info = env.reset(seed=42)
        
        # Take step with identity action
        action = np.array([1.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep result:")
        print(f"  obs shape: {obs.shape}")
        print(f"  reward: {reward:.6f}")
        print(f"  terminated: {terminated}")
        print(f"  truncated: {truncated}")
        print(f"  time: {info['time']:.6e}")
        
        assert obs.shape == (19,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_multiple_steps(self):
        """Multiple steps should work correctly."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=5)
        obs, info = env.reset(seed=42)
        
        print(f"\nMultiple steps test:")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.4f}, time={info['time']:.4e}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {i+1}")
                break
        
        # Should reach max_steps or terminate
        assert env.current_step <= env.max_steps


class TestActionIntegration:
    """Test 3: Action handling."""
    
    def test_action_space_sampling(self):
        """Action space should allow sampling."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv()
        
        print(f"\nAction space sampling:")
        for i in range(5):
            action = env.action_space.sample()
            print(f"  Sample {i+1}: {action}")
            assert env.action_space.contains(action)
    
    def test_action_modulation_effect(self):
        """Different actions should produce different results."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env1 = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=10)
        env2 = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=10)
        
        # Same initial state
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Different actions
        action_low = np.array([0.5, 0.5])   # Low dissipation
        action_high = np.array([2.0, 2.0])  # High dissipation
        
        # Run for a few steps
        for _ in range(5):
            obs1, r1, _, _, _ = env1.step(action_low)
            obs2, r2, _, _, _ = env2.step(action_high)
        
        # Results should be different
        obs_diff = np.linalg.norm(obs1 - obs2)
        
        print(f"\nAction modulation effect:")
        print(f"  Low dissipation action: {action_low}")
        print(f"  High dissipation action: {action_high}")
        print(f"  |obs_low - obs_high|: {obs_diff:.3e}")
        
        assert obs_diff > 1e-6  # Significant difference


class TestObservationIntegration:
    """Test 4: Observation handling."""
    
    def test_observation_structure(self):
        """Observation should have correct structure."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv()
        obs, _ = env.reset(seed=42)
        
        print(f"\nObservation structure:")
        print(f"  Total dimension: {obs.shape[0]}")
        print(f"  psi_modes (16D): {obs[:16]}")
        print(f"  energy (1D): {obs[16]}")
        print(f"  energy_drift (1D): {obs[17]}")
        print(f"  div_B_max (1D): {obs[18]}")
        
        assert obs.shape == (19,)
        assert np.all(np.isfinite(obs))
    
    def test_observation_evolution(self):
        """Observation should evolve with steps."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64)
        obs0, _ = env.reset(seed=42)
        
        action = np.array([1.5, 0.8])  # Modulated action
        
        obs_history = [obs0.copy()]
        for _ in range(5):
            obs, _, _, _, _ = env.step(action)
            obs_history.append(obs.copy())
        
        print(f"\nObservation evolution:")
        for i, obs in enumerate(obs_history):
            energy_drift = obs[17]
            print(f"  Step {i}: energy_drift={energy_drift:.6f}")
        
        # Observations should change
        obs_diff = np.linalg.norm(obs_history[-1] - obs_history[0])
        assert obs_diff > 1e-6


class TestRewardComputation:
    """Test 5: Reward function."""
    
    def test_reward_components(self):
        """Reward should have all components."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv()
        obs, _ = env.reset(seed=42)
        
        action = np.array([1.2, 0.9])
        obs, reward, _, _, info = env.step(action)
        
        print(f"\nReward components:")
        print(f"  reward_energy: {info['reward_energy']:.6f}")
        print(f"  reward_action: {info['reward_action']:.6f}")
        print(f"  reward_constraint: {info['reward_constraint']:.6f}")
        print(f"  reward_total: {info['reward_total']:.6f}")
        
        assert 'reward_energy' in info
        assert 'reward_action' in info
        assert 'reward_constraint' in info
        assert np.isclose(
            info['reward_total'],
            info['reward_energy'] + info['reward_action'] + info['reward_constraint']
        )


class TestTerminationConditions:
    """Test 6: Termination and truncation."""
    
    def test_max_steps_truncation(self):
        """Episode should truncate at max_steps."""
        from pytokmhd.rl.mhd_env_v1_2 import ToroidalMHDEnv
        
        env = ToroidalMHDEnv(nr=32, ntheta=64, max_steps=5)
        obs, _ = env.reset(seed=42)
        
        truncated = False
        for i in range(10):  # Try to exceed max_steps
            obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
            if truncated:
                print(f"\nTruncated at step {i+1} (max_steps={env.max_steps})")
                break
        
        assert truncated
        assert env.current_step == env.max_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
