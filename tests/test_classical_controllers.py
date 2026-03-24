"""
Tests for Classical Control Baselines

Issue #28: Validate baseline controller implementations

Author: 小A 🤖
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytest
import numpy as np
import gymnasium as gym

from pytokmhd.rl.classical_controllers import (
    NoControlAgent,
    RandomAgent,
    PIDController,
    make_baseline_agent
)


class TestBaselineAgents:
    """Test baseline agent implementations."""
    
    @pytest.fixture
    def action_space(self):
        """Create mock action space."""
        return gym.spaces.Box(
            low=np.array([0.5, 0.5], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
    
    @pytest.fixture
    def mock_obs(self):
        """Create mock observation (23D)."""
        return np.random.randn(23).astype(np.float32)
    
    def test_no_control_agent(self, action_space, mock_obs):
        """Test NoControlAgent returns neutral action."""
        agent = NoControlAgent(action_space)
        
        action = agent.act(mock_obs)
        
        # Should return [1.0, 1.0] (neutral)
        assert action.shape == (2,)
        assert np.allclose(action, [1.0, 1.0])
        
        print("✅ NoControlAgent: Returns neutral action")
    
    def test_random_agent(self, action_space, mock_obs):
        """Test RandomAgent samples valid actions."""
        agent = RandomAgent(action_space, seed=42)
        
        # Sample multiple actions
        actions = [agent.act(mock_obs) for _ in range(10)]
        
        for action in actions:
            # Check shape
            assert action.shape == (2,)
            
            # Check bounds
            assert np.all(action >= action_space.low)
            assert np.all(action <= action_space.high)
        
        # Check actions are different (random)
        actions_array = np.array(actions)
        assert np.std(actions_array) > 0.1
        
        print("✅ RandomAgent: Samples valid random actions")
    
    def test_pid_controller_basic(self, action_space, mock_obs):
        """Test PID controller basic functionality."""
        agent = PIDController(
            action_space,
            target=0.0,
            Kp=5.0,
            Ki=0.5,
            Kd=0.01,
            dt=1e-4
        )
        
        # Reset
        agent.reset()
        
        # Act
        action = agent.act(mock_obs)
        
        # Check action shape
        assert action.shape == (2,)
        
        # Check bounds
        assert np.all(action >= action_space.low)
        assert np.all(action <= action_space.high)
        
        # Check nu_mult unchanged (should be 1.0)
        assert np.isclose(action[1], 1.0)
        
        print("✅ PID: Basic functionality works")
    
    def test_pid_anti_windup(self, action_space):
        """Test PID anti-windup protection."""
        agent = PIDController(
            action_space,
            target=0.0,
            Kp=100.0,  # Large gain to trigger saturation
            Ki=10.0,
            Kd=0.0,
            dt=1e-4
        )
        
        agent.reset()
        
        # Large error (should saturate)
        obs_large_error = np.zeros(23, dtype=np.float32)
        obs_large_error[7] = 10.0  # Large m1_amp
        
        # Act multiple times
        for _ in range(10):
            action = agent.act(obs_large_error)
            
            # Should saturate at upper bound
            assert action[0] <= action_space.high[0]
        
        # Integral should not blow up (anti-windup working)
        assert np.abs(agent.error_int) < 100.0
        
        print("✅ PID: Anti-windup protection works")
    
    def test_make_baseline_agent(self, action_space):
        """Test factory function."""
        # No control
        agent_nc = make_baseline_agent('no_control', action_space)
        assert isinstance(agent_nc, NoControlAgent)
        
        # Random
        agent_rand = make_baseline_agent('random', action_space, seed=42)
        assert isinstance(agent_rand, RandomAgent)
        
        # PID
        agent_pid = make_baseline_agent('pid', action_space, Kp=10.0)
        assert isinstance(agent_pid, PIDController)
        assert agent_pid.Kp == 10.0
        
        # Unknown
        with pytest.raises(ValueError):
            make_baseline_agent('unknown', action_space)
        
        print("✅ make_baseline_agent: Factory works")


class TestPIDTuning:
    """Test PID controller with different tunings."""
    
    @pytest.fixture
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([0.5, 0.5], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def test_conservative_tuning(self, action_space):
        """Test 小P's conservative tuning."""
        agent = PIDController(
            action_space,
            Kp=5.0,   # 小P recommended
            Ki=0.5,
            Kd=0.01,
            dt=1e-4
        )
        
        # Should be stable for moderate errors
        obs = np.zeros(23, dtype=np.float32)
        obs[7] = 0.1  # Moderate m1_amp
        
        agent.reset()
        
        for _ in range(100):
            action = agent.act(obs)
            
            # Should stay in bounds
            assert np.all(action >= action_space.low)
            assert np.all(action <= action_space.high)
        
        print("✅ PID: Conservative tuning stable")
    
    def test_response_to_error(self, action_space):
        """Test PID responds correctly to error."""
        agent = PIDController(
            action_space,
            target=0.0,
            Kp=5.0,
            Ki=0.0,  # No integral for simplicity
            Kd=0.0,  # No derivative
            dt=1e-4
        )
        
        agent.reset()
        
        # High m1_amp → should increase eta_mult (小P physics ⚛️)
        obs_high = np.zeros(23, dtype=np.float32)
        obs_high[7] = 0.5  # High m1 amplitude
        
        action_high = agent.act(obs_high)
        
        # Physics (小P ⚛️): Higher η suppresses tearing mode
        # error = m1_amp - target = 0.5 - 0.0 = 0.5 (positive)
        # u = Kp * error = 5.0 * 0.5 = 2.5
        # eta_mult = 1.0 + u = 3.5 → clipped to 2.0 ✅
        
        # Verify eta_mult increased
        assert action_high[0] > 1.0, "High m1 should increase eta"
        assert action_high[0] == 2.0, "Should saturate at upper bound"
        
        # Low m1_amp → should decrease eta_mult
        obs_low = np.zeros(23, dtype=np.float32)
        obs_low[7] = 0.0  # Zero amplitude
        
        agent.reset()
        action_low = agent.act(obs_low)
        
        # error = 0.0 - 0.0 = 0.0
        # u = 0.0
        # eta_mult = 1.0 (neutral)
        assert np.isclose(action_low[0], 1.0), "Zero m1 should give neutral eta"
        
        print("✅ PID: Correct response to error (小P verified ⚛️)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
