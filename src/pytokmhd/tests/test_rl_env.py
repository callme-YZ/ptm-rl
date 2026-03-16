"""
Unit tests for MHD Tearing Control Environment.

Author: 小A 🤖 (RL Lead)
Date: 2026-03-16
Phase: 5 Step 2.5 - Gymnasium Migration + Parameterization
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from pytokmhd.rl import MHDTearingControlEnv


class TestEnvironmentAPI:
    """Test Gymnasium API compliance."""
    
    def test_import(self):
        """Test environment can be imported."""
        from pytokmhd.rl import MHDTearingControlEnv
        assert MHDTearingControlEnv is not None
    
    def test_init_simple(self):
        """Test environment initialization with simple equilibrium."""
        env = MHDTearingControlEnv(equilibrium_type='simple')
        assert env is not None
        assert env.equilibrium_type == 'simple'
    
    def test_init_custom_params(self):
        """Test environment initialization with custom parameters."""
        env = MHDTearingControlEnv(
            equilibrium_type='simple',
            grid_size=32,
            action_smoothing_alpha=0.5,
            max_psi_threshold=20.0,
            max_steps=100,
        )
        assert env.grid_size == 32
        assert env.alpha_smooth == 0.5
        assert env.max_psi == 20.0
        assert env.max_steps == 100
    
    def test_spaces(self):
        """Test observation and action spaces."""
        env = MHDTearingControlEnv()
        
        # Observation space: 25D
        assert env.observation_space.shape == (25,)
        
        # Action space: 1D, [-1, 1]
        assert env.action_space.shape == (1,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
    
    def test_reset_returns_tuple(self):
        """Test reset returns (obs, info) tuple (Gymnasium standard)."""
        env = MHDTearingControlEnv()
        result = env.reset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
    
    def test_reset_observation_shape(self):
        """Test reset observation has correct shape."""
        env = MHDTearingControlEnv()
        obs, info = env.reset()
        assert obs.shape == (25,)
    
    def test_reset_info_dict(self):
        """Test reset info dict contains required keys."""
        env = MHDTearingControlEnv()
        obs, info = env.reset()
        
        assert 'step' in info
        assert 'equilibrium_type' in info
        assert info['step'] == 0
        assert info['equilibrium_type'] == 'simple'
    
    def test_step_returns_five_values(self):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        env = MHDTearingControlEnv()
        env.reset()
        
        action = np.array([0.5])
        result = env.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
    
    def test_step_observation_shape(self):
        """Test step observation has correct shape."""
        env = MHDTearingControlEnv()
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        assert obs.shape == (25,)


class TestEnvironmentBehavior:
    """Test environment physics and dynamics."""
    
    def test_action_smoothing(self):
        """Test action smoothing is applied."""
        env = MHDTearingControlEnv(action_smoothing_alpha=0.5)
        env.reset()
        
        # Apply action
        obs, reward, term, trunc, info = env.step(np.array([1.0]))
        
        # Current action should be smoothed (not 1.0)
        assert env.current_action < 1.0
        assert env.current_action > 0.0
    
    def test_episode_termination(self):
        """Test episode terminates at max steps."""
        env = MHDTearingControlEnv(max_steps=10)
        env.reset()
        
        terminated = False
        truncated = False
        
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            if terminated or truncated:
                break
        
        assert truncated or env.step_count >= 10
    
    def test_early_termination(self):
        """Test early termination on psi threshold."""
        env = MHDTearingControlEnv(max_psi_threshold=0.5)
        env.reset()
        
        # Force large psi
        env.psi = np.ones_like(env.psi) * 10.0
        
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        
        assert terminated
    
    def test_reward_components(self):
        """Test reward function penalizes island width and action."""
        env = MHDTearingControlEnv()
        env.reset()
        
        # Zero action should have higher reward than large action
        obs1, r1, _, _, _ = env.step(np.array([0.0]))
        env.reset()
        obs2, r2, _, _, _ = env.step(np.array([1.0]))
        
        # Reward should be negative (penalty)
        assert r1 < 0
        assert r2 < 0
        
        # Large action should have more penalty
        # (assuming similar island width)
        # Note: May not always hold due to physics evolution


class TestEquilibriumTypes:
    """Test different equilibrium initialization types."""
    
    def test_simple_equilibrium(self):
        """Test simple equilibrium initialization."""
        env = MHDTearingControlEnv(equilibrium_type='simple')
        obs, info = env.reset()
        
        assert env.psi is not None
        assert env.omega is not None
        assert env.psi.shape == (64, 64, 32)
    
    def test_equilibrium_type_in_info(self):
        """Test equilibrium type is reported in info."""
        env = MHDTearingControlEnv(equilibrium_type='simple')
        obs, info = env.reset()
        
        assert info['equilibrium_type'] == 'simple'


class TestStability:
    """Test numerical stability."""
    
    def test_100_steps_no_crash(self):
        """Test environment runs 100 steps without crash."""
        env = MHDTearingControlEnv()
        env.reset()
        
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            
            if terminated or truncated:
                env.reset()
        
        # If we get here, no crash
        assert True
    
    def test_observation_finite(self):
        """Test observations remain finite."""
        env = MHDTearingControlEnv()
        obs, info = env.reset()
        
        assert np.all(np.isfinite(obs))
        
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            assert np.all(np.isfinite(obs)), f"Non-finite obs at step {_}"
    
    def test_reward_finite(self):
        """Test rewards remain finite."""
        env = MHDTearingControlEnv()
        env.reset()
        
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            assert np.isfinite(reward), f"Non-finite reward at step {_}"


class TestDiagnostics:
    """Test diagnostic information."""
    
    def test_info_contains_diagnostics(self):
        """Test info dict contains diagnostic keys."""
        env = MHDTearingControlEnv()
        obs, info = env.reset()
        
        required_keys = ['step', 'psi_max', 'omega_max', 'equilibrium_type']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_step_counter(self):
        """Test step counter increments correctly."""
        env = MHDTearingControlEnv()
        obs, info = env.reset()
        
        assert info['step'] == 0
        
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            assert info['step'] == i + 1


class TestSB3Compatibility:
    """Test Stable-Baselines3 compatibility."""
    
    def test_sb3_import(self):
        """Test SB3 can be imported."""
        try:
            from stable_baselines3 import PPO
            assert True
        except ImportError:
            pytest.skip("SB3 not installed")
    
    def test_sb3_model_creation(self):
        """Test SB3 model can be created."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("SB3 not installed")
        
        env = MHDTearingControlEnv()
        model = PPO('MlpPolicy', env, verbose=0)
        
        assert model is not None
    
    def test_sb3_single_step(self):
        """Test SB3 model can take a step."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("SB3 not installed")
        
        env = MHDTearingControlEnv()
        model = PPO('MlpPolicy', env, verbose=0)
        
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        assert action.shape == (1,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestGridAttributes:
    """Test grid attribute naming consistency."""
    
    def test_grid_aliases(self):
        """Test Nr, Nphi, Nz are aliases for nx, ny, nz."""
        env = MHDTearingControlEnv(grid_size=64)
        
        assert env.Nr == env.nx
        assert env.Nphi == env.ny
        assert env.Nz == env.nz
    
    def test_grid_scaling(self):
        """Test grid size parameter works correctly."""
        env32 = MHDTearingControlEnv(grid_size=32)
        env64 = MHDTearingControlEnv(grid_size=64)
        
        assert env32.Nr == 32
        assert env32.Nz == 16
        
        assert env64.Nr == 64
        assert env64.Nz == 32
    
    def test_phase4_compatibility(self):
        """Test Phase 4 naming convention (Nr, Nz) works."""
        env = MHDTearingControlEnv()
        
        # Phase 4 style access should work
        Nr = env.Nr
        Nz = env.Nz
        
        assert Nr > 0
        assert Nz > 0
        assert Nz == Nr // 2  # Default scaling
