"""
M3 Step 3.2: Action Space Tests

Validation of action processing and parameter modulation.

Author: 小P ⚛️
Created: 2026-03-18
"""

import numpy as np
import pytest
import gymnasium as gym


class TestMHDActionBasics:
    """Test 1: MHDAction class basic functionality."""
    
    def test_initialization(self):
        """MHDAction should initialize correctly."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        
        print(f"\nAction handler initialized:")
        print(f"  eta_base: {handler.eta_base}")
        print(f"  nu_base: {handler.nu_base}")
        print(f"  action_bounds: {handler.action_bounds}")
        
        assert handler.eta_base == 1e-5
        assert handler.nu_base == 1e-4
        assert handler.action_bounds == (0.5, 2.0)
    
    def test_apply_identity(self):
        """Identity action [1.0, 1.0] should give base parameters."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        
        action = np.array([1.0, 1.0])
        eta_eff, nu_eff = handler.apply(action)
        
        print(f"\nIdentity action:")
        print(f"  action: {action}")
        print(f"  eta_eff: {eta_eff} (expected: {handler.eta_base})")
        print(f"  nu_eff: {nu_eff} (expected: {handler.nu_base})")
        
        assert eta_eff == handler.eta_base
        assert nu_eff == handler.nu_base
    
    def test_apply_modulation(self):
        """Non-identity action should modulate parameters."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        
        action = np.array([1.5, 0.8])
        eta_eff, nu_eff = handler.apply(action)
        
        expected_eta = 1e-5 * 1.5
        expected_nu = 1e-4 * 0.8
        
        print(f"\nModulated action:")
        print(f"  action: {action}")
        print(f"  eta_eff: {eta_eff:.3e} (expected: {expected_eta:.3e})")
        print(f"  nu_eff: {nu_eff:.3e} (expected: {expected_nu:.3e})")
        
        assert np.isclose(eta_eff, expected_eta)
        assert np.isclose(nu_eff, expected_nu)
    
    def test_apply_clipping(self):
        """Out-of-bounds action should be clipped."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4, action_bounds=(0.5, 2.0))
        
        # Test low bound
        action_low = np.array([0.3, 0.3])  # Below 0.5
        eta_low, nu_low = handler.apply(action_low)
        
        expected_eta_low = 1e-5 * 0.5  # Clipped to 0.5
        expected_nu_low = 1e-4 * 0.5
        
        print(f"\nLow clipping:")
        print(f"  action: {action_low} → clipped to [0.5, 0.5]")
        print(f"  eta_eff: {eta_low:.3e} (expected: {expected_eta_low:.3e})")
        
        assert np.isclose(eta_low, expected_eta_low)
        assert np.isclose(nu_low, expected_nu_low)
        
        # Test high bound
        action_high = np.array([3.0, 3.0])  # Above 2.0
        eta_high, nu_high = handler.apply(action_high)
        
        expected_eta_high = 1e-5 * 2.0  # Clipped to 2.0
        expected_nu_high = 1e-4 * 2.0
        
        print(f"\nHigh clipping:")
        print(f"  action: {action_high} → clipped to [2.0, 2.0]")
        print(f"  eta_eff: {eta_high:.3e} (expected: {expected_eta_high:.3e})")
        
        assert np.isclose(eta_high, expected_eta_high)
        assert np.isclose(nu_high, expected_nu_high)


class TestActionSpace:
    """Test 2: Gymnasium action space compatibility."""
    
    def test_get_action_space(self):
        """get_action_space should return valid Box."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        action_space = handler.get_action_space()
        
        print(f"\nAction space: {action_space}")
        print(f"  low: {action_space.low}")
        print(f"  high: {action_space.high}")
        print(f"  shape: {action_space.shape}")
        print(f"  dtype: {action_space.dtype}")
        
        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.shape == (2,)
        assert action_space.dtype == np.float32
        assert np.allclose(action_space.low, [0.5, 0.5])
        assert np.allclose(action_space.high, [2.0, 2.0])
    
    def test_sample_from_space(self):
        """Action space should allow sampling."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        action_space = handler.get_action_space()
        
        # Sample multiple times
        samples = [action_space.sample() for _ in range(10)]
        
        print(f"\nSampled actions (10):")
        for i, sample in enumerate(samples):
            print(f"  {i}: {sample}")
        
        # All samples should be in bounds
        for sample in samples:
            assert action_space.contains(sample)
            assert np.all(sample >= 0.5)
            assert np.all(sample <= 2.0)
    
    def test_default_action(self):
        """get_default_action should return identity."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        default = handler.get_default_action()
        
        print(f"\nDefault action: {default}")
        
        assert np.allclose(default, [1.0, 1.0])
        assert default.dtype == np.float32


class TestActionNormalization:
    """Test 3: Action normalization for RL algorithms."""
    
    def test_normalize_from_unit(self):
        """normalize_from_unit should map [-1,1] to [low,high]."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4, action_bounds=(0.5, 2.0))
        
        # Test extremes
        action_min = np.array([-1.0, -1.0])
        action_scaled_min = handler.normalize_from_unit(action_min)
        
        action_max = np.array([1.0, 1.0])
        action_scaled_max = handler.normalize_from_unit(action_max)
        
        action_zero = np.array([0.0, 0.0])
        action_scaled_zero = handler.normalize_from_unit(action_zero)
        
        print(f"\nNormalization test:")
        print(f"  [-1, -1] → {action_scaled_min} (expected: [0.5, 0.5])")
        print(f"  [0, 0]   → {action_scaled_zero} (expected: [1.25, 1.25])")
        print(f"  [1, 1]   → {action_scaled_max} (expected: [2.0, 2.0])")
        
        assert np.allclose(action_scaled_min, [0.5, 0.5])
        assert np.allclose(action_scaled_max, [2.0, 2.0])
        assert np.allclose(action_scaled_zero, [1.25, 1.25])  # Midpoint


class TestActionInfo:
    """Test 4: Action information utilities."""
    
    def test_get_action_info(self):
        """get_action_info should provide interpretation."""
        from pytokmhd.rl.actions import MHDAction
        
        handler = MHDAction(eta_base=1e-5, nu_base=1e-4)
        
        action = np.array([1.5, 0.8])
        info = handler.get_action_info(action)
        
        print(f"\nAction info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Check structure
        assert 'eta_multiplier' in info
        assert 'nu_multiplier' in info
        assert 'eta_effective' in info
        assert 'nu_effective' in info
        assert 'eta_change' in info
        assert 'nu_change' in info
        
        # Check values
        assert info['eta_multiplier'] == 1.5
        assert info['nu_multiplier'] == 0.8
        assert np.isclose(info['eta_effective'], 1.5e-5)
        assert np.isclose(info['nu_effective'], 0.8e-4)
        assert np.isclose(info['eta_change'], 50.0)  # +50%
        assert np.isclose(info['nu_change'], -20.0)  # -20%


class TestFactoryFunctions:
    """Test 5: Convenience factory functions."""
    
    def test_create_action_handler(self):
        """create_action_handler should create MHDAction."""
        from pytokmhd.rl.actions import create_action_handler
        
        handler = create_action_handler(eta=1e-5, nu=1e-4)
        
        print(f"\nFactory-created handler:")
        print(f"  type: {type(handler).__name__}")
        print(f"  eta_base: {handler.eta_base}")
        print(f"  nu_base: {handler.nu_base}")
        
        assert handler.eta_base == 1e-5
        assert handler.nu_base == 1e-4
    
    def test_get_action_space_v1_1(self):
        """get_action_space_v1_1 should return standard space."""
        from pytokmhd.rl.actions import get_action_space_v1_1
        
        action_space = get_action_space_v1_1()
        
        print(f"\nv1.1 standard action space: {action_space}")
        
        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.shape == (2,)
        assert np.allclose(action_space.low, [0.5, 0.5])
        assert np.allclose(action_space.high, [2.0, 2.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
