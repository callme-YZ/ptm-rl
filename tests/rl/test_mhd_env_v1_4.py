"""
Unit tests for 3D MHD Gym Environment (v1.4)

Tests:
1. test_reset: Environment initializes correctly
2. test_step: Single step executes
3. test_random_rollout: 10-step episode with random actions
4. test_energy_tracking: Reward reflects energy change

Author: 小A 🤖
Created: 2026-03-20
Phase: 3 (RL Environment Testing)
"""

import pytest
import numpy as np
import gymnasium as gym
from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D, make_env


class TestMHDEnv3D:
    """Test suite for 3D MHD RL environment."""
    
    @pytest.fixture
    def env(self):
        """Create test environment with small grid for speed."""
        return MHDEnv3D(
            grid_size=(16, 32, 16),  # Small grid for fast tests
            eta=1e-4,
            dt=0.01,
            max_steps=10,
            I_max=1.0,
            n_coils=5
        )
    
    def test_reset(self, env):
        """
        Test 1: Environment initializes correctly.
        
        Checks:
        - Observation dict has correct keys
        - Field shapes match grid size
        - Normalization factors are positive
        - Initial energy is positive
        """
        obs, info = env.reset(seed=42)
        
        # Check observation structure
        assert isinstance(obs, dict), "Observation should be a dict"
        expected_keys = {'psi', 'omega', 'energy', 'max_psi', 'max_omega'}
        assert set(obs.keys()) == expected_keys, f"Missing keys: {expected_keys - set(obs.keys())}"
        
        # Check field shapes
        nr, ntheta, nzeta = env.grid.nr, env.grid.ntheta, env.grid.nzeta
        assert obs['psi'].shape == (nr, ntheta, nzeta), "psi shape mismatch"
        assert obs['omega'].shape == (nr, ntheta, nzeta), "omega shape mismatch"
        
        # Check scalar observations
        assert obs['energy'].shape == (), "energy should be scalar"
        assert obs['max_psi'].shape == (), "max_psi should be scalar"
        assert obs['max_omega'].shape == (), "max_omega should be scalar"
        
        # Check info metadata
        assert 'E0' in info, "Missing initial energy E0"
        assert 'psi_max' in info, "Missing psi_max"
        assert 'omega_max' in info, "Missing omega_max"
        
        # Check normalization factors are positive
        assert info['E0'] > 0, "Initial energy should be positive"
        assert info['psi_max'] > 0, "psi_max should be positive"
        assert info['omega_max'] > 0, "omega_max should be positive"
        
        # Check initial normalized energy is ~1
        assert np.abs(obs['energy'] - 1.0) < 0.1, "Initial normalized energy should be ~1"
        
        print("✅ test_reset passed")
    
    def test_step(self, env):
        """
        Test 2: Single step executes without errors.
        
        Checks:
        - Step returns correct tuple (obs, reward, terminated, truncated, info)
        - Observation structure is preserved
        - Reward is a scalar float
        - Info dict contains diagnostics
        """
        obs, info = env.reset(seed=42)
        
        # Take random action
        action = env.action_space.sample()
        obs_new, reward, terminated, truncated, info = env.step(action)
        
        # Check return types
        assert isinstance(obs_new, dict), "Observation should be dict"
        assert isinstance(reward, (float, np.floating)), "Reward should be float"
        assert isinstance(terminated, bool), "terminated should be bool"
        assert isinstance(truncated, bool), "truncated should be bool"
        assert isinstance(info, dict), "info should be dict"
        
        # Check observation structure
        expected_keys = {'psi', 'omega', 'energy', 'max_psi', 'max_omega'}
        assert set(obs_new.keys()) == expected_keys, "Observation keys changed after step"
        
        # Check reward is finite
        assert np.isfinite(reward), "Reward should be finite"
        
        # Check info diagnostics
        required_info = {'time', 'energy', 'energy_drift', 'coil_currents'}
        assert required_info.issubset(info.keys()), f"Missing info keys: {required_info - set(info.keys())}"
        
        # Check time incremented
        assert info['time'] == env.dt, "Time should be dt after 1 step"
        
        # Check coil currents match action
        assert np.allclose(info['coil_currents'], action * env.I_max), "Coil currents should match scaled action"
        
        print("✅ test_step passed")
    
    def test_random_rollout(self, env):
        """
        Test 3: 10-step episode with random actions completes.
        
        Checks:
        - Episode runs to completion without errors
        - Energy evolves (changes over time)
        - Observations remain valid (no NaN/inf)
        - Episode truncates at max_steps
        """
        obs, info = env.reset(seed=42)
        
        energies = [info['E0']]
        rewards = []
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record diagnostics
            energies.append(info['energy'])
            rewards.append(reward)
            
            # Check observation validity
            assert np.all(np.isfinite(obs['psi'])), f"psi contains NaN/inf at step {step}"
            assert np.all(np.isfinite(obs['omega'])), f"omega contains NaN/inf at step {step}"
            assert np.isfinite(obs['energy']), f"energy is NaN/inf at step {step}"
            
            # Check not terminated early (no failure condition)
            assert not terminated, "Episode should not terminate before max_steps"
            
            # Check truncation at last step
            if step == 9:
                assert truncated, "Episode should truncate at max_steps=10"
            else:
                assert not truncated, "Episode should not truncate before max_steps"
        
        # Check energy evolved
        energy_std = np.std(energies)
        assert energy_std > 0, "Energy should change during episode"
        
        # Check rewards are finite
        assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"
        
        print(f"✅ test_random_rollout passed: {len(energies)} steps, ΔE/E₀ = {energy_std/energies[0]:.2e}")
    
    def test_energy_tracking(self, env):
        """
        Test 4: Reward correctly tracks energy conservation.
        
        Checks:
        - Reward = -|ΔE/E₀| per step
        - Energy drift accumulates over episode
        - Zero action (I=0) should minimize energy change
        """
        obs, info = env.reset(seed=42)
        E0 = info['E0']
        
        # Test with zero action (no control)
        action_zero = np.zeros(env.n_coils)
        obs1, reward1, _, _, info1 = env.step(action_zero)
        
        # Check reward formula: r = -|ΔE/E₀|
        E1 = info1['energy']
        expected_reward1 = -abs(E1 - E0) / E0
        assert np.isclose(reward1, expected_reward1, rtol=1e-5), \
            f"Reward mismatch: {reward1} vs {expected_reward1}"
        
        # Take another step
        obs2, reward2, _, _, info2 = env.step(action_zero)
        
        # Check reward is relative to previous step, not initial
        E2 = info2['energy']
        expected_reward2 = -abs(E2 - E1) / E0
        assert np.isclose(reward2, expected_reward2, rtol=1e-5), \
            f"Reward mismatch at step 2: {reward2} vs {expected_reward2}"
        
        # Check energy drift accumulates
        drift = abs(E2 - E0) / E0
        assert drift == info2['energy_drift'], "energy_drift should be |E-E₀|/E₀"
        
        # For resistive MHD, energy should decrease (dissipation)
        if env.eta > 0:
            assert E2 < E0, "Energy should dissipate with resistivity η>0"
        
        print(f"✅ test_energy_tracking passed: drift = {drift:.2e}, dissipation verified")
    
    def test_action_space(self, env):
        """
        Test 5: Action space is correctly configured.
        
        Checks:
        - Action space is Box with correct shape
        - Bounds are [-1, 1]
        - Sampling produces valid actions
        """
        assert isinstance(env.action_space, gym.spaces.Box), "Action space should be Box"
        assert env.action_space.shape == (env.n_coils,), "Action shape should match n_coils"
        
        # Check bounds
        assert np.all(env.action_space.low == -1.0), "Action lower bound should be -1"
        assert np.all(env.action_space.high == 1.0), "Action upper bound should be +1"
        
        # Test sampling
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action), "Sampled action should be valid"
            assert np.all(action >= -1.0) and np.all(action <= 1.0), "Action out of bounds"
        
        print("✅ test_action_space passed")
    
    def test_observation_space(self, env):
        """
        Test 6: Observation space is correctly configured.
        
        Checks:
        - Observation space is Dict with correct keys
        - Field shapes match grid
        - Actual observations satisfy space
        """
        obs_space = env.observation_space
        
        assert isinstance(obs_space, gym.spaces.Dict), "Observation space should be Dict"
        assert set(obs_space.spaces.keys()) == {'psi', 'omega', 'energy', 'max_psi', 'max_omega'}, \
            "Observation space keys mismatch"
        
        # Check field shapes
        nr, ntheta, nzeta = env.grid.nr, env.grid.ntheta, env.grid.nzeta
        assert obs_space['psi'].shape == (nr, ntheta, nzeta), "psi space shape mismatch"
        assert obs_space['omega'].shape == (nr, ntheta, nzeta), "omega space shape mismatch"
        
        # Get actual observation
        obs, _ = env.reset(seed=42)
        
        # Check observation satisfies space
        assert obs_space.contains(obs), "Observation should satisfy observation space"
        
        print("✅ test_observation_space passed")


def test_make_env():
    """Test convenience function make_env."""
    env = make_env(grid_size=(16, 32, 16), max_steps=5)
    
    assert isinstance(env, MHDEnv3D), "make_env should return MHDEnv3D instance"
    assert env.max_steps == 5, "make_env should respect kwargs"
    assert env.grid.nr == 16, "Grid size should be set correctly"
    
    # Run quick episode
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, (float, np.floating)), "Step should return valid reward"
    
    print("✅ test_make_env passed")


if __name__ == "__main__":
    """Run tests manually (pytest also works)."""
    import sys
    sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl')
    
    # Import after path fix
    import gymnasium as gym
    
    print("Running MHDEnv3D test suite...")
    print("=" * 60)
    
    env = MHDEnv3D(
        grid_size=(16, 32, 16),
        eta=1e-4,
        dt=0.01,
        max_steps=10,
        I_max=1.0,
        n_coils=5
    )
    
    test_suite = TestMHDEnv3D()
    
    try:
        test_suite.test_reset(env)
        test_suite.test_step(env)
        test_suite.test_random_rollout(env)
        test_suite.test_energy_tracking(env)
        test_suite.test_action_space(env)
        test_suite.test_observation_space(env)
        test_make_env()
        
        print("=" * 60)
        print("🎉 All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
