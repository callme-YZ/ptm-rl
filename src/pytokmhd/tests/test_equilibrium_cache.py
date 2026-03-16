"""
Equilibrium Cache Performance Tests

Tests for Phase 2: Cache performance and hit rate

Author: 小P ⚛️
"""

import numpy as np
import pytest
import time
from pytokmhd.solver.equilibrium_cache import EquilibriumCache
from pytokmhd.solver.initial_conditions import solovev_equilibrium


class TestCachePerformance:
    """Test cache performance metrics"""
    
    def test_cache_population_time(self):
        """
        Test 3: Verify cache population time < 5min
        
        Uses mock equilibrium solver (Solovev) for speed
        """
        # Mock equilibrium solver
        def mock_equilibrium_solver(q0, beta_p, target_grid):
            """Fast mock solver using Solovev"""
            r, z = target_grid
            psi, omega = solovev_equilibrium(r, z)
            
            # Scale by parameters
            psi = psi * q0 * beta_p
            
            return {
                'psi_eq': psi,
                'j_eq': np.gradient(np.gradient(psi, axis=0), axis=0),
                'p_eq': 0.01 * psi**2,
                'q_profile': np.linspace(q0, q0 + 2, len(r))
            }
        
        # Cache setup
        cache = EquilibriumCache(cache_size=50)
        
        # Target grid
        r = np.linspace(0.5, 1.5, 64)
        z = np.linspace(-0.5, 0.5, 128)
        
        # Parameter ranges
        param_ranges = {
            'q0': (0.8, 1.2),
            'beta_p': (0.5, 2.0)
        }
        
        # Populate cache and time it
        print("\nPopulating cache...")
        total_time = cache.populate_cache(
            mock_equilibrium_solver,
            param_ranges,
            (r, z),
            verbose=True
        )
        
        print(f"\nCache population results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Time per equilibrium: {total_time/50:.3f}s")
        print(f"  Cache size: {len(cache)}")
        
        # Verification
        assert total_time < 300, f"Cache population took {total_time:.1f}s > 5min"
        assert len(cache) == 50, f"Cache size {len(cache)} != 50"
    
    def test_reset_time(self):
        """
        Test reset time < 0.1s (vs 1s without cache)
        """
        # Setup cache
        cache = EquilibriumCache(cache_size=50)
        
        def mock_solver(q0, beta_p, target_grid):
            r, z = target_grid
            psi, _ = solovev_equilibrium(r, z)
            return {
                'psi_eq': psi * q0 * beta_p,
                'j_eq': np.zeros_like(psi),
                'p_eq': np.zeros_like(psi),
                'q_profile': np.linspace(q0, q0 + 2, len(r))
            }
        
        r = np.linspace(0.5, 1.5, 64)
        z = np.linspace(-0.5, 0.5, 128)
        
        cache.populate_cache(
            mock_solver,
            {'q0': (0.8, 1.2), 'beta_p': (0.5, 2.0)},
            (r, z),
            verbose=False
        )
        
        # Time cache access
        n_samples = 1000
        t0 = time.time()
        
        for _ in range(n_samples):
            eq = cache.get_equilibrium(perturb=True)
        
        t1 = time.time()
        
        avg_time = (t1 - t0) / n_samples
        
        print(f"\nReset time test:")
        print(f"  Average get_equilibrium time: {avg_time*1000:.3f}ms")
        print(f"  Target: <1ms")
        
        # Verification
        assert avg_time < 0.001, f"Reset time {avg_time*1000:.2f}ms exceeds 1ms"
    
    def test_cache_hit_rate(self):
        """
        Test cache hit rate > 99%
        
        With random sampling, hit rate should be 100%
        """
        cache = EquilibriumCache(cache_size=50)
        
        def mock_solver(q0, beta_p, target_grid):
            r, z = target_grid
            psi, _ = solovev_equilibrium(r, z)
            return {
                'psi_eq': psi,
                'j_eq': np.zeros_like(psi),
                'p_eq': np.zeros_like(psi),
                'q_profile': np.linspace(1.0, 3.0, len(r))
            }
        
        r = np.linspace(0.5, 1.5, 32)
        z = np.linspace(-0.5, 0.5, 64)
        
        cache.populate_cache(
            mock_solver,
            {'q0': (0.8, 1.2), 'beta_p': (0.5, 2.0)},
            (r, z),
            verbose=False
        )
        
        # Sample many times
        cache.reset_stats()
        for _ in range(1000):
            eq = cache.get_equilibrium()
        
        hit_rate = cache.get_hit_rate()
        
        print(f"\nHit rate test:")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  Target: >99%")
        
        # With random sampling from cache, hit rate is always 100%
        assert hit_rate > 0.99, f"Hit rate {hit_rate:.2%} below 99%"
    
    def test_speedup_vs_no_cache(self):
        """
        Test that cache provides >10× speedup
        """
        # Mock slow solver (simulates 100ms PyTokEq call)
        def slow_solver(q0, beta_p, target_grid):
            time.sleep(0.1)  # 100ms delay
            r, z = target_grid
            psi, _ = solovev_equilibrium(r, z)
            return {
                'psi_eq': psi,
                'j_eq': np.zeros_like(psi),
                'p_eq': np.zeros_like(psi),
                'q_profile': np.linspace(1.0, 3.0, len(r))
            }
        
        r = np.linspace(0.5, 1.5, 32)
        z = np.linspace(-0.5, 0.5, 64)
        
        # Time without cache (direct solver calls)
        n_calls = 10
        t0 = time.time()
        for _ in range(n_calls):
            eq = slow_solver(1.0, 1.0, (r, z))
        t_no_cache = (time.time() - t0) / n_calls
        
        # Time with cache
        cache = EquilibriumCache(cache_size=20)
        cache.populate_cache(
            slow_solver,
            {'q0': (0.8, 1.2), 'beta_p': (0.5, 2.0)},
            (r, z),
            verbose=False
        )
        
        t0 = time.time()
        for _ in range(n_calls):
            eq = cache.get_equilibrium()
        t_cache = (time.time() - t0) / n_calls
        
        speedup = t_no_cache / t_cache
        
        print(f"\nSpeedup test:")
        print(f"  No cache: {t_no_cache*1000:.1f}ms per call")
        print(f"  With cache: {t_cache*1000:.3f}ms per call")
        print(f"  Speedup: {speedup:.1f}×")
        print(f"  Target: >10×")
        
        assert speedup > 10, f"Speedup {speedup:.1f}× below 10×"


class TestCacheUtilities:
    """Test cache utility functions"""
    
    def test_latin_hypercube_sampling(self):
        """
        Test that Latin Hypercube Sampling covers parameter space
        """
        cache = EquilibriumCache(cache_size=100)
        
        param_ranges = {
            'q0': (0.8, 1.2),
            'beta_p': (0.5, 2.0)
        }
        
        params_list = cache._generate_params(param_ranges, 100)
        
        # Extract parameters
        q0_samples = [p['q0'] for p in params_list]
        beta_p_samples = [p['beta_p'] for p in params_list]
        
        print(f"\nLatin Hypercube Sampling:")
        print(f"  q0 range: [{min(q0_samples):.3f}, {max(q0_samples):.3f}]")
        print(f"  Expected: [0.800, 1.200]")
        print(f"  beta_p range: [{min(beta_p_samples):.3f}, {max(beta_p_samples):.3f}]")
        print(f"  Expected: [0.500, 2.000]")
        
        # Check coverage
        assert min(q0_samples) >= 0.8
        assert max(q0_samples) <= 1.2
        assert min(beta_p_samples) >= 0.5
        assert max(beta_p_samples) <= 2.0
    
    def test_perturbation(self):
        """
        Test equilibrium perturbation (±5%)
        """
        cache = EquilibriumCache(cache_size=1)
        
        # Create mock equilibrium
        r = np.linspace(0.5, 1.5, 32)
        z = np.linspace(-0.5, 0.5, 64)
        psi, _ = solovev_equilibrium(r, z)
        
        eq_original = {
            'psi_eq': psi,
            'j_eq': np.zeros_like(psi),
            'p_eq': np.zeros_like(psi)
        }
        
        # Perturb
        eq_perturbed = cache._perturb_equilibrium(eq_original)
        
        # Check perturbation amplitude
        rel_change = np.abs(eq_perturbed['psi_eq'] - eq_original['psi_eq']) / (np.abs(eq_original['psi_eq']) + 1e-10)
        max_change = np.max(rel_change)
        
        print(f"\nPerturbation test:")
        print(f"  Max relative change: {max_change:.3f}")
        print(f"  Expected: ~0.05 (5%)")
        
        # Should be within ±5%
        assert max_change < 0.1, f"Perturbation {max_change:.3f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
