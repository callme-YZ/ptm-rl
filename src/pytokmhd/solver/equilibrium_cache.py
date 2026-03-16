"""
Equilibrium Cache for Fast Reset

Pre-computes and caches PyTokEq equilibria to avoid 1s bottleneck on reset.
Uses Latin Hypercube Sampling for parameter space coverage.

Phase 2 Component - Performance Optimization
Author: 小P ⚛️
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import time
from scipy.stats import qmc


class EquilibriumCache:
    """
    Cache PyTokEq equilibria for fast reset
    
    Strategy:
    - Pre-compute equilibria with different parameters
    - Parameters: q0 ∈ [0.8, 1.2], beta_p ∈ [0.5, 2.0]
    - On reset: random sample from cache + small perturbation
    - Hit rate target: >99%
    
    Attributes:
        cache: List of cached equilibria
        cache_size: Number of equilibria to cache
        param_ranges: Parameter ranges for sampling
        hit_count: Number of cache hits
        total_count: Total number of get_equilibrium calls
    """
    
    def __init__(self, cache_size: int = 50):
        """
        Initialize equilibrium cache
        
        Args:
            cache_size: Number of equilibria to pre-compute
        """
        self.cache: List[Dict] = []
        self.cache_size = cache_size
        self.param_ranges: Optional[Dict] = None
        self.hit_count = 0
        self.total_count = 0
        
    def populate_cache(
        self,
        equilibrium_solver: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        target_grid: Tuple[np.ndarray, np.ndarray],
        verbose: bool = True
    ) -> float:
        """
        Pre-compute cache using Latin Hypercube Sampling
        
        Args:
            equilibrium_solver: Function(q0, beta_p) -> equilibrium_dict
            param_ranges: Dict with 'q0' and 'beta_p' ranges
            target_grid: MHD grid (r, z) for interpolation
            verbose: Print progress
            
        Returns:
            total_time: Time taken to populate cache (seconds)
        """
        self.param_ranges = param_ranges
        
        # Generate parameter samples using Latin Hypercube
        params_list = self._generate_params(param_ranges, self.cache_size)
        
        if verbose:
            print(f"Populating cache with {self.cache_size} equilibria...")
        
        t0 = time.time()
        
        for i, params in enumerate(params_list):
            # Solve equilibrium
            eq = equilibrium_solver(
                q0=params['q0'],
                beta_p=params['beta_p'],
                target_grid=target_grid
            )
            
            # Store in cache
            self.cache.append({
                'params': params,
                'equilibrium': eq
            })
            
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (self.cache_size - i - 1)
                print(f"  {i+1}/{self.cache_size} completed | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        total_time = time.time() - t0
        
        if verbose:
            print(f"Cache populated in {total_time:.1f}s "
                  f"({total_time/self.cache_size:.2f}s per equilibrium)")
        
        return total_time
    
    def get_equilibrium(self, perturb: bool = True, seed: Optional[int] = None) -> Dict:
        """
        Random sample from cache with optional perturbation
        
        Args:
            perturb: Add small random perturbation (±5%)
            seed: Random seed for reproducibility
            
        Returns:
            equilibrium: Sampled (and perturbed) equilibrium dict
        """
        if len(self.cache) == 0:
            raise RuntimeError("Cache is empty. Call populate_cache() first.")
        
        self.total_count += 1
        self.hit_count += 1
        
        # Random sample
        if seed is not None:
            np.random.seed(seed)
        
        idx = np.random.randint(len(self.cache))
        eq = self.cache[idx]['equilibrium'].copy()
        
        if perturb:
            eq = self._perturb_equilibrium(eq)
        
        return eq
    
    def _generate_params(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int
    ) -> List[Dict[str, float]]:
        """
        Generate parameter samples using Latin Hypercube Sampling
        
        Ensures good coverage of parameter space with few samples.
        
        Args:
            param_ranges: Dict with parameter names and (min, max) tuples
            n_samples: Number of samples to generate
            
        Returns:
            params_list: List of parameter dicts
        """
        # Number of parameters
        n_params = len(param_ranges)
        param_names = list(param_ranges.keys())
        
        # Latin Hypercube Sampler
        sampler = qmc.LatinHypercube(d=n_params)
        samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        params_list = []
        for sample in samples:
            params = {}
            for i, name in enumerate(param_names):
                min_val, max_val = param_ranges[name]
                params[name] = min_val + sample[i] * (max_val - min_val)
            params_list.append(params)
        
        return params_list
    
    def _perturb_equilibrium(self, eq: Dict) -> Dict:
        """
        Add small random perturbation to equilibrium
        
        Perturbation amplitude: ±5% for psi, j_tor, pressure
        
        Args:
            eq: Equilibrium dict
            
        Returns:
            eq_perturbed: Perturbed equilibrium
        """
        eq_perturbed = eq.copy()
        
        # Perturbation amplitude
        amp = 0.05
        
        # Perturb psi
        if 'psi_eq' in eq:
            perturbation = 1.0 + amp * (2 * np.random.random(eq['psi_eq'].shape) - 1)
            eq_perturbed['psi_eq'] = eq['psi_eq'] * perturbation
        
        # Perturb j_tor
        if 'j_eq' in eq:
            perturbation = 1.0 + amp * (2 * np.random.random(eq['j_eq'].shape) - 1)
            eq_perturbed['j_eq'] = eq['j_eq'] * perturbation
        
        # Perturb pressure
        if 'p_eq' in eq:
            perturbation = 1.0 + amp * (2 * np.random.random(eq['p_eq'].shape) - 1)
            eq_perturbed['p_eq'] = eq['p_eq'] * perturbation
        
        return eq_perturbed
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate
        
        Returns:
            hit_rate: Fraction of successful cache hits
        """
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count
    
    def reset_stats(self):
        """Reset hit rate statistics"""
        self.hit_count = 0
        self.total_count = 0
    
    def __len__(self) -> int:
        """Number of cached equilibria"""
        return len(self.cache)
    
    def __repr__(self) -> str:
        return (f"EquilibriumCache(size={len(self.cache)}, "
                f"hit_rate={self.get_hit_rate():.2%})")
