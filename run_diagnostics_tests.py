#!/usr/bin/env python3
"""
Standalone test runner for diagnostics module
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now run the tests
from pytokmhd.tests.test_diagnostics import *

if __name__ == '__main__':
    print("=" * 70)
    print("Running PyTokMHD Diagnostics Tests")
    print("=" * 70)
    
    # Rational surface tests
    print("\n--- Test 1: Rational Surface ---")
    test_rational_surface_solovev()
    test_rational_surface_linear_q()
    test_rational_surface_out_of_range()
    print("✓ All rational surface tests passed")
    
    # Island width tests
    print("\n--- Test 2: Island Width ---")
    test_island_width_perturbed_solovev()
    test_island_width_scaling()
    print("✓ All island width tests passed")
    
    # Growth rate tests
    print("\n--- Test 3: Growth Rate ---")
    test_growth_rate_exponential()
    test_growth_rate_with_noise()
    test_growth_rate_negative()
    print("✓ All growth rate tests passed")
    
    # Monitor integration tests
    print("\n--- Test 4: Monitor Integration ---")
    test_monitor_integration()
    test_monitor_reset()
    print("✓ All monitor tests passed")
    
    print("\n" + "=" * 70)
    print("ALL DIAGNOSTICS TESTS PASSED! ✓")
    print("=" * 70)
