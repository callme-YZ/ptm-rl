"""
PyTokMHD Control Module

RMP-based control for tearing mode suppression.

Modules:
- rmp_field: RMP field generation
- rmp_coupling: RMP-MHD coupling
- controller: Control interface (P, PID, RL)
- validation: Control effectiveness tests

Author: 小P ⚛️
Phase: 4
"""

from .rmp_field import (
    generate_rmp_field,
    generate_multimode_rmp,
    compute_rmp_current,
    compute_rmp_helicity,
    validate_rmp_field,
)

from .rmp_coupling import (
    rhs_psi_with_rmp,
    rhs_omega_with_rmp,
    rk4_step_with_rmp,
    compute_rmp_effectiveness,
)

from .controller import (
    RMPController,
    compute_control_metrics,
    validate_controller,
)

from .validation import (
    test_rmp_suppression_open_loop,
    test_proportional_control,
    test_pid_control,
    test_phase_scan,
    benchmark_rmp_overhead,
)

__all__ = [
    # RMP field
    'generate_rmp_field',
    'generate_multimode_rmp',
    'compute_rmp_current',
    'compute_rmp_helicity',
    'validate_rmp_field',
    
    # RMP coupling
    'rhs_psi_with_rmp',
    'rhs_omega_with_rmp',
    'rk4_step_with_rmp',
    'compute_rmp_effectiveness',
    
    # Controller
    'RMPController',
    'compute_control_metrics',
    'validate_controller',
    
    # Validation
    'test_rmp_suppression_open_loop',
    'test_proportional_control',
    'test_pid_control',
    'test_phase_scan',
    'benchmark_rmp_overhead',
]
