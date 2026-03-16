"""
RMP Control Validation

Tests RMP control effectiveness with full MHD evolution.

Validation Tests:
1. Open-loop: RMP suppression with constant amplitude
2. Closed-loop P-control: proportional feedback convergence
3. Closed-loop PID-control: PID feedback with overshoot check
4. Phase scan: verify phase dependence

Author: 小P ⚛️
Created: 2026-03-16
Phase: 4
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time

from .rmp_coupling import rk4_step_with_rmp
from .controller import RMPController
from ..diagnostics.monitor import TearingModeMonitor
from ..solver.initial_conditions import setup_tearing_mode


def test_rmp_suppression_open_loop(
    Nr: int = 64,
    Nz: int = 128,
    Lr: float = 1.0,
    Lz: float = 2*np.pi,
    eta: float = 1e-3,
    nu: float = 0.0,
    rmp_amplitude: float = 0.05,
    m: int = 2,
    n: int = 1,
    n_steps: int = 100,
    dt: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Test open-loop RMP suppression.
    
    Setup:
    1. Initialize tearing mode with w_0 = 0.01
    2. Run without RMP → measure γ_free
    3. Run with constant RMP → measure γ_rmp
    
    Success Criterion:
    γ_rmp < 0.5 * γ_free (50% reduction in growth rate)
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    Lr, Lz : float
        Domain size
    eta : float
        Resistivity
    nu : float
        Viscosity
    rmp_amplitude : float
        RMP control amplitude
    m, n : int
        Mode numbers
    n_steps : int
        Number of timesteps
    dt : float
        Timestep size
    
    Returns
    -------
    success : bool
        True if RMP achieves >50% reduction
    diagnostics : dict
        Test results
    
    Examples
    --------
    >>> success, diag = test_rmp_suppression_open_loop(rmp_amplitude=0.05)
    >>> print(f"Reduction: {diag['reduction']*100:.1f}%")
    Reduction: 65.3%
    """
    # Setup grid
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # Initialize tearing mode
    q_profile = 1.5 + 1.5 * (r / Lr)**2  # q(r) ∈ [1.5, 3.0]
    psi0, omega0, r_s = setup_tearing_mode(R, Z, q_profile, r, m=m, n=n, w_0=0.01)
    
    # Monitor
    monitor_free = TearingModeMonitor(m=m, n=n)
    monitor_rmp = TearingModeMonitor(m=m, n=n)
    
    # 1. Free evolution (no RMP)
    psi_free = psi0.copy()
    omega_free = omega0.copy()
    
    for step in range(n_steps):
        t = step * dt
        psi_free, omega_free = rk4_step_with_rmp(
            psi_free, omega_free, dt, dr, dz, R, eta, nu,
            rmp_amplitude=0.0  # No control
        )
        monitor_free.update(psi_free, omega_free, t, r, z, q_profile)
    
    # Measure growth rate (from last 50 steps)
    w_free = np.array(monitor_free.w_history)
    if len(w_free) > 50:
        # Fit exponential: w = w0 * exp(γ*t)
        t_fit = np.arange(len(w_free) - 50, len(w_free)) * dt
        w_fit = w_free[-50:]
        
        # Avoid log of zero
        w_fit = np.maximum(w_fit, 1e-10)
        
        # Linear fit in log space
        log_w = np.log(w_fit)
        gamma_free = np.polyfit(t_fit, log_w, 1)[0]
    else:
        gamma_free = 0.0
    
    # 2. With RMP control
    psi_rmp = psi0.copy()
    omega_rmp = omega0.copy()
    
    for step in range(n_steps):
        t = step * dt
        psi_rmp, omega_rmp = rk4_step_with_rmp(
            psi_rmp, omega_rmp, dt, dr, dz, R, eta, nu,
            rmp_amplitude=rmp_amplitude,  # Constant RMP
            m=m, n=n
        )
        monitor_rmp.update(psi_rmp, omega_rmp, t, r, z, q_profile)
    
    # Measure growth rate with RMP
    w_rmp = np.array(monitor_rmp.w_history)
    if len(w_rmp) > 50:
        t_fit = np.arange(len(w_rmp) - 50, len(w_rmp)) * dt
        w_fit = w_rmp[-50:]
        w_fit = np.maximum(w_fit, 1e-10)
        log_w = np.log(w_fit)
        gamma_rmp = np.polyfit(t_fit, log_w, 1)[0]
    else:
        gamma_rmp = 0.0
    
    # Compute reduction
    if abs(gamma_free) < 1e-10:
        reduction = 0.0
    else:
        reduction = (gamma_free - gamma_rmp) / gamma_free
    
    # Success criterion: >50% reduction
    success = reduction > 0.5
    
    diagnostics = {
        'gamma_free': gamma_free,
        'gamma_rmp': gamma_rmp,
        'reduction': reduction,
        'w_free_final': w_free[-1],
        'w_rmp_final': w_rmp[-1],
        'w_free_history': w_free,
        'w_rmp_history': w_rmp,
        'success': success
    }
    
    return success, diagnostics


def test_proportional_control(
    Nr: int = 64,
    Nz: int = 128,
    Lr: float = 1.0,
    Lz: float = 2*np.pi,
    eta: float = 1e-3,
    nu: float = 0.0,
    A_max: float = 0.1,
    m: int = 2,
    n: int = 1,
    n_steps: int = 200,
    dt: float = 0.01,
    setpoint: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Test proportional control convergence.
    
    Setup:
    1. Initialize unstable tearing mode (w_0 = 0.05)
    2. Apply proportional control
    3. Check convergence to setpoint
    
    Success Criterion:
    |w_final - setpoint| < 0.005 within 200 steps
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    Lr, Lz : float
        Domain size
    eta : float
        Resistivity
    nu : float
        Viscosity
    A_max : float
        Maximum RMP amplitude
    m, n : int
        Mode numbers
    n_steps : int
        Number of timesteps
    dt : float
        Timestep size
    setpoint : float
        Target island width
    
    Returns
    -------
    success : bool
        True if controller converges
    diagnostics : dict
        Test results
    
    Examples
    --------
    >>> success, diag = test_proportional_control(setpoint=0.01)
    >>> print(f"Final error: {diag['final_error']:.5f}")
    Final error: 0.00234
    """
    # Setup grid
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # Initialize tearing mode
    q_profile = 1.5 + 1.5 * (r / Lr)**2
    psi, omega, r_s = setup_tearing_mode(R, Z, q_profile, r, m=m, n=n, w_0=0.05)
    
    # Controller
    controller = RMPController(m=m, n=n, A_max=A_max, control_type='proportional')
    monitor = TearingModeMonitor(m=m, n=n)
    
    # Evolution with control
    action_history = []
    
    for step in range(n_steps):
        t = step * dt
        
        # Diagnostics
        diag = monitor.update(psi, omega, t, R, Z, q_profile)
        
        # Compute control action
        action = controller.compute_action(diag, setpoint=setpoint)
        action_history.append(action)
        
        # MHD step with RMP
        psi, omega = rk4_step_with_rmp(
            psi, omega, dt, dr, dz, R, eta, nu,
            rmp_amplitude=action,
            m=m, n=n
        )
    
    # Check convergence
    w_final = monitor.w_history[-1]
    final_error = abs(w_final - setpoint)
    success = final_error < 0.005
    
    diagnostics = {
        'w_history': np.array(monitor.w_history),
        'action_history': np.array(action_history),
        'final_error': final_error,
        'w_final': w_final,
        'setpoint': setpoint,
        'success': success
    }
    
    return success, diagnostics


def test_pid_control(
    Nr: int = 64,
    Nz: int = 128,
    Lr: float = 1.0,
    Lz: float = 2*np.pi,
    eta: float = 1e-3,
    nu: float = 0.0,
    A_max: float = 0.1,
    m: int = 2,
    n: int = 1,
    n_steps: int = 200,
    dt: float = 0.01,
    setpoint: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Test PID control with overshoot check.
    
    Setup:
    1. Initialize unstable tearing mode
    2. Apply PID control
    3. Check convergence and overshoot
    
    Success Criteria:
    1. Convergence: |w_final - setpoint| < 0.005
    2. Overshoot: < 20% beyond setpoint
    
    Parameters
    ----------
    Similar to test_proportional_control
    
    Returns
    -------
    success : bool
        True if both criteria met
    diagnostics : dict
        Test results including overshoot
    
    Examples
    --------
    >>> success, diag = test_pid_control(setpoint=0.01)
    >>> print(f"Overshoot: {diag['overshoot']*100:.1f}%")
    Overshoot: 12.3%
    """
    # Setup grid
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # Initialize tearing mode
    q_profile = 1.5 + 1.5 * (r / Lr)**2
    psi, omega, r_s = setup_tearing_mode(R, Z, q_profile, r, m=m, n=n, w_0=0.05)
    
    # PID Controller
    controller = RMPController(m=m, n=n, A_max=A_max, control_type='pid')
    monitor = TearingModeMonitor(m=m, n=n)
    
    # Evolution with PID control
    action_history = []
    
    for step in range(n_steps):
        t = step * dt
        
        # Diagnostics
        diag = monitor.update(psi, omega, t, R, Z, q_profile)
        
        # Compute control action (with time)
        action = controller.compute_action(diag, setpoint=setpoint, t=t)
        action_history.append(action)
        
        # MHD step
        psi, omega = rk4_step_with_rmp(
            psi, omega, dt, dr, dz, R, eta, nu,
            rmp_amplitude=action,
            m=m, n=n
        )
    
    # Check convergence
    w_history = np.array(monitor.w_history)
    w_final = w_history[-1]
    final_error = abs(w_final - setpoint)
    converged = final_error < 0.005
    
    # Check overshoot
    initial_w = w_history[0]
    if setpoint < initial_w:
        # Suppression case: check undershoot
        min_w = np.min(w_history)
        overshoot = max(0, setpoint - min_w) / (initial_w - setpoint)
    else:
        # Tracking case
        max_w = np.max(w_history)
        overshoot = max(0, max_w - setpoint) / (setpoint - initial_w)
    
    overshoot_ok = overshoot < 0.2  # < 20%
    
    success = converged and overshoot_ok
    
    diagnostics = {
        'w_history': w_history,
        'action_history': np.array(action_history),
        'final_error': final_error,
        'w_final': w_final,
        'overshoot': overshoot,
        'converged': converged,
        'overshoot_ok': overshoot_ok,
        'success': success
    }
    
    return success, diagnostics


def test_phase_scan(
    Nr: int = 64,
    Nz: int = 128,
    Lr: float = 1.0,
    Lz: float = 2*np.pi,
    eta: float = 1e-3,
    rmp_amplitude: float = 0.05,
    m: int = 2,
    n_phases: int = 8,
    n_steps: int = 100,
    dt: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Test RMP phase dependence.
    
    Physics:
    RMP effectiveness depends on phase relative to island:
    - Optimal phase minimizes island growth
    - Wrong phase can enhance growth
    
    Test:
    Scan phases [0, 2π] and verify:
    1. Phase variation exists
    2. Optimal phase suppresses better than phase=0
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    Lr, Lz : float
        Domain size
    eta : float
        Resistivity
    rmp_amplitude : float
        RMP amplitude
    m : int
        Mode number
    n_phases : int
        Number of phases to scan
    n_steps : int
        Evolution steps
    dt : float
        Timestep
    
    Returns
    -------
    success : bool
        True if phase dependence observed
    diagnostics : dict
        Phase scan results
    
    Examples
    --------
    >>> success, diag = test_phase_scan(n_phases=8)
    >>> best_phase = diag['phases'][np.argmin(diag['final_widths'])]
    >>> print(f"Optimal phase: {best_phase:.2f} rad")
    Optimal phase: 3.14 rad
    """
    # Setup grid
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    # Initialize tearing mode
    q_profile = 1.5 + 1.5 * (r / Lr)**2
    psi0, omega0, r_s = setup_tearing_mode(R, Z, q_profile, r, m=m, n=1, w_0=0.01)
    
    # Phase scan
    phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)
    final_widths = []
    
    for phase in phases:
        psi = psi0.copy()
        omega = omega0.copy()
        monitor = TearingModeMonitor(m=m, n=1)
        
        for step in range(n_steps):
            t = step * dt
            psi, omega = rk4_step_with_rmp(
                psi, omega, dt, dr, dz, R, eta, 0.0,
                rmp_amplitude=rmp_amplitude,
                rmp_phase=phase,
                m=m, n=1
            )
            monitor.update(psi, omega, t, R, Z, q_profile)
        
        final_widths.append(monitor.w_history[-1])
    
    final_widths = np.array(final_widths)
    
    # Check phase variation
    variation = (np.max(final_widths) - np.min(final_widths)) / np.mean(final_widths)
    phase_dependence_exists = variation > 0.1  # 10% variation
    
    # Check optimal phase better than phase=0
    w_phase0 = final_widths[0]
    w_optimal = np.min(final_widths)
    improvement = (w_phase0 - w_optimal) / w_phase0
    has_improvement = improvement > 0.05  # 5% improvement
    
    success = phase_dependence_exists and has_improvement
    
    diagnostics = {
        'phases': phases,
        'final_widths': final_widths,
        'variation': variation,
        'w_phase0': w_phase0,
        'w_optimal': w_optimal,
        'optimal_phase': phases[np.argmin(final_widths)],
        'improvement': improvement,
        'success': success
    }
    
    return success, diagnostics


# =============================================================================
# Performance Benchmarks
# =============================================================================

def benchmark_rmp_overhead(
    Nr: int = 64,
    Nz: int = 128,
    n_steps: int = 100
) -> Dict:
    """
    Measure RMP computation overhead.
    
    Compares:
    - Baseline: no RMP
    - With RMP: full RMP computation
    
    Acceptable overhead: < 10%
    
    Parameters
    ----------
    Nr, Nz : int
        Grid resolution
    n_steps : int
        Number of steps to benchmark
    
    Returns
    -------
    diagnostics : dict
        Benchmark results
    
    Examples
    --------
    >>> diag = benchmark_rmp_overhead(Nr=64, Nz=128, n_steps=100)
    >>> print(f"RMP overhead: {diag['overhead']:.1f}%")
    RMP overhead: 7.3%
    """
    # Setup
    Lr, Lz = 1.0, 2*np.pi
    r = np.linspace(0, Lr, Nr)
    z = np.linspace(0, Lz, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    
    q_profile = 1.5 + 1.5 * (r / Lr)**2
    psi0, omega0, r_s = setup_tearing_mode(R, Z, q_profile, r, m=2, n=1, w_0=0.01)
    
    eta, nu, dt = 1e-3, 0.0, 0.01
    
    # Baseline (no RMP)
    psi = psi0.copy()
    omega = omega0.copy()
    
    t_start = time.time()
    for _ in range(n_steps):
        psi, omega = rk4_step_with_rmp(psi, omega, dt, dr, dz, R, eta, nu, rmp_amplitude=0.0)
    t_baseline = time.time() - t_start
    
    # With RMP
    psi = psi0.copy()
    omega = omega0.copy()
    
    t_start = time.time()
    for _ in range(n_steps):
        psi, omega = rk4_step_with_rmp(psi, omega, dt, dr, dz, R, eta, nu, rmp_amplitude=0.05)
    t_rmp = time.time() - t_start
    
    # Compute overhead
    overhead = (t_rmp - t_baseline) / t_baseline * 100
    
    diagnostics = {
        't_baseline': t_baseline,
        't_rmp': t_rmp,
        'overhead': overhead,
        'overhead_ok': overhead < 10,
        'time_per_step_baseline': t_baseline / n_steps,
        'time_per_step_rmp': t_rmp / n_steps
    }
    
    return diagnostics
