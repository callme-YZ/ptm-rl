"""
Energy Conservation Diagnostics for v1.2.1 Benchmark

Implements energy tracking and drift measurement for validating
PyTokEq equilibrium quality.

Author: 小P ⚛️
Date: 2026-03-18
Phase: v1.2.1 Phase 1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import warnings


@dataclass
class EnergyMetrics:
    """Container for energy conservation metrics"""
    time: float
    total_energy: float
    magnetic_energy: float
    kinetic_energy: float
    dissipation_cumulative: float
    drift_relative: float
    drift_absolute: float


class EnergyConservationTracker:
    """
    Track energy conservation during MHD evolution
    
    Implements Phase 0 monitoring configuration:
    - Total energy: every step
    - Energy components: every 50 steps
    - Alerts on threshold violations
    
    Attributes:
        E0: Initial total energy (reference)
        times: Array of time values
        energies: Array of total energies
        metrics: List of EnergyMetrics
        thresholds: Alert thresholds from Phase 0
    """
    
    def __init__(
        self,
        E0: float,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize tracker
        
        Args:
            E0: Initial total energy (t=0)
            thresholds: Alert thresholds (defaults from Phase 0)
        """
        self.E0 = E0
        self.times: List[float] = [0.0]
        self.energies: List[float] = [E0]
        self.metrics: List[EnergyMetrics] = []
        
        # Default thresholds from Phase 0 design
        self.thresholds = thresholds or {
            'drift_tier1': 0.01,    # 1%
            'drift_tier2': 0.05,    # 5%
            'drift_fail': 0.10,     # 10%
        }
        
        self.warnings_issued = {
            'tier1': False,
            'tier2': False,
            'fail': False
        }
        
    def record_step(
        self,
        t: float,
        E_total: float,
        E_mag: Optional[float] = None,
        E_kin: Optional[float] = None,
        dissipation: Optional[float] = None
    ):
        """
        Record energy at current timestep
        
        Args:
            t: Current time
            E_total: Total energy
            E_mag: Magnetic energy (optional)
            E_kin: Kinetic energy (optional)
            dissipation: Cumulative dissipation (optional)
        """
        self.times.append(t)
        self.energies.append(E_total)
        
        # Compute drift
        drift_abs = abs(E_total - self.E0)
        drift_rel = drift_abs / abs(self.E0)
        
        # Check thresholds and issue warnings
        self._check_alerts(drift_rel, t)
        
        # Store detailed metrics if components provided
        if E_mag is not None or E_kin is not None:
            metric = EnergyMetrics(
                time=t,
                total_energy=E_total,
                magnetic_energy=E_mag or 0.0,
                kinetic_energy=E_kin or 0.0,
                dissipation_cumulative=dissipation or 0.0,
                drift_relative=drift_rel,
                drift_absolute=drift_abs
            )
            self.metrics.append(metric)
    
    def _check_alerts(self, drift: float, t: float):
        """Check drift against thresholds and issue warnings"""
        
        if drift > self.thresholds['drift_fail'] and not self.warnings_issued['fail']:
            warnings.warn(
                f"⚠️ CRITICAL: Energy drift {drift:.1%} > 10% at t={t:.3f}. "
                f"Investigation required!",
                category=RuntimeWarning
            )
            self.warnings_issued['fail'] = True
            
        elif drift > self.thresholds['drift_tier2'] and not self.warnings_issued['tier2']:
            warnings.warn(
                f"⚠️ Energy drift {drift:.1%} > 5% (Tier 2) at t={t:.3f}. "
                f"Still acceptable but not ideal.",
                category=RuntimeWarning
            )
            self.warnings_issued['tier2'] = True
            
        elif drift > self.thresholds['drift_tier1'] and not self.warnings_issued['tier1']:
            print(f"ℹ️ Energy drift {drift:.1%} > 1% (Tier 1 missed) at t={t:.3f}")
            self.warnings_issued['tier1'] = True
    
    def get_final_drift(self) -> Tuple[float, float]:
        """
        Get final energy drift
        
        Returns:
            (drift_relative, drift_absolute)
        """
        if len(self.energies) == 0:
            return 0.0, 0.0
        
        E_final = self.energies[-1]
        drift_abs = abs(E_final - self.E0)
        drift_rel = drift_abs / abs(self.E0)
        
        return drift_rel, drift_abs
    
    def get_max_drift(self) -> Tuple[float, float, float]:
        """
        Get maximum energy drift over evolution
        
        Returns:
            (max_drift_relative, max_drift_absolute, time_of_max)
        """
        if len(self.energies) == 0:
            return 0.0, 0.0, 0.0
        
        energies = np.array(self.energies)
        times = np.array(self.times)
        
        drifts_abs = np.abs(energies - self.E0)
        drifts_rel = drifts_abs / abs(self.E0)
        
        idx_max = np.argmax(drifts_rel)
        
        return drifts_rel[idx_max], drifts_abs[idx_max], times[idx_max]
    
    def check_monotonic_decrease(self) -> Tuple[bool, List[int]]:
        """
        Check if energy monotonically decreases (dissipative system)
        
        Phase 2.5 Check 2 requirement
        
        Returns:
            (is_monotonic, violation_indices)
        """
        if len(self.energies) < 2:
            return True, []
        
        energies = np.array(self.energies)
        dE = np.diff(energies)
        
        violations = np.where(dE > 0)[0].tolist()
        is_monotonic = len(violations) == 0
        
        return is_monotonic, violations
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics for reporting
        
        Returns:
            Dictionary with all key metrics
        """
        drift_final, drift_final_abs = self.get_final_drift()
        drift_max, drift_max_abs, t_max = self.get_max_drift()
        monotonic, violations = self.check_monotonic_decrease()
        
        # Determine success tier
        if drift_final < self.thresholds['drift_tier1']:
            tier = 1
            status = "EXCELLENT"
        elif drift_final < self.thresholds['drift_tier2']:
            tier = 2
            status = "ACCEPTABLE"
        else:
            tier = 3
            status = "NEEDS_DEBUG"
        
        return {
            'drift_final': drift_final,
            'drift_final_abs': drift_final_abs,
            'drift_max': drift_max,
            'drift_max_abs': drift_max_abs,
            'time_of_max_drift': t_max,
            'n_steps': len(self.energies) - 1,
            'total_time': self.times[-1] if self.times else 0.0,
            'monotonic_decrease': monotonic,
            'n_violations': len(violations),
            'tier': tier,
            'status': status,
            'E0': self.E0,
            'E_final': self.energies[-1] if self.energies else self.E0
        }
    
    def get_evolution_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get energy evolution curve for plotting
        
        Returns:
            (times, energies_normalized)
        """
        times = np.array(self.times)
        energies = np.array(self.energies) / self.E0  # Normalize
        
        return times, energies


def compute_total_energy(
    psi: np.ndarray,
    omega: np.ndarray,
    grid
) -> Tuple[float, float, float]:
    """
    Compute total energy (magnetic + kinetic)
    
    E_total = E_mag + E_kin
    E_mag = ∫ |∇ψ|² / (2μ₀) dV
    E_kin = ∫ ω² / 2 dV
    
    Args:
        psi: Poloidal flux (nr, ntheta)
        omega: Vorticity (nr, ntheta)
        grid: ToroidalGrid instance
        
    Returns:
        (E_total, E_mag, E_kin)
    """
    from ..operators.toroidal_operators import gradient_toroidal
    
    # Magnetic energy: |∇ψ|²
    grad_psi_r, grad_psi_theta = gradient_toroidal(psi, grid)
    grad_psi_sq = grad_psi_r**2 + grad_psi_theta**2
    
    # Integration with Jacobian
    dV = grid.jacobian() * grid.dr * grid.dtheta
    E_mag = 0.5 * np.sum(grad_psi_sq * dV)
    
    # Kinetic energy: ω²
    E_kin = 0.5 * np.sum(omega**2 * dV)
    
    E_total = E_mag + E_kin
    
    return E_total, E_mag, E_kin


def track_energy_drift(
    solver,
    n_steps: int,
    monitor_components_every: int = 50,
    verbose: bool = True
) -> EnergyConservationTracker:
    """
    Run evolution and track energy drift
    
    Implements Phase 0 monitoring configuration:
    - Energy: every step
    - Components: every 50 steps
    
    Args:
        solver: MHD solver instance (must have .step() method)
        n_steps: Number of steps to evolve
        monitor_components_every: Frequency for component monitoring
        verbose: Print progress
        
    Returns:
        EnergyConservationTracker with results
    """
    # Get initial state
    psi0 = solver.psi
    omega0 = solver.omega
    grid = solver.grid
    
    # Compute initial energy
    E0_total, E0_mag, E0_kin = compute_total_energy(psi0, omega0, grid)
    
    if verbose:
        print(f"Initial energy: E0 = {E0_total:.6e}")
        print(f"  E_mag = {E0_mag:.6e}, E_kin = {E0_kin:.6e}")
    
    # Initialize tracker
    tracker = EnergyConservationTracker(E0_total)
    
    # Record initial state
    tracker.record_step(
        t=0.0,
        E_total=E0_total,
        E_mag=E0_mag,
        E_kin=E0_kin,
        dissipation=0.0
    )
    
    # Evolution loop
    t0 = time.time()
    
    for step in range(n_steps):
        # Evolve one step
        solver.step()
        
        # Get current state
        psi = solver.psi
        omega = solver.omega
        t = solver.time
        
        # Monitor energy
        E_total, E_mag, E_kin = compute_total_energy(psi, omega, grid)
        
        # Record
        if (step + 1) % monitor_components_every == 0:
            # Full metrics
            tracker.record_step(
                t=t,
                E_total=E_total,
                E_mag=E_mag,
                E_kin=E_kin
            )
        else:
            # Just total energy
            tracker.record_step(t=t, E_total=E_total)
        
        # Progress reporting
        if verbose and (step + 1) % 100 == 0:
            drift, _ = tracker.get_final_drift()
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (n_steps - step - 1)
            print(f"Step {step+1}/{n_steps} | "
                  f"t={t:.3f} | "
                  f"drift={drift:.2%} | "
                  f"ETA={eta:.1f}s")
    
    if verbose:
        total_time = time.time() - t0
        print(f"\nEvolution complete in {total_time:.1f}s")
        print(f"Steps/second: {n_steps/total_time:.1f}")
    
    return tracker


def plot_energy_evolution(
    tracker: EnergyConservationTracker,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot energy evolution curve
    
    Args:
        tracker: EnergyConservationTracker instance
        save_path: Path to save figure (optional)
        show: Display plot
    """
    import matplotlib.pyplot as plt
    
    times, E_norm = tracker.get_evolution_curve()
    drift_final, _ = tracker.get_final_drift()
    summary = tracker.get_summary()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy
    ax.plot(times, E_norm, 'b-', linewidth=2, label='E(t) / E(0)')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='E(0)')
    
    # Tier thresholds
    ax.axhline(1 - tracker.thresholds['drift_tier1'], 
               color='g', linestyle=':', alpha=0.5, label='Tier 1 (1%)')
    ax.axhline(1 - tracker.thresholds['drift_tier2'], 
               color='orange', linestyle=':', alpha=0.5, label='Tier 2 (5%)')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('E(t) / E(0)', fontsize=12)
    ax.set_title(f'Energy Conservation: drift = {drift_final:.2%} ({summary["status"]})', 
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
