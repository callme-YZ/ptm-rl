"""
Test 2: Energy Budget
Verify energy dissipation: dH/dt = -η∫J² dV

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Validation
"""
import numpy as np
import matplotlib.pyplot as plt
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal
from pytokmhd.physics import compute_current_density, compute_hamiltonian
from pytokmhd.equilibrium import load_solovev_equilibrium


def test_energy_budget():
    """Test energy budget: dH/dt = -η∫J² dV"""
    
    # Test parameters
    eta = 1e-4
    n_steps = 200
    dt = 1e-3
    
    # Grid setup
    nr, nth = 64, 64
    R0, a = 1.0, 0.3
    grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=nth)
    
    print(f"\n{'='*60}")
    print(f"Energy Budget Test (η = {eta:.1e})")
    print(f"{'='*60}")
    
    # Initialize solver
    solver = HamiltonianMHDIMEX(
        grid,
        dt=dt,
        eta=eta,
        nu=0.0,
        use_imex=True,
        verbose=False
    )
    
    # Initial condition: simple parabolic + perturbation
    psi0 = grid.r_grid**2 * (1 - grid.r_grid/a)**2
    pert = 0.01 * np.sin(2*grid.theta_grid) * grid.r_grid**2 * (1 - grid.r_grid/a)**2
    psi = psi0 + pert
    omega = -laplacian_toroidal(psi, grid)
    
    # Storage
    time_history = []
    energy_history = []
    dH_dt_numerical = []
    dH_dt_theory = []
    J2_history = []
    
    # Volume element
    dV = (R0 + grid.r_grid * np.cos(grid.theta_grid)) * grid.dr * grid.dtheta
    
    # Initial energy
    H0 = compute_hamiltonian(psi, omega, grid)
    energy_history.append(H0)
    time_history.append(0.0)
    
    print(f"Initial energy: H0 = {H0:.6e}")
    
    # Time evolution
    for step in range(n_steps):
        # Compute J before step
        J = compute_current_density(psi, grid)
        J2_int = np.sum(J**2 * dV)
        J2_history.append(J2_int)
        
        # Step
        psi, omega = solver.step(psi, omega)
        
        # Energy
        H = compute_hamiltonian(psi, omega, grid)
        energy_history.append(H)
        time_history.append((step+1) * dt)
        
        # Dissipation rate
        dH_numerical = (H - energy_history[-2]) / dt
        dH_theory = -eta * J2_int
        dH_dt_numerical.append(dH_numerical)
        dH_dt_theory.append(dH_theory)
        
        if step % 40 == 0:
            rel_err = abs((dH_numerical - dH_theory) / dH_theory) if dH_theory != 0 else 0
            print(f"Step {step:3d}: dH/dt = {dH_numerical:.3e} (theory: {dH_theory:.3e}, error: {rel_err:.1%})")
    
    # Convert to arrays
    time_history = np.array(time_history)
    energy_history = np.array(energy_history)
    dH_dt_numerical = np.array(dH_dt_numerical)
    dH_dt_theory = np.array(dH_dt_theory)
    J2_history = np.array(J2_history)
    
    # Check energy always decreases
    energy_diff = np.diff(energy_history)
    if np.all(energy_diff <= 0):
        print("\n✅ Energy monotonically decreasing")
    else:
        increases = np.sum(energy_diff > 0)
        print(f"\n⚠️  Energy increased {increases} times")
    
    # Check relative error
    valid_idx = np.abs(dH_dt_theory) > 1e-10  # Avoid division by zero
    if np.sum(valid_idx) > 0:
        rel_errors = np.abs((dH_dt_numerical[valid_idx] - dH_dt_theory[valid_idx]) / dH_dt_theory[valid_idx])
        mean_error = np.mean(rel_errors)
        max_error = np.max(rel_errors)
        
        print(f"\nDissipation rate error:")
        print(f"  Mean: {mean_error:.1%}")
        print(f"  Max:  {max_error:.1%}")
        
        if mean_error < 0.05:
            print("  ✅ Theory match within 5%")
            success = True
        elif mean_error < 0.10:
            print("  ⚠️  Theory match within 10%")
            success = True
        else:
            print("  ❌ Theory error > 10%")
            success = False
    else:
        print("\n⚠️  Theory values too small to compare")
        success = True
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Energy vs time
    axes[0, 0].plot(time_history, energy_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Hamiltonian H')
    axes[0, 0].set_title('Energy Evolution')
    axes[0, 0].grid(True)
    
    # dH/dt comparison
    axes[0, 1].plot(time_history[1:], dH_dt_numerical, 'b-', label='Numerical', alpha=0.7)
    axes[0, 1].plot(time_history[1:], dH_dt_theory, 'r--', label='Theory', alpha=0.7)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('dH/dt')
    axes[0, 1].set_title('Dissipation Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Relative error
    if np.sum(valid_idx) > 0:
        axes[1, 0].semilogy(time_history[1:][valid_idx], rel_errors, 'g-', linewidth=1)
        axes[1, 0].axhline(0.05, color='orange', linestyle='--', label='5% tolerance')
        axes[1, 0].axhline(0.10, color='red', linestyle='--', label='10% tolerance')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Relative Error')
        axes[1, 0].set_title('|(dH/dt)_num - (dH/dt)_theory| / |(dH/dt)_theory|')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # J² integral
    axes[1, 1].plot(time_history[1:], J2_history, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('∫J² dV')
    axes[1, 1].set_title('Current Density Squared')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/test_imex_energy_budget.png', dpi=150)
    print(f"\n📊 Plot saved: results/test_imex_energy_budget.png")
    
    assert success, "Energy budget test failed"
    print("\n🎉 Energy budget test PASSED!")


if __name__ == "__main__":
    test_energy_budget()
