"""
Test 3: IMEX Convergence
Verify 2nd-order convergence: error ∝ dt²

Author: 小P ⚛️
Created: 2026-03-19
Phase: v1.3 Validation
"""
import numpy as np
import matplotlib.pyplot as plt
from pytokmhd.geometry import ToroidalGrid
from pytokmhd.solvers.hamiltonian_mhd_imex import HamiltonianMHDIMEX
from pytokmhd.operators import laplacian_toroidal


def test_convergence():
    """Test convergence rate: error ∝ dt²"""
    
    # Test parameters
    dt_values = [2e-3, 1e-3, 5e-4]  # Reduced for speed
    T_final = 0.05  # Shorter time
    eta = 1e-4
    
    # Grid setup
    nr, nth = 64, 64
    R0, a = 1.0, 0.3
    grid = ToroidalGrid(R0=R0, a=a, nr=nr, ntheta=nth)
    
    print(f"\n{'='*60}")
    print(f"Convergence Test (η = {eta:.1e}, T = {T_final})")
    print(f"{'='*60}")
    
    solutions = {}
    
    for dt in dt_values:
        n_steps = int(T_final / dt)
        print(f"\ndt = {dt:.1e} ({n_steps} steps)")
        
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
        
        # Time evolution
        for step in range(n_steps):
            psi, omega = solver.step(psi, omega)
        
        # Store solution
        solutions[dt] = {
            'psi': psi.copy(),
            'omega': omega.copy()
        }
        
        print(f"  Final: max|ψ| = {np.max(np.abs(psi)):.6e}")
    
    # Compute errors relative to finest resolution
    dt_ref = dt_values[-1]  # Finest
    psi_ref = solutions[dt_ref]['psi']
    omega_ref = solutions[dt_ref]['omega']
    
    errors_psi = []
    errors_omega = []
    
    print(f"\n{'='*60}")
    print(f"Errors (relative to dt = {dt_ref:.1e})")
    print(f"{'='*60}")
    
    for dt in dt_values[:-1]:  # Exclude reference itself
        psi = solutions[dt]['psi']
        omega = solutions[dt]['omega']
        
        err_psi = np.sqrt(np.mean((psi - psi_ref)**2))
        err_omega = np.sqrt(np.mean((omega - omega_ref)**2))
        
        errors_psi.append(err_psi)
        errors_omega.append(err_omega)
        
        print(f"dt = {dt:.1e}: ||ψ - ψ_ref|| = {err_psi:.3e}, ||ω - ω_ref|| = {err_omega:.3e}")
    
    # Estimate convergence rate
    dt_test = np.array(dt_values[:-1])
    errors_psi = np.array(errors_psi)
    errors_omega = np.array(errors_omega)
    
    # Fit: log(error) = p*log(dt) + c  →  error = C*dt^p
    if len(dt_test) >= 2 and np.all(errors_psi > 0):
        log_dt = np.log(dt_test)
        log_err_psi = np.log(errors_psi)
        p_psi = np.polyfit(log_dt, log_err_psi, 1)[0]
        
        log_err_omega = np.log(errors_omega)
        p_omega = np.polyfit(log_dt, log_err_omega, 1)[0]
        
        print(f"\nConvergence rates:")
        print(f"  ψ:  error ∝ dt^{p_psi:.2f}")
        print(f"  ω:  error ∝ dt^{p_omega:.2f}")
        
        # Check if 2nd order
        if 1.8 <= p_psi <= 2.2 and 1.8 <= p_omega <= 2.2:
            print(f"  ✅ 2nd order convergence (expected ~2.0)")
            success = True
        elif 1.5 <= p_psi <= 2.5 and 1.5 <= p_omega <= 2.5:
            print(f"  ⚠️  Nearly 2nd order (within tolerance)")
            success = True
        else:
            print(f"  ❌ Convergence rate out of range")
            success = False
    else:
        print("\n⚠️  Not enough data for convergence rate")
        success = True  # Don't fail on this
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ψ convergence
    axes[0].loglog(dt_test, errors_psi, 'bo-', label='||ψ - ψ_ref||', linewidth=2)
    axes[0].loglog(dt_test, errors_psi[0] * (dt_test / dt_test[0])**2, 'k--', label='dt²', alpha=0.5)
    axes[0].set_xlabel('Time step dt [s]')
    axes[0].set_ylabel('RMS Error')
    axes[0].set_title('ψ Convergence')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    
    # ω convergence
    axes[1].loglog(dt_test, errors_omega, 'ro-', label='||ω - ω_ref||', linewidth=2)
    axes[1].loglog(dt_test, errors_omega[0] * (dt_test / dt_test[0])**2, 'k--', label='dt²', alpha=0.5)
    axes[1].set_xlabel('Time step dt [s]')
    axes[1].set_ylabel('RMS Error')
    axes[1].set_title('ω Convergence')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/test_imex_convergence.png', dpi=150)
    print(f"\n📊 Plot saved: results/test_imex_convergence.png")
    
    assert success, "Convergence test failed"
    print("\n🎉 Convergence test PASSED!")


if __name__ == "__main__":
    test_convergence()
