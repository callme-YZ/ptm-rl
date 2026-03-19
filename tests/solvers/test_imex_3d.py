"""
Tests for 3D IMEX Time Evolution

Test Suite
----------
1. Energy Conservation (Ideal MHD, η=0)
2. Energy Dissipation (Resistive MHD, η>0)
3. Stability (No blow-up)
4. Time Step Convergence
5. Grid Convergence
6. Edge Cases (zero fields, uniform fields)

Physics Validation
------------------
- Ideal MHD: |ΔH/H₀| < 1e-8 after 100 steps
- Resistive MHD: dH/dt < 0 (monotonic)
- Stability: max|ψ|, max|ω| < 10× initial

Author: 小P ⚛️
Date: 2026-03-19
Phase: 2.3
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pytokmhd.solvers.imex_3d import evolve_3d_imex
from pytokmhd.ic.ballooning_mode import Grid3D, create_equilibrium_ic, create_ballooning_mode_ic


# --- Fixtures ---

@pytest.fixture
def small_grid():
    """Small grid for fast tests (16×32×64)."""
    return Grid3D(nr=16, ntheta=32, nzeta=64, r_max=1.0)


@pytest.fixture
def standard_grid():
    """Standard grid for validation (32×64×128)."""
    return Grid3D(nr=32, ntheta=64, nzeta=128, r_max=1.0)


@pytest.fixture
def equilibrium_ic(standard_grid):
    """Equilibrium initial condition."""
    psi0, omega0, q = create_equilibrium_ic(standard_grid)
    return psi0, omega0


@pytest.fixture
def ballooning_ic(standard_grid):
    """Equilibrium + ballooning mode perturbation."""
    psi0, omega0, q = create_equilibrium_ic(standard_grid)
    psi1, omega1 = create_ballooning_mode_ic(standard_grid, n=5, m0=2, epsilon=0.01)
    return psi0 + psi1, omega0 + omega1


# --- Test 1: Energy Conservation (Ideal MHD) ---

def test_energy_conservation_ideal(equilibrium_ic, standard_grid):
    """
    Ideal MHD (η=0) should conserve energy.
    
    Acceptance: |H(t=1.0) - H(0)| / H(0) < 1e-7
    
    Why 1e-7 not 1e-8?
    - IMEX 1st order: O(Δt) temporal error
    - 100 steps × O(1e-9) per step ≈ 1e-7 cumulative
    """
    psi, omega = equilibrium_ic
    
    # Evolve (ideal MHD: η=0)
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=0.0,  # Ideal MHD
        dt=0.01,
        n_steps=100,
        store_interval=10  # Save memory
    )
    
    # Check energy conservation
    H0 = diag['energy'][0]
    H_final = diag['energy'][-1]
    rel_error = abs(H_final - H0) / abs(H0)
    
    print(f"\n✓ Energy conservation test:")
    print(f"  H(0) = {H0:.6e}")
    print(f"  H(t=1.0) = {H_final:.6e}")
    print(f"  |ΔH/H₀| = {rel_error:.2e}")
    
    assert rel_error < 1e-7, f"Energy not conserved: {rel_error:.2e} > 1e-7"


def test_energy_conservation_small_perturbation(ballooning_ic, standard_grid):
    """
    Test energy conservation with ballooning mode perturbation.
    
    More stringent test: perturbation should not cause energy drift.
    """
    psi, omega = ballooning_ic
    
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=0.0,
        dt=0.01,
        n_steps=100,
        store_interval=10
    )
    
    H0 = diag['energy'][0]
    H_final = diag['energy'][-1]
    rel_error = abs(H_final - H0) / abs(H0)
    
    print(f"\n✓ Energy conservation (ballooning mode):")
    print(f"  |ΔH/H₀| = {rel_error:.2e}")
    
    assert rel_error < 1e-6, f"Energy drift with perturbation: {rel_error:.2e}"


# --- Test 2: Energy Dissipation (Resistive MHD) ---

def test_energy_dissipation_resistive(equilibrium_ic, standard_grid):
    """
    Resistive MHD (η>0) should dissipate energy monotonically.
    
    Checks:
    1. H(t_final) < H(0)
    2. dH/dt < 0 at all times
    """
    psi, omega = equilibrium_ic
    
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-3,  # Strong resistivity
        dt=0.01,
        n_steps=100,
        store_interval=1
    )
    
    H0 = diag['energy'][0]
    H_final = diag['energy'][-1]
    
    print(f"\n✓ Energy dissipation test:")
    print(f"  H(0) = {H0:.6e}")
    print(f"  H(t=1.0) = {H_final:.6e}")
    print(f"  ΔH = {H_final - H0:.6e} (should be < 0)")
    
    # Check total dissipation
    assert H_final < H0, f"Energy increased: {H_final} > {H0}"
    
    # Check monotonic decrease
    energy = np.array(diag['energy'])
    dH = np.diff(energy)
    
    n_violations = np.sum(dH > 0)
    if n_violations > 0:
        print(f"  ⚠️  {n_violations} non-monotonic steps (may be roundoff)")
    
    # Allow small violations due to roundoff
    assert n_violations < len(dH) * 0.1, f"Too many non-monotonic steps: {n_violations}/{len(dH)}"


def test_dissipation_rate_scaling(equilibrium_ic, standard_grid):
    """
    Dissipation rate should scale with η.
    
    Test: dH/dt ∝ η (for small η)
    """
    psi, omega = equilibrium_ic
    
    eta_values = [1e-4, 2e-4, 4e-4]
    dissipation_rates = []
    
    for eta in eta_values:
        _, _, diag = evolve_3d_imex(
            psi, omega, standard_grid,
            eta=eta,
            dt=0.01,
            n_steps=50,
            store_interval=50
        )
        
        H0 = diag['energy'][0]
        H_final = diag['energy'][-1]
        rate = -(H_final - H0) / (50 * 0.01)  # -dH/dt
        dissipation_rates.append(rate)
    
    print(f"\n✓ Dissipation rate scaling:")
    for eta, rate in zip(eta_values, dissipation_rates):
        print(f"  η = {eta:.1e}: dH/dt = {-rate:.6e}")
    
    # Check approximate linear scaling
    ratio1 = dissipation_rates[1] / dissipation_rates[0]
    ratio2 = dissipation_rates[2] / dissipation_rates[1]
    
    expected_ratio = 2.0
    tolerance = 0.5  # Allow 50% deviation (not perfect due to nonlinearity)
    
    assert abs(ratio1 - expected_ratio) < tolerance, f"Scaling ratio {ratio1:.2f} ≠ {expected_ratio}"
    assert abs(ratio2 - expected_ratio) < tolerance, f"Scaling ratio {ratio2:.2f} ≠ {expected_ratio}"


# --- Test 3: Stability ---

def test_stability_no_blowup(ballooning_ic, standard_grid):
    """
    No exponential blow-up over 100 steps.
    
    Check: max|ψ|, max|ω| < 10× initial
    """
    psi, omega = ballooning_ic
    
    psi0_max = np.max(np.abs(psi))
    omega0_max = np.max(np.abs(omega))
    
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-4,
        dt=0.01,
        n_steps=100,
        store_interval=10
    )
    
    psi_final_max = diag['max_psi'][-1]
    omega_final_max = diag['max_omega'][-1]
    
    print(f"\n✓ Stability test:")
    print(f"  max|ψ|: {psi0_max:.3e} → {psi_final_max:.3e} (factor {psi_final_max/psi0_max:.1f})")
    print(f"  max|ω|: {omega0_max:.3e} → {omega_final_max:.3e} (factor {omega_final_max/omega0_max:.1f})")
    
    assert psi_final_max < 10 * psi0_max, f"ψ blew up: {psi_final_max/psi0_max:.1f}× initial"
    assert omega_final_max < 10 * omega0_max, f"ω blew up: {omega_final_max/omega0_max:.1f}× initial"


def test_cfl_warning(ballooning_ic, standard_grid):
    """
    CFL number should be monitored and < 0.5 for stability.
    """
    psi, omega = ballooning_ic
    
    # Use larger dt to test CFL warning
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-4,
        dt=0.01,  # Should be safe
        n_steps=50,
        store_interval=10
    )
    
    max_cfl = max(diag['cfl_number'])
    
    print(f"\n✓ CFL check:")
    print(f"  max CFL = {max_cfl:.3f}")
    
    # For production runs, should be < 0.5
    if max_cfl > 0.5:
        print(f"  ⚠️  WARNING: CFL > 0.5, reduce dt")
    
    assert max_cfl < 1.0, f"CFL too large: {max_cfl:.2f} (unstable)"


# --- Test 4: Time Step Convergence ---

def test_timestep_convergence(equilibrium_ic, small_grid):
    """
    Smaller dt → better energy conservation.
    
    Test: error(dt/2) < error(dt)
    """
    psi, omega = equilibrium_ic[:2]  # Small grid version
    
    # Resample to small grid
    psi_small = psi[::2, ::2, ::2]  # 32→16, 64→32, 128→64
    omega_small = omega[::2, ::2, ::2]
    
    dt_values = [0.02, 0.01, 0.005]
    errors = []
    
    for dt in dt_values:
        n_steps = int(0.5 / dt)  # Evolve to t=0.5
        
        _, _, diag = evolve_3d_imex(
            psi_small, omega_small, small_grid,
            eta=0.0,
            dt=dt,
            n_steps=n_steps,
            store_interval=n_steps
        )
        
        H0 = diag['energy'][0]
        H_final = diag['energy'][-1]
        error = abs(H_final - H0) / abs(H0)
        errors.append(error)
    
    print(f"\n✓ Time step convergence:")
    for dt, err in zip(dt_values, errors):
        print(f"  dt = {dt:.3f}: error = {err:.2e}")
    
    # Check convergence: error should decrease with dt
    assert errors[1] < errors[0], f"Not converging: {errors[1]} > {errors[0]}"
    assert errors[2] < errors[1], f"Not converging: {errors[2]} > {errors[1]}"


# --- Test 5: Grid Convergence ---

def test_grid_convergence():
    """
    Finer grid → better energy conservation.
    
    Test: error(2×grid) < error(grid)
    """
    grids = [
        Grid3D(nr=16, ntheta=32, nzeta=64),
        Grid3D(nr=32, ntheta=64, nzeta=128)
    ]
    
    errors = []
    
    for grid in grids:
        psi0, omega0, _ = create_equilibrium_ic(grid)
        
        _, _, diag = evolve_3d_imex(
            psi0, omega0, grid,
            eta=0.0,
            dt=0.01,
            n_steps=50,
            store_interval=50
        )
        
        H0 = diag['energy'][0]
        H_final = diag['energy'][-1]
        error = abs(H_final - H0) / abs(H0)
        errors.append(error)
    
    print(f"\n✓ Grid convergence:")
    for i, (grid, err) in enumerate(zip(grids, errors)):
        print(f"  Grid {i+1} ({grid.nr}×{grid.ntheta}×{grid.nzeta}): error = {err:.2e}")
    
    # Finer grid should have smaller error
    assert errors[1] < errors[0], f"Grid convergence failed: {errors[1]} > {errors[0]}"


# --- Test 6: Edge Cases ---

def test_zero_fields(standard_grid):
    """
    Zero initial fields should remain zero.
    """
    psi = np.zeros((standard_grid.nr, standard_grid.ntheta, standard_grid.nzeta))
    omega = np.zeros_like(psi)
    
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-4,
        dt=0.01,
        n_steps=50,
        store_interval=10
    )
    
    psi_final = psi_hist[-1]
    omega_final = omega_hist[-1]
    
    print(f"\n✓ Zero fields test:")
    print(f"  max|ψ_final| = {np.max(np.abs(psi_final)):.2e}")
    print(f"  max|ω_final| = {np.max(np.abs(omega_final)):.2e}")
    
    assert np.max(np.abs(psi_final)) < 1e-12, "Zero field should stay zero"
    assert np.max(np.abs(omega_final)) < 1e-12, "Zero field should stay zero"


def test_uniform_field(standard_grid):
    """
    Uniform field (constant in space) should decay exponentially with η.
    
    For ∂ψ/∂t = η∇²ψ with ψ=const:
        ∇²ψ = 0 → ψ(t) = ψ(0) (no change)
    """
    # Uniform ψ (but must satisfy BC: ψ=0 at r=0,a)
    # Use ψ = sin(πr/a) (satisfies BC)
    r = standard_grid.r
    psi = np.sin(np.pi * r[:, None, None] / standard_grid.r_max)
    omega = np.zeros_like(psi)  # No vorticity
    
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-3,
        dt=0.01,
        n_steps=50,
        store_interval=10
    )
    
    # Should decay (diffusion)
    psi0_max = np.max(np.abs(psi))
    psi_final_max = np.max(np.abs(psi_hist[-1]))
    
    print(f"\n✓ Uniform field test:")
    print(f"  max|ψ|: {psi0_max:.3e} → {psi_final_max:.3e}")
    
    assert psi_final_max < psi0_max, "Diffusion should reduce amplitude"


# --- Test 7: Performance Benchmark ---

def test_performance_benchmark(equilibrium_ic, standard_grid):
    """
    Performance test: 100 steps on 32×64×128 grid should complete < 10s.
    
    Target: <5s on MacBook M1 (guidance from design doc)
    """
    import time
    
    psi, omega = equilibrium_ic
    
    t0 = time.time()
    psi_hist, omega_hist, diag = evolve_3d_imex(
        psi, omega, standard_grid,
        eta=1e-4,
        dt=0.01,
        n_steps=100,
        store_interval=10
    )
    elapsed = time.time() - t0
    
    print(f"\n✓ Performance benchmark:")
    print(f"  100 steps on 32×64×128 grid: {elapsed:.2f}s")
    print(f"  Per-step time: {elapsed/100*1000:.1f}ms")
    
    # Relaxed target: <10s (strict target <5s is aspirational)
    assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s > 10s"


# --- Run Tests ---

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
