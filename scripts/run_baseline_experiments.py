"""
Run baseline controller experiments for Issue #28

Compares:
- No control
- Random control  
- PID control

Metrics:
- Success rate (3 tiers)
- Energy efficiency
- Stability
- Control effort

Author: 小A 🤖
Date: 2026-03-24
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from pytokmhd.rl.hamiltonian_env import make_hamiltonian_mhd_env
from pytokmhd.rl.classical_controllers import make_baseline_agent


def reset_tearing_mode(env, seed=None):
    """
    Reset environment with Harris sheet tearing mode IC.
    
    Physics (小P ⚛️, Issue #29):
    - Harris sheet equilibrium + m=1 tearing perturbation
    - Expected growth: ~11% in 0.1s (γ ≈ 1.05 s⁻¹)
    - Proper unstable tearing mode (FKR theory)
    
    Fixes Issue #28 IC problem (was decay, now growth).
    """
    import jax.numpy as jnp
    from pim_rl.physics.v2.tearing_ic import create_tearing_ic
    
    # Set seed if provided
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    
    # Create Harris sheet tearing IC (Issue #29 ⚛️)
    nr, ntheta = env.grid.nr, env.grid.ntheta
    psi, phi = create_tearing_ic(nr=nr, ntheta=ntheta)
    
    # Convert to JAX arrays
    psi = jnp.array(psi, dtype=jnp.float32)
    phi = jnp.array(phi, dtype=jnp.float32)
    
    # Initialize solver
    env.mhd_solver.initialize(psi, phi)
    
    # Compute initial observation
    obs = env.obs_computer.compute_observation(psi, phi)
    
    # Initialize observation cache (Issue #26 sparse obs mode)
    env._last_obs = obs
    env._last_psi = psi
    env._last_phi = phi
    env.psi = psi
    env.phi = phi
    
    # Reset counters
    env.current_step = 0
    env.obs_computer.reset()
    
    return obs, {}


def run_episode(env, agent, seed=None, record_every=100):
    """
    Run single episode and collect diagnostics.
    
    Performance optimization (小P ⚛️):
    - Only compute full observation every N steps
    - Drastically reduces Poisson solver calls
    - ~100× speedup for baseline experiments
    
    Parameters
    ----------
    record_every : int
        Compute full observation every N steps (default: 100)
        Set to 1 for full recording (slow!)
    
    Returns
    -------
    diagnostics : dict
        m1_amp_history, H_history, action_history, etc.
    """
    # Use custom tearing mode reset (小P fix ⚛️)
    obs, info = reset_tearing_mode(env, seed=seed)
    agent.reset()
    
    # Record initial
    # obs[7] = m=0, obs[8] = m=1 (小P correction ⚛️)
    m1_initial = np.abs(obs[8])
    H_initial = obs[0]
    
    # Storage (sparse recording)
    m1_amps = [m1_initial]
    H_values = [H_initial]
    actions = []
    rewards = []
    record_steps = [0]
    
    done = False
    step = 0
    
    while not done and step < env.max_steps:
        # Get action
        action = agent.act(obs)
        
        # Step environment (Issue #26 fix: sparse obs for speed ⚛️)
        # Only compute full observation at recording intervals
        need_obs = ((step + 1) % record_every == 0) or (step + 1 >= env.max_steps)
        obs_next, reward, terminated, truncated, info = env.step(action, compute_obs=need_obs)
        done = terminated or truncated
        
        # Always record action & reward (cheap)
        actions.append(action.copy())
        rewards.append(reward)
        
        # Record observation when computed
        if need_obs:
            # obs[7] = m=0, obs[8] = m=1 (小P correction ⚛️)
            m1_amp = np.abs(obs_next[8])
            H = obs_next[0]
            
            m1_amps.append(m1_amp)
            H_values.append(H)
            record_steps.append(step + 1)
        
        obs = obs_next
        step += 1
    
    return {
        'm1_amp': np.array(m1_amps),
        'H': np.array(H_values),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'steps': step,
        'record_steps': np.array(record_steps)
    }


def compute_metrics(diagnostics):
    """
    Compute performance metrics from episode diagnostics.
    
    Works with sparse observation recording (小P optimization ⚛️)
    
    Parameters
    ----------
    diagnostics : dict
        Episode data (possibly sparse)
        
    Returns
    -------
    metrics : dict
        Success metrics, efficiency, stability, etc.
    """
    m1_amp = diagnostics['m1_amp']
    H = diagnostics['H']
    actions = diagnostics['actions']
    
    initial_m1 = m1_amp[0]
    final_m1 = m1_amp[-1]
    
    # Success tiers (小P recommendation)
    tier1 = final_m1 < initial_m1  # Stabilization
    tier2 = final_m1 < 0.5 * initial_m1  # Suppression
    tier3 = final_m1 < 0.1 * initial_m1  # Quenching
    
    # Energy efficiency (total |dH/dt|)
    dH = np.diff(H)
    total_dissipation = np.sum(np.abs(dH))
    
    # Stability
    max_m1 = np.max(m1_amp)
    std_H = np.std(H)
    
    # Control effort
    mean_action = np.mean(np.abs(actions - 1.0))
    
    # Reduction ratio
    reduction = (initial_m1 - final_m1) / initial_m1
    
    return {
        'tier1_stabilization': tier1,
        'tier2_suppression': tier2,
        'tier3_quenching': tier3,
        'initial_m1': initial_m1,
        'final_m1': final_m1,
        'max_m1': max_m1,
        'reduction': reduction,
        'total_dissipation': total_dissipation,
        'std_H': std_H,
        'mean_control_effort': mean_action
    }


def run_experiments(
    agent_type: str,
    n_trials: int = 5,
    max_steps: int = 1000,  # Increased to see tearing growth (小P ⚛️)
    save_dir: str = "results/issue28"
):
    """
    Run experiments for one agent type.
    
    Parameters
    ----------
    agent_type : str
        'no_control', 'random', 'pid'
    n_trials : int
        Number of trials
    max_steps : int
        Episode length
    save_dir : str
        Results directory
    """
    print(f"\n{'='*60}")
    print(f"Running {agent_type} experiments ({n_trials} trials)")
    print(f"{'='*60}\n")
    
    # Create environment
    # Note: eta=0.05 matches Issue #29 Harris sheet IC design (小P ⚛️)
    # This produces observable tearing growth (~11% in 0.1s)
    env = make_hamiltonian_mhd_env(
        nr=32,
        ntheta=64,
        nz=8,
        dt=1e-4,
        max_steps=max_steps,
        eta=0.05,  # Issue #29 design parameter
        nu=1e-4,
        normalize_obs=False  # Disable for baseline comparison
    )
    
    # Create agent
    if agent_type == 'pid':
        # 小P conservative tuning
        agent = make_baseline_agent(
            'pid',
            env.action_space,
            Kp=5.0,
            Ki=0.5,
            Kd=0.01,
            target=0.0,
            dt=1e-4
        )
    else:
        agent = make_baseline_agent(agent_type, env.action_space, seed=42)
    
    # Run trials
    all_diagnostics = []
    all_metrics = []
    
    for trial in tqdm(range(n_trials), desc=f"{agent_type}"):
        seed = 1000 + trial  # Deterministic seeds
        
        # Run episode (sparse recording for speed)
        diag = run_episode(env, agent, seed=seed, record_every=100)
        
        # Compute metrics
        metrics = compute_metrics(diag)
        
        all_diagnostics.append(diag)
        all_metrics.append(metrics)
    
    # Aggregate results
    results = aggregate_metrics(all_metrics)
    results['agent_type'] = agent_type
    results['n_trials'] = n_trials
    
    # Save
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics (convert numpy types to native Python)
    results_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in results.items()
    }
    with open(save_dir_path / f"{agent_type}_metrics.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save diagnostics (first 3 trials for plotting)
    np.savez(
        save_dir_path / f"{agent_type}_diagnostics.npz",
        **{f"trial_{i}": diag for i, diag in enumerate(all_diagnostics[:3])}
    )
    
    print(f"\n{agent_type} Results:")
    print(f"  Tier 1 (Stabilization): {results['tier1_rate']:.1%}")
    print(f"  Tier 2 (Suppression):   {results['tier2_rate']:.1%}")
    print(f"  Tier 3 (Quenching):     {results['tier3_rate']:.1%}")
    print(f"  Mean reduction: {results['mean_reduction']:.2%} ± {results['std_reduction']:.2%}")
    
    return results, all_diagnostics


def aggregate_metrics(metrics_list):
    """Aggregate metrics across trials."""
    # Success rates
    tier1_rate = np.mean([m['tier1_stabilization'] for m in metrics_list])
    tier2_rate = np.mean([m['tier2_suppression'] for m in metrics_list])
    tier3_rate = np.mean([m['tier3_quenching'] for m in metrics_list])
    
    # Reduction
    reductions = [m['reduction'] for m in metrics_list]
    mean_reduction = np.mean(reductions)
    std_reduction = np.std(reductions)
    
    # Final m1
    final_m1s = [m['final_m1'] for m in metrics_list]
    mean_final_m1 = np.mean(final_m1s)
    std_final_m1 = np.std(final_m1s)
    
    # Max m1
    max_m1s = [m['max_m1'] for m in metrics_list]
    mean_max_m1 = np.mean(max_m1s)
    
    # Dissipation
    dissipations = [m['total_dissipation'] for m in metrics_list]
    mean_dissipation = np.mean(dissipations)
    
    # Control effort
    efforts = [m['mean_control_effort'] for m in metrics_list]
    mean_effort = np.mean(efforts)
    
    return {
        'tier1_rate': tier1_rate,
        'tier2_rate': tier2_rate,
        'tier3_rate': tier3_rate,
        'mean_reduction': mean_reduction,
        'std_reduction': std_reduction,
        'mean_final_m1': mean_final_m1,
        'std_final_m1': std_final_m1,
        'mean_max_m1': mean_max_m1,
        'mean_dissipation': mean_dissipation,
        'mean_control_effort': mean_effort
    }


def plot_comparison(save_dir="results/issue28"):
    """Generate comparison plots."""
    save_dir_path = Path(save_dir)
    
    # Load all results
    agent_types = ['no_control', 'random', 'pid']
    all_diags = {}
    
    for agent_type in agent_types:
        data = np.load(save_dir_path / f"{agent_type}_diagnostics.npz", allow_pickle=True)
        all_diags[agent_type] = data['trial_0'].item()  # First trial
    
    # Plot m1 amplitude evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, agent_type in enumerate(agent_types):
        ax = axes[i]
        diag = all_diags[agent_type]
        
        m1_amp = diag['m1_amp']
        steps = np.arange(len(m1_amp))
        
        ax.plot(steps, m1_amp, 'b-', linewidth=2)
        ax.axhline(m1_amp[0], color='k', linestyle='--', alpha=0.5, label='Initial')
        ax.axhline(0.5 * m1_amp[0], color='orange', linestyle='--', alpha=0.5, label='Tier 2')
        ax.axhline(0.1 * m1_amp[0], color='green', linestyle='--', alpha=0.5, label='Tier 3')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('m=1 Amplitude')
        ax.set_title(f'{agent_type.replace("_", " ").title()}')
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir_path / 'comparison_m1_evolution.png', dpi=150)
    print(f"\nPlot saved: {save_dir_path / 'comparison_m1_evolution.png'}")
    
    # Summary table plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Load metrics
    metrics_data = []
    for agent_type in agent_types:
        with open(save_dir_path / f"{agent_type}_metrics.json", 'r') as f:
            metrics = json.load(f)
        metrics_data.append([
            agent_type.replace('_', ' ').title(),
            f"{metrics['tier1_rate']:.1%}",
            f"{metrics['tier2_rate']:.1%}",
            f"{metrics['tier3_rate']:.1%}",
            f"{metrics['mean_reduction']:.1%} ± {metrics['std_reduction']:.1%}"
        ])
    
    table = ax.table(
        cellText=metrics_data,
        colLabels=['Agent', 'Tier 1', 'Tier 2', 'Tier 3', 'Reduction'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Baseline Controller Comparison', fontsize=14, pad=20)
    plt.savefig(save_dir_path / 'comparison_table.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: {save_dir_path / 'comparison_table.png'}")


def main():
    """Run all baseline experiments."""
    save_dir = "results/issue28"
    n_trials = 5  # Reduced from 10 (小P optimization ⚛️)
    
    # Run experiments for each agent
    results_all = {}
    
    for agent_type in ['no_control', 'random', 'pid']:
        results, _ = run_experiments(
            agent_type=agent_type,
            n_trials=n_trials,
            save_dir=save_dir
        )
        results_all[agent_type] = results
    
    # Generate plots
    plot_comparison(save_dir=save_dir)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {save_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
