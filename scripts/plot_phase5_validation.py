"""
Phase 5 Validation Visualization

Create 3 plots:
1. Robustness heatmap (ε vs n performance)
2. Generalization box plot (train vs interpolate vs extrapolate)
3. Action stability over long episodes

Author: 小A 🤖
Date: 2026-03-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_robustness_heatmap(csv_file, output_file):
    """Plot robustness heatmap: success rate vs ε and n"""
    df = pd.read_csv(csv_file)
    
    # Create pivot table: rows=ε, columns=n
    pivot = df.pivot_table(
        values='success',
        index='epsilon',
        columns='n',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.0%', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
    ax.set_title('PPO Robustness: Success Rate vs IC Parameters', fontsize=14, fontweight='bold')
    ax.set_xlabel('Toroidal Mode Number (n)', fontsize=12)
    ax.set_ylabel('Perturbation Amplitude (ε)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_generalization_boxplot(csv_file, output_file):
    """Plot generalization performance: train vs interpolation vs extrapolation"""
    try:
        df = pd.read_csv(csv_file)
        
        # Filter successful episodes
        df_success = df[df['success'] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of mean rewards
        sns.boxplot(x='category', y='mean_reward', data=df_success, ax=ax,
                   order=['Training', 'Interpolation', 'Extrapolation'])
        
        # Add success rate annotations
        for i, category in enumerate(['Training', 'Interpolation', 'Extrapolation']):
            cat_df = df[df['category'] == category]
            success_rate = cat_df['success'].mean()
            n_success = cat_df['success'].sum()
            n_total = len(cat_df)
            ax.text(i, ax.get_ylim()[1] * 0.95, 
                   f'{success_rate:.0%}\n({n_success}/{n_total})',
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        ax.set_title('PPO Generalization: Performance on Seen vs Unseen ICs', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('IC Category', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    except FileNotFoundError:
        print(f"Generalization test not yet completed, skipping plot")

def plot_long_episode_stability(csv_file, output_file):
    """Plot action stability over long episodes"""
    try:
        df = pd.read_csv(csv_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Completion rate
        completion_data = df.groupby('policy')['completion_rate'].agg(['mean', 'std'])
        policies = completion_data.index
        means = completion_data['mean']
        stds = completion_data['std']
        
        axes[0].bar(policies, means, yerr=stds, capsize=5,
                   color=['#2ecc71', '#95a5a6', '#e74c3c'])
        axes[0].set_ylabel('Episode Completion Rate', fontsize=12)
        axes[0].set_title('500-Step Episode Completion', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Action variance (stability)
        df_success = df[df['success'] == 1]
        action_data = df_success.groupby('policy')['action_variance'].agg(['mean', 'std'])
        
        axes[1].bar(action_data.index, action_data['mean'], yerr=action_data['std'], 
                   capsize=5, color=['#2ecc71', '#95a5a6', '#e74c3c'])
        axes[1].set_ylabel('Action Variance', fontsize=12)
        axes[1].set_title('Control Stability (Lower = More Stable)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Long Episode Performance (500 steps, 5× training)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    except FileNotFoundError:
        print(f"Long episode test not yet completed, skipping plot")

def plot_robustness_by_parameter(csv_file, output_file):
    """Plot success rate grouped by each parameter"""
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # By epsilon
    eps_success = df.groupby('epsilon')['success'].mean()
    axes[0].bar(range(len(eps_success)), eps_success.values, 
               color='#3498db', edgecolor='black')
    axes[0].set_xticks(range(len(eps_success)))
    axes[0].set_xticklabels([f'{e:.5f}' for e in eps_success.index], rotation=45)
    axes[0].set_ylabel('Success Rate', fontsize=12)
    axes[0].set_xlabel('Perturbation ε', fontsize=12)
    axes[0].set_title('Success Rate vs ε', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].axhline(0.8, color='red', linestyle='--', label='80% threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # By n
    n_success = df.groupby('n')['success'].mean()
    axes[1].bar(n_success.index, n_success.values, 
               color='#2ecc71', edgecolor='black')
    axes[1].set_ylabel('Success Rate', fontsize=12)
    axes[1].set_xlabel('Toroidal Mode n', fontsize=12)
    axes[1].set_title('Success Rate vs n', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1.1])
    axes[1].axhline(0.8, color='red', linestyle='--', label='80% threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # By m0
    m_success = df.groupby('m0')['success'].mean()
    axes[2].bar(m_success.index, m_success.values, 
               color='#e74c3c', edgecolor='black')
    axes[2].set_ylabel('Success Rate', fontsize=12)
    axes[2].set_xlabel('Poloidal Mode m₀', fontsize=12)
    axes[2].set_title('Success Rate vs m₀', fontsize=12, fontweight='bold')
    axes[2].set_ylim([0, 1.1])
    axes[2].axhline(0.8, color='red', linestyle='--', label='80% threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    output_dir = "results/phase5"
    
    # Plot 1: Robustness heatmap
    plot_robustness_heatmap(
        f"{output_dir}/robustness_results.csv",
        f"{output_dir}/robustness_heatmap.png"
    )
    
    # Plot 1b: Robustness by parameter
    plot_robustness_by_parameter(
        f"{output_dir}/robustness_results.csv",
        f"{output_dir}/robustness_by_param.png"
    )
    
    # Plot 2: Generalization box plot
    plot_generalization_boxplot(
        f"{output_dir}/generalization_results.csv",
        f"{output_dir}/generalization_boxplot.png"
    )
    
    # Plot 3: Long episode stability
    plot_long_episode_stability(
        f"{output_dir}/long_episode_results.csv",
        f"{output_dir}/long_episode_stability.png"
    )
    
    print("\nAll plots completed!")
