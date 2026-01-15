"""
Generate Publication-Quality Plots
===================================
Creates figures for the few-shot SCA paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("GENERATING PUBLICATION PLOTS")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv('few_shot_sca_results.csv')

# Try to load aggregated statistics if available
try:
    stats_df = pd.read_csv('experiment_results/aggregated_statistics.csv')
    has_stats = True
    print("✓ Found aggregated statistics from multiple runs")
except:
    has_stats = False
    print("! Using single run results (run run_multiple_experiments.py for error bars)")

# Try to load baseline
try:
    baseline_df = pd.read_csv('baseline_standard_cnn_results.csv')
    has_baseline = True
    print("✓ Found baseline results")
except:
    has_baseline = False
    print("! No baseline found (run baseline_standard_training.py)")

print()

# ============================================================================
# Figure 1: Key Rank vs Number of Attack Traces (for each k-shot)
# ============================================================================
print("Generating Figure 1: Key Rank vs Attack Traces...")

# Use clean scientific style
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

k_shots = [5, 10, 15, 20]
# Updated colors: MAML=blue, ProtoNet=orange, Siamese=green
colors = {'MAML': '#1f77b4', 'ProtoNet': '#ff7f0e', 'Siamese': '#2ca02c'}
methods = ['MAML', 'ProtoNet', 'Siamese']

for idx, k_shot in enumerate(k_shots):
    ax = axes[idx]

    for method in methods:
        method_data = df[(df['Method'] == method) & (df['K-Shot'] == k_shot)]

        if has_stats:
            # Plot with error bars
            stats_subset = stats_df[(stats_df['Method'] == method) & (stats_df['K-Shot'] == k_shot)]
            ax.errorbar(
                stats_subset['Attack_Traces'],
                stats_subset['Mean_Rank'],
                yerr=stats_subset['Std_Rank'],
                marker='o',
                label=method,
                linewidth=2,
                capsize=5,
                color=colors[method],
                markersize=6
            )
        else:
            # Plot single run
            ax.plot(
                method_data['Attack Traces'],
                method_data['Key Rank'],
                marker='o',
                label=method,
                linewidth=2,
                color=colors[method],
                markersize=6
            )

    # Add horizontal dashed red line at y=10 (practical success threshold)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
               label='Practical success threshold' if idx == 0 else '')

    ax.set_xlabel('Number of Attack Traces', fontsize=11)
    ax.set_ylabel('Key Rank', fontsize=11)
    ax.set_title(f'{k_shot}-shot', fontsize=12, fontweight='bold')

    # Set y-axis to linear scale from 0 to 256
    ax.set_ylim(0, 256)

    # Set x-axis to logarithmic scale
    ax.set_xscale('log')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Only show legend in first subplot to avoid clutter
    if idx == 0:
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Add overall figure title
fig.suptitle('Key Rank vs. Number of Attack Traces for Different Few-Shot Settings\n(with Minimal Variance Sampling)',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_key_rank_vs_traces.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_key_rank_vs_traces.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig1_key_rank_vs_traces.png")

# ============================================================================
# Figure 2: Comparison across k-shot values (at 1000 traces)
# ============================================================================
print("Generating Figure 2: K-Shot Ablation Study...")

fig, ax = plt.subplots(figsize=(10, 6))

subset_1000 = df[df['Attack Traces'] == 1000]

if has_stats:
    stats_1000 = stats_df[stats_df['Attack_Traces'] == 1000]

    for method in methods:
        method_stats = stats_1000[stats_1000['Method'] == method]
        ax.errorbar(
            method_stats['K-Shot'],
            method_stats['Mean_Rank'],
            yerr=method_stats['Std_Rank'],
            marker='o',
            label=method,
            linewidth=2,
            markersize=8,
            capsize=5,
            color=colors[method]
        )
else:
    for method in methods:
        method_data = subset_1000[subset_1000['Method'] == method]
        ax.plot(
            method_data['K-Shot'],
            method_data['Key Rank'],
            marker='o',
            label=method,
            linewidth=2,
            markersize=8,
            color=colors[method]
        )

ax.set_xlabel('K-Shot (training examples per class)')
ax.set_ylabel('Key Rank at 1000 Attack Traces (lower is better)')
ax.set_title('Ablation Study: Effect of K-Shot on Attack Performance')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 15, 20])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_kshot_ablation.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_kshot_ablation.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig2_kshot_ablation.png")

# ============================================================================
# Figure 3: Heatmap of all results
# ============================================================================
print("Generating Figure 3: Results Heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, method in enumerate(methods):
    ax = axes[idx]

    method_data = df[df['Method'] == method]
    pivot = method_data.pivot(index='K-Shot', columns='Attack Traces', values='Key Rank')

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Key Rank'},
        ax=ax,
        vmin=0,
        vmax=255
    )
    ax.set_title(f'{method} Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Attack Traces')
    ax.set_ylabel('K-Shot')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_results_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_results_heatmap.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig3_results_heatmap.png")

# ============================================================================
# Figure 4: Comparison with Baseline (if available)
# ============================================================================
if has_baseline:
    print("Generating Figure 4: Few-Shot vs Standard Training...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Few-shot methods at 1000 traces
    for method in methods:
        method_data = subset_1000[subset_1000['Method'] == method]
        ax.plot(
            method_data['K-Shot'] * 256,  # Convert to total training samples
            method_data['Key Rank'],
            marker='o',
            label=f'{method} (Few-Shot)',
            linewidth=2,
            markersize=8,
            color=colors[method]
        )

    # Baseline
    baseline_1000 = baseline_df[baseline_df['Attack_Traces'] == 1000]
    ax.plot(
        baseline_1000['Training_Size'],
        baseline_1000['Key_Rank'],
        marker='s',
        label='Standard CNN',
        linewidth=2,
        markersize=8,
        color='#2ECC71',
        linestyle='--'
    )

    ax.set_xlabel('Number of Training Traces')
    ax.set_ylabel('Key Rank at 1000 Attack Traces (lower is better)')
    ax.set_title('Few-Shot Meta-Learning vs Standard Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_fewshot_vs_baseline.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_fewshot_vs_baseline.pdf'), bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig4_fewshot_vs_baseline.png")

# ============================================================================
# Figure 5: Success Rate (Key Rank < 50)
# ============================================================================
print("Generating Figure 5: Success Rate Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

# Use Key Rank < 50 as success threshold (more realistic)
SUCCESS_THRESHOLD = 50

for method in methods:
    success_rates = []
    for k_shot in k_shots:
        method_k = df[(df['Method'] == method) & (df['K-Shot'] == k_shot)]
        # Success = key rank < threshold
        successes = (method_k['Key Rank'] < SUCCESS_THRESHOLD).sum()
        total = len(method_k)
        success_rate = (successes / total) * 100 if total > 0 else 0
        success_rates.append(success_rate)

    ax.plot(
        k_shots,
        success_rates,
        marker='o',
        label=method,
        linewidth=2,
        markersize=8,
        color=colors[method]
    )

ax.set_xlabel('K-Shot', fontsize=11)
ax.set_ylabel(f'Success Rate (%) - Key Rank < {SUCCESS_THRESHOLD}', fontsize=11)
ax.set_title(f'Attack Success Rate Across K-Shot Values (Threshold: Rank < {SUCCESS_THRESHOLD})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([5, 10, 15, 20])
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_success_rate.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_success_rate.pdf'), bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR}/fig5_success_rate.png")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ All figures saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  - fig1_key_rank_vs_traces.png/pdf")
print("  - fig2_kshot_ablation.png/pdf")
print("  - fig3_results_heatmap.png/pdf")
if has_baseline:
    print("  - fig4_fewshot_vs_baseline.png/pdf")
print("  - fig5_success_rate.png/pdf")

print("\nFor your paper:")
print("  1. Use PNG for quick viewing")
print("  2. Use PDF for LaTeX inclusion (vector graphics, better quality)")
print("  3. Reference figures in your paper text")
print("  4. Add captions explaining each figure")
