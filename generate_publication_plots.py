"""
Generate Publication-Quality Plots for Few-Shot SCA Paper
==========================================================
Comprehensive plotting script covering all essential figures for the paper.

Figures generated:
  Fig 1: Key Rank vs Attack Traces — convergence curves (ASCAD HW, per k-shot)
  Fig 2: Key Rank vs Attack Traces — convergence curves (CHES CTF HW, per k-shot)
  Fig 3: Data Efficiency — Few-shot vs Template vs CNN (both datasets)
  Fig 4: SNR Profile — ASCAD and CHES CTF side by side
  Fig 5: Cross-Dataset Comparison — grouped bar chart (ASCAD vs CHES CTF)
  Fig 6: Box Plots — Key rank distribution from 10-run data
  Fig 7: K-Shot Ablation — effect of K on attack performance

Usage:
  python generate_publication_plots.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Publication Style Configuration
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (7, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

METHOD_COLORS = {
    'MAML': '#1f77b4',
    'ProtoNet': '#ff7f0e',
    'Siamese': '#2ca02c',
    'Template Attack': '#d62728',
    'Standard CNN': '#9467bd',
}
METHOD_MARKERS = {
    'MAML': 'o',
    'ProtoNet': 's',
    'Siamese': '^',
    'Template Attack': 'D',
    'Standard CNN': 'v',
}

SCA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCA_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Data Loading Helpers
# =============================================================================
def load_csv_safe(path, description=""):
    """Load CSV if it exists, return None otherwise."""
    full_path = os.path.join(SCA_DIR, path)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        print(f"  [OK] {description}: {path} ({len(df)} rows)")
        return df
    else:
        print(f"  [--] {description}: {path} (not found)")
        return None


print("=" * 70)
print("GENERATING PUBLICATION PLOTS")
print("=" * 70)
print("\nLoading data files...")

# --- ASCAD HW ---
ascad_hw_fewshot = load_csv_safe('hw_few_shot_sca_results.csv', 'ASCAD HW few-shot')
ascad_hw_template = load_csv_safe('hw_template_attack_results.csv', 'ASCAD HW template')
ascad_hw_cnn = load_csv_safe('hw_baseline_standard_cnn_results.csv', 'ASCAD HW CNN')

# --- CHES CTF HW ---
ches_hw_fewshot = load_csv_safe('ches_ctf_hw_few_shot_results.csv', 'CHES CTF HW few-shot')
ches_hw_template = load_csv_safe('ches_ctf_hw_template_attack_results.csv', 'CHES CTF HW template')
ches_hw_cnn = load_csv_safe('ches_ctf_hw_baseline_standard_cnn_results.csv', 'CHES CTF HW CNN')

# --- ASCAD ID (original) ---
ascad_id_fewshot = load_csv_safe('few_shot_sca_results.csv', 'ASCAD ID few-shot')
ascad_id_cnn = load_csv_safe('baseline_standard_cnn_results.csv', 'ASCAD ID CNN')
ascad_id_template = load_csv_safe('template_attack_results.csv', 'ASCAD ID template')

# --- 10-run aggregated statistics ---
ascad_id_fewshot_agg = load_csv_safe('experiment_results/aggregated_statistics.csv', 'ASCAD ID few-shot (10-run)')
ascad_id_cnn_agg = load_csv_safe('baseline_results/baseline_aggregated_statistics.csv', 'ASCAD ID CNN (10-run)')
ascad_id_template_agg = load_csv_safe('template_results/template_aggregated_statistics.csv', 'ASCAD ID template (10-run)')

# --- 10-run combined (for box plots) ---
ascad_id_fewshot_combined = load_csv_safe('experiment_results/all_runs_combined.csv', 'ASCAD ID few-shot combined')
ascad_id_cnn_combined = load_csv_safe('baseline_results/all_baseline_runs_combined.csv', 'ASCAD ID CNN combined')
ascad_id_template_combined = load_csv_safe('template_results/all_template_runs_combined.csv', 'ASCAD ID template combined')

# ASCAD HW 10-run (may not exist yet)
ascad_hw_fewshot_agg = load_csv_safe('hw_results/hw_fewshot_aggregated_statistics.csv', 'ASCAD HW few-shot (10-run)')
ascad_hw_cnn_agg = load_csv_safe('hw_results/hw_cnn_aggregated_statistics.csv', 'ASCAD HW CNN (10-run)')
ascad_hw_template_agg = load_csv_safe('hw_results/hw_template_aggregated_statistics.csv', 'ASCAD HW template (10-run)')

# CHES CTF HW 10-run (may not exist yet)
ches_hw_fewshot_agg = load_csv_safe('ches_ctf_hw_results/ches_ctf_hw_fewshot_aggregated_statistics.csv', 'CHES CTF HW few-shot (10-run)')
ches_hw_cnn_agg = load_csv_safe('ches_ctf_hw_results/ches_ctf_hw_cnn_aggregated_statistics.csv', 'CHES CTF HW CNN (10-run)')
ches_hw_template_agg = load_csv_safe('ches_ctf_hw_results/ches_ctf_hw_template_aggregated_statistics.csv', 'CHES CTF HW template (10-run)')

# CHES CTF HW 10-run combined (for box plots)
ches_hw_fewshot_combined = load_csv_safe('ches_ctf_hw_results/ches_ctf_hw_fewshot_all_runs_combined.csv', 'CHES CTF HW few-shot combined')

# ASCAD HW 10-run combined
ascad_hw_fewshot_combined = load_csv_safe('hw_results/hw_fewshot_all_runs_combined.csv', 'ASCAD HW few-shot combined')

print()

FEWSHOT_METHODS = ['MAML', 'ProtoNet', 'Siamese']
K_SHOTS = [5, 10, 15, 20]
ATTACK_TRACES = [100, 500, 1000, 2000, 5000]

generated = []


# =============================================================================
# FIGURE 1: Key Rank vs Attack Traces — ASCAD HW (convergence curves)
# =============================================================================
def plot_convergence_curves(fewshot_df, fewshot_agg, dataset_name, fig_name, fig_num):
    """Plot key rank vs attack traces for all k-shots and methods."""
    if fewshot_df is None and fewshot_agg is None:
        print(f"  [SKIP] Fig {fig_num}: No data for {dataset_name}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, k_shot in enumerate(K_SHOTS):
        ax = axes[idx]

        for method in FEWSHOT_METHODS:
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]

            if fewshot_agg is not None:
                # Use aggregated 10-run data with CI bands
                # Handle different column name formats
                cols = fewshot_agg.columns.tolist()
                at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
                mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
                sr_col = next((c for c in cols if c in ['Std_Rank', 'Std_Key_Rank']), None)
                ci_lo = next((c for c in cols if c in ['CI_Lower', 'CI95_Lower']), None)
                ci_hi = next((c for c in cols if c in ['CI_Upper', 'CI95_Upper']), None)
                ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

                if at_col and mr_col and ks_col:
                    subset = fewshot_agg[
                        (fewshot_agg['Method'] == method) &
                        (fewshot_agg[ks_col] == k_shot)
                    ].sort_values(at_col)

                    if len(subset) > 0:
                        x = subset[at_col].values
                        y = subset[mr_col].values

                        ax.plot(x, y, marker=marker, color=color, label=method,
                                linewidth=2, markersize=6)

                        # Add CI shading
                        if ci_lo and ci_hi:
                            lo = subset[ci_lo].values
                            hi = subset[ci_hi].values
                            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
                        elif sr_col:
                            std = subset[sr_col].values
                            ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
            elif fewshot_df is not None:
                # Single-run fallback
                subset = fewshot_df[
                    (fewshot_df['Method'] == method) &
                    (fewshot_df['K-Shot'] == k_shot)
                ].sort_values('Attack Traces')

                if len(subset) > 0:
                    ax.plot(subset['Attack Traces'], subset['Key Rank'],
                            marker=marker, color=color, label=method,
                            linewidth=2, markersize=6)

        # Threshold line
        ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label='Rank 10 threshold' if idx == 0 else '')

        ax.set_xlabel('Number of Attack Traces')
        ax.set_ylabel('Key Rank')
        ax.set_title(f'{k_shot}-shot', fontweight='bold')
        ax.set_ylim(-5, 260)
        ax.set_xscale('log')
        ax.set_xticks(ATTACK_TRACES)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    suffix = "(Mean ± 95% CI, 10 runs)" if fewshot_agg is not None else "(single seed)"
    fig.suptitle(f'Key Rank Convergence — {dataset_name} (HW Model)\n{suffix}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fig_name}.png/.pdf')
    print(f"  [OK] Fig {fig_num}: {fig_name}")


print("--- Generating Figures ---")
plot_convergence_curves(ascad_hw_fewshot, ascad_hw_fewshot_agg,
                        'ASCAD v1', 'fig1_convergence_ascad_hw', 1)
plot_convergence_curves(ches_hw_fewshot, ches_hw_fewshot_agg,
                        'CHES CTF 2018', 'fig2_convergence_ches_ctf_hw', 2)


# =============================================================================
# FIGURE 3: Data Efficiency — Few-shot vs Template vs CNN
# =============================================================================
def plot_data_efficiency(fewshot_df, template_df, cnn_df,
                         fewshot_agg, template_agg, cnn_agg,
                         dataset_name, fig_name, fig_num):
    """Key rank at 1000 traces vs number of profiling traces used."""
    if fewshot_df is None and fewshot_agg is None:
        print(f"  [SKIP] Fig {fig_num}: No data for {dataset_name}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- Few-shot methods: profiling traces = K * 9 (HW classes) ---
    for method in FEWSHOT_METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        prof_traces = []
        ranks = []
        stds = []

        if fewshot_agg is not None:
            cols = fewshot_agg.columns.tolist()
            at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
            mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
            sr_col = next((c for c in cols if c in ['Std_Rank', 'Std_Key_Rank']), None)
            ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

            for k in K_SHOTS:
                match = fewshot_agg[
                    (fewshot_agg['Method'] == method) &
                    (fewshot_agg[ks_col] == k) &
                    (fewshot_agg[at_col] == 1000)
                ]
                if len(match) > 0:
                    prof_traces.append(k * 9)
                    ranks.append(match.iloc[0][mr_col])
                    stds.append(match.iloc[0][sr_col] if sr_col else 0)

            if prof_traces:
                ax.errorbar(prof_traces, ranks, yerr=stds, marker=marker, color=color,
                            label=f'{method} (few-shot)', linewidth=2, capsize=4, markersize=7)
        elif fewshot_df is not None:
            for k in K_SHOTS:
                match = fewshot_df[
                    (fewshot_df['Method'] == method) &
                    (fewshot_df['K-Shot'] == k) &
                    (fewshot_df['Attack Traces'] == 1000)
                ]
                if len(match) > 0:
                    prof_traces.append(k * 9)
                    ranks.append(match.iloc[0]['Key Rank'])

            if prof_traces:
                ax.plot(prof_traces, ranks, marker=marker, color=color,
                        label=f'{method} (few-shot)', linewidth=2, markersize=7)

    # --- Template Attack ---
    if template_agg is not None:
        cols = template_agg.columns.tolist()
        at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
        mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
        sr_col = next((c for c in cols if c in ['Std_Rank', 'Std_Key_Rank']), None)

        if at_col and mr_col:
            t_data = template_agg[template_agg[at_col] == 1000].sort_values('Training_Size')
            if len(t_data) > 0:
                ax.errorbar(t_data['Training_Size'], t_data[mr_col],
                            yerr=t_data[sr_col] if sr_col else None,
                            marker='D', color=METHOD_COLORS['Template Attack'],
                            label='Template Attack', linewidth=2, linestyle='--',
                            capsize=4, markersize=7)
    elif template_df is not None:
        t_data = template_df[template_df['Attack_Traces'] == 1000].sort_values('Training_Size')
        if len(t_data) > 0:
            ax.plot(t_data['Training_Size'], t_data['Key_Rank'],
                    marker='D', color=METHOD_COLORS['Template Attack'],
                    label='Template Attack', linewidth=2, linestyle='--', markersize=7)

    # --- Standard CNN ---
    if cnn_agg is not None:
        cols = cnn_agg.columns.tolist()
        at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
        mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
        sr_col = next((c for c in cols if c in ['Std_Rank', 'Std_Key_Rank']), None)

        if at_col and mr_col:
            c_data = cnn_agg[cnn_agg[at_col] == 1000].sort_values('Training_Size')
            if len(c_data) > 0:
                ax.errorbar(c_data['Training_Size'], c_data[mr_col],
                            yerr=c_data[sr_col] if sr_col else None,
                            marker='v', color=METHOD_COLORS['Standard CNN'],
                            label='Standard CNN', linewidth=2, linestyle=':',
                            capsize=4, markersize=7)
    elif cnn_df is not None:
        c_data = cnn_df[cnn_df['Attack_Traces'] == 1000].sort_values('Training_Size')
        if len(c_data) > 0:
            ax.plot(c_data['Training_Size'], c_data['Key_Rank'],
                    marker='v', color=METHOD_COLORS['Standard CNN'],
                    label='Standard CNN', linewidth=2, linestyle=':', markersize=7)

    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Rank 10')
    ax.set_xlabel('Number of Profiling Traces (log scale)')
    ax.set_ylabel('Key Rank at 1000 Attack Traces')
    ax.set_title(f'Data Efficiency — {dataset_name} (HW Model)', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim(-5, 260)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fig_name}.png/.pdf')
    print(f"  [OK] Fig {fig_num}: {fig_name}")


plot_data_efficiency(ascad_hw_fewshot, ascad_hw_template, ascad_hw_cnn,
                     ascad_hw_fewshot_agg, ascad_hw_template_agg, ascad_hw_cnn_agg,
                     'ASCAD v1', 'fig3a_data_efficiency_ascad_hw', '3a')
plot_data_efficiency(ches_hw_fewshot, ches_hw_template, ches_hw_cnn,
                     ches_hw_fewshot_agg, ches_hw_template_agg, ches_hw_cnn_agg,
                     'CHES CTF 2018', 'fig3b_data_efficiency_ches_ctf_hw', '3b')


# =============================================================================
# FIGURE 4: SNR Profile — ASCAD and CHES CTF
# =============================================================================
print("\nGenerating Fig 4: SNR Profiles...")

sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def compute_snr(traces, labels, n_classes=9):
    """Compute SNR per time sample."""
    n_samples = traces.shape[1]
    class_means = np.zeros((n_classes, n_samples))
    class_vars = np.zeros((n_classes, n_samples))
    class_counts = np.zeros(n_classes)

    for c in range(n_classes):
        mask = labels == c
        count = mask.sum()
        if count > 1:
            class_means[c] = traces[mask].mean(axis=0)
            class_vars[c] = traces[mask].var(axis=0)
            class_counts[c] = count
        elif count == 1:
            class_means[c] = traces[mask][0]
            class_counts[c] = count

    weights = class_counts / class_counts.sum()
    signal = np.average(
        (class_means - np.average(class_means, axis=0, weights=weights))**2,
        axis=0, weights=weights
    )
    noise = np.average(class_vars, axis=0, weights=weights)
    noise[noise == 0] = 1e-10
    return signal / noise


try:
    import h5py

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    snr_data = {}
    n_poi = 200

    # --- ASCAD ---
    ascad_path = os.path.join(SCA_DIR, 'ASCAD_data', 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5')
    if os.path.exists(ascad_path):
        with h5py.File(ascad_path, 'r') as f:
            traces = np.array(f['Profiling_traces']['traces'], dtype=np.float32)
            metadata = f['Profiling_traces']['metadata'][:]
            pt = metadata['plaintext'][:, 0]
            key = metadata['key'][:, 0]
            masks = metadata['masks'][:, 0]
            labels = HW[sbox[pt ^ key] ^ masks]

        snr_ascad = compute_snr(traces, labels)
        poi_indices_ascad = np.sort(np.argsort(snr_ascad)[-n_poi:])
        snr_data['ascad'] = (snr_ascad, poi_indices_ascad, traces.shape[1])

        ax = axes[0]
        ax.plot(snr_ascad, color='steelblue', linewidth=0.5, alpha=0.8)
        # Highlight POI
        poi_mask = np.zeros(len(snr_ascad), dtype=bool)
        poi_mask[poi_indices_ascad] = True
        ax.fill_between(range(len(snr_ascad)), 0, snr_ascad,
                         where=poi_mask, color='red', alpha=0.3, label=f'Top-{n_poi} POI')
        ax.set_xlabel('Time Sample Index')
        ax.set_ylabel('SNR')
        ax.set_title('ASCAD v1 (ATMega8515, 8-bit AVR)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(snr_ascad))
        print(f"  [OK] ASCAD SNR: max={snr_ascad.max():.4f}, {traces.shape[1]} samples")
    else:
        axes[0].text(0.5, 0.5, 'ASCAD dataset not found', transform=axes[0].transAxes,
                     ha='center', va='center', fontsize=12, color='gray')
        print(f"  [--] ASCAD dataset not found at {ascad_path}")

    # --- CHES CTF ---
    ches_path = os.path.join(SCA_DIR, 'datasets', 'ches_ctf.h5')
    if os.path.exists(ches_path):
        with h5py.File(ches_path, 'r') as f:
            traces = np.array(f['profiling_traces'], dtype=np.float32)
            prof_data = np.array(f['profiling_data'], dtype=np.uint8)
            pt = prof_data[:, 0]
            key_byte = prof_data[0, 32]
            labels = HW[sbox[pt ^ key_byte]]

        snr_ches = compute_snr(traces, labels)
        poi_indices_ches = np.sort(np.argsort(snr_ches)[-n_poi:])
        snr_data['ches'] = (snr_ches, poi_indices_ches, traces.shape[1])

        ax = axes[1]
        ax.plot(snr_ches, color='steelblue', linewidth=0.5, alpha=0.8)
        poi_mask = np.zeros(len(snr_ches), dtype=bool)
        poi_mask[poi_indices_ches] = True
        ax.fill_between(range(len(snr_ches)), 0, snr_ches,
                         where=poi_mask, color='red', alpha=0.3, label=f'Top-{n_poi} POI')
        ax.set_xlabel('Time Sample Index')
        ax.set_ylabel('SNR')
        ax.set_title('CHES CTF 2018 (STM32, 32-bit ARM)', fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlim(0, len(snr_ches))
        print(f"  [OK] CHES CTF SNR: max={snr_ches.max():.6f}, {traces.shape[1]} samples")
    else:
        axes[1].text(0.5, 0.5, 'CHES CTF dataset not found', transform=axes[1].transAxes,
                     ha='center', va='center', fontsize=12, color='gray')
        print(f"  [--] CHES CTF dataset not found at {ches_path}")

    fig.suptitle('Signal-to-Noise Ratio (HW Model) with Selected Points of Interest',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_snr_profiles.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_snr_profiles.pdf'), bbox_inches='tight')
    plt.close()
    generated.append('fig4_snr_profiles.png/.pdf')
    print(f"  [OK] Fig 4: fig4_snr_profiles")

except ImportError:
    print("  [SKIP] Fig 4: h5py not installed, cannot generate SNR plots")
except Exception as e:
    print(f"  [SKIP] Fig 4: Error generating SNR plots: {e}")


# =============================================================================
# FIGURE 5: Cross-Dataset Comparison — Grouped Bar Chart
# =============================================================================
print("\nGenerating Fig 5: Cross-Dataset Comparison...")

# Gather best few-shot results at 20-shot, 1000 traces for each dataset
datasets_for_comparison = {}

for label, fs_df, fs_agg, tmpl_df, cnn_df in [
    ('ASCAD v1\n(AVR)', ascad_hw_fewshot, ascad_hw_fewshot_agg, ascad_hw_template, ascad_hw_cnn),
    ('CHES CTF\n(ARM)', ches_hw_fewshot, ches_hw_fewshot_agg, ches_hw_template, ches_hw_cnn),
]:
    data = {}

    # Few-shot methods at 20-shot, 1000 traces
    for method in FEWSHOT_METHODS:
        if fs_agg is not None:
            cols = fs_agg.columns.tolist()
            at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
            mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
            ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)
            match = fs_agg[
                (fs_agg['Method'] == method) &
                (fs_agg[ks_col] == 20) &
                (fs_agg[at_col] == 1000)
            ]
            if len(match) > 0:
                data[method] = match.iloc[0][mr_col]
        elif fs_df is not None:
            match = fs_df[
                (fs_df['Method'] == method) &
                (fs_df['K-Shot'] == 20) &
                (fs_df['Attack Traces'] == 1000)
            ]
            if len(match) > 0:
                data[method] = match.iloc[0]['Key Rank']

    # Template at 20-shot equivalent, 1000 traces
    if tmpl_df is not None:
        t_match = tmpl_df[
            (tmpl_df['Attack_Traces'] == 1000) &
            (tmpl_df['Training_Config'].str.contains('20-shot'))
        ]
        if len(t_match) > 0:
            data['Template'] = t_match.iloc[0]['Key_Rank']

    # CNN at 20-shot equivalent, 1000 traces
    if cnn_df is not None:
        c_match = cnn_df[
            (cnn_df['Attack_Traces'] == 1000) &
            (cnn_df['Training_Config'].str.contains('20-shot'))
        ]
        if len(c_match) > 0:
            data['CNN'] = c_match.iloc[0]['Key_Rank']

    if data:
        datasets_for_comparison[label] = data

if len(datasets_for_comparison) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    all_methods = ['MAML', 'ProtoNet', 'Siamese', 'Template', 'CNN']
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    dataset_labels = list(datasets_for_comparison.keys())
    n_datasets = len(dataset_labels)
    n_methods = len(all_methods)

    x = np.arange(n_datasets)
    width = 0.15

    for i, (method, color) in enumerate(zip(all_methods, bar_colors)):
        values = []
        for ds_label in dataset_labels:
            ds_data = datasets_for_comparison[ds_label]
            values.append(ds_data.get(method, np.nan))

        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=color, alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        f'{int(val)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Key Rank at 20-shot, 1000 Attack Traces')
    ax.set_title('Cross-Dataset Comparison (HW Model, 20-shot, 1000 Traces)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.legend(loc='upper left', ncol=3)
    ax.set_ylim(0, 270)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cross_dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cross_dataset_comparison.pdf'), bbox_inches='tight')
    plt.close()
    generated.append('fig5_cross_dataset_comparison.png/.pdf')
    print(f"  [OK] Fig 5: fig5_cross_dataset_comparison")
else:
    print(f"  [SKIP] Fig 5: Insufficient data for cross-dataset comparison")


# =============================================================================
# FIGURE 6: Box Plots — Key Rank Distribution (10-run data)
# =============================================================================
print("\nGenerating Fig 6: Box Plots...")

# Try to build box plot data from 10-run combined files
box_data_available = False

for ds_name, combined_df, fig_suffix in [
    ('ASCAD v1 (HW)', ascad_hw_fewshot_combined, 'ascad_hw'),
    ('CHES CTF (HW)', ches_hw_fewshot_combined, 'ches_ctf_hw'),
    ('ASCAD v1 (ID)', ascad_id_fewshot_combined, 'ascad_id'),
]:
    if combined_df is None:
        continue

    # Filter to 20-shot, 1000 traces
    cols = combined_df.columns.tolist()
    at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
    kr_col = next((c for c in cols if c in ['Key_Rank', 'Key Rank']), None)
    ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

    if not (at_col and kr_col and ks_col):
        continue

    subset = combined_df[
        (combined_df[ks_col] == 20) &
        (combined_df[at_col] == 1000)
    ]

    if len(subset) == 0:
        continue

    box_data_available = True
    fig, ax = plt.subplots(figsize=(8, 5))

    methods_present = [m for m in FEWSHOT_METHODS if m in subset['Method'].values]
    box_data = [subset[subset['Method'] == m][kr_col].values for m in methods_present]
    colors_list = [METHOD_COLORS[m] for m in methods_present]

    bp = ax.boxplot(box_data, labels=methods_present, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6))

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (data, color) in enumerate(zip(box_data, colors_list)):
        jitter = np.random.uniform(-0.1, 0.1, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   color=color, alpha=0.5, s=30, zorder=3, edgecolors='black', linewidth=0.5)

    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Rank 10')
    ax.set_ylabel('Key Rank')
    ax.set_title(f'Key Rank Distribution — {ds_name}\n(20-shot, 1000 traces, 10 runs)',
                 fontweight='bold')
    ax.set_ylim(-5, max(max(d.max() for d in box_data if len(d) > 0) + 20, 50))
    ax.legend(loc='upper right')

    plt.tight_layout()
    fname = f'fig6_boxplot_{fig_suffix}'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fname}.png/.pdf')
    print(f"  [OK] Fig 6: {fname}")

if not box_data_available:
    print("  [SKIP] Fig 6: No 10-run combined data found. Run 10-run experiments first.")


# =============================================================================
# FIGURE 7: K-Shot Ablation — Effect of K on Attack Performance
# =============================================================================
print("\nGenerating Fig 7: K-Shot Ablation...")


def plot_kshot_ablation(fewshot_df, fewshot_agg, dataset_name, fig_name, fig_num):
    """Plot key rank at 1000 traces vs k-shot for all methods."""
    if fewshot_df is None and fewshot_agg is None:
        print(f"  [SKIP] Fig {fig_num}: No data for {dataset_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in FEWSHOT_METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        if fewshot_agg is not None:
            cols = fewshot_agg.columns.tolist()
            at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
            mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
            sr_col = next((c for c in cols if c in ['Std_Rank', 'Std_Key_Rank']), None)
            ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

            subset = fewshot_agg[
                (fewshot_agg['Method'] == method) &
                (fewshot_agg[at_col] == 1000)
            ].sort_values(ks_col)

            if len(subset) > 0:
                ax.errorbar(subset[ks_col], subset[mr_col],
                            yerr=subset[sr_col] if sr_col else None,
                            marker=marker, color=color, label=method,
                            linewidth=2, capsize=5, markersize=8)
        elif fewshot_df is not None:
            subset = fewshot_df[
                (fewshot_df['Method'] == method) &
                (fewshot_df['Attack Traces'] == 1000)
            ].sort_values('K-Shot')

            if len(subset) > 0:
                ax.plot(subset['K-Shot'], subset['Key Rank'],
                        marker=marker, color=color, label=method,
                        linewidth=2, markersize=8)

    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('K-Shot (training examples per class)')
    ax.set_ylabel('Key Rank at 1000 Attack Traces')
    ax.set_title(f'K-Shot Ablation — {dataset_name} (HW Model)', fontweight='bold')
    ax.set_xticks(K_SHOTS)
    ax.legend(loc='upper right')
    ax.set_ylim(-5, 260)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fig_name}.png/.pdf')
    print(f"  [OK] Fig {fig_num}: {fig_name}")


plot_kshot_ablation(ascad_hw_fewshot, ascad_hw_fewshot_agg,
                    'ASCAD v1', 'fig7a_kshot_ablation_ascad_hw', '7a')
plot_kshot_ablation(ches_hw_fewshot, ches_hw_fewshot_agg,
                    'CHES CTF 2018', 'fig7b_kshot_ablation_ches_ctf_hw', '7b')


# =============================================================================
# FIGURE 8: Results Heatmap — All methods/k-shots/traces
# =============================================================================
print("\nGenerating Fig 8: Results Heatmaps...")


def plot_heatmap(fewshot_df, fewshot_agg, dataset_name, fig_name, fig_num):
    """Heatmap of key rank for all k-shot x attack-traces combinations per method."""
    if fewshot_df is None and fewshot_agg is None:
        print(f"  [SKIP] Fig {fig_num}: No data for {dataset_name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, method in enumerate(FEWSHOT_METHODS):
        ax = axes[idx]

        if fewshot_agg is not None:
            cols = fewshot_agg.columns.tolist()
            at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
            mr_col = next((c for c in cols if c in ['Mean_Rank', 'Mean_Key_Rank']), None)
            ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

            method_data = fewshot_agg[fewshot_agg['Method'] == method]
            pivot = method_data.pivot(index=ks_col, columns=at_col, values=mr_col)
            title_suffix = '(Mean, 10 runs)'
        elif fewshot_df is not None:
            method_data = fewshot_df[fewshot_df['Method'] == method]
            pivot = method_data.pivot(index='K-Shot', columns='Attack Traces', values='Key Rank')
            title_suffix = '(single seed)'
        else:
            continue

        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'Key Rank'}, ax=ax, vmin=0, vmax=255,
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'{method} {title_suffix}', fontweight='bold')
        ax.set_xlabel('Attack Traces')
        ax.set_ylabel('K-Shot')

    fig.suptitle(f'Key Rank Heatmaps — {dataset_name} (HW Model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fig_name}.png/.pdf')
    print(f"  [OK] Fig {fig_num}: {fig_name}")


plot_heatmap(ascad_hw_fewshot, ascad_hw_fewshot_agg,
             'ASCAD v1', 'fig8a_heatmap_ascad_hw', '8a')
plot_heatmap(ches_hw_fewshot, ches_hw_fewshot_agg,
             'CHES CTF 2018', 'fig8b_heatmap_ches_ctf_hw', '8b')


# =============================================================================
# FIGURE 9: Success Rate — % of configurations reaching rank < threshold
# =============================================================================
print("\nGenerating Fig 9: Success Rate...")

SUCCESS_THRESHOLDS = [1, 5, 10, 25, 50]


def plot_success_rate(fewshot_df, fewshot_combined, dataset_name, fig_name, fig_num):
    """Success rate across k-shot values at 1000 traces."""
    source_df = fewshot_combined if fewshot_combined is not None else fewshot_df
    if source_df is None:
        print(f"  [SKIP] Fig {fig_num}: No data for {dataset_name}")
        return

    cols = source_df.columns.tolist()
    at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
    kr_col = next((c for c in cols if c in ['Key_Rank', 'Key Rank']), None)
    ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)

    if not (at_col and kr_col and ks_col):
        print(f"  [SKIP] Fig {fig_num}: Column mismatch")
        return

    subset = source_df[source_df[at_col] == 1000]
    if len(subset) == 0:
        print(f"  [SKIP] Fig {fig_num}: No data at 1000 traces")
        return

    fig, axes = plt.subplots(1, len(FEWSHOT_METHODS), figsize=(15, 5), sharey=True)

    for idx, method in enumerate(FEWSHOT_METHODS):
        ax = axes[idx]
        method_data = subset[subset['Method'] == method]

        for threshold in SUCCESS_THRESHOLDS:
            rates = []
            for k in K_SHOTS:
                k_data = method_data[method_data[ks_col] == k]
                if len(k_data) > 0:
                    rate = (k_data[kr_col] < threshold).mean() * 100
                    rates.append(rate)
                else:
                    rates.append(0)

            ax.plot(K_SHOTS, rates, marker='o', linewidth=2, markersize=6,
                    label=f'Rank < {threshold}')

        ax.set_xlabel('K-Shot')
        if idx == 0:
            ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{method}', fontweight='bold')
        ax.set_xticks(K_SHOTS)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if idx == len(FEWSHOT_METHODS) - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    n_runs = "10 runs" if fewshot_combined is not None else "single seed"
    fig.suptitle(f'Attack Success Rate — {dataset_name} at 1000 Traces ({n_runs})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.close()
    generated.append(f'{fig_name}.png/.pdf')
    print(f"  [OK] Fig {fig_num}: {fig_name}")


plot_success_rate(ascad_id_fewshot, ascad_id_fewshot_combined,
                  'ASCAD v1 (ID)', 'fig9a_success_rate_ascad_id', '9a')
plot_success_rate(ascad_hw_fewshot, ascad_hw_fewshot_combined,
                  'ASCAD v1 (HW)', 'fig9b_success_rate_ascad_hw', '9b')
plot_success_rate(ches_hw_fewshot, ches_hw_fewshot_combined,
                  'CHES CTF (HW)', 'fig9c_success_rate_ches_ctf_hw', '9c')


# =============================================================================
# FIGURE 10: HW Class Distribution — Both Datasets
# =============================================================================
print("\nGenerating Fig 10: HW Class Distribution...")

try:
    import h5py as h5py_fig10

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hw_dist_generated = False

    for ax, ds_name, ds_path, load_fn in [
        (axes[0], 'ASCAD v1 (ATMega8515)',
         os.path.join(SCA_DIR, 'ASCAD_data', 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5'),
         'ascad'),
        (axes[1], 'CHES CTF 2018 (STM32)',
         os.path.join(SCA_DIR, 'datasets', 'ches_ctf.h5'),
         'ches'),
    ]:
        if not os.path.exists(ds_path):
            ax.text(0.5, 0.5, 'Dataset not found', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            continue

        with h5py_fig10.File(ds_path, 'r') as f:
            if load_fn == 'ascad':
                metadata = f['Profiling_traces']['metadata'][:]
                pt = metadata['plaintext'][:, 0]
                key = metadata['key'][:, 0]
                masks = metadata['masks'][:, 0]
                labels = HW[sbox[pt ^ key] ^ masks]
            else:
                prof_data = np.array(f['profiling_data'], dtype=np.uint8)
                pt = prof_data[:, 0]
                key_byte = prof_data[0, 32]
                labels = HW[sbox[pt ^ key_byte]]

        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        percentages = counts / total * 100

        colors_hw = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique)))
        bars = ax.bar(unique, counts, color=colors_hw, edgecolor='black', linewidth=0.5)

        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Hamming Weight Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'{ds_name}\n(N={total:,})', fontweight='bold')
        ax.set_xticks(range(9))
        hw_dist_generated = True

    if hw_dist_generated:
        fig.suptitle('Hamming Weight Class Distribution (Profiling Traces)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_hw_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_hw_class_distribution.pdf'), bbox_inches='tight')
        plt.close()
        generated.append('fig10_hw_class_distribution.png/.pdf')
        print(f"  [OK] Fig 10: fig10_hw_class_distribution")
    else:
        plt.close()
        print(f"  [SKIP] Fig 10: No datasets found")

except ImportError:
    print("  [SKIP] Fig 10: h5py not installed")
except Exception as e:
    print(f"  [SKIP] Fig 10: Error: {e}")


# =============================================================================
# FIGURE 11: Statistical Significance — Wilcoxon Signed-Rank Tests
# =============================================================================
print("\nGenerating Fig 11: Statistical Significance...")

from scipy import stats as scipy_stats

sig_results = []

for ds_name, combined_df, fig_suffix in [
    ('ASCAD v1 (ID)', ascad_id_fewshot_combined, 'ascad_id'),
    ('ASCAD v1 (HW)', ascad_hw_fewshot_combined, 'ascad_hw'),
    ('CHES CTF (HW)', ches_hw_fewshot_combined, 'ches_ctf_hw'),
]:
    if combined_df is None:
        continue

    cols = combined_df.columns.tolist()
    at_col = next((c for c in cols if c in ['Attack_Traces', 'Attack Traces']), None)
    kr_col = next((c for c in cols if c in ['Key_Rank', 'Key Rank']), None)
    ks_col = next((c for c in cols if c in ['K-Shot', 'K_Shot']), None)
    run_col = next((c for c in cols if c in ['Run', 'run']), None)

    if not (at_col and kr_col and ks_col and run_col):
        continue

    subset = combined_df[
        (combined_df[ks_col] == 20) &
        (combined_df[at_col] == 1000)
    ]

    if len(subset) == 0:
        continue

    # Compare each method pair
    methods_present = [m for m in FEWSHOT_METHODS if m in subset['Method'].values]
    for i in range(len(methods_present)):
        for j in range(i + 1, len(methods_present)):
            m1, m2 = methods_present[i], methods_present[j]
            d1 = subset[subset['Method'] == m1].sort_values(run_col)[kr_col].values
            d2 = subset[subset['Method'] == m2].sort_values(run_col)[kr_col].values

            if len(d1) >= 5 and len(d2) >= 5 and len(d1) == len(d2):
                # Wilcoxon signed-rank test (paired)
                try:
                    stat_w, p_w = scipy_stats.wilcoxon(d1, d2)
                except:
                    stat_w, p_w = np.nan, np.nan

                # Paired t-test
                stat_t, p_t = scipy_stats.ttest_rel(d1, d2)

                # Effect size (Cohen's d for paired samples)
                diff = d1 - d2
                cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

                sig_results.append({
                    'Dataset': ds_name,
                    'Method_1': m1, 'Method_2': m2,
                    'Mean_1': np.mean(d1), 'Mean_2': np.mean(d2),
                    'Wilcoxon_stat': stat_w, 'Wilcoxon_p': p_w,
                    'TTest_stat': stat_t, 'TTest_p': p_t,
                    'Cohens_d': cohens_d,
                    'N': len(d1),
                })

if sig_results:
    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_significance_tests.csv'), index=False)

    # Create significance heatmap per dataset
    for ds_name in sig_df['Dataset'].unique():
        ds_sig = sig_df[sig_df['Dataset'] == ds_name]
        methods_in_ds = sorted(set(ds_sig['Method_1'].tolist() + ds_sig['Method_2'].tolist()))
        n_m = len(methods_in_ds)

        p_matrix = np.ones((n_m, n_m))
        d_matrix = np.zeros((n_m, n_m))

        for _, row in ds_sig.iterrows():
            i = methods_in_ds.index(row['Method_1'])
            j = methods_in_ds.index(row['Method_2'])
            p_matrix[i, j] = row['Wilcoxon_p']
            p_matrix[j, i] = row['Wilcoxon_p']
            d_matrix[i, j] = row['Cohens_d']
            d_matrix[j, i] = -row['Cohens_d']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # P-value heatmap
        mask = np.eye(n_m, dtype=bool)
        annot_p = np.where(mask, '', np.vectorize(lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A')(p_matrix))
        p_display = np.where(mask, np.nan, p_matrix)
        sns.heatmap(p_display, annot=annot_p, fmt='', cmap='RdYlGn', ax=ax1,
                    xticklabels=methods_in_ds, yticklabels=methods_in_ds,
                    vmin=0, vmax=0.1, linewidths=1, linecolor='white',
                    cbar_kws={'label': 'p-value'}, mask=mask)
        ax1.set_title('Wilcoxon p-values\n(green = significant)', fontweight='bold')

        # Effect size heatmap
        annot_d = np.where(mask, '', np.vectorize(lambda x: f'{x:.2f}')(d_matrix))
        d_display = np.where(mask, np.nan, d_matrix)
        sns.heatmap(d_display, annot=annot_d, fmt='', cmap='coolwarm', ax=ax2,
                    xticklabels=methods_in_ds, yticklabels=methods_in_ds,
                    center=0, linewidths=1, linecolor='white',
                    cbar_kws={'label': "Cohen's d"}, mask=mask)
        ax2.set_title("Cohen's d Effect Size\n(row vs column)", fontweight='bold')

        safe_name = ds_name.replace(' ', '_').replace('(', '').replace(')', '')
        fig.suptitle(f'Statistical Significance — {ds_name}\n(20-shot, 1000 traces, paired across 10 runs)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fname = f'fig11_significance_{safe_name}'
        plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.pdf'), bbox_inches='tight')
        plt.close()
        generated.append(f'{fname}.png/.pdf')
        print(f"  [OK] Fig 11: {fname}")

    # Print table
    print("\n  Statistical Significance Summary (20-shot, 1000 traces):")
    print(f"  {'Dataset':18s} {'M1':10s} {'M2':10s} {'Mean1':>7s} {'Mean2':>7s} {'Wilcoxon p':>11s} {'Cohen d':>8s} {'Sig?':>5s}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*11} {'-'*8} {'-'*5}")
    for _, row in sig_df.iterrows():
        sig = '*' if row['Wilcoxon_p'] < 0.05 else ''
        sig += '*' if row['Wilcoxon_p'] < 0.01 else ''
        sig += '*' if row['Wilcoxon_p'] < 0.001 else ''
        print(f"  {row['Dataset']:18s} {row['Method_1']:10s} {row['Method_2']:10s} "
              f"{row['Mean_1']:7.1f} {row['Mean_2']:7.1f} "
              f"{row['Wilcoxon_p']:11.4f} {row['Cohens_d']:8.2f} {sig:>5s}")
else:
    print("  [SKIP] Fig 11: No 10-run combined data available for significance tests")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nGenerated {len(generated)} figure(s) in {OUTPUT_DIR}/:")
for name in generated:
    print(f"  - {name}")

print("\nFor LaTeX, use the PDF versions for vector graphics.")
print("For quick viewing, use the PNG versions.")

if not any([ascad_hw_fewshot_agg, ches_hw_fewshot_agg]):
    print("\nNOTE: 10-run aggregated statistics not found for HW experiments.")
    print("  Run the following for CI bands and box plots:")
    print("    python run_hw_experiments.py         (ASCAD HW)")
    print("    python run_ches_ctf_hw_experiments.py (CHES CTF HW)")

print("\nFor t-SNE, confusion matrix, loss curves, and POI ablation:")
print("  python generate_deep_analysis_plots.py")
print("\n" + "=" * 70)
