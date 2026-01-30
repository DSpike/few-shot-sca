"""
Run All CHES CTF HW (Hamming Weight) Experiments Multiple Times
================================================================
Runs all three CHES CTF HW experiment scripts 10 times each (seeds 42-51),
then aggregates results and compares with ASCAD HW results.

Scripts run:
  1. ches_ctf_hw_few_shot_study.py      - Few-shot meta-learning (MAML, ProtoNet, Siamese)
  2. ches_ctf_hw_template_attack.py     - Classical Template Attack baseline
  3. ches_ctf_hw_baseline_cnn.py        - Standard CNN baseline
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

NUM_RUNS = 10  # Same as ASCAD experiments
OUTPUT_DIR = 'ches_ctf_hw_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================
EXPERIMENTS = [
    {
        'name': 'CHES CTF HW Few-Shot',
        'script': 'ches_ctf_hw_few_shot_study.py',
        'output_csv': 'ches_ctf_hw_few_shot_results.csv',
        'prefix': 'ches_ctf_hw_fewshot',
        'timeout': 7200,  # 2 hours per run
    },
    {
        'name': 'CHES CTF HW Template Attack',
        'script': 'ches_ctf_hw_template_attack.py',
        'output_csv': 'ches_ctf_hw_template_attack_results.csv',
        'prefix': 'ches_ctf_hw_template',
        'timeout': 3600,
    },
    {
        'name': 'CHES CTF HW Standard CNN',
        'script': 'ches_ctf_hw_baseline_cnn.py',
        'output_csv': 'ches_ctf_hw_baseline_standard_cnn_results.csv',
        'prefix': 'ches_ctf_hw_cnn',
        'timeout': 3600,
    },
]


def run_experiment(experiment, num_runs=NUM_RUNS):
    """Run a single experiment script multiple times."""
    name = experiment['name']
    script = experiment['script']
    output_csv = experiment['output_csv']
    prefix = experiment['prefix']
    timeout = experiment['timeout']

    print(f"\n{'#' * 70}")
    print(f"# {name} - {num_runs} RUNS (seeds 42-{41 + num_runs})")
    print(f"{'#' * 70}")

    all_results = []

    for run_id in range(1, num_runs + 1):
        seed = 41 + run_id  # Seeds 42-51

        print(f"\n{'=' * 70}")
        print(f"RUN {run_id}/{num_runs} - {name} (seed={seed})")
        print(f"{'=' * 70}")

        try:
            result = subprocess.run(
                [sys.executable, script, "--seed", str(seed)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Print last 500 chars of output
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            if result.returncode != 0:
                print(f"ERROR in run {run_id}:")
                print(result.stderr[-500:])
                continue

            # Read and save per-run results
            run_df = pd.read_csv(output_csv)
            run_df['Run'] = run_id
            run_df['Seed'] = seed

            run_file = os.path.join(OUTPUT_DIR, f'{prefix}_run_{run_id:02d}_results.csv')
            run_df.to_csv(run_file, index=False)
            print(f"Saved: {run_file}")

            all_results.append(run_df)

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT in run {run_id} (>{timeout}s)")
        except Exception as e:
            print(f"Error in run {run_id}: {e}")

    return all_results


def aggregate_results(all_results, experiment):
    """Compute aggregated statistics from multiple runs."""
    if not all_results:
        print(f"\nNo results to aggregate for {experiment['name']}")
        return None

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_file = os.path.join(OUTPUT_DIR, f"{experiment['prefix']}_all_runs_combined.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\nSaved combined: {combined_file}")

    # Detect grouping columns based on experiment type
    if 'K-Shot' in combined_df.columns:
        # Few-shot experiment
        group_cols = ['Method', 'K-Shot', 'Attack Traces']
        rank_col = 'Key Rank'
    elif 'Training_Config' in combined_df.columns:
        # Baseline experiment
        group_cols = ['Method', 'Training_Config', 'Training_Size', 'Attack_Traces']
        rank_col = 'Key_Rank'
    else:
        print(f"Unknown CSV format for {experiment['name']}")
        return None

    stats_records = []
    grouped = combined_df.groupby(group_cols)

    for group_key, group in grouped:
        ranks = group[rank_col].values
        n = len(ranks)
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks, ddof=1) if n > 1 else 0.0
        median_rank = np.median(ranks)
        min_rank = np.min(ranks)
        max_rank = np.max(ranks)

        # 95% CI
        if n > 1:
            ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean_rank,
                                        scale=std_rank/np.sqrt(n))
        else:
            ci = (mean_rank, mean_rank)

        record = {
            'Mean_Rank': mean_rank,
            'Std_Rank': std_rank,
            'Median_Rank': median_rank,
            'Min_Rank': min_rank,
            'Max_Rank': max_rank,
            'CI_Lower': ci[0],
            'CI_Upper': ci[1],
            'N_Runs': n,
            'Leakage_Model': 'HW',
            'Dataset': 'CHES_CTF'
        }

        # Add group columns
        if isinstance(group_key, tuple):
            for col, val in zip(group_cols, group_key):
                record[col] = val
        else:
            record[group_cols[0]] = group_key

        stats_records.append(record)

    stats_df = pd.DataFrame(stats_records)
    stats_file = os.path.join(OUTPUT_DIR, f"{experiment['prefix']}_aggregated_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics: {stats_file}")

    return stats_df


def print_summary(stats_df, experiment):
    """Print a summary of aggregated statistics."""
    if stats_df is None:
        return

    print(f"\n{'=' * 70}")
    print(f"{experiment['name']} SUMMARY (Mean +/- Std across runs)")
    print(f"{'=' * 70}")

    if 'K-Shot' in stats_df.columns:
        # Few-shot format
        for method in stats_df['Method'].unique():
            method_data = stats_df[stats_df['Method'] == method]
            print(f"\n  {method}:")
            for _, row in method_data.sort_values(['K-Shot', 'Attack Traces']).iterrows():
                if row['Attack Traces'] == 1000:
                    print(f"    {int(row['K-Shot']):2d}-shot, {int(row['Attack Traces']):>6d} traces -> "
                          f"Rank: {row['Mean_Rank']:.1f} +/- {row['Std_Rank']:.1f} "
                          f"(95% CI: [{row['CI_Lower']:.1f}, {row['CI_Upper']:.1f}])")
    elif 'Training_Config' in stats_df.columns:
        # Baseline format
        for config in stats_df['Training_Config'].unique():
            config_data = stats_df[stats_df['Training_Config'] == config]
            print(f"\n  {config}:")
            for _, row in config_data.iterrows():
                at_col = 'Attack_Traces' if 'Attack_Traces' in stats_df.columns else 'Attack Traces'
                if row[at_col] == 1000:
                    print(f"    {int(row[at_col]):>6d} traces -> "
                          f"Rank: {row['Mean_Rank']:.1f} +/- {row['Std_Rank']:.1f} "
                          f"(95% CI: [{row['CI_Lower']:.1f}, {row['CI_Upper']:.1f}])")


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("CHES CTF 2018 - HW LEAKAGE MODEL - FULL EXPERIMENT SUITE")
print(f"({NUM_RUNS} runs per experiment, seeds 42-{41 + NUM_RUNS})")
print("=" * 70)

all_stats = {}

for experiment in EXPERIMENTS:
    results = run_experiment(experiment)
    stats = aggregate_results(results, experiment)
    print_summary(stats, experiment)
    all_stats[experiment['name']] = stats

# =============================================================================
# Cross-comparison: CHES CTF vs ASCAD (if ASCAD HW results available)
# =============================================================================
print("\n" + "#" * 70)
print("# CROSS-DATASET COMPARISON: CHES CTF vs ASCAD (HW Model)")
print("#" * 70)

# Try to load ASCAD HW few-shot results
try:
    ascad_fewshot = pd.read_csv('hw_results/hw_fewshot_aggregated_statistics.csv')
    ches_fewshot = all_stats.get('CHES CTF HW Few-Shot')

    if ches_fewshot is not None and ascad_fewshot is not None:
        print("\n  Few-Shot Methods at 1000 Attack Traces:")
        print(f"  {'Method':15s} {'K-Shot':>6s}  {'ASCAD (AVR)':>20s}  {'CHES CTF (ARM)':>20s}")
        print(f"  {'-'*15} {'-'*6}  {'-'*20}  {'-'*20}")

        for method in ['MAML', 'ProtoNet', 'Siamese']:
            for k in [5, 10, 15, 20]:
                # ASCAD result
                ascad_match = ascad_fewshot[
                    (ascad_fewshot['Method'] == method) &
                    (ascad_fewshot['K-Shot'] == k) &
                    (ascad_fewshot['Attack Traces'] == 1000)
                ]
                # CHES CTF result
                ches_match = ches_fewshot[
                    (ches_fewshot['Method'] == method) &
                    (ches_fewshot['K-Shot'] == k) &
                    (ches_fewshot['Attack Traces'] == 1000)
                ]

                ascad_str = f"{ascad_match.iloc[0]['Mean_Rank']:.1f} +/- {ascad_match.iloc[0]['Std_Rank']:.1f}" if len(ascad_match) > 0 else "N/A"
                ches_str = f"{ches_match.iloc[0]['Mean_Rank']:.1f} +/- {ches_match.iloc[0]['Std_Rank']:.1f}" if len(ches_match) > 0 else "N/A"

                print(f"  {method:15s} {k:>6d}  {ascad_str:>20s}  {ches_str:>20s}")

except FileNotFoundError:
    print("\n  ASCAD HW few-shot results not found (hw_results/hw_fewshot_aggregated_statistics.csv)")
    print("  Run ASCAD HW experiments first to compare across datasets.")

# Try to load ASCAD HW template results
try:
    ascad_template = pd.read_csv('hw_results/hw_template_aggregated_statistics.csv')
    ches_template = all_stats.get('CHES CTF HW Template Attack')

    if ches_template is not None and ascad_template is not None:
        print("\n  Template Attack at 1000 Traces (HW Model):")
        print(f"  {'Config':25s}  {'ASCAD (AVR)':>20s}  {'CHES CTF (ARM)':>20s}")
        print(f"  {'-'*25}  {'-'*20}  {'-'*20}")

        for _, ches_row in ches_template[ches_template['Attack_Traces'] == 1000].iterrows():
            config = ches_row['Training_Config']
            ches_str = f"{ches_row['Mean_Rank']:.1f} +/- {ches_row['Std_Rank']:.1f}"

            ascad_match = ascad_template[
                (ascad_template['Training_Config'] == config) &
                (ascad_template['Attack_Traces'] == 1000)
            ]
            ascad_str = f"{ascad_match.iloc[0]['Mean_Rank']:.1f} +/- {ascad_match.iloc[0]['Std_Rank']:.1f}" if len(ascad_match) > 0 else "N/A"

            print(f"  {config:25s}  {ascad_str:>20s}  {ches_str:>20s}")

except FileNotFoundError:
    print("\n  ASCAD HW template results not found.")

# Try to load ASCAD HW CNN results
try:
    ascad_cnn = pd.read_csv('hw_results/hw_cnn_aggregated_statistics.csv')
    ches_cnn = all_stats.get('CHES CTF HW Standard CNN')

    if ches_cnn is not None and ascad_cnn is not None:
        print("\n  Standard CNN at 1000 Traces (HW Model):")
        print(f"  {'Config':25s}  {'ASCAD (AVR)':>20s}  {'CHES CTF (ARM)':>20s}")
        print(f"  {'-'*25}  {'-'*20}  {'-'*20}")

        for _, ches_row in ches_cnn[ches_cnn['Attack_Traces'] == 1000].iterrows():
            config = ches_row['Training_Config']
            ches_str = f"{ches_row['Mean_Rank']:.1f} +/- {ches_row['Std_Rank']:.1f}"

            ascad_match = ascad_cnn[
                (ascad_cnn['Training_Config'] == config) &
                (ascad_cnn['Attack_Traces'] == 1000)
            ]
            ascad_str = f"{ascad_match.iloc[0]['Mean_Rank']:.1f} +/- {ascad_match.iloc[0]['Std_Rank']:.1f}" if len(ascad_match) > 0 else "N/A"

            print(f"  {config:25s}  {ascad_str:>20s}  {ches_str:>20s}")

except FileNotFoundError:
    print("\n  ASCAD HW CNN results not found.")

print("\n" + "=" * 70)
print("CHES CTF HW EXPERIMENT SUITE COMPLETE")
print("=" * 70)
print(f"\nAll results saved in: {OUTPUT_DIR}/")
print(f"  Few-shot:        {OUTPUT_DIR}/ches_ctf_hw_fewshot_*")
print(f"  Template Attack: {OUTPUT_DIR}/ches_ctf_hw_template_*")
print(f"  Standard CNN:    {OUTPUT_DIR}/ches_ctf_hw_cnn_*")
