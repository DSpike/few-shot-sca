"""
Run All HW (Hamming Weight) Experiments Multiple Times
======================================================
Runs all three HW experiment scripts 10 times each (seeds 42-51),
then aggregates results and compares HW vs ID models.

Scripts run:
  1. hw_few_shot_study.py        - Few-shot meta-learning (MAML, ProtoNet, Siamese)
  2. hw_template_attack_baseline.py - Classical Template Attack baseline
  3. hw_baseline_standard_training.py - Standard CNN baseline
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

NUM_RUNS = 10  # Same as ID experiments
OUTPUT_DIR = 'hw_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================
EXPERIMENTS = [
    {
        'name': 'HW Few-Shot',
        'script': 'hw_few_shot_study.py',
        'output_csv': 'hw_few_shot_sca_results.csv',
        'prefix': 'hw_fewshot',
        'timeout': 7200,  # 2 hours per run
    },
    {
        'name': 'HW Template Attack',
        'script': 'hw_template_attack_baseline.py',
        'output_csv': 'hw_template_attack_results.csv',
        'prefix': 'hw_template',
        'timeout': 3600,
    },
    {
        'name': 'HW Standard CNN',
        'script': 'hw_baseline_standard_training.py',
        'output_csv': 'hw_baseline_standard_cnn_results.csv',
        'prefix': 'hw_cnn',
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
            'Leakage_Model': 'HW'
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
print("HW LEAKAGE MODEL - FULL EXPERIMENT SUITE")
print(f"({NUM_RUNS} runs per experiment, seeds 42-{41 + NUM_RUNS})")
print("=" * 70)

all_stats = {}

for experiment in EXPERIMENTS:
    results = run_experiment(experiment)
    stats = aggregate_results(results, experiment)
    print_summary(stats, experiment)
    all_stats[experiment['name']] = stats

# =============================================================================
# Cross-comparison: HW vs ID (if ID results available)
# =============================================================================
print("\n" + "#" * 70)
print("# CROSS-COMPARISON: HW vs ID Leakage Model")
print("#" * 70)

# Try to load ID few-shot results
try:
    id_fewshot = pd.read_csv('experiment_results/aggregated_statistics.csv')
    hw_fewshot = all_stats.get('HW Few-Shot')

    if hw_fewshot is not None and id_fewshot is not None:
        print("\n  Few-Shot Methods at 1000 Attack Traces:")
        print(f"  {'Method':15s} {'K-Shot':>6s}  {'ID Model':>20s}  {'HW Model':>20s}  {'Improvement':>12s}")
        print(f"  {'-'*15} {'-'*6}  {'-'*20}  {'-'*20}  {'-'*12}")

        # Detect ID column names
        id_cols = id_fewshot.columns.tolist()
        id_at_col = next((c for c in id_cols if c in ['Attack_Traces', 'Attack Traces']), None)
        id_mean_col = next((c for c in id_cols if c in ['Mean_Rank', 'Mean_Key_Rank', 'Mean Key Rank']), None)
        id_ks_col = next((c for c in id_cols if c in ['K-Shot', 'K_Shot']), None)

        if id_at_col and id_mean_col and id_ks_col:
            for method in ['MAML', 'ProtoNet', 'Siamese']:
                for k in [5, 10, 15, 20]:
                    # ID result
                    id_match = id_fewshot[
                        (id_fewshot['Method'] == method) &
                        (id_fewshot[id_ks_col] == k) &
                        (id_fewshot[id_at_col] == 1000)
                    ]
                    # HW result
                    hw_match = hw_fewshot[
                        (hw_fewshot['Method'] == method) &
                        (hw_fewshot['K-Shot'] == k) &
                        (hw_fewshot['Attack Traces'] == 1000)
                    ]

                    if len(id_match) > 0 and len(hw_match) > 0:
                        id_rank = id_match.iloc[0][id_mean_col]
                        hw_rank = hw_match.iloc[0]['Mean_Rank']
                        improvement = id_rank - hw_rank
                        sign = "+" if improvement > 0 else ""
                        print(f"  {method:15s} {k:>6d}  "
                              f"{id_rank:>8.1f}              "
                              f"{hw_rank:>8.1f}              "
                              f"{sign}{improvement:>8.1f}")

except FileNotFoundError:
    print("\n  ID few-shot results not found (experiment_results/aggregated_statistics.csv)")
    print("  Run ID experiments first to compare.")

# Try to load ID template results
try:
    id_template = pd.read_csv('template_results/template_aggregated_statistics.csv')
    hw_template = all_stats.get('HW Template Attack')

    if hw_template is not None and id_template is not None:
        print("\n  Template Attack at 1000 Traces:")
        print(f"  {'Config':25s}  {'ID Model':>20s}  {'HW Model':>20s}")
        print(f"  {'-'*25}  {'-'*20}  {'-'*20}")

        for _, hw_row in hw_template[hw_template['Attack_Traces'] == 1000].iterrows():
            config = hw_row['Training_Config']
            hw_rank = hw_row['Mean_Rank']

            # Find matching ID config
            id_match = id_template[
                (id_template['Training_Config'].str.contains('shot', case=False)) &
                (id_template['Attack_Traces'] == 1000)
            ]

            print(f"  {config:25s}  {'N/A':>20s}  {hw_rank:>8.1f}")

except FileNotFoundError:
    print("\n  ID template results not found.")

print("\n" + "=" * 70)
print("HW EXPERIMENT SUITE COMPLETE")
print("=" * 70)
print(f"\nAll results saved in: {OUTPUT_DIR}/")
print(f"  Few-shot:        {OUTPUT_DIR}/hw_fewshot_*")
print(f"  Template Attack: {OUTPUT_DIR}/hw_template_*")
print(f"  Standard CNN:    {OUTPUT_DIR}/hw_cnn_*")
