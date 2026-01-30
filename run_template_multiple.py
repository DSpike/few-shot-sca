"""
Run Template Attack Baseline Multiple Times
=============================================
Runs template attack 10 times with seeds 42-51
(same as few-shot and CNN baseline experiments)
Produces aggregated statistics for fair comparison.
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

NUM_RUNS = 10  # Same as few-shot experiments
OUTPUT_DIR = 'template_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("TEMPLATE ATTACK - MULTIPLE RUNS (10 runs, seeds 42-51)")
print("=" * 70)

all_results = []

for run_id in range(1, NUM_RUNS + 1):
    seed = 41 + run_id  # Seeds 42-51 (same as few-shot)
    print(f"\n{'=' * 70}")
    print(f"RUN {run_id}/{NUM_RUNS} (seed={seed})")
    print(f"{'=' * 70}")

    result = subprocess.run(
        [sys.executable, "template_attack_baseline.py", "--seed", str(seed)],
        capture_output=True,
        text=True,
        timeout=3600
    )

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"ERROR in run {run_id}:")
        print(result.stderr[-500:])
        continue

    # Read results and save per-run file
    try:
        run_df = pd.read_csv('template_attack_results.csv')
        run_df['Run'] = run_id
        run_df['Seed'] = seed

        run_file = os.path.join(OUTPUT_DIR, f'template_run_{run_id:02d}_results.csv')
        run_df.to_csv(run_file, index=False)
        print(f"Saved: {run_file}")

        all_results.append(run_df)
    except Exception as e:
        print(f"Error reading results for run {run_id}: {e}")

if not all_results:
    print("\nERROR: No successful runs!")
    sys.exit(1)

# =============================================================================
# Aggregate Results
# =============================================================================
print("\n" + "=" * 70)
print("AGGREGATING TEMPLATE ATTACK RESULTS")
print("=" * 70)

combined_df = pd.concat(all_results, ignore_index=True)
combined_file = os.path.join(OUTPUT_DIR, 'all_template_runs_combined.csv')
combined_df.to_csv(combined_file, index=False)
print(f"\nSaved combined results: {combined_file}")

# Compute statistics
stats_records = []
grouped = combined_df.groupby(['Training_Config', 'Training_Size', 'Attack_Traces'])

for (config, train_size, n_traces), group in grouped:
    ranks = group['Key_Rank'].values
    n = len(ranks)
    mean_rank = np.mean(ranks)
    std_rank = np.std(ranks, ddof=1)
    median_rank = np.median(ranks)
    min_rank = np.min(ranks)
    max_rank = np.max(ranks)

    # 95% confidence interval
    if n > 1:
        ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean_rank, scale=std_rank/np.sqrt(n))
    else:
        ci = (mean_rank, mean_rank)

    stats_records.append({
        'Method': 'Template Attack',
        'Training_Config': config,
        'Training_Size': train_size,
        'Attack_Traces': n_traces,
        'Mean_Key_Rank': mean_rank,
        'Std_Key_Rank': std_rank,
        'Median_Key_Rank': median_rank,
        'Min_Key_Rank': min_rank,
        'Max_Key_Rank': max_rank,
        'CI_Lower': ci[0],
        'CI_Upper': ci[1],
        'N_Runs': n
    })

stats_df = pd.DataFrame(stats_records)
stats_file = os.path.join(OUTPUT_DIR, 'template_aggregated_statistics.csv')
stats_df.to_csv(stats_file, index=False)

# Print summary
print(f"\nSaved aggregated statistics: {stats_file}")
print("\n" + "=" * 70)
print("TEMPLATE ATTACK SUMMARY (Mean +/- Std across 10 runs)")
print("=" * 70)

for config in stats_df['Training_Config'].unique():
    config_data = stats_df[stats_df['Training_Config'] == config]
    print(f"\n  {config}:")
    for _, row in config_data.iterrows():
        print(f"    {int(row['Attack_Traces']):>6d} traces -> "
              f"Key Rank: {row['Mean_Key_Rank']:.1f} +/- {row['Std_Key_Rank']:.1f} "
              f"(95% CI: [{row['CI_Lower']:.1f}, {row['CI_Upper']:.1f}])")

# =============================================================================
# Compare with few-shot results (if available)
# =============================================================================
try:
    fewshot_stats = pd.read_csv('experiment_results/aggregated_statistics.csv')
    print("\n" + "=" * 70)
    print("COMPARISON: Template Attack vs Few-Shot Meta-Learning")
    print("=" * 70)

    # Compare at 1000 traces
    template_1000 = stats_df[stats_df['Attack_Traces'] == 1000]

    # Detect column names in few-shot stats (handle different naming conventions)
    fs_at_col = 'Attack_Traces' if 'Attack_Traces' in fewshot_stats.columns else 'Attack Traces'
    fs_mean_col = 'Mean_Rank' if 'Mean_Rank' in fewshot_stats.columns else (
        'Mean_Key_Rank' if 'Mean_Key_Rank' in fewshot_stats.columns else 'Mean Key Rank')
    fs_std_col = 'Std_Rank' if 'Std_Rank' in fewshot_stats.columns else (
        'Std_Key_Rank' if 'Std_Key_Rank' in fewshot_stats.columns else 'Std Key Rank')

    for _, t_row in template_1000.iterrows():
        config = t_row['Training_Config']
        k_shot = t_row['Training_Size'] // 256

        print(f"\n  {config} (k={k_shot}):")
        print(f"    Template Attack:  {t_row['Mean_Key_Rank']:.1f} +/- {t_row['Std_Key_Rank']:.1f}")

        # Find matching few-shot results
        for method in ['MAML', 'ProtoNet', 'Siamese']:
            fs_match = fewshot_stats[
                (fewshot_stats['Method'] == method) &
                (fewshot_stats['K-Shot'] == k_shot) &
                (fewshot_stats[fs_at_col] == 1000)
            ]
            if len(fs_match) > 0:
                fs_row = fs_match.iloc[0]
                fs_mean = fs_row[fs_mean_col]
                fs_std = fs_row[fs_std_col]
                print(f"    {method:15s}: {fs_mean:.1f} +/- {fs_std:.1f}")

except FileNotFoundError:
    print("\nFew-shot results not found. Run experiments first to compare.")

print("\nTemplate attack baseline complete!")
print("\nOutput files:")
print(f"  Individual runs: {OUTPUT_DIR}/template_run_XX_results.csv")
print(f"  Combined:        {combined_file}")
print(f"  Statistics:      {stats_file}")
