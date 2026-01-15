"""
Run Multiple Experiments for Confidence Intervals
==================================================
Runs the comprehensive few-shot study 5-10 times and aggregates results
with mean ± standard deviation for publication
"""

import subprocess
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Configuration
NUM_RUNS = 10  # 10 runs for robust statistics and tighter confidence intervals
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print(f"RUNNING {NUM_RUNS} INDEPENDENT EXPERIMENTS")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

all_results = []

for run_id in range(1, NUM_RUNS + 1):
    # Use standard ML seeds: 42, 43, 44, 45, 46
    # Seed 42 is widely used in ML research (reference: Hitchhiker's Guide)
    seed = 41 + run_id

    print(f"\n{'='*70}")
    print(f"RUN {run_id}/{NUM_RUNS} (seed={seed})")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Run the comprehensive study with fixed seed for reproducibility
    # Uses seeds 42-46 following ML community convention
    result = subprocess.run(
        ["python", "comprehensive_few_shot_study.py", "--seed", str(seed)],
        cwd=r"C:\Users\Dspike\Documents\sca",
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"✓ Run {run_id} completed successfully ({elapsed/60:.1f} minutes)")

        # Read results
        df = pd.read_csv("few_shot_sca_results.csv")
        df['Run'] = run_id
        all_results.append(df)

        # Save individual run
        run_file = os.path.join(OUTPUT_DIR, f"run_{run_id:02d}_results.csv")
        df.to_csv(run_file, index=False)
        print(f"  Saved to: {run_file}")

    else:
        print(f"✗ Run {run_id} failed!")
        print(f"Error: {result.stderr}")
        continue

# Aggregate results
if len(all_results) > 0:
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    # Combine all runs
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "all_runs_combined.csv"), index=False)

    # Calculate statistics with 95% CI
    from scipy import stats as scipy_stats

    def calc_ci95(group):
        n = len(group)
        mean = group.mean()
        std = group.std()
        # 95% CI using t-distribution
        ci = scipy_stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
        return pd.Series({
            'Mean_Rank': mean,
            'Std_Rank': std,
            'Min_Rank': group.min(),
            'Max_Rank': group.max(),
            'Count': n,
            'CI95_Lower': ci[0],
            'CI95_Upper': ci[1],
            'CI95_Width': ci[1] - ci[0]
        })

    # Group and calculate statistics
    grouped = combined_df.groupby(['Method', 'K-Shot', 'Attack Traces'])['Key Rank']
    stats_list = []

    for (method, k_shot, traces), group in grouped:
        stats = calc_ci95(group)
        stats['Method'] = method
        stats['K-Shot'] = k_shot
        stats['Attack_Traces'] = traces
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Reorder columns
    cols = ['Method', 'K-Shot', 'Attack_Traces', 'Mean_Rank', 'Std_Rank', 'Min_Rank',
            'Max_Rank', 'Count', 'CI95_Lower', 'CI95_Upper', 'CI95_Width']
    stats_df = stats_df[cols]

    # Save statistics
    stats_file = os.path.join(OUTPUT_DIR, "aggregated_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to: {stats_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Key Rank at 1000 Attack Traces (Mean ± Std)")
    print("="*70)

    # Format for publication
    for method in ['MAML', 'ProtoNet', 'Siamese']:
        if method not in stats_df['Method'].values:
            continue
        print(f"\n{method}:")
        for k_shot in [5, 10, 15, 20]:
            row = stats_df[(stats_df['Method'] == method) &
                          (stats_df['K-Shot'] == k_shot) &
                          (stats_df['Attack_Traces'] == 1000)]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {k_shot}-shot: {r['Mean_Rank']:.1f} ± {r['Std_Rank']:.1f} "
                      f"[95% CI: {r['CI95_Lower']:.1f} - {r['CI95_Upper']:.1f}]")

    # Find best configuration overall
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS (Lowest Mean Rank)")
    print("="*70)

    for method in ['MAML', 'ProtoNet', 'Siamese']:
        if method not in stats_df['Method'].values:
            continue
        method_stats = stats_df[stats_df['Method'] == method]
        best_idx = method_stats['Mean_Rank'].idxmin()
        best = method_stats.loc[best_idx]
        print(f"{method:10s}: {best['Mean_Rank']:.1f} ± {best['Std_Rank']:.1f} "
              f"[95% CI: {best['CI95_Lower']:.1f} - {best['CI95_Upper']:.1f}] "
              f"({int(best['K-Shot'])}-shot, {int(best['Attack_Traces'])} traces)")

    print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
    print(f"✓ Total time: {sum([time.time() - start_time])/60:.1f} minutes")

else:
    print("\n✗ No successful runs to aggregate!")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Review aggregated_statistics.csv")
print("2. Run generate_plots.py to create publication figures")
print("3. Use statistics in your paper (mean ± std for each configuration)")
