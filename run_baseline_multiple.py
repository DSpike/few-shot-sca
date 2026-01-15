"""
Run Baseline Experiments Multiple Times
========================================
Runs standard CNN baseline 10 times with seeds 42-51
to match the few-shot experiments for fair comparison
"""

import subprocess
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Configuration - MUST MATCH few-shot experiments!
NUM_RUNS = 10  # Same as run_multiple_experiments.py
OUTPUT_DIR = "baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print(f"RUNNING {NUM_RUNS} BASELINE EXPERIMENTS (STANDARD CNN)")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Seeds: 42-51 (matching few-shot experiments)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

all_results = []

for run_id in range(1, NUM_RUNS + 1):
    # Use SAME seed range as few-shot experiments (42-51)
    seed = 41 + run_id

    print(f"\n{'='*70}")
    print(f"BASELINE RUN {run_id}/{NUM_RUNS} (seed={seed})")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Run baseline with fixed seed
    result = subprocess.run(
        ["python", "baseline_standard_training.py", "--seed", str(seed)],
        cwd=r"C:\Users\Dspike\Documents\sca",
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"✓ Run {run_id} completed successfully ({elapsed/60:.1f} minutes)")

        # Read results
        df = pd.read_csv("baseline_standard_cnn_results.csv")
        df['Run'] = run_id
        df['Seed'] = seed
        all_results.append(df)

        # Save individual run
        run_file = os.path.join(OUTPUT_DIR, f"baseline_run_{run_id:02d}_results.csv")
        df.to_csv(run_file, index=False)
        print(f"  Saved to: {run_file}")

    else:
        print(f"✗ Run {run_id} failed!")
        print(f"Error: {result.stderr}")
        continue

# Aggregate results
if len(all_results) > 0:
    print(f"\n{'='*70}")
    print("AGGREGATING BASELINE RESULTS")
    print(f"{'='*70}\n")

    # Combine all runs
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, "all_baseline_runs_combined.csv"), index=False)

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

    # Group by Training_Size and Attack_Traces
    grouped = combined_df.groupby(['Training_Size', 'Attack_Traces'])['Key_Rank']
    stats_list = []

    for (training_size, attack_traces), group in grouped:
        stats = calc_ci95(group)
        stats['Training_Size'] = training_size
        stats['Attack_Traces'] = attack_traces
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Reorder columns
    cols = ['Training_Size', 'Attack_Traces', 'Mean_Rank', 'Std_Rank', 'Min_Rank',
            'Max_Rank', 'Count', 'CI95_Lower', 'CI95_Upper', 'CI95_Width']
    stats_df = stats_df[cols]

    # Save statistics
    stats_file = os.path.join(OUTPUT_DIR, "baseline_aggregated_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to: {stats_file}")

    # Print summary table
    print("\n" + "="*70)
    print("BASELINE SUMMARY: Key Rank at 1000 Attack Traces (Mean ± Std)")
    print("="*70)

    # Map training sizes to k-shot equivalents
    training_size_map = {
        1280: '5-shot',
        2560: '10-shot',
        3840: '15-shot',
        5120: '20-shot',
        50000: 'Full (50k)'
    }

    for training_size in [1280, 2560, 3840, 5120, 50000]:
        row = stats_df[(stats_df['Training_Size'] == training_size) &
                      (stats_df['Attack_Traces'] == 1000)]
        if len(row) > 0:
            r = row.iloc[0]
            k_shot_label = training_size_map.get(training_size, f'{training_size}')
            print(f"  {k_shot_label:12s} ({training_size:5d} traces): "
                  f"{r['Mean_Rank']:.1f} ± {r['Std_Rank']:.1f} "
                  f"[95% CI: {r['CI95_Lower']:.1f} - {r['CI95_Upper']:.1f}]")

    # Compare with few-shot results if available
    print("\n" + "="*70)
    print("COMPARISON WITH FEW-SHOT META-LEARNING")
    print("="*70)

    try:
        fewshot_stats = pd.read_csv('experiment_results/aggregated_statistics.csv')
        print("\n✓ Found few-shot results for comparison\n")

        # Compare at 1000 attack traces
        print("At 1000 Attack Traces:")
        print(f"{'Training Size':<15} {'Baseline CNN':<25} {'MAML':<25} {'ProtoNet':<25} {'Siamese':<25}")
        print("-" * 115)

        for training_size, k_shot in [(1280, 5), (2560, 10), (3840, 15), (5120, 20)]:
            # Baseline
            baseline_row = stats_df[(stats_df['Training_Size'] == training_size) &
                                   (stats_df['Attack_Traces'] == 1000)]

            # Few-shot methods
            maml_row = fewshot_stats[(fewshot_stats['Method'] == 'MAML') &
                                    (fewshot_stats['K-Shot'] == k_shot) &
                                    (fewshot_stats['Attack_Traces'] == 1000)]

            protonet_row = fewshot_stats[(fewshot_stats['Method'] == 'ProtoNet') &
                                        (fewshot_stats['K-Shot'] == k_shot) &
                                        (fewshot_stats['Attack_Traces'] == 1000)]

            siamese_row = fewshot_stats[(fewshot_stats['Method'] == 'Siamese') &
                                       (fewshot_stats['K-Shot'] == k_shot) &
                                       (fewshot_stats['Attack_Traces'] == 1000)]

            if len(baseline_row) > 0:
                b = baseline_row.iloc[0]
                baseline_str = f"{b['Mean_Rank']:.1f} ± {b['Std_Rank']:.1f}"

                maml_str = "N/A"
                protonet_str = "N/A"
                siamese_str = "N/A"

                if len(maml_row) > 0:
                    m = maml_row.iloc[0]
                    maml_str = f"{m['Mean_Rank']:.1f} ± {m['Std_Rank']:.1f}"

                if len(protonet_row) > 0:
                    p = protonet_row.iloc[0]
                    protonet_str = f"{p['Mean_Rank']:.1f} ± {p['Std_Rank']:.1f}"

                if len(siamese_row) > 0:
                    s = siamese_row.iloc[0]
                    siamese_str = f"{s['Mean_Rank']:.1f} ± {s['Std_Rank']:.1f}"

                label = f"{training_size} ({k_shot}-shot)"
                print(f"{label:<15} {baseline_str:<25} {maml_str:<25} {protonet_str:<25} {siamese_str:<25}")

        # Statistical comparison
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE (vs Baseline)")
        print("="*70)
        print("\nIndependent t-tests (Few-Shot vs Baseline):")

        for training_size, k_shot in [(1280, 5), (2560, 10), (3840, 15), (5120, 20)]:
            print(f"\n{k_shot}-shot ({training_size} traces):")

            # Get baseline runs
            baseline_runs = combined_df[(combined_df['Training_Size'] == training_size) &
                                       (combined_df['Attack_Traces'] == 1000)]['Key_Rank']

            # Load few-shot combined results
            try:
                fewshot_combined = pd.read_csv('experiment_results/all_runs_combined.csv')

                for method in ['MAML', 'ProtoNet', 'Siamese']:
                    fewshot_runs = fewshot_combined[(fewshot_combined['Method'] == method) &
                                                   (fewshot_combined['K-Shot'] == k_shot) &
                                                   (fewshot_combined['Attack Traces'] == 1000)]['Key Rank']

                    if len(baseline_runs) > 0 and len(fewshot_runs) > 0:
                        t_stat, p_value = scipy_stats.ttest_ind(fewshot_runs, baseline_runs)

                        if p_value < 0.001:
                            sig = "***"
                        elif p_value < 0.01:
                            sig = "**"
                        elif p_value < 0.05:
                            sig = "*"
                        else:
                            sig = "ns"

                        # Calculate improvement
                        baseline_mean = baseline_runs.mean()
                        fewshot_mean = fewshot_runs.mean()
                        improvement = ((baseline_mean - fewshot_mean) / baseline_mean) * 100

                        print(f"  {method:10s}: improvement={improvement:+6.1f}%, t={t_stat:6.2f}, p={p_value:.4f} {sig}")

            except FileNotFoundError:
                print("    (Few-shot combined results not found)")
                break

        print("\n  * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")

    except FileNotFoundError:
        print("\n! Few-shot results not found")
        print("  Run: python run_multiple_experiments.py")

    print(f"\n✓ All baseline results saved to: {OUTPUT_DIR}/")

else:
    print("\n✗ No successful runs to aggregate!")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Review baseline_aggregated_statistics.csv")
print("2. Compare with few-shot results (above)")
print("3. Use generate_plots.py to create comparison figures")
print("4. Include baseline comparison in your paper")
print("\nBaseline experiments complete!")
