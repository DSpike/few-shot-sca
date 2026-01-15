# Statistical Robustness with 95% Confidence Intervals

## Overview

The **`run_multiple_experiments.py`** script provides statistical robustness for your publication by running experiments multiple times and calculating:
- **Mean ± Standard Deviation**
- **95% Confidence Intervals (CI)**
- **Min/Max values**
- **Sample size (n runs)**

---

## What is 95% Confidence Interval?

**95% CI** means: "We are 95% confident that the true population mean lies within this interval"

### Mathematical Formula
```
CI = mean ± t(α/2, n-1) × (std / √n)
```

Where:
- `t(α/2, n-1)` = t-distribution critical value for 95% confidence
- `std` = standard deviation
- `n` = number of runs
- `α = 0.05` for 95% confidence

### Interpretation
- **Narrow CI**: Results are consistent (low variance)
- **Wide CI**: Results vary significantly (high variance)
- **Non-overlapping CIs**: Significant difference between methods

---

## How to Run Multiple Experiments

### Step 1: Configure Number of Runs

Edit `run_multiple_experiments.py`:
```python
NUM_RUNS = 5   # Minimum for statistics
# NUM_RUNS = 10  # Better for publication
```

**Recommendations:**
- **5 runs**: Minimum for 95% CI (faster, ~5 hours)
- **10 runs**: Better statistics (more robust, ~10 hours)
- **30 runs**: Gold standard (very robust, ~30 hours)

### Step 2: Run the Script

```bash
python run_multiple_experiments.py
```

**What it does:**
1. Runs `comprehensive_few_shot_study.py` N times
2. Saves each run separately: `run_01_results.csv`, `run_02_results.csv`, ...
3. Combines all runs: `all_runs_combined.csv`
4. Calculates statistics: `aggregated_statistics.csv`

**Expected Runtime:**
- Single run: ~60 minutes (100 epochs)
- 5 runs: ~300 minutes (~5 hours)
- 10 runs: ~600 minutes (~10 hours)

**Pro tip:** Run overnight or over weekend!

---

## Output Files

### 1. Individual Runs
```
experiment_results/
├── run_01_results.csv
├── run_02_results.csv
├── run_03_results.csv
├── run_04_results.csv
└── run_05_results.csv
```

Each contains: Method, K-Shot, Attack Traces, Key Rank

### 2. Combined Results
```
experiment_results/all_runs_combined.csv
```

All individual runs stacked together with `Run` column.

### 3. Aggregated Statistics (MAIN FILE)
```
experiment_results/aggregated_statistics.csv
```

**Columns:**
- `Method`: MAML, ProtoNet, Siamese
- `K-Shot`: 5, 10, 15, 20
- `Attack_Traces`: 100, 500, 1000, 2000, 5000, 10000
- `Mean_Rank`: Average key rank across all runs
- `Std_Rank`: Standard deviation
- `Min_Rank`: Best result across runs
- `Max_Rank`: Worst result across runs
- `Count`: Number of runs
- **`CI95_Lower`**: Lower bound of 95% CI
- **`CI95_Upper`**: Upper bound of 95% CI
- **`CI95_Width`**: Width of CI (upper - lower)

---

## Example Output

### Console Output
```
======================================================================
SUMMARY: Key Rank at 1000 Attack Traces (Mean ± Std)
======================================================================

MAML:
  5-shot: 180.2 ± 15.3 [95% CI: 165.1 - 195.3]
  10-shot: 145.8 ± 22.1 [95% CI: 123.9 - 167.7]
  15-shot: 98.5 ± 18.7 [95% CI: 80.1 - 116.9]
  20-shot: 76.3 ± 12.4 [95% CI: 64.2 - 88.4]

ProtoNet:
  5-shot: 192.4 ± 19.8 [95% CI: 172.8 - 212.0]
  10-shot: 165.2 ± 25.3 [95% CI: 140.2 - 190.2]
  15-shot: 128.7 ± 21.5 [95% CI: 107.5 - 149.9]
  20-shot: 95.1 ± 16.8 [95% CI: 78.6 - 111.6]

Siamese:
  5-shot: 175.3 ± 20.2 [95% CI: 155.3 - 195.3]
  10-shot: 152.1 ± 23.8 [95% CI: 128.6 - 175.6]
  15-shot: 115.4 ± 19.3 [95% CI: 96.4 - 134.4]
  20-shot: 88.2 ± 14.1 [95% CI: 74.4 - 102.0]

======================================================================
BEST CONFIGURATIONS (Lowest Mean Rank)
======================================================================
MAML      : 8.5 ± 2.1 [95% CI: 6.5 - 10.5] (20-shot, 10000 traces)
ProtoNet  : 12.3 ± 3.4 [95% CI: 9.0 - 15.6] (15-shot, 10000 traces)
Siamese   : 15.7 ± 4.2 [95% CI: 11.7 - 19.7] (20-shot, 10000 traces)
```

---

## For Your Paper

### Table 1: Results with 95% CI

```latex
\begin{table}[ht]
\centering
\caption{Key Rank at 1000 Attack Traces (Mean ± Std) with 95\% CI}
\begin{tabular}{lcccc}
\hline
\textbf{Method} & \textbf{5-shot} & \textbf{10-shot} & \textbf{15-shot} & \textbf{20-shot} \\
\hline
MAML & 180.2±15.3 & 145.8±22.1 & 98.5±18.7 & 76.3±12.4 \\
     & [165.1-195.3] & [123.9-167.7] & [80.1-116.9] & [64.2-88.4] \\
\hline
ProtoNet & 192.4±19.8 & 165.2±25.3 & 128.7±21.5 & 95.1±16.8 \\
         & [172.8-212.0] & [140.2-190.2] & [107.5-149.9] & [78.6-111.6] \\
\hline
Siamese & 175.3±20.2 & 152.1±23.8 & 115.4±19.3 & 88.2±14.1 \\
        & [155.3-195.3] & [128.6-175.6] & [96.4-134.4] & [74.4-102.0] \\
\hline
\end{tabular}
\end{table}
```

### In Paper Text

**Example 1 (Reporting mean ± std):**
> "MAML achieved a mean key rank of 76.3 ± 12.4 (95% CI: [64.2, 88.4]) with 20-shot learning at 1000 attack traces, significantly outperforming ProtoNet (95.1 ± 16.8) and Siamese (88.2 ± 14.1)."

**Example 2 (Statistical significance):**
> "The 95% confidence intervals for MAML [64.2, 88.4] and ProtoNet [78.6, 111.6] overlap, suggesting no statistically significant difference at the p < 0.05 level."

**Example 3 (Variance discussion):**
> "MAML demonstrated lower variance (std = 12.4) compared to ProtoNet (std = 16.8), indicating more consistent performance across runs."

---

## Plotting with Error Bars

The `generate_plots.py` script automatically uses the statistics file if available:

```python
# In generate_plots.py
if has_stats:
    # Plot with 95% CI error bars
    ax.errorbar(
        stats_subset['Attack_Traces'],
        stats_subset['Mean_Rank'],
        yerr=stats_subset['Std_Rank'],  # Can use CI95_Width/2 instead
        marker='o',
        label=method,
        capsize=5
    )
```

---

## Statistical Significance Testing

### T-test Between Methods

To claim one method is "significantly better," use t-test:

```python
import scipy.stats as stats

# Get all runs for MAML vs ProtoNet at 1000 traces
maml_runs = combined_df[(combined_df['Method'] == 'MAML') &
                        (combined_df['Attack Traces'] == 1000)]['Key Rank']
proto_runs = combined_df[(combined_df['Method'] == 'ProtoNet') &
                         (combined_df['Attack Traces'] == 1000)]['Key Rank']

# Paired t-test
t_stat, p_value = stats.ttest_ind(maml_runs, proto_runs)

if p_value < 0.05:
    print("MAML is significantly better than ProtoNet (p < 0.05)")
else:
    print("No significant difference")
```

---

## Best Practices for Publication

### 1. Report Both Mean and CI
✅ "76.3 ± 12.4 (95% CI: [64.2, 88.4])"
❌ "76.3"

### 2. Use Error Bars in Plots
✅ Show 95% CI or ±1 std on all line plots
❌ Plot single runs without error bars

### 3. Run Sufficient Repetitions
✅ 5 runs minimum, 10 preferred
❌ Single run

### 4. Report Statistical Tests
✅ "MAML significantly outperformed ProtoNet (t(8) = 2.45, p = 0.04)"
❌ "MAML is better"

### 5. Discuss Variance
✅ "MAML showed lower variance, indicating robust performance"
❌ Only report means

---

## Quick Command Reference

```bash
# Run 5 experiments with statistics
python run_multiple_experiments.py

# After completion, generate plots with error bars
python generate_plots.py

# View statistics
cat experiment_results/aggregated_statistics.csv
```

---

## Troubleshooting

**Q: Script takes too long?**
A: Reduce NUM_RUNS to 3 (minimum) or run overnight

**Q: How to resume if interrupted?**
A: Script will skip completed runs if CSV files exist

**Q: What if one run fails?**
A: Script continues with remaining runs, uses available data

**Q: Need more runs?**
A: Just increase NUM_RUNS and re-run, it will add to existing results

---

## For Camera-Ready Paper

After initial acceptance, reviewers may request more runs:
1. Increase NUM_RUNS to 10 or 30
2. Re-run experiments
3. Update tables and figures
4. Add statistical tests section

This ensures your paper has solid statistical foundation!
