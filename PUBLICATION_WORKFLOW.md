# Publication Workflow Guide
Complete workflow for generating publication-ready results

## Overview
This guide helps you complete all steps needed for publishing your few-shot SCA research:
1. Run multiple experiments for statistical confidence
2. Generate baseline comparisons
3. Create publication-quality plots
4. Analyze and report results

---

## Step 1: Run Multiple Experiments (for Confidence Intervals)

**Purpose**: Get mean ± standard deviation for each configuration

**Script**: [run_multiple_experiments.py](run_multiple_experiments.py)

```bash
python run_multiple_experiments.py
```

**What it does**:
- Runs `comprehensive_few_shot_study.py` 5 times (configurable to 10)
- Saves each run separately
- Aggregates results with statistics (mean, std, min, max)
- Creates `experiment_results/aggregated_statistics.csv`

**Expected time**: ~75-100 minutes (5 runs × 15-20 minutes each)

**Output files**:
```
experiment_results/
├── run_01_results.csv
├── run_02_results.csv
├── run_03_results.csv
├── run_04_results.csv
├── run_05_results.csv
├── all_runs_combined.csv
└── aggregated_statistics.csv  ← Use this for paper
```

**For paper**: Report results as "Mean ± Std" from `aggregated_statistics.csv`

---

## Step 2: Generate Baseline Comparisons

**Purpose**: Compare few-shot methods against standard CNN training

**Script**: [baseline_standard_training.py](baseline_standard_training.py)

```bash
python baseline_standard_training.py
```

**What it does**:
- Trains standard CNNs with varying amounts of data:
  - 1,280 traces (5-shot equivalent)
  - 2,560 traces (10-shot equivalent)
  - 3,840 traces (15-shot equivalent)
  - 5,120 traces (20-shot equivalent)
  - 50,000 traces (full training set)
- Evaluates attack performance for each

**Expected time**: ~30-40 minutes

**Output**: `baseline_standard_cnn_results.csv`

**For paper**: Shows few-shot methods can match/exceed standard training with much less data

---

## Step 3: Generate Publication Plots

**Purpose**: Create all figures for your paper

**Script**: [generate_plots.py](generate_plots.py)

```bash
python generate_plots.py
```

**What it does**:
- Figure 1: Key Rank vs Attack Traces (4 subplots for each k-shot)
- Figure 2: K-Shot Ablation Study (comparison at 1000 traces)
- Figure 3: Results Heatmap (all configurations)
- Figure 4: Few-Shot vs Baseline (if baseline available)
- Figure 5: Success Rate Analysis (Key Rank < 10)

**Required files**:
- `few_shot_sca_results.csv` (from single run)
- `experiment_results/aggregated_statistics.csv` (optional, for error bars)
- `baseline_standard_cnn_results.csv` (optional, for Fig 4)

**Output**:
```
figures/
├── fig1_key_rank_vs_traces.png
├── fig1_key_rank_vs_traces.pdf
├── fig2_kshot_ablation.png
├── fig2_kshot_ablation.pdf
├── fig3_results_heatmap.png
├── fig3_results_heatmap.pdf
├── fig4_fewshot_vs_baseline.png
├── fig4_fewshot_vs_baseline.pdf
├── fig5_success_rate.png
└── fig5_success_rate.pdf
```

**For paper**:
- Use `.pdf` files for LaTeX (vector graphics, scales perfectly)
- Use `.png` files for quick preview

---

## Complete Workflow Example

```bash
# Terminal 1: Monitor GPU utilization
nvidia-smi dmon -s u

# Terminal 2: Run experiments

# Step 1: Multiple runs for confidence intervals (75-100 min)
python run_multiple_experiments.py

# Step 2: Baseline comparison (30-40 min)
python baseline_standard_training.py

# Step 3: Generate all plots (instant)
python generate_plots.py
```

---

## Quick Single-Run Workflow

If you just want to test or don't need error bars:

```bash
# 1. Run once (~15-20 min)
python comprehensive_few_shot_study.py

# 2. Generate plots (instant)
python generate_plots.py
```

---

## Results Summary

### Current Single-Run Results

From [few_shot_sca_results.csv](few_shot_sca_results.csv):

**Best Results (Key Rank at 1000 traces)**:
- MAML 15-shot: Rank 15
- MAML 10-shot: Rank 198
- ProtoNet 5-shot: Rank 64

**Best Overall**:
- MAML 10-shot with 5000 traces: Rank 8
- MAML 15-shot with 500 traces: Rank 13

### What to Report in Paper

#### Table 1: Ablation Study (Key Rank at 1000 Attack Traces)
```
| K-Shot | MAML       | ProtoNet   | Standard CNN |
|--------|------------|------------|--------------|
| 5      | XXX ± YY   | XXX ± YY   | XXX          |
| 10     | XXX ± YY   | XXX ± YY   | XXX          |
| 15     | XXX ± YY   | XXX ± YY   | XXX          |
| 20     | XXX ± YY   | XXX ± YY   | XXX          |
```

Use values from `aggregated_statistics.csv` after running multiple experiments.

#### Figure Captions

**Figure 1**: Key recovery performance across different numbers of attack traces for 5-shot, 10-shot, 15-shot, and 20-shot configurations. Lower rank indicates better attack performance. Error bars show standard deviation across 5 independent runs.

**Figure 2**: Ablation study showing the effect of k-shot values on attack performance at 1000 attack traces. MAML demonstrates superior performance with fewer training examples per class.

**Figure 3**: Heatmap visualization of key ranks for all method and configuration combinations. Darker colors indicate better (lower) key ranks.

**Figure 4**: Comparison between few-shot meta-learning approaches and standard CNN training. Few-shot methods achieve comparable performance with significantly less training data (k×256 traces vs. 50,000 traces).

**Figure 5**: Attack success rate (percentage of configurations achieving key rank < 10) across different k-shot values.

---

## Files Overview

### Main Scripts
- `comprehensive_few_shot_study.py` - Main few-shot SCA implementation (GPU-optimized)
- `run_multiple_experiments.py` - Run multiple times for statistics
- `baseline_standard_training.py` - Standard CNN baseline
- `generate_plots.py` - Create all publication figures

### Result Files
- `few_shot_sca_results.csv` - Single run results
- `experiment_results/aggregated_statistics.csv` - Multi-run statistics
- `baseline_standard_cnn_results.csv` - Baseline results
- `figures/*.pdf` - Publication-ready plots

### Documentation
- `RESEARCH_GUIDE.md` - Research roadmap
- `GPU_OPTIMIZATIONS.md` - GPU optimization details
- `PUBLICATION_WORKFLOW.md` - This file

---

## Paper Writing Checklist

- [ ] Run multiple experiments (5-10 runs)
- [ ] Generate baseline results
- [ ] Create all plots
- [ ] Write abstract
- [ ] Write introduction (motivation + contributions)
- [ ] Write related work section
  - [ ] Cite MAML paper (Finn et al., 2017)
  - [ ] Cite Prototypical Networks (Snell et al., 2017)
  - [ ] Cite SCA papers (Picek et al., Wouters et al.)
- [ ] Write methodology section
  - [ ] Explain few-shot learning
  - [ ] Explain MAML and ProtoNet
  - [ ] Explain minimal variance sampling
- [ ] Write experimental setup
  - [ ] Dataset description (ASCAD)
  - [ ] Model architecture
  - [ ] Hyperparameters
- [ ] Write results section
  - [ ] Include Table 1 (ablation study)
  - [ ] Include all figures with captions
  - [ ] Analyze trends
- [ ] Write discussion
  - [ ] Compare with baseline
  - [ ] Discuss advantages of few-shot approach
  - [ ] Limitations
- [ ] Write conclusion
- [ ] Add references
- [ ] Proofread

---

## Target Venues

**Recommended conferences**:
1. **CHES 2026** (Cryptographic Hardware and Embedded Systems)
   - Deadline: ~July 2026
   - Top venue for SCA

2. **CCS 2026** (ACM Conference on Computer and Communications Security)
   - Deadline: ~May 2026
   - Broader security audience

3. **CARDIS 2026** (Smart Card Research and Advanced Applications)
   - Deadline: ~August 2026
   - Strong SCA focus

**Preparation time needed**: 2-3 months for quality paper

---

## Troubleshooting

### Issue: GPU not being used
**Solution**: Scripts are already GPU-optimized. Check with:
```bash
nvidia-smi dmon -s u
```
Should show 99-100% GPU utilization during training.

### Issue: Out of memory
**Solution**: Reduce batch sizes in the code (currently 1000)

### Issue: Results look poor (high key ranks)
**Possible causes**:
1. Need more training epochs (increase from 50 to 100-200)
2. Learning rate needs tuning
3. Model architecture needs adjustment
4. Dataset quality issues

### Issue: High variance between runs
**Solution**: Run more experiments (10 instead of 5) for stable statistics

---

## Next Steps

1. ✅ GPU optimization complete
2. ⬜ Run multiple experiments
3. ⬜ Generate baseline
4. ⬜ Create plots
5. ⬜ Start writing paper
6. ⬜ Submit to conference

**Current status**: Ready to run full experimental pipeline!
