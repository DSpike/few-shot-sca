# Baseline Comparison: Standard CNN vs Few-Shot Meta-Learning

## Purpose

Train the **same CNN backbone** with standard supervised learning (no meta-learning) to compare against your few-shot results. This proves whether meta-learning provides any benefit over just training normally with the same amount of data.

## The Comparison

### Few-Shot Meta-Learning (Your Main Experiments)
- **Methods**: MAML, ProtoNet, Siamese
- **Training**: Meta-learning with support/query sets
- **Data per training**:
  - 5-shot = 1,280 traces (5 × 256 classes)
  - 10-shot = 2,560 traces (10 × 256 classes)
  - 15-shot = 3,840 traces (15 × 256 classes)
  - 20-shot = 5,120 traces (20 × 256 classes)

### Standard CNN (Baseline)
- **Method**: Standard supervised learning
- **Training**: Normal cross-entropy loss
- **Data per training**: Same amounts (1,280 / 2,560 / 3,840 / 5,120 traces)

**Key point**: Both use the **exact same CNN architecture** and **exact same amount of training data**!

## Running the Baseline

### RECOMMENDED: Multiple Runs (Matches Few-Shot Experiments)
```bash
# Run 10 times with seeds 42-51 (SAME as few-shot experiments!)
python run_baseline_multiple.py
```

**This automatically**:
- Runs baseline 10 times with seeds 42-51
- Same seed range as `run_multiple_experiments.py`
- Aggregates results with Mean ± Std
- Calculates 95% confidence intervals
- **Directly compares with few-shot results** (if available)

### Single Run (Quick Test)
```bash
python baseline_standard_training.py --seed 42
```

### What It Does

**Training sizes tested**:
- 1,280 traces (5-shot equivalent)
- 2,560 traces (10-shot equivalent)
- 3,840 traces (15-shot equivalent)
- 4,120 traces (20-shot equivalent)
- 50,000 traces (full profiling set - to show upper bound)

For each training size:
1. Samples training data (stratified - balanced across 256 classes)
2. Trains the CNN with standard supervised learning
3. Evaluates attack at different trace counts (100, 500, 1000, 2000, 5000, 10000)
4. Records key ranks

**Output**: `baseline_standard_cnn_results.csv`

## Expected Results

### Scenario 1: Meta-Learning Helps ✅ (Good for Your Paper!)

```
At 3,840 traces (15-shot equivalent), 1000 attack traces:

Standard CNN:     Rank 180
MAML (15-shot):   Rank 141  (22% better!)
ProtoNet (15-shot): Rank 136  (24% better!)
Siamese (15-shot):  Rank 125  (31% better!)
```

**Interpretation**: Meta-learning provides **genuine benefit** over standard training with same data!

**Paper claim**:
> "Our meta-learning approach outperforms standard supervised training with the same amount of training data, demonstrating that few-shot learning is not merely data efficiency but learns better representations for rapid adaptation."

### Scenario 2: Meta-Learning Comparable (Still OK)

```
At 3,840 traces (15-shot equivalent), 1000 attack traces:

Standard CNN:     Rank 140
MAML (15-shot):   Rank 141  (comparable)
ProtoNet (15-shot): Rank 136  (3% better)
Siamese (15-shot):  Rank 145  (3% worse)
```

**Interpretation**: Meta-learning is **data-efficient** but not necessarily better.

**Paper claim**:
> "Our few-shot meta-learning approach achieves comparable performance to standard supervised training while requiring minimal adaptation examples at test time, making it practical for scenarios where rapid deployment is needed."

### Scenario 3: Meta-Learning Worse ❌ (Problem!)

```
At 3,840 traces (15-shot equivalent), 1000 attack traces:

Standard CNN:     Rank 120
MAML (15-shot):   Rank 180  (50% worse!)
ProtoNet (15-shot): Rank 160  (33% worse!)
```

**Interpretation**: Meta-learning **doesn't help** - standard training is better!

**What to do**:
- ⚠️ **Don't claim meta-learning is better**
- ✅ Focus on **rapid adaptation** as the benefit
- ✅ Emphasize: "With only 5-20 examples per class **at test time**, the model can adapt to new scenarios"

## For Your Paper

### If Meta-Learning Wins (Best Case)

**Methodology section**:
> "To validate the effectiveness of meta-learning over standard supervised training, we train a baseline CNN with the same architecture using standard cross-entropy loss. We compare performance at equivalent training data sizes (1,280 to 5,120 traces)."

**Results table**:
| Training Size | Standard CNN | MAML | ProtoNet | Siamese |
|---------------|--------------|------|----------|---------|
| 1,280 traces  | 185 ± X      | 165 ± Y | 155 ± Z | 160 ± W |
| 2,560 traces  | 160 ± X      | 140 ± Y | 135 ± Z | 138 ± W |
| 3,840 traces  | 145 ± X      | 130 ± Y | 125 ± Z | 128 ± W |
| 5,120 traces  | 135 ± X      | 120 ± Y | 115 ± Z | 118 ± W |

**Discussion**:
> "Meta-learning consistently outperforms standard supervised training across all data regimes, demonstrating that the meta-learning objective learns better representations for few-shot SCA. The improvement ranges from 15-25%, showing that few-shot learning provides genuine benefit beyond simple data efficiency."

### If Meta-Learning Comparable (Okay Case)

**Focus on**: Rapid adaptation capability

> "While meta-learning achieves comparable key ranks to standard training with the same amount of profiling data, the key advantage lies in rapid adaptation: meta-learned models can adapt to new attack scenarios with only 5-20 examples per class at test time, whereas standard models require retraining from scratch."

### If Meta-Learning Worse (Problematic Case)

**Pivot focus**: Application novelty, not method superiority

> "We compare few-shot meta-learning against standard supervised training as a baseline. While standard training achieves lower key ranks with equivalent profiling data, meta-learning provides advantages in scenarios requiring rapid deployment or adaptation to new devices. Our contribution lies in demonstrating the feasibility of few-shot learning for SCA in resource-constrained IoT environments."

## Why This Comparison Matters

**Reviewers will ask**:
> "You claim meta-learning is better, but couldn't you just train a standard CNN with the same data?"

**Your answer with baseline**:
> "We compare against a standard CNN baseline (Table X). Meta-learning outperforms standard training by 15-25% with the same training data, showing genuine benefit of the meta-learning objective."

**Without baseline**:
> "Uh... we didn't test that..." ❌

## Running Everything

### Complete Experimental Pipeline (RECOMMENDED)

```bash
# 1. Run few-shot experiments (10 runs, seeds 42-51)
python run_multiple_experiments.py

# 2. Run baseline (10 runs, seeds 42-51) - SAME SEEDS!
python run_baseline_multiple.py

# 3. Generate comparison plots
python generate_plots.py
```

**Time estimate**:
- Few-shot experiments: ~6-10 hours (10 runs × 12 experiments each)
- Baseline: ~5-8 hours (10 runs × 5 training sizes each)
- Plotting: ~1 minute

**Output structure**:
```
experiment_results/          # Few-shot results
├── run_01_results.csv       (seeds 42-51)
├── ...
├── all_runs_combined.csv
└── aggregated_statistics.csv

baseline_results/            # Baseline results
├── baseline_run_01_results.csv  (seeds 42-51)
├── ...
├── all_baseline_runs_combined.csv
└── baseline_aggregated_statistics.csv
```

### Quick Test (Optional First)

```bash
# Test few-shot (1 run)
python comprehensive_few_shot_study.py --seed 42

# Test baseline (1 run)
python baseline_standard_training.py --seed 42
```

**Time**: ~1-1.5 hours total

## Key Differences Between Methods

| Aspect | Standard CNN | Few-Shot Meta-Learning |
|--------|--------------|------------------------|
| **Training objective** | Cross-entropy loss | Meta-learning objective (task adaptation) |
| **Training procedure** | Standard SGD | Inner/outer loop (MAML) or metric learning (ProtoNet) |
| **At test time** | Use trained model directly | Adapt to support set, then predict |
| **Strength** | Simple, well-understood | Designed for few-shot scenarios |
| **Weakness** | Requires retraining for new scenarios | More complex, higher variance |

## Summary

**You NEED this baseline** to:
1. ✅ Prove meta-learning provides benefit
2. ✅ Answer reviewer questions
3. ✅ Understand where your contribution lies
4. ✅ Make honest claims about performance

**Run it now**:
```bash
python baseline_standard_training.py --seed 42
```

Then compare with your few-shot results! 🎯
