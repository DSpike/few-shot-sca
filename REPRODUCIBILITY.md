# Reproducibility and Random Seeding

## Overview

Your experiments now use **deterministic seeding** for reproducibility while still measuring genuine variance across different initializations.

---

## How It Works

### Fixed Seed Ranges
Each experimental run uses a **different but predetermined seed**:

```
Run 1: seed = 42
Run 2: seed = 43
Run 3: seed = 44
Run 4: seed = 45
Run 5: seed = 46
```

### What This Achieves

✅ **Reproducible**: Anyone can recreate your exact results
✅ **Measures variance**: Different seeds = different random initializations
✅ **Transparent**: Clear methodology for reviewers
✅ **Debuggable**: Can re-run specific failed runs

---

## Running Experiments

### Multiple Runs (Recommended)
```bash
python run_multiple_experiments.py
```

This automatically:
- Runs 5 experiments with seeds 42, 43, 44, 45, 46
- Saves each run separately
- Aggregates statistics with mean ± std
- Calculates 95% confidence intervals

**Output**:
```
RUN 1/5 (seed=42)
RUN 2/5 (seed=43)
RUN 3/5 (seed=44)
RUN 4/5 (seed=45)
RUN 5/5 (seed=46)
```

### Single Run with Specific Seed
```bash
# Run with seed 42
python comprehensive_few_shot_study.py --seed 42

# Run with seed 100
python comprehensive_few_shot_study.py --seed 100

# Run without seed (random)
python comprehensive_few_shot_study.py
```

### Using Safe Runner
```bash
# With seed
python run_single_experiment_safe.py 42

# Random seed
python run_single_experiment_safe.py
```

---

## What Gets Seeded

When you specify `--seed`, the following are set deterministically:

### Python Random
```python
random.seed(seed)
```
- Random number generation
- Random sampling

### NumPy Random
```python
np.random.seed(seed)
```
- Array initialization
- Random permutations
- K-means initialization

### PyTorch Random
```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```
- Weight initialization
- Dropout
- Data shuffling
- Random tensor operations

### CUDA Determinism
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
- Ensures GPU operations are deterministic
- **Slightly slower** but fully reproducible

---

## Variance Interpretation

### What the Variance Represents

Your results with seeds 42-46 show variance because:

1. **Different weight initialization** - Each seed starts with different random weights
2. **Different k-means initialization** - Minimal Variance Sampling clusters differ
3. **Different task sampling** - Meta-learning samples different tasks during training

**This is real variance** that an attacker would experience!

### Example Results
```
MAML 10-shot at 1000 traces:
- Seed 42: rank 120
- Seed 43: rank 85
- Seed 44: rank 95
- Seed 45: rank 75
- Seed 46: rank 90
Mean: 93 ± 17
```

The std=17 represents **genuine variability** due to initialization, not measurement error.

---

## For Your Paper

### Methodology Section

**Add this paragraph**:

> To ensure reproducibility while measuring genuine algorithmic variance, we conduct 5 independent runs with fixed random seeds (42-46). Each run uses a different seed for weight initialization, k-means clustering, and task sampling, representing the variability an attacker would encounter when applying these methods with different initializations. All random number generators (Python's random, NumPy, PyTorch) are seeded deterministically, and CUDA operations are configured for deterministic behavior.

### Reproducibility Statement

> All experiments are fully reproducible. Running `python run_multiple_experiments.py` will generate identical results to those reported in this paper. Individual runs can be reproduced using `python comprehensive_few_shot_study.py --seed N` where N ∈ {42, 43, 44, 45, 46}. Code is available at [your repository].

---

## Comparing to Other Work

### Your Approach ✅
- Fixed seeds: {42, 43, 44, 45, 46}
- Reproducible variance measurement
- Transparent methodology

### Common Alternatives

**Bad: Single seed** ❌
```python
# Only reports results from seed=42
np.random.seed(42)
```
Problem: Doesn't measure variance, cherry-picking

**Bad: Random seeds** ❌
```python
# No seed control
# Results change every time
```
Problem: Not reproducible

**Good: Fixed seed range** ✅ (Your approach)
```python
# Seeds: 42, 43, 44, 45, 46
```
Best of both worlds!

---

## Extending to More Runs

### For 10 Runs
Edit `run_multiple_experiments.py`:
```python
NUM_RUNS = 10  # Seeds 42-51
```

### For 20 Runs (Publication Quality)
```python
NUM_RUNS = 20  # Seeds 42-61
```

This will:
- ✅ Reduce confidence interval width by ~50%
- ✅ Give more reliable mean estimates
- ✅ Look more rigorous to reviewers
- ❌ **Not** reduce the standard deviation (variance is real!)

---

## CI Width vs Number of Runs

| Runs | Seeds | CI Width Multiplier | Example (std=45) |
|------|-------|---------------------|------------------|
| 5 | 42-46 | 1.24 × std | ±56 |
| 10 | 42-51 | 0.72 × std | ±32 |
| 20 | 42-61 | 0.46 × std | ±21 |

Note: **Std stays ~45**, only CI narrows!

---

## Debugging Failed Runs

If a specific run fails, you can re-run just that seed:

```bash
# Run 3 failed? Re-run with same seed
python comprehensive_few_shot_study.py --seed 44

# Debug with different seed
python comprehensive_few_shot_study.py --seed 100
```

---

## Performance Impact

### Deterministic Mode
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Impact**: ~5-10% slower
**Benefit**: Fully reproducible GPU operations

### Without Determinism
- Faster (~5-10%)
- Still mostly reproducible (CPU operations fixed)
- GPU operations may vary slightly

**Recommendation**: Keep determinism ON for publication experiments, turn OFF for development/debugging if you need speed.

---

## Summary

✅ **You now have reproducible experiments**
✅ **Each run uses a different seed (42-46)**
✅ **Variance measurement is scientifically valid**
✅ **Anyone can reproduce your exact results**

Run with:
```bash
python run_multiple_experiments.py
```

And you'll get the same results every time! 🎉
