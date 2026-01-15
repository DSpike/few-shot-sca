# Refactor: Stratified Random Sampling

## Summary

Removed MVS (Minimal Variance Sampling) and basic random sampling implementations, replacing them with **stratified random sampling** as the single, principled baseline for few-shot learning.

## Motivation

### Why Remove MVS?

The ablation study results showed:
- ❌ **No statistical significance** (all p > 0.05)
- ❌ **Mixed/inconsistent results** (sometimes worse than random)
- ❌ **High variance in both MVS and Random** (std 40-90)
- ❌ **Cannot claim MVS as a contribution** without statistical evidence

### Why Stratified Random Sampling?

Stratified random sampling is:
- ✅ **Standard practice** in few-shot learning literature
- ✅ **Balanced representation** (exactly k samples per class)
- ✅ **Reproducible** with seeding
- ✅ **Simple and principled** baseline
- ✅ **No need for ablation study** (it's the standard approach)

## Changes Made

### 1. Removed Code

**Deleted implementations**:
- `kmeans_gpu()` - GPU k-means clustering
- `MinimalVarianceSampler` class - MVS with k-means
- `RandomSampler` class - Unbalanced random sampling
- `get_sampler()` factory function
- `sample_support_set()` helper function

**Removed command-line arguments**:
- `--sampling` argument (no longer needed)

### 2. New Implementation

**Added**: `StratifiedSampler` class (lines 111-163)

```python
class StratifiedSampler:
    """
    Stratified random sampling for few-shot learning
    Ensures balanced representation across all 256 classes
    Each class contributes exactly k_shot samples
    """
    def sample_support_set(self, k_shot):
        # Sample k_shot examples from each class
        for cls in range(256):
            if len(self.class_idx[cls]) >= k_shot:
                # Stratified random selection
                indices = np.random.choice(self.class_idx[cls], k_shot, replace=False)
                support_idx.extend(indices.tolist())
```

### 3. Updated Functions

**Training functions** now use `StratifiedSampler` directly:
- `train_maml()` - line 214
- `train_protonet()` - line 275
- `train_siamese()` - line 361

**Before**:
```python
def train_maml(X_train, y_train, k_shot, epochs=100, sampling_strategy='mvs'):
    sampler = get_sampler(X_train, y_train, sampling_strategy)
    support_x, support_y = sample_support_set(sampler, k_shot, sampling_strategy)
```

**After**:
```python
def train_maml(X_train, y_train, k_shot, epochs=100):
    sampler = StratifiedSampler(X_train, y_train)
    support_x, support_y = sampler.sample_support_set(k_shot)
```

### 4. Simplified Main Loop

**Before**:
```python
print(f"K-SHOT = {k_shot} | Sampling: {args.sampling.upper()}")
sampler = get_sampler(X_test, y_test, args.sampling)
support_x, support_y = sample_support_set(sampler, k_shot, args.sampling)
model = train_maml(X_train, y_train, k_shot, epochs=500, sampling_strategy=args.sampling)
```

**After**:
```python
print(f"K-SHOT = {k_shot}")
sampler = StratifiedSampler(X_test, y_test)
support_x, support_y = sampler.sample_support_set(k_shot)
model = train_maml(X_train, y_train, k_shot, epochs=500)
```

## Usage

### Running Experiments

**Single run with seed**:
```bash
python comprehensive_few_shot_study.py --seed 42
```

**Multiple runs for statistics**:
```bash
python run_multiple_experiments.py
```

**Output**: Uses stratified random sampling automatically (no arguments needed)

### Expected Sample Sizes

For k-shot sampling with 256 classes:
- 5-shot: 1,280 samples (5 × 256)
- 10-shot: 2,560 samples (10 × 256)
- 15-shot: 3,840 samples (15 × 256)
- 20-shot: 5,120 samples (20 × 256)

Each class contributes exactly k samples (balanced representation).

## Impact on Paper

### Revised Contributions

**Before** (with MVS):
1. ❌ Minimal Variance Sampling via GPU k-means (NOT VALIDATED)
2. ✅ First few-shot SCA for wearable IoT
3. ✅ GPU-optimized implementation
4. ✅ Comprehensive method comparison

**After** (without MVS):
1. ✅ **First application of few-shot meta-learning to wearable IoT SCA**
2. ✅ **GPU-optimized implementation for practical deployment**
3. ✅ **Comprehensive comparison of three meta-learning methods**
4. ✅ **Analysis of few-shot learning for 256-way SCA classification**

### Methodology Section

**Add**:
> "We use stratified random sampling to construct support sets, ensuring balanced representation across all 256 S-box output classes. For k-shot learning, each class contributes exactly k examples, resulting in support sets of size 256k. This follows standard practice in few-shot learning literature and ensures reproducibility when combined with fixed random seeds."

### No Ablation Study Needed

Since stratified random sampling is:
- The **standard baseline** in few-shot learning
- **Balanced by design** (not a novel contribution)
- **Well-established** in the literature

You **do NOT need an ablation study** comparing it to other sampling strategies. It's simply the correct way to sample for few-shot learning.

## Advantages of This Approach

### 1. Scientific Integrity ✅
- Not claiming unvalidated contributions
- Using established best practices
- Honest about what works

### 2. Simpler Implementation ✅
- ~150 fewer lines of code
- No complex k-means implementation
- No sampling strategy arguments
- Easier to understand and maintain

### 3. Clearer Paper Focus ✅
- Focus on **application domain** (wearable IoT)
- Focus on **method comparison** (MAML vs ProtoNet vs Siamese)
- Focus on **GPU optimization** for practical deployment
- No need to defend MVS

### 4. Follows Best Practices ✅
- Stratified sampling is standard in few-shot learning
- Used in Prototypical Networks, MAML, and other seminal papers
- Reviewers will recognize and accept it

## What This Means for Results

### Variance Will Still Be High

Stratified sampling won't reduce variance because:
- Variance comes from **algorithm sensitivity**, not sampling
- 256-way few-shot learning is **inherently difficult**
- Different random seeds → different initializations → different results

**This is expected and acceptable!**

### Report Variance Honestly

In your paper:
> "Due to the difficulty of 256-way few-shot classification and sensitivity to random initialization, we observe high variance across runs (std 40-90 key ranks). This is consistent with the challenging nature of few-shot SCA and highlights the need for robust meta-learning methods. We report mean ± std across 10 independent runs with fixed seeds (42-51) for reproducibility."

## Files Modified

1. **comprehensive_few_shot_study.py**:
   - Removed: MVS, RandomSampler, factory functions (~160 lines)
   - Added: StratifiedSampler class (~55 lines)
   - Simplified: Training functions and main loop
   - Net reduction: ~105 lines

## Next Steps

1. ✅ **Code refactored** to use stratified sampling
2. ⏭️ **Run experiments**: `python run_multiple_experiments.py` (10 runs with seeds 42-51)
3. ⏭️ **Analyze results**: Focus on method comparison, not sampling
4. ⏭️ **Update paper**: Emphasize application domain and method comparison
5. ⏭️ **Generate plots**: Compare MAML vs ProtoNet vs Siamese

## Verification

```bash
# Check syntax
python -m py_compile comprehensive_few_shot_study.py

# Test run
python comprehensive_few_shot_study.py --seed 42

# Help output
python comprehensive_few_shot_study.py --help
```

**Expected output**:
```
usage: comprehensive_few_shot_study.py [-h] [--seed SEED]

options:
  --seed SEED  Random seed for reproducibility (default: None = random)
```

## Summary

**Clean, principled, publishable approach**:
- ✅ Standard stratified random sampling
- ✅ No unvalidated claims
- ✅ Simpler codebase
- ✅ Clear paper focus on application and method comparison
- ✅ Honest reporting of variance

**This is the right scientific approach!** 🎯
