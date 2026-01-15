# Variance Reduction Updates

## Problem Identified

Your initial 5-run results showed **unacceptably high variance**:
- MAML 5-shot: 109.0 ± 87.4 (**80% relative std**)
- Confidence intervals spanning **negative to 200+**
- Standard deviation often 60-80% of the mean

**Root cause**: Under-training with only 100 epochs led to inconsistent convergence across runs.

---

## Changes Made

### 1. **Increased Training Epochs: 100 → 300** ✅
```python
# Before:
model = train_maml(X_train, y_train, k_shot, epochs=100)

# After:
model = train_maml(X_train, y_train, k_shot, epochs=300)
```

**Impact**:
- More time for models to converge
- Reduces sensitivity to random initialization
- Expected variance reduction: **50-70%**

### 2. **Added Early Stopping** ✅
```python
# Early stopping with patience=50
best_loss = float('inf')
patience = 50
patience_counter = 0

# Stop if no improvement for 50 epochs
if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch+1}")
    break

# Restore best model weights
model.load_state_dict(best_model_state)
```

**Impact**:
- Prevents overfitting
- Stops training when converged
- Returns best model, not final model

### 3. **Applied to All Three Methods** ✅
- MAML: epochs=300, early stopping with patience=50
- ProtoNet: epochs=300, early stopping with patience=50
- Siamese: epochs=300, early stopping with patience=50

---

## Expected Runtime Impact

| Configuration | Before (100 epochs) | After (300 epochs, early stopping) |
|---------------|---------------------|-------------------------------------|
| Single k-shot | ~3-4 minutes | ~5-8 minutes (may stop earlier) |
| Full experiment (4 k-shots × 3 methods) | ~20-25 minutes | ~30-45 minutes |
| 5 runs total | ~100-125 minutes | ~150-225 minutes (~2.5-4 hours) |

**Note**: Early stopping may reduce actual runtime if models converge before 300 epochs.

---

## Expected Variance Improvement

### Target Metrics (Acceptable for Publication)

| Metric | Before | Target After | Status |
|--------|--------|--------------|--------|
| Relative Std (std/mean) | 60-80% | **< 30%** | Need to verify |
| 95% CI Width | 200+ | **< 100** | Need to verify |
| Negative CI bounds | Yes ❌ | No ✅ | Should be fixed |

### What to Look For in Next Run

✅ **Good signs**:
- Std drops to 20-40 (from 60-90)
- Relative std < 30-40%
- CI width < 100
- No negative CI bounds
- More consistent mean across runs

⚠️ **Still needs work if**:
- Relative std still > 40%
- CI bounds still include negative values
- Variance hasn't reduced by at least 40%

---

## Next Steps

### Option 1: Test with Current Changes (Recommended First)
```bash
# Run 3-5 experiments to test variance reduction
python run_multiple_experiments.py  # Keep NUM_RUNS = 5
```

**Wait time**: ~2.5-4 hours

**Decision point**:
- If variance drops to acceptable levels (relative std < 30-40%) → Proceed to 10 runs
- If variance still high (relative std > 50%) → Need further investigation

### Option 2: If Variance Still High

Additional measures to try:
1. **Increase epochs to 500** (but may overfit)
2. **Add learning rate scheduling**:
   ```python
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
   ```
3. **Increase number of runs to 10** (won't reduce variance, but tighter CI)
4. **Check training loss curves** (are models actually converging?)

---

## How to Interpret New Results

### Example of Good Results:
```
MAML:
  5-shot: 109.0 ± 25.0 [95% CI: 78.0 - 140.0]  ✅ Relative std: 23%
  10-shot: 95.0 ± 22.0 [95% CI: 68.0 - 122.0]  ✅ Relative std: 23%
```

### Example of Still-Too-High Variance:
```
MAML:
  5-shot: 109.0 ± 60.0 [95% CI: 34.0 - 184.0]  ❌ Relative std: 55%
  10-shot: 95.0 ± 55.0 [95% CI: 27.0 - 163.0]  ❌ Relative std: 58%
```

---

## Why This Happens (Technical Context)

### 256-way Classification is Hard
- You're distinguishing between 256 classes with only 5-20 examples each
- Much harder than typical 5-way few-shot learning
- Small changes in initialization can lead to different local minima

### Meta-Learning Sensitivity
- MAML is particularly sensitive to initialization
- 100 epochs may not be enough to escape poor local minima
- 300 epochs gives more chances to find better solutions

### Random Factors
1. **Weight initialization**: Different random seeds
2. **Task sampling**: Which tasks are sampled during meta-training
3. **Support set selection**: Which k shots are selected (though MVS reduces this)
4. **Query set selection**: Random query batches during training

---

## For Your Paper

### If Variance Improves ✅
Report it honestly:
> "To ensure statistical robustness, we conducted 10 independent runs with different random initializations. Our training procedure uses 300 epochs with early stopping (patience=50) to ensure consistent convergence. Results are reported as mean ± standard deviation with 95% confidence intervals computed using the t-distribution."

### If Variance Remains High ⚠️
You have two options:
1. **Investigate and fix** (preferred)
2. **Acknowledge in limitations**:
   > "We observe moderate variance across runs (relative std 30-40%), which we attribute to the challenging 256-way few-shot classification problem. This is significantly higher than typical 5-way few-shot benchmarks but reflects the inherent difficulty of side-channel analysis."

---

## Summary

**Changes**: Increased epochs to 300, added early stopping to all methods
**Expected impact**: 50-70% variance reduction
**Runtime**: ~2.5-4 hours for 5 runs
**Next action**: Run `python run_multiple_experiments.py` and check if variance is acceptable

**Acceptable threshold for publication**: Relative std < 30-40%
**Your previous results**: Relative std 60-80% ❌
**Target**: Relative std 20-30% ✅
