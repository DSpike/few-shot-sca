# Complete Few-Shot SCA Study - Three Methods

## Overview

Your comprehensive study now includes **THREE** state-of-the-art few-shot learning methods:

### 1. MAML (Model-Agnostic Meta-Learning)
- **Paper**: Finn et al., ICML 2017
- **Key idea**: Learn good initialization for fast adaptation
- **How it works**:
  - Meta-training: Update model parameters across multiple tasks
  - Meta-testing: Fine-tune on support set, evaluate on attack traces
- **Advantage**: Can adapt to new classes with very few gradient steps

### 2. Prototypical Networks
- **Paper**: Snell et al., NeurIPS 2017
- **Key idea**: Learn a metric space where classification uses distances to class prototypes
- **How it works**:
  - Compute prototype (mean embedding) for each class from support set
  - Classify query samples by finding nearest prototype
- **Advantage**: Simple, effective, works well with few examples

### 3. Siamese Networks (NEW!)
- **Paper**: Koch et al., ICML 2015
- **Key idea**: Learn similarity metric between pairs of examples
- **How it works**:
  - Train with triplet loss (anchor, positive, negative)
  - Learn embeddings where same-class samples are close, different-class samples are far
  - Classify by finding closest support sample in embedding space
- **Advantage**: Classic few-shot learning baseline, metric learning approach

## Your Novel Contribution

**Minimal Variance Sampling (MVS)** using GPU-accelerated k-means:
- Selects most representative k shots per class
- Reduces variance in support set selection
- GPU implementation for fast sampling
- **This is your novel contribution!**

## Complete Experimental Design

### Methods Compared
1. MAML + MVS
2. ProtoNet + MVS
3. Siamese + MVS

### Ablation Study
- K-shot values: 5, 10, 15, 20
- Attack traces: 100, 500, 1000, 2000, 5000, 10000
- **Total configurations**: 3 methods × 4 k-shots × 6 trace counts = 72 results

### What Makes This Publication-Ready

1. **Novel sampling**: MVS with k-means (your contribution)
2. **Comprehensive comparison**: Three established methods
3. **Rigorous evaluation**: Multiple k-shots and trace counts
4. **Domain-specific**: Applied to side-channel analysis (SCA)
5. **Statistical rigor**: Mean ± std from multiple runs

## Paper Structure Recommendation

### Title
"Minimal Variance Sampling for Few-Shot Side-Channel Analysis"

### Abstract
- Problem: SCA requires many traces, impractical for real attacks
- Solution: Few-shot learning with novel MVS strategy
- Methods: Compare MAML, ProtoNet, Siamese with MVS
- Results: Achieve rank X with only Y traces
- Contribution: MVS improves sample efficiency

### Main Contributions
1. **First application of few-shot meta-learning to SCA** (if true - check literature!)
2. **Novel MVS strategy** using k-means for representative shot selection
3. **Comprehensive comparison** of three meta-learning methods
4. **Strong empirical results** on ASCAD benchmark

### Related Work
#### Few-Shot Learning
- MAML (Finn et al., 2017)
- Prototypical Networks (Snell et al., 2017)
- Siamese Networks (Koch et al., 2015)

#### Side-Channel Analysis
- Deep learning for SCA (Picek et al., Wouters et al.)
- Standard CNN attacks (Zaid et al., ASCAD paper)
- Profiling attacks

### Methodology
1. **Problem Formulation**
   - K-shot N-way classification for SCA
   - Support set: k examples per class
   - Query set: Attack traces

2. **Minimal Variance Sampling**
   - GPU k-means clustering per class
   - Select k samples closest to centroids
   - Reduces intra-class variance

3. **Meta-Learning Methods**
   - MAML: Meta-optimization algorithm
   - ProtoNet: Prototype-based classification
   - Siamese: Metric learning with triplets

4. **Experimental Setup**
   - Dataset: ASCAD (masked AES)
   - Architecture: CNN (SimpleCNN)
   - Training: 50 epochs, Adam optimizer
   - Evaluation: Key rank metric

### Results
- **Table 1**: Key ranks at 1000 attack traces (all methods, all k-shots)
- **Figure 1**: Key rank vs attack traces (learning curves)
- **Figure 2**: K-shot ablation study
- **Figure 3**: Heatmap of all results
- **Figure 4**: Comparison with standard CNN baseline
- **Figure 5**: Success rate analysis

### Discussion
- MVS improves performance across all methods
- MAML shows best adaptation capability
- ProtoNet provides good balance of speed and accuracy
- Siamese networks competitive but slightly slower to train
- Few-shot learning reduces traces needed by X%

## Expected Runtime

With your optimized code (single process, no competition):

| Phase | K-SHOT=5 | K-SHOT=10 | K-SHOT=15 | K-SHOT=20 |
|-------|----------|-----------|-----------|-----------|
| Sampling (MVS) | 30s | 60s* | 90s* | 120s* |
| MAML Training | 3-4 min | 3-4 min | 3-4 min | 3-4 min |
| ProtoNet Training | 3-4 min | 3-4 min | 3-4 min | 3-4 min |
| Siamese Training | 3-4 min | 3-4 min | 3-4 min | 3-4 min |
| Evaluation | 2 min | 2 min | 2 min | 2 min |
| **Subtotal** | ~12 min | ~13 min | ~14 min | ~15 min |

*Cached after first sampling per k-shot

**Total for full experiment**: ~20-25 minutes

## Next Steps

1. ✅ Kill all competing Python processes
2. ✅ Run safe experiment: `python run_single_experiment_safe.py`
3. ⬜ Wait 20-25 minutes for completion
4. ⬜ Run multiple times (5 runs) for statistics
5. ⬜ Generate baseline comparison
6. ⬜ Create publication plots
7. ⬜ Write paper

## Target Venues

**Primary**: CHES 2026 (Cryptographic Hardware and Embedded Systems)
**Secondary**: CCS 2026, CARDIS 2026, IEEE S&P

## Citation Template

```bibtex
@inproceedings{yourname2026minimal,
  title={Minimal Variance Sampling for Few-Shot Side-Channel Analysis},
  author={Your Name},
  booktitle={CHES 2026},
  year={2026}
}
```

Key references to cite:
- MAML: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- ProtoNet: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
- Siamese: Koch et al., "Siamese Neural Networks for One-shot Image Recognition", ICML 2015
- ASCAD: Benadjila et al., "Deep learning for side-channel analysis and introduction to ASCAD database", 2018
