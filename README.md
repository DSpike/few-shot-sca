# Few-Shot Meta-Learning for Side-Channel Analysis of Wearable IoT Devices

Official implementation of "Few-Shot Meta-Learning for Side-Channel Analysis of Wearable IoT Devices"

## Overview

This repository contains the code for applying few-shot meta-learning to power analysis attacks on wearable IoT cryptographic implementations. We implement and compare three meta-learning methods (MAML, Prototypical Networks, Siamese Networks) for 256-way few-shot side-channel analysis.

## Key Features

- **Three Meta-Learning Methods**: MAML, Prototypical Networks, Siamese Networks
- **256-way Classification**: All AES S-box outputs (much harder than typical 5-way few-shot benchmarks)
- **GPU-Optimized**: Efficient CUDA implementations for practical deployment
- **Stratified Random Sampling**: Balanced support sets across all classes
- **Reproducible**: Fixed seeds (42-51) for all experiments
- **Baseline Comparison**: Standard CNN training for fair comparison
- **Statistical Analysis**: Mean ± Std across 10 independent runs with 95% confidence intervals

## Requirements

### Software
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA-capable GPU (tested on NVIDIA GPUs)

### Python Packages
```bash
pip install torch numpy pandas h5py scipy matplotlib seaborn
```

### Dataset
- ASCAD dataset (download from: https://github.com/ANSSI-FR/ASCAD)
- Place `ASCAD.h5` in: `ASCAD_data/ASCAD_data/ASCAD_databases/`

## Project Structure

```
sca/
├── comprehensive_few_shot_study.py    # Main experiment script
├── run_multiple_experiments.py        # Run 10 experiments with seeds 42-51
├── baseline_standard_training.py      # Standard CNN baseline (single run)
├── run_baseline_multiple.py           # Baseline 10 runs with seeds 42-51
├── generate_plots.py                  # Generate publication figures
├── experiment_results/                # Few-shot experiment outputs
│   ├── run_01_results.csv            # Individual runs
│   ├── all_runs_combined.csv         # Combined results
│   └── aggregated_statistics.csv     # Mean ± Std statistics
├── baseline_results/                  # Baseline experiment outputs
│   ├── baseline_run_01_results.csv
│   ├── all_baseline_runs_combined.csv
│   └── baseline_aggregated_statistics.csv
└── figures/                           # Publication-quality plots
```

## Usage

### Quick Start (Single Run)

```bash
# Few-shot experiment (one seed)
python comprehensive_few_shot_study.py --seed 42

# Baseline experiment (one seed)
python baseline_standard_training.py --seed 42
```

### Full Experiments (Recommended for Publication)

```bash
# 1. Run few-shot experiments (10 runs, ~6-10 hours)
python run_multiple_experiments.py

# 2. Run baseline experiments (10 runs, ~5-8 hours)
python run_baseline_multiple.py

# 3. Generate figures
python generate_plots.py
```

### Output

Each script produces CSV files with key rank results at different attack trace counts (100, 500, 1000, 2000, 5000, 10000).

**Key files for paper**:
- `experiment_results/aggregated_statistics.csv` - Few-shot Mean ± Std
- `baseline_results/baseline_aggregated_statistics.csv` - Baseline Mean ± Std
- `figures/*.pdf` - Publication-quality figures

## Methods

### Few-Shot Meta-Learning

**MAML (Model-Agnostic Meta-Learning)**:
- Inner-outer loop optimization
- 500 epochs with early stopping (patience=50)
- Fast adaptation to new support sets

**Prototypical Networks**:
- Metric learning approach
- Classification via distance to class prototypes
- 300 epochs with early stopping

**Siamese Networks**:
- Triplet loss for embedding learning
- Similarity-based classification
- 300 epochs with early stopping

### Baseline

**Standard CNN**:
- Same architecture as meta-learning methods
- Standard supervised learning with cross-entropy loss
- Trained with 1,280 / 2,560 / 3,840 / 5,120 traces for fair comparison

### CNN Architecture

```
Conv1D(64) → BatchNorm → AvgPool(2) →
Conv1D(128) → BatchNorm → AvgPool(2) →
Conv1D(256) → BatchNorm → GlobalAvgPool →
Fully Connected(256)
```

## Experimental Setup

### Few-Shot Learning

- **N-way**: 256 classes (all S-box outputs)
- **K-shot**: 5, 10, 15, 20 examples per class
- **Support set sizes**: 1,280 / 2,560 / 3,840 / 5,120 samples
- **Attack traces**: 100 to 10,000

### Reproducibility

- **10 independent runs** with fixed seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- **Deterministic CUDA operations** for GPU reproducibility
- **Stratified sampling** ensures balanced class representation
- **Statistical analysis** with 95% confidence intervals

## Results

Results are reported as **Mean ± Std** across 10 runs.

**Example at 1000 attack traces** (your actual results will vary):

| Method | 5-shot | 10-shot | 15-shot | 20-shot |
|--------|--------|---------|---------|---------|
| MAML | X ± Y | X ± Y | X ± Y | X ± Y |
| ProtoNet | X ± Y | X ± Y | X ± Y | X ± Y |
| Siamese | X ± Y | X ± Y | X ± Y | X ± Y |
| Baseline CNN | X ± Y | X ± Y | X ± Y | X ± Y |

*Lower rank is better (0 = correct key at top)*

## Documentation

- `GIT_SETUP_GUIDE.md` - How to push to GitHub
- `NEW_PAPER_DIRECTION.md` - Paper structure and writing guide
- `STRATIFIED_SAMPLING_REFACTOR.md` - Technical details of sampling approach
- `BASELINE_COMPARISON.md` - Baseline experiment details
- `REPRODUCIBILITY.md` - Seeding and reproducibility details

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024fewshot,
  title={Few-Shot Meta-Learning for Side-Channel Analysis of Wearable IoT Devices},
  author={Your Name and Co-authors},
  journal={Venue},
  year={2024}
}
```

## License

[Add your license here - e.g., MIT, Apache 2.0, or "Code available upon paper acceptance"]

## Contact

[Your Name] - [your.email@example.com]

Project Link: [https://github.com/DSpike/few-shot-sca](https://github.com/DSpike/few-shot-sca)

## Acknowledgments

- ANSSI for the ASCAD dataset
- [Add funding acknowledgments]
- [Add institutional acknowledgments]
