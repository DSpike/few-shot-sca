"""
CHES CTF 2018 - Baseline: Template Attack with Hamming Weight Leakage Model
=============================================================================
Same Template Attack (Chari et al., 2002) with HW model applied to CHES CTF.

Key differences from ASCAD hw_template_attack_baseline.py:
- Device: STM32 (32-bit ARM) vs ATMega8515 (8-bit AVR)
- Trace length: 2200 samples (vs 700)
- No masks: y = HW(Sbox(pt XOR key))
- 45,000 profiling + 5,000 attack traces
"""

import h5py
import numpy as np
import pandas as pd
import time
import argparse
import random
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
parser.add_argument('--n_components', type=int, default=8,
                   help='Number of PCA components (default: 8, max N_CLASSES-1)')
parser.add_argument('--pooled_cov', action='store_true', default=True,
                   help='Use pooled covariance matrix')
parser.add_argument('--byte', type=int, default=0,
                   help='Target key byte (default: 0 for CHES CTF)')
parser.add_argument('--n_poi', type=int, default=200,
                   help='Number of Points of Interest to select via SNR (default: 200)')
args = parser.parse_args()

# Set random seeds
print(f"Setting random seed: {args.seed}")
random.seed(args.seed)
np.random.seed(args.seed)

# =============================================================================
# Constants
# =============================================================================
N_CLASSES = 9  # HW classes 0-8

# S-box
sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

# Hamming Weight lookup table
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def compute_snr(traces, labels, n_classes=9):
    """
    Compute Signal-to-Noise Ratio per time sample.
    SNR(t) = Var_Y(E[X_t | Y]) / E_Y(Var[X_t | Y])
    Reference: Mangard et al., "Power Analysis Attacks", Springer 2007
    """
    n_samples = traces.shape[1]
    class_means = np.zeros((n_classes, n_samples))
    class_vars = np.zeros((n_classes, n_samples))
    class_counts = np.zeros(n_classes)

    for c in range(n_classes):
        mask = labels == c
        count = mask.sum()
        if count > 1:
            class_means[c] = traces[mask].mean(axis=0)
            class_vars[c] = traces[mask].var(axis=0)
            class_counts[c] = count
        elif count == 1:
            class_means[c] = traces[mask][0]
            class_vars[c] = 0.0
            class_counts[c] = count

    weights = class_counts / class_counts.sum()
    signal = np.average((class_means - np.average(class_means, axis=0, weights=weights))**2,
                        axis=0, weights=weights)
    noise = np.average(class_vars, axis=0, weights=weights)
    noise[noise == 0] = 1e-10

    return signal / noise


def select_poi(traces_prof, traces_attack, labels, n_poi=200, n_classes=9):
    """Select top-N Points of Interest based on SNR computed from profiling traces."""
    print(f"\n  Computing SNR for POI selection...")
    snr = compute_snr(traces_prof, labels, n_classes)

    poi_indices = np.argsort(snr)[-n_poi:]
    poi_indices = np.sort(poi_indices)

    max_snr = snr[poi_indices].max()
    mean_snr = snr[poi_indices].mean()
    print(f"  SNR: max={max_snr:.4f}, mean(top-{n_poi})={mean_snr:.4f}")
    print(f"  POI range: [{poi_indices[0]}, {poi_indices[-1]}] (span: {poi_indices[-1]-poi_indices[0]+1} samples)")
    print(f"  Trace reduction: {traces_prof.shape[1]} -> {n_poi} samples ({n_poi/traces_prof.shape[1]*100:.1f}%)")

    return traces_prof[:, poi_indices], traces_attack[:, poi_indices], poi_indices


# =============================================================================
# 1. Load CHES CTF Dataset (HW labels, NO masks)
# =============================================================================
target_byte = args.byte
print(f"Loading CHES CTF dataset (target byte: {target_byte}, leakage model: HW)...")
file_path = r'C:\Users\Dspike\Documents\sca\datasets\ches_ctf.h5'

with h5py.File(file_path, 'r') as f:
    # CHES CTF flat structure: profiling_traces (45000,2200), profiling_data (45000,48)
    # profiling_data columns: [0:16]=plaintext, [16:32]=ciphertext, [32:48]=key
    X_prof = np.array(f['profiling_traces'], dtype=np.float32)
    prof_data = np.array(f['profiling_data'], dtype=np.uint8)
    X_attack = np.array(f['attacking_traces'], dtype=np.float32)
    atk_data = np.array(f['attacking_data'], dtype=np.uint8)

    plaintext_prof = prof_data[:, target_byte]           # plaintext byte
    key_byte = prof_data[0, 32 + target_byte]            # key byte (constant)
    # CHES CTF: no masks -> y = HW(Sbox(pt XOR key))
    y_prof = HW[sbox[plaintext_prof ^ key_byte]]

    plaintext_attack = atk_data[:, target_byte]           # plaintext byte
    correct_key = atk_data[0, 32 + target_byte]           # attacking key byte

print(f"Loaded: {X_prof.shape}, Correct key byte[{target_byte}]: {hex(correct_key)}")
print(f"Raw trace length: {X_prof.shape[1]}")
print(f"Leakage model: Hamming Weight ({N_CLASSES} classes)")

# Print HW class distribution
unique, counts = np.unique(y_prof, return_counts=True)
for hw_val, cnt in zip(unique, counts):
    print(f"  HW={hw_val}: {cnt} samples ({cnt/len(y_prof)*100:.1f}%)")

# SNR-based POI selection (reduces 2200 -> n_poi informative samples)
X_prof, X_attack, poi_indices = select_poi(X_prof, X_attack, y_prof,
                                            n_poi=args.n_poi, n_classes=N_CLASSES)
TRACE_LENGTH = X_prof.shape[1]
print(f"Trace length after POI selection: {TRACE_LENGTH}")

# Normalize
mean = X_prof.mean(axis=0)
std_val = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std_val
X_attack = (X_attack - mean) / std_val
print(f"Normalized\n")

# =============================================================================
# 2. Template Attack Implementation (HW model)
# =============================================================================
class TemplateAttackHW:
    """
    Template Attack with Hamming Weight leakage model.
    Builds 9 templates (HW 0-8) instead of 256.
    """

    def __init__(self, n_components=8, pooled_cov=True):
        self.n_components = n_components
        self.pooled_cov = pooled_cov
        self.pca = None
        self.templates = {}
        self.pooled_cov_matrix = None

    def fit(self, X_train, y_train):
        """Build templates for 9 HW classes."""
        print(f"  Building HW templates with {len(X_train)} traces...")
        start_time = time.time()

        # PCA dimensionality reduction
        n_comp = min(self.n_components, X_train.shape[0] - 1, X_train.shape[1])
        n_comp = max(n_comp, 2)
        print(f"    PCA: {X_train.shape[1]} -> {n_comp} dimensions")
        self.pca = PCA(n_components=n_comp)
        X_reduced = self.pca.fit_transform(X_train)
        self.n_components = n_comp

        # Build templates for each HW class
        classes_found = 0
        all_class_data = {}

        for cls in range(N_CLASSES):  # 0-8
            mask = (y_train == cls)
            n_cls = mask.sum()

            if n_cls >= 2:
                X_cls = X_reduced[mask]
                all_class_data[cls] = X_cls
                self.templates[cls] = {
                    'mean': X_cls.mean(axis=0),
                    'n_samples': n_cls
                }
                classes_found += 1
            elif n_cls == 1:
                X_cls = X_reduced[mask]
                all_class_data[cls] = X_cls
                self.templates[cls] = {
                    'mean': X_cls[0],
                    'n_samples': n_cls
                }
                classes_found += 1
            else:
                self.templates[cls] = {
                    'mean': X_reduced.mean(axis=0),
                    'n_samples': 0
                }

        # Compute covariance
        if self.pooled_cov:
            residuals = []
            for cls, X_cls in all_class_data.items():
                if len(X_cls) >= 2:
                    cls_mean = self.templates[cls]['mean']
                    residuals.append(X_cls - cls_mean)

            if residuals:
                all_residuals = np.vstack(residuals)
                self.pooled_cov_matrix = np.cov(all_residuals, rowvar=False)
                self.pooled_cov_matrix += np.eye(self.n_components) * 1e-6
            else:
                self.pooled_cov_matrix = np.eye(self.n_components)

            for cls in range(N_CLASSES):
                self.templates[cls]['cov'] = self.pooled_cov_matrix
        else:
            for cls in range(N_CLASSES):
                if cls in all_class_data and len(all_class_data[cls]) > self.n_components:
                    cov = np.cov(all_class_data[cls], rowvar=False)
                    cov += np.eye(self.n_components) * 1e-6
                    self.templates[cls] = {**self.templates[cls], 'cov': cov}
                else:
                    if self.pooled_cov_matrix is None:
                        residuals = []
                        for c, X_c in all_class_data.items():
                            if len(X_c) >= 2:
                                residuals.append(X_c - self.templates[c]['mean'])
                        all_residuals = np.vstack(residuals) if residuals else np.zeros((1, self.n_components))
                        self.pooled_cov_matrix = np.cov(all_residuals, rowvar=False)
                        self.pooled_cov_matrix += np.eye(self.n_components) * 1e-6
                    self.templates[cls] = {**self.templates[cls], 'cov': self.pooled_cov_matrix}

        elapsed = time.time() - start_time
        print(f"    Templates built: {classes_found}/{N_CLASSES} HW classes ({elapsed:.1f}s)")
        return self

    def predict_log_proba(self, X_attack_traces):
        """Compute log-probabilities for each HW class."""
        X_reduced = self.pca.transform(X_attack_traces)

        n_traces = len(X_reduced)
        log_probs = np.zeros((n_traces, N_CLASSES))

        for cls in range(N_CLASSES):
            template_mean = self.templates[cls]['mean']
            template_cov = self.templates[cls]['cov']

            try:
                rv = multivariate_normal(mean=template_mean, cov=template_cov,
                                        allow_singular=True)
                log_probs[:, cls] = rv.logpdf(X_reduced)
            except Exception:
                diff = X_reduced - template_mean
                log_probs[:, cls] = -0.5 * np.sum(diff ** 2, axis=1)

        return log_probs


def evaluate_template_attack(ta, X_attack_traces, plaintext_attack, correct_key):
    """Evaluate key recovery using HW template attack (CHES CTF: no masks)."""
    print("  Computing log-probabilities on attack traces...")
    start_time = time.time()

    log_probs = ta.predict_log_proba(X_attack_traces)  # shape: (n_traces, 9)

    elapsed = time.time() - start_time
    print(f"    Log-probs computed ({elapsed:.1f}s)")

    # Key ranking: 256 candidates, map through HW (NO masks for CHES CTF)
    traces_list = [100, 500, 1000, 2000, 5000]
    ranks = []

    for n in traces_list:
        if n > len(log_probs):
            break
        scores = np.zeros(256)
        for k in range(256):
            intermediate_hw = HW[sbox[plaintext_attack[:n] ^ k]]  # No masks!
            scores[k] = np.sum(log_probs[np.arange(n), intermediate_hw])
        rank = np.argsort(-scores).tolist().index(correct_key)
        ranks.append(rank)

    return traces_list[:len(ranks)], ranks


# =============================================================================
# 3. Main Experiment
# =============================================================================
print("=" * 70)
print("CHES CTF 2018 - BASELINE: TEMPLATE ATTACK (HW Leakage Model)")
print("=" * 70)
print(f"Dataset: CHES CTF 2018 (STM32, masked AES)")
print(f"Target byte: {target_byte}")
print(f"Trace length: {TRACE_LENGTH}")
print()

# Training sizes: same k-shot values as few-shot, plus ID-equivalent sizes
training_sizes = {
    '5-shot (HW)': 5 * N_CLASSES,          # 45 traces
    '10-shot (HW)': 10 * N_CLASSES,         # 90 traces
    '15-shot (HW)': 15 * N_CLASSES,         # 135 traces
    '20-shot (HW)': 20 * N_CLASSES,         # 180 traces
    'ID-5-shot equiv': 5 * 256,             # 1280 traces
    'ID-10-shot equiv': 10 * 256,           # 2560 traces
    'ID-15-shot equiv': 15 * 256,           # 3840 traces
    'ID-20-shot equiv': 20 * 256,           # 5120 traces
    'Full training set': len(X_prof)        # 45,000 traces
}

results = []

for name, n_samples in training_sizes.items():
    print(f"\n{'=' * 70}")
    print(f"HW Template Attack with {n_samples} profiling traces ({name})")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    # Stratified sampling over HW classes
    if n_samples < len(X_prof):
        indices = []
        samples_per_class = n_samples // N_CLASSES
        for cls in range(N_CLASSES):
            class_indices = np.where(y_prof == cls)[0]
            if len(class_indices) >= samples_per_class:
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
            else:
                indices.extend(class_indices)
        indices = np.array(indices)
        X_train_subset = X_prof[indices]
        y_train_subset = y_prof[indices]
    else:
        X_train_subset = X_prof
        y_train_subset = y_prof

    # PCA components: at most N_CLASSES-1 for small datasets
    n_components = min(args.n_components, len(X_train_subset) // N_CLASSES - 1, X_train_subset.shape[1])
    n_components = max(n_components, 2)

    print(f"  PCA components: {n_components}")
    print(f"  Profiling traces: {len(X_train_subset)}")
    print(f"  Samples per HW class: ~{len(X_train_subset) // N_CLASSES}")

    # Build templates
    ta = TemplateAttackHW(n_components=n_components, pooled_cov=args.pooled_cov)
    ta.fit(X_train_subset, y_train_subset)

    elapsed_build = time.time() - start_time
    print(f"  Template build time: {elapsed_build:.1f}s")

    # Evaluate (CHES CTF: no masks)
    traces_list, ranks = evaluate_template_attack(
        ta, X_attack, plaintext_attack, correct_key
    )

    for n_traces, rank in zip(traces_list, ranks):
        results.append({
            'Method': 'Template Attack (HW)',
            'Training_Size': n_samples,
            'Training_Config': name,
            'Attack_Traces': n_traces,
            'Key_Rank': rank,
            'PCA_Components': n_components,
            'Leakage_Model': 'HW',
            'Dataset': 'CHES_CTF'
        })

    elapsed_total = time.time() - start_time
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Results: {dict(zip(traces_list, ranks))}")

# =============================================================================
# 4. Save Results
# =============================================================================
df = pd.DataFrame(results)
df.to_csv('ches_ctf_hw_template_attack_results.csv', index=False)

print("\n" + "=" * 70)
print("CHES CTF HW TEMPLATE ATTACK RESULTS SUMMARY")
print("=" * 70)

print("\nKey Rank at 1000 Attack Traces:")
pivot = df[df['Attack_Traces'] == 1000][['Training_Config', 'Key_Rank']]
for _, row in pivot.iterrows():
    print(f"  {row['Training_Config']:25s} -> Key Rank {int(row['Key_Rank']):3d}")

print(f"\nResults saved to: ches_ctf_hw_template_attack_results.csv")
