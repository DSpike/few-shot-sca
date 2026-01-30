"""
Baseline: Classical Template Attack (Gaussian Template Analysis)
================================================================
Implements the standard Template Attack from Chari et al. (2002)
This is the gold-standard classical SCA method for the low-data regime.

Template Attack builds multivariate Gaussian templates for each key hypothesis
during profiling, then uses maximum likelihood to classify attack traces.

Reference:
    Chari, S., Rao, J. R., & Rohatgi, P. (2002).
    Template attacks. CHES 2002.
"""

import h5py
import numpy as np
import pandas as pd
import time
import argparse
import random
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
parser.add_argument('--n_components', type=int, default=20,
                   help='Number of PCA components for dimensionality reduction (default: 20)')
parser.add_argument('--pooled_cov', action='store_true', default=True,
                   help='Use pooled covariance matrix (more robust with few samples)')
args = parser.parse_args()

# Set random seeds
print(f"Setting random seed: {args.seed}")
random.seed(args.seed)
np.random.seed(args.seed)

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

# =============================================================================
# 1. Load ASCAD Dataset
# =============================================================================
print("Loading ASCAD dataset...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_attack = f['Attack_traces/metadata']

    # Use byte 3 (same as few-shot experiments)
    plaintext_prof = np.array([m['plaintext'][3] for m in metadata_prof])
    masks_prof = np.array([m['masks'][3] for m in metadata_prof])
    key_byte = metadata_prof[0]['key'][3]
    y_prof = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m['plaintext'][3] for m in metadata_attack])
    masks_attack = np.array([m['masks'][3] for m in metadata_attack])
    correct_key = metadata_attack[0]['key'][3]

print(f"Loaded: {X_prof.shape}, Correct key: {hex(correct_key)}")

# Normalize (same as few-shot experiments)
mean = X_prof.mean(axis=0)
std_val = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std_val
X_attack = (X_attack - mean) / std_val

print(f"Normalized\n")

# =============================================================================
# 2. Template Attack Implementation
# =============================================================================
class TemplateAttack:
    """
    Classical Template Attack (Chari et al., 2002)

    Profiling phase:
        - For each class (S-box output value 0-255), compute:
          - Mean trace (template center)
          - Covariance matrix (template spread)
        - Optionally use PCA for dimensionality reduction

    Attack phase:
        - For each attack trace and key hypothesis:
          - Compute the intermediate value (S-box output)
          - Evaluate the log-likelihood under the corresponding template
          - Accumulate log-likelihoods across traces
        - The key with the highest cumulative log-likelihood wins
    """

    def __init__(self, n_components=20, pooled_cov=True):
        self.n_components = n_components
        self.pooled_cov = pooled_cov
        self.pca = None
        self.templates = {}  # {class_label: {'mean': ..., 'cov': ...}}
        self.pooled_cov_matrix = None

    def fit(self, X_train, y_train):
        """
        Build templates from profiling traces.

        Args:
            X_train: Profiling traces (n_samples, n_features)
            y_train: Labels (S-box output values, 0-255)
        """
        print(f"  Building templates with {len(X_train)} traces...")
        start_time = time.time()

        # Step 1: PCA dimensionality reduction
        print(f"    PCA: {X_train.shape[1]} -> {self.n_components} dimensions")
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X_train)

        # Step 2: Build templates for each class
        classes_found = 0
        all_class_data = {}

        for cls in range(256):
            mask = (y_train == cls)
            n_cls = mask.sum()

            if n_cls >= 2:  # Need at least 2 samples for covariance
                X_cls = X_reduced[mask]
                all_class_data[cls] = X_cls
                self.templates[cls] = {
                    'mean': X_cls.mean(axis=0),
                    'n_samples': n_cls
                }
                classes_found += 1
            elif n_cls == 1:
                # Single sample: use just the mean, will use pooled covariance
                X_cls = X_reduced[mask]
                all_class_data[cls] = X_cls
                self.templates[cls] = {
                    'mean': X_cls[0],
                    'n_samples': n_cls
                }
                classes_found += 1
            else:
                # No samples for this class: use global mean
                self.templates[cls] = {
                    'mean': X_reduced.mean(axis=0),
                    'n_samples': 0
                }

        # Step 3: Compute covariance
        if self.pooled_cov:
            # Pooled covariance: single covariance across all classes
            # More robust when few samples per class (typical in SCA)
            residuals = []
            for cls, X_cls in all_class_data.items():
                if len(X_cls) >= 2:
                    cls_mean = self.templates[cls]['mean']
                    residuals.append(X_cls - cls_mean)

            if residuals:
                all_residuals = np.vstack(residuals)
                self.pooled_cov_matrix = np.cov(all_residuals, rowvar=False)
                # Add small regularization for numerical stability
                self.pooled_cov_matrix += np.eye(self.n_components) * 1e-6
            else:
                self.pooled_cov_matrix = np.eye(self.n_components)

            for cls in range(256):
                self.templates[cls]['cov'] = self.pooled_cov_matrix
        else:
            # Per-class covariance (requires more data per class)
            for cls in range(256):
                if cls in all_class_data and len(all_class_data[cls]) > self.n_components:
                    cov = np.cov(all_class_data[cls], rowvar=False)
                    cov += np.eye(self.n_components) * 1e-6
                    self.templates[cls] = {**self.templates[cls], 'cov': cov}
                else:
                    # Fall back to pooled covariance
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
        print(f"    Templates built: {classes_found}/256 classes ({elapsed:.1f}s)")

        return self

    def predict_log_proba(self, X_attack_traces):
        """
        Compute log-probabilities for each class on attack traces.

        Args:
            X_attack_traces: Attack traces (n_traces, n_features)

        Returns:
            log_probs: (n_traces, 256) log-likelihood for each class
        """
        # Apply same PCA transformation
        X_reduced = self.pca.transform(X_attack_traces)

        n_traces = len(X_reduced)
        log_probs = np.zeros((n_traces, 256))

        for cls in range(256):
            template_mean = self.templates[cls]['mean']
            template_cov = self.templates[cls]['cov']

            try:
                rv = multivariate_normal(mean=template_mean, cov=template_cov,
                                        allow_singular=True)
                log_probs[:, cls] = rv.logpdf(X_reduced)
            except Exception:
                # Fallback: use Euclidean distance
                diff = X_reduced - template_mean
                log_probs[:, cls] = -0.5 * np.sum(diff ** 2, axis=1)

        return log_probs


def evaluate_template_attack(ta, X_attack_traces, plaintext_attack, masks_attack, correct_key):
    """Evaluate key recovery using template attack log-probabilities."""
    print("  Computing log-probabilities on attack traces...")
    start_time = time.time()

    log_probs = ta.predict_log_proba(X_attack_traces)

    elapsed = time.time() - start_time
    print(f"    Log-probs computed ({elapsed:.1f}s)")

    # Key ranking (same as few-shot evaluation)
    traces_list = [100, 500, 1000, 2000, 5000, 10000]
    ranks = []

    for n in traces_list:
        if n > len(log_probs):
            break
        scores = np.zeros(256)
        for k in range(256):
            intermediate = sbox[plaintext_attack[:n] ^ k ^ masks_attack[:n]]
            scores[k] = np.sum(log_probs[np.arange(n), intermediate])
        rank = np.argsort(-scores).tolist().index(correct_key)
        ranks.append(rank)

    return traces_list[:len(ranks)], ranks


# =============================================================================
# 3. Main Experiment
# =============================================================================
print("=" * 70)
print("BASELINE: CLASSICAL TEMPLATE ATTACK (Gaussian Template Analysis)")
print("=" * 70)

# Test with same training sizes as few-shot and CNN baseline
training_sizes = {
    '5-shot equivalent': 5 * 256,     # 1,280 traces
    '10-shot equivalent': 10 * 256,   # 2,560 traces
    '15-shot equivalent': 15 * 256,   # 3,840 traces
    '20-shot equivalent': 20 * 256,   # 5,120 traces
    'Full training set': len(X_prof)  # 50,000 traces
}

results = []

for name, n_samples in training_sizes.items():
    print(f"\n{'=' * 70}")
    print(f"Template Attack with {n_samples} profiling traces ({name})")
    print(f"{'=' * 70}\n")

    start_time = time.time()

    # Stratified sampling (same as other baselines)
    if n_samples < len(X_prof):
        indices = []
        samples_per_class = n_samples // 256
        for cls in range(256):
            class_indices = np.where(y_prof == cls)[0]
            if len(class_indices) >= samples_per_class:
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                indices.extend(selected)
        indices = np.array(indices)
        X_train_subset = X_prof[indices]
        y_train_subset = y_prof[indices]
    else:
        X_train_subset = X_prof
        y_train_subset = y_prof

    # Adjust PCA components based on available data
    n_components = min(args.n_components, n_samples // 256 - 1, X_train_subset.shape[1])
    n_components = max(n_components, 5)  # Minimum 5 components

    print(f"  PCA components: {n_components}")
    print(f"  Profiling traces: {len(X_train_subset)}")
    print(f"  Samples per class: {n_samples // 256}")

    # Build templates
    ta = TemplateAttack(n_components=n_components, pooled_cov=args.pooled_cov)
    ta.fit(X_train_subset, y_train_subset)

    elapsed_build = time.time() - start_time
    print(f"  Template build time: {elapsed_build:.1f}s")

    # Evaluate attack
    traces_list, ranks = evaluate_template_attack(
        ta, X_attack, plaintext_attack, masks_attack, correct_key
    )

    for n_traces, rank in zip(traces_list, ranks):
        results.append({
            'Method': 'Template Attack',
            'Training_Size': n_samples,
            'Training_Config': name,
            'Attack_Traces': n_traces,
            'Key_Rank': rank,
            'PCA_Components': n_components
        })

    elapsed_total = time.time() - start_time
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Results: {dict(zip(traces_list, ranks))}")

# =============================================================================
# 4. Save Results
# =============================================================================
df = pd.DataFrame(results)
df.to_csv('template_attack_results.csv', index=False)

print("\n" + "=" * 70)
print("TEMPLATE ATTACK RESULTS SUMMARY")
print("=" * 70)

print("\nKey Rank at 1000 Attack Traces:")
pivot = df[df['Attack_Traces'] == 1000].pivot(
    index='Training_Config',
    columns='Method',
    values='Key_Rank'
)
print(pivot)

print("\nKey Rank across all trace counts:")
for name in training_sizes:
    config_data = df[df['Training_Config'] == name]
    print(f"\n  {name}:")
    for _, row in config_data.iterrows():
        print(f"    {int(row['Attack_Traces']):>6d} traces -> Key Rank {int(row['Key_Rank']):3d}")

print(f"\nResults saved to: template_attack_results.csv")
