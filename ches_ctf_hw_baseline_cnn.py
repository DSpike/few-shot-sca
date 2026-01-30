"""
CHES CTF 2018 - Baseline: Standard CNN Training with Hamming Weight Leakage Model
==================================================================================
Same standard CNN baseline with HW model applied to CHES CTF.

Key differences from ASCAD hw_baseline_standard_training.py:
- Device: STM32 (32-bit ARM) vs ATMega8515 (8-bit AVR)
- Trace length: 2200 samples (vs 700)
- No masks: y = HW(Sbox(pt XOR key))
- 45,000 profiling + 5,000 attack traces
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time
import argparse
import random

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
parser.add_argument('--byte', type=int, default=0,
                   help='Target key byte (default: 0 for CHES CTF)')
parser.add_argument('--n_poi', type=int, default=200,
                   help='Number of Points of Interest to select via SNR (default: 200)')
args = parser.parse_args()

# Set random seeds
print(f"Setting random seed: {args.seed}")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

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

# SNR-based POI selection (reduces 2200 -> n_poi informative samples)
X_prof, X_attack, poi_indices = select_poi(X_prof, X_attack, y_prof,
                                            n_poi=args.n_poi, n_classes=N_CLASSES)
TRACE_LENGTH = X_prof.shape[1]
print(f"Trace length after POI selection: {TRACE_LENGTH}")

# Normalize
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std
print(f"Normalized\n")

# =============================================================================
# 2. CNN Model (9-class output, adapted for CHES CTF trace length)
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_size=2200):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, N_CLASSES)  # 9 HW classes

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        features = self.pool(x).squeeze(-1)
        return self.fc(features)

# =============================================================================
# 3. Training
# =============================================================================
def train_standard_cnn(X_train, y_train, epochs=100, batch_size=128):
    """Standard supervised training with early stopping."""
    model = SimpleCNN(input_size=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best loss: {best_loss:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model

# =============================================================================
# 4. Attack Evaluation (CHES CTF: NO masks)
# =============================================================================
def evaluate_attack(model, X_attack, plaintext_attack, correct_key):
    """Evaluate key recovery attack with HW model (CHES CTF: no masks)."""
    model.eval()

    all_probs = []
    batch_size = 1000
    for i in range(0, len(X_attack), batch_size):
        batch = torch.FloatTensor(X_attack[i:i+batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    probs = np.vstack(all_probs)  # shape: (n_attack, 9)
    log_probs = np.log(probs + 1e-40)

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
# 5. Main Experiment
# =============================================================================
print("=" * 70)
print("CHES CTF 2018 - BASELINE: STANDARD CNN TRAINING (HW Leakage Model)")
print("=" * 70)
print(f"Dataset: CHES CTF 2018 (STM32, masked AES)")
print(f"Target byte: {target_byte}")
print(f"Trace length: {TRACE_LENGTH}")
print()

# Training sizes: HW k-shot equivalents + ID-equivalent sizes + full
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
    print(f"Training with {n_samples} samples ({name})")
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

    print(f"  Training CNN (HW, {N_CLASSES} classes)...")
    print(f"  Training samples: {len(X_train_subset)}")
    print(f"  Samples per HW class: ~{len(X_train_subset) // N_CLASSES}")
    model = train_standard_cnn(X_train_subset, y_train_subset, epochs=100)

    elapsed = time.time() - start_time
    print(f"  Training complete ({elapsed:.1f}s)")

    # Evaluate (CHES CTF: no masks)
    print(f"  Evaluating attack...")
    traces_list, ranks = evaluate_attack(model, X_attack, plaintext_attack, correct_key)

    for n_traces, rank in zip(traces_list, ranks):
        results.append({
            'Method': 'Standard CNN (HW)',
            'Training_Size': n_samples,
            'Training_Config': name,
            'Attack_Traces': n_traces,
            'Key_Rank': rank,
            'Leakage_Model': 'HW',
            'Dataset': 'CHES_CTF'
        })

    print(f"  Results: {dict(zip(traces_list, ranks))}")

# =============================================================================
# 6. Save Results
# =============================================================================
df = pd.DataFrame(results)
df.to_csv('ches_ctf_hw_baseline_standard_cnn_results.csv', index=False)

print("\n" + "=" * 70)
print("CHES CTF HW STANDARD CNN BASELINE RESULTS SUMMARY")
print("=" * 70)

print("\nKey Rank at 1000 Attack Traces:")
for _, row in df[df['Attack_Traces'] == 1000].iterrows():
    print(f"  {row['Training_Config']:25s} -> Key Rank {int(row['Key_Rank']):3d}")

print(f"\nResults saved to: ches_ctf_hw_baseline_standard_cnn_results.csv")
