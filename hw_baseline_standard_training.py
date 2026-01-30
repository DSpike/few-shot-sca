"""
Baseline: Standard CNN Training with Hamming Weight Leakage Model
=================================================================
Same standard CNN baseline but with HW model (9 classes).

Key difference from baseline_standard_training.py:
- Labels: y = HW(Sbox(pt XOR key XOR mask)) -> 9 classes
- CNN output: 9 classes instead of 256
- Attack: ranks 256 key candidates by mapping through HW
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

# =============================================================================
# 1. Load Data (HW labels)
# =============================================================================
print("Loading ASCAD dataset (HW leakage model)...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_attack = f['Attack_traces/metadata']

    plaintext_prof = np.array([m['plaintext'][3] for m in metadata_prof])
    masks_prof = np.array([m['masks'][3] for m in metadata_prof])
    key_byte = metadata_prof[0]['key'][3]
    # HW leakage model
    y_prof = HW[sbox[plaintext_prof ^ key_byte ^ masks_prof]]

    plaintext_attack = np.array([m['plaintext'][3] for m in metadata_attack])
    masks_attack = np.array([m['masks'][3] for m in metadata_attack])
    correct_key = metadata_attack[0]['key'][3]

print(f"Loaded: {X_prof.shape}, Correct key: {hex(correct_key)}")
print(f"Leakage model: Hamming Weight ({N_CLASSES} classes)")

# Normalize
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std
print(f"Normalized\n")

# =============================================================================
# 2. CNN Model (9-class output)
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_size=700):
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
# 4. Attack Evaluation (HW model)
# =============================================================================
def evaluate_attack(model, X_attack, plaintext_attack, masks_attack, correct_key):
    """Evaluate key recovery attack with HW model."""
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

    # Key ranking: 256 candidates, map through HW
    traces_list = [100, 500, 1000, 2000, 5000, 10000]
    ranks = []

    for n in traces_list:
        if n > len(log_probs):
            break
        scores = np.zeros(256)
        for k in range(256):
            intermediate_hw = HW[sbox[plaintext_attack[:n] ^ k ^ masks_attack[:n]]]
            scores[k] = np.sum(log_probs[np.arange(n), intermediate_hw])
        rank = np.argsort(-scores).tolist().index(correct_key)
        ranks.append(rank)

    return traces_list[:len(ranks)], ranks

# =============================================================================
# 5. Main Experiment
# =============================================================================
print("=" * 70)
print("BASELINE: STANDARD CNN TRAINING (HW Leakage Model)")
print("=" * 70)

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
    'Full training set': len(X_prof)        # 50,000 traces
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

    print(f"  Evaluating attack...")
    traces_list, ranks = evaluate_attack(model, X_attack, plaintext_attack, masks_attack, correct_key)

    for n_traces, rank in zip(traces_list, ranks):
        results.append({
            'Method': 'Standard CNN (HW)',
            'Training_Size': n_samples,
            'Training_Config': name,
            'Attack_Traces': n_traces,
            'Key_Rank': rank,
            'Leakage_Model': 'HW'
        })

    print(f"  Results: {dict(zip(traces_list, ranks))}")

# =============================================================================
# 6. Save Results
# =============================================================================
df = pd.DataFrame(results)
df.to_csv('hw_baseline_standard_cnn_results.csv', index=False)

print("\n" + "=" * 70)
print("HW STANDARD CNN BASELINE RESULTS SUMMARY")
print("=" * 70)

print("\nKey Rank at 1000 Attack Traces:")
for _, row in df[df['Attack_Traces'] == 1000].iterrows():
    print(f"  {row['Training_Config']:25s} -> Key Rank {int(row['Key_Rank']):3d}")

print(f"\nResults saved to: hw_baseline_standard_cnn_results.csv")
