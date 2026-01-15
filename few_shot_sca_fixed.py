"""
Few-Shot SCA - FIXED VERSION
=============================
Uses SNR for feature selection + better hyperparameters
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# IMPROVED Hyperparameters
K_SHOT = 10          # Start with 10-shot (easier to learn)
N_FEATURES = 500     # Fewer but better features
META_ITERS = 200     # More iterations
INNER_STEPS = 10     # More adaptation steps
INNER_LR = 0.01
META_LR = 0.001
ATTACK_BATCH_SIZE = 100

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

# IMPROVED: SNR-based feature selection
def compute_snr(X, y):
    """Compute Signal-to-Noise Ratio for each feature"""
    n_features = X.shape[1]
    snr = np.zeros(n_features)

    for i in range(n_features):
        # Mean of each class
        class_means = []
        class_vars = []

        for c in range(256):
            mask = (y == c)
            if mask.sum() > 0:
                class_means.append(np.mean(X[mask, i]))
                class_vars.append(np.var(X[mask, i]))

        if len(class_means) > 1:
            # SNR = variance of means / mean of variances
            snr[i] = np.var(class_means) / (np.mean(class_vars) + 1e-12)

    return snr

print("Loading ASCAD dataset...")
file_path = r'C:\Users\Dspike\Documents\sca\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5'

with h5py.File(file_path, 'r') as f:
    X_prof = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    metadata_prof = f['Profiling_traces/metadata']
    X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
    metadata_attack = f['Attack_traces/metadata']

    plaintext_prof = np.array([m['plaintext'][2] for m in metadata_prof])
    masks_prof = np.array([m['masks'][2] for m in metadata_prof])
    key_byte = metadata_prof[0]['key'][2]
    y_prof = sbox[plaintext_prof ^ key_byte ^ masks_prof]

    plaintext_attack = np.array([m['plaintext'][2] for m in metadata_attack])
    masks_attack = np.array([m['masks'][2] for m in metadata_attack])
    correct_key = metadata_attack[0]['key'][2]

print(f"✓ Loaded: {X_prof.shape[0]} profiling, {X_attack.shape[0]} attack traces")
print(f"✓ Correct key: {hex(correct_key)}")

# CRITICAL: Use SNR-based feature selection
print(f"\nComputing SNR for all {X_prof.shape[1]} features...")
snr_scores = compute_snr(X_prof, y_prof)
print(f"  SNR range: {snr_scores.min():.4f} to {snr_scores.max():.4f}")

# Select top N_FEATURES by SNR
top_idx = np.argsort(snr_scores)[-N_FEATURES:]
selected_points = np.sort(top_idx)  # Keep time order

print(f"✓ Selected top {N_FEATURES} features by SNR")
print(f"  Selected SNR range: {snr_scores[selected_points].min():.4f} to {snr_scores[selected_points].max():.4f}")

X_prof = X_prof[:, selected_points]
X_attack = X_attack[:, selected_points]

# IMPROVED: Better normalization (global, not per-trace)
mean = X_prof.mean(axis=0)
std = X_prof.std(axis=0) + 1e-8
X_prof = (X_prof - mean) / std
X_attack = (X_attack - mean) / std

print(f"✓ Normalized features")

# Split
split = int(0.8 * len(X_prof))
X_train, y_train = X_prof[:split], y_prof[:split]
X_test, y_test = X_prof[split:], y_prof[split:]

print(f"✓ Meta-train: {len(X_train)}, Meta-test: {len(X_test)}\n")

# IMPROVED: Slightly larger model
class ImprovedCNN(nn.Module):
    def __init__(self):
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
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Task sampler
class TaskSampler:
    def __init__(self, X, y, k_shot=10):
        self.X = X
        self.y = y
        self.k_shot = k_shot

        self.class_idx = defaultdict(list)
        for i, label in enumerate(y):
            self.class_idx[label].append(i)

        self.valid_classes = [c for c in range(256)
                             if len(self.class_idx[c]) >= k_shot + 3]
        print(f"  Valid classes: {len(self.valid_classes)}/256")

    def sample(self):
        sup_x, sup_y, qry_x, qry_y = [], [], [], []

        for cls in self.valid_classes:
            idx = self.class_idx[cls]
            sel = np.random.choice(idx, self.k_shot + 3, replace=False)

            sup_x.append(self.X[sel[:self.k_shot]])
            sup_y.append([cls] * self.k_shot)
            qry_x.append(self.X[sel[self.k_shot:]])
            qry_y.append([cls] * 3)

        return (torch.FloatTensor(np.vstack(sup_x)),
                torch.LongTensor(np.hstack(sup_y)),
                torch.FloatTensor(np.vstack(qry_x)),
                torch.LongTensor(np.hstack(qry_y)))

# MAML
def maml_inner_loop(model, sup_x, sup_y, steps=10, lr=0.01):
    adapted = ImprovedCNN().to(device)
    adapted.load_state_dict(model.state_dict())
    optimizer = optim.SGD(adapted.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(adapted(sup_x), sup_y)
        loss.backward()
        optimizer.step()

    return adapted

def maml_train_step(model, meta_opt, sampler):
    meta_opt.zero_grad()
    total_loss = 0

    for _ in range(4):
        sup_x, sup_y, qry_x, qry_y = sampler.sample()
        sup_x, sup_y = sup_x.to(device), sup_y.to(device)
        qry_x, qry_y = qry_x.to(device), qry_y.to(device)

        adapted = maml_inner_loop(model, sup_x, sup_y, steps=INNER_STEPS, lr=INNER_LR)
        qry_loss = F.cross_entropy(adapted(qry_x), qry_y)
        total_loss += qry_loss

    total_loss = total_loss / 4
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    meta_opt.step()

    return total_loss.item()

# Train MAML
print("="*60)
print(f"Training MAML ({K_SHOT}-shot, {META_ITERS} iterations)")
print("="*60)

sampler = TaskSampler(X_train, y_train, k_shot=K_SHOT)
model = ImprovedCNN().to(device)
meta_optimizer = optim.Adam(model.parameters(), lr=META_LR)

losses = []
for i in range(META_ITERS):
    loss = maml_train_step(model, meta_optimizer, sampler)
    losses.append(loss)

    if (i + 1) % 20 == 0:
        print(f"  Iter {i+1}/{META_ITERS}: Loss = {loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Meta-iteration')
plt.ylabel('Meta-loss')
plt.title('MAML Training')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('maml_training_fixed.png', dpi=100)
print("\n✓ Saved: maml_training_fixed.png")

# Evaluate
print("\n" + "="*60)
print("Attack Evaluation")
print("="*60)

test_sampler = TaskSampler(X_test, y_test, k_shot=K_SHOT)
sup_x, sup_y, _, _ = test_sampler.sample()

print(f"Support set: {len(sup_y)} examples ({K_SHOT} per class)")
print("Adapting model to support set...")
adapted_model = maml_inner_loop(model, sup_x.to(device), sup_y.to(device),
                               steps=INNER_STEPS, lr=INNER_LR)

adapted_model.eval()

all_probs = []
n_batches = (len(X_attack) + ATTACK_BATCH_SIZE - 1) // ATTACK_BATCH_SIZE

print("Running attack...")
with torch.no_grad():
    for batch_idx in range(n_batches):
        start_idx = batch_idx * ATTACK_BATCH_SIZE
        end_idx = min(start_idx + ATTACK_BATCH_SIZE, len(X_attack))

        batch_traces = torch.FloatTensor(X_attack[start_idx:end_idx]).to(device)
        batch_logits = adapted_model(batch_traces)
        batch_probs = F.softmax(batch_logits, dim=1).cpu().numpy()
        all_probs.append(batch_probs)

        del batch_traces, batch_logits
        torch.cuda.empty_cache()

probs = np.vstack(all_probs)
log_probs = np.log(probs + 1e-40)

# Key ranking
traces_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ranks = []

for n_traces in traces_list:
    if n_traces > len(log_probs):
        break

    scores = np.zeros(256)
    for key_guess in range(256):
        intermediate = sbox[plaintext_attack[:n_traces] ^ key_guess ^ masks_attack[:n_traces]]
        scores[key_guess] = np.sum(log_probs[np.arange(n_traces), intermediate])

    rank = np.argsort(-scores).tolist().index(correct_key)
    ranks.append(rank)
    status = "✓ SUCCESS!" if rank == 0 else ("✓ GOOD!" if rank < 10 else "")
    print(f"  {n_traces:5d} traces → rank {rank:3d}  {status}")

plt.figure(figsize=(8, 5))
plt.semilogx(traces_list[:len(ranks)], ranks, 'o-', linewidth=2, markersize=8, color='darkblue')
plt.xlabel('Number of attack traces')
plt.ylabel('Key rank')
plt.title(f'MAML Attack Results ({K_SHOT}-shot, SNR features)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('attack_results_fixed.png', dpi=100)
print("\n✓ Saved: attack_results_fixed.png")

print(f"\n{'='*60}")
print(f"Best rank: {min(ranks)}")
print(f"Final rank: {ranks[-1]}")
if min(ranks) < 10:
    print("✓ EXCELLENT! This is publication-worthy!")
elif min(ranks) < 50:
    print("✓ GOOD! Getting close to publication quality.")
else:
    print("  Still learning. Try increasing META_ITERS to 500-1000")
print('='*60)
